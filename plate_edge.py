"""
PLATESCOPE EDGE — Live RTSP license plate service.

The camera-side counterpart to plate_scan.py. Where plate_scan.py reads a video
file end to end and votes once at the finish, this runs continuously against
entry/exit RTSP cameras, logs each vehicle locally, and syncs to a central
server — the same shape as the face/QR attendance edge service.

Detection, OCR, text cleaning and vote clustering are imported from
plate_scan.py, not reimplemented. That file is not modified.

  Adapting the vote to a live stream
  ----------------------------------
  A file has an end, so plate_scan.py can accumulate every read and cluster
  once. A camera does not. Here the vote is closed per *tracked vehicle*: the
  vehicle detector runs with persist=True, reads accumulate against the track
  ID while the vehicle is in frame, and the plate is committed the moment its
  cluster reaches MIN_VOTES. One event per vehicle, decided by several reads
  rather than by whichever read landed first.

  Threading 
  ---------
  Each camera runs on its own thread, not on the event loop. A plate pass is
  YOLO + YOLO + PaddleOCR and takes far longer than a frame interval; running
  it in an asyncio task would stall every other camera and every HTTP request.
  Threads detect and push events into an asyncio queue; the loop only does I/O.

  Each worker owns its own model instances. ultralytics keeps tracker state on
  the model object, so two cameras sharing one model would interleave into a
  single tracker and shuffle each other's IDs.

Run: python plate_edge.py
"""

import os
from pathlib import Path

# Paddle reads these at import time, so they must be set before paddleocr is
# imported anywhere in the process. plate_scan.py sets the same block at its
# own top; duplicated here rather than relying on import order, which a
# formatter or linter is free to rearrange.
#
# PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT is the load-bearing one: without it
# PaddleX picks the oneDNN CPU backend and the first real OCR call dies with
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#                        [pir::ArrayAttribute<pir::DoubleAttribute>]
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("FLAGS_log_level", "3")

# RTSP over UDP drops packets under load, and a lost packet shows up as
#   [h264] error while decoding MB 4 42, bytestream -5
# with visibly corrupted frames — which then get fed to OCR. TCP costs a little
# latency and removes the loss. FFmpeg reads this at capture-open time, so it
# has to be in the environment before cv2.VideoCapture is constructed.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|max_delay;500000",
)


# Load .env before importing plate_scan — it reads PLATE_MODEL_PATH and
# VEHICLE_MODEL_PATH from the environment at import time.
from dotenv import load_dotenv

_env = Path(__file__).parent / ".env"
load_dotenv(_env if _env.exists() else Path(__file__).parent / ".env.example")

import re
import cv2
import json
import time
import inspect
import base64
import sqlite3
import asyncio
import logging
import threading
import numpy as np
import aiohttp

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Imports that only exist inside the virtual environment. Running this with the
# system python is the easy mistake to make, and the bare ModuleNotFoundError
# does not hint at the cause, so say it plainly.
try:
    import torch
    from ultralytics import YOLO
    from paddleocr import PaddleOCR
except ModuleNotFoundError as e:
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _venv = os.path.join(_here, "venv", "bin", "python")
    print(f"\nMissing package: {e.name}\n", file=_sys.stderr)
    if os.path.exists(_venv) and _sys.executable != os.path.realpath(_venv):
        print("You are running the system python. Use the virtual "
              "environment instead:\n", file=_sys.stderr)
        print(f"  {os.path.relpath(_venv, os.getcwd())} "
              f"{os.path.basename(__file__)} " + " ".join(_sys.argv[1:]),
              file=_sys.stderr)
        print("\nor activate it first:\n", file=_sys.stderr)
        print("  source venv/bin/activate", file=_sys.stderr)
    else:
        print("Install the requirements:\n", file=_sys.stderr)
        print("  pip install -r requirements.txt", file=_sys.stderr)
    print("", file=_sys.stderr)
    _sys.exit(1)

# Everything plate-specific comes from the existing scanner.
import plate_scan
import plate_format
from plate_scan import (
    clean_plate_text,
    normalize_for_match,
    edit_distance,
    cluster_reads,
    draw_plate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PlateEdge")


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Service configuration, all overridable from .env"""

    SITE_ID: str = os.getenv("SITE_ID", "").strip()
    SITE_NAME: str = os.getenv("SITE_NAME", "Edge Site")

    CENTRAL_API_URL: str = os.getenv("CENTRAL_API_URL", "").strip()
    API_KEY: str = os.getenv("EDGE_API_KEY", "").strip()

    # Central endpoints. EP_DETECTIONS is the one the backend actually
    # implements (see python-developer-api 1.md): a single route taking the
    # plate number plus the image it was read from. The rest are optional —
    # each self-disables if the server answers 404.
    EP_DETECTIONS: str = os.getenv("EP_DETECTIONS", "/edge/detections")
    EP_CAMERAS: str = os.getenv("EP_CAMERAS", "/edge/cameras")
    EP_REGISTERED: str = os.getenv("EP_REGISTERED", "/edge/vehicles/sync")
    EP_HEARTBEAT: str = os.getenv("EP_HEARTBEAT", "/edge/heartbeat")

    # Image sent with each detection. jpg or png, per the API contract.
    DETECTION_FORMAT: str = os.getenv("DETECTION_FORMAT", "jpg").lower()
    SNAPSHOT_WIDTH: int = int(os.getenv("SNAPSHOT_WIDTH", "640"))

    # The server rejects bodies over 10MB with a 413.
    MAX_BODY_BYTES: int = int(os.getenv("MAX_BODY_BYTES", str(10 * 1024 * 1024)))

    # --- Testing with a video file instead of a camera --------------------
    # A source with no "://" is treated as a file on disk. Everything else in
    # the pipeline is identical, so this exercises detection, voting, the
    # local database and the upload path without needing a camera.
    TEST_VIDEO: str = os.getenv("TEST_VIDEO", "")
    TEST_VIDEO_ID: str = os.getenv("TEST_VIDEO_ID", "test-video")

    # Off by default: a looping test video would post a detection to the real
    # server on every lap. Turn it on deliberately, with a real cameraId.
    TEST_VIDEO_UPLOAD: bool = os.getenv(
        "TEST_VIDEO_UPLOAD", "false").strip().lower() in ("1", "true", "yes")

    TEST_VIDEO_LOOP: bool = os.getenv(
        "TEST_VIDEO_LOOP", "true").strip().lower() in ("1", "true", "yes")

    # --- Cameras ----------------------------------------------------------
    # Cameras are no longer configured here. The service asks the central
    # server for them:
    #
    #   GET {CENTRAL_API_URL}/edge/detections/cameras
    #   -> data.cameras[] of { id, name, streamUrl }
    #
    # That endpoint is unauthenticated by design (see
    # "python-camera-list-api 1.md"), and its `id` is exactly the cameraId the
    # detections endpoint expects — which removes the whole class of
    # "404 No camera found" mistakes that hand-copied ids caused.
    EP_CAMERA_LIST: str = os.getenv(
        "EP_CAMERA_LIST", "/edge/detections/cameras")

    # How often to re-read the list, so cameras added or removed on the server
    # are picked up without restarting the service.
    CAMERA_SYNC_INTERVAL: int = int(os.getenv("CAMERA_SYNC_INTERVAL", "120"))

    # Last known camera list, so a restart while the server is unreachable
    # still brings the cameras up.
    CAMERA_CACHE_PATH: str = os.getenv(
        "CAMERA_CACHE_PATH", "./data/cameras.json")

    # Every camera thread needs its own vehicle model (the tracker keeps state
    # on it), but the plate model and OCR engine are stateless per call, so
    # they can be shared behind a lock. With a model set costing well over a
    # gigabyte, sharing is what makes more than two or three cameras possible.
    SHARE_MODELS: bool = os.getenv(
        "SHARE_MODELS", "true").strip().lower() in ("1", "true", "yes")

    # Seconds between starting one camera and the next. Loading several model
    # sets at once spikes memory and CPU hard enough to stall the whole box.
    CAMERA_START_STAGGER: float = float(
        os.getenv("CAMERA_START_STAGGER", "3.0"))

    # Minimum seconds between detection passes on one camera — a throttle, not
    # a target. It was 0.4 while capture and detection shared a thread, where
    # pacing was the only thing keeping the decoder from falling behind. The
    # grabber does that job now, so the default is 0: look at every frame the
    # detector has time for. A vehicle crossing the frame in a second gets
    # every pass the CPU can give it instead of at most two.
    #
    # Raise it to hand cycles back — to other cameras, or to the rest of the
    # box — at the cost of looks per vehicle.
    DETECT_INTERVAL: float = float(os.getenv("DETECT_INTERVAL", "0"))

    # Model tuning — forwarded into plate_scan's module globals below.
    VEHICLE_IMGSZ: int = int(os.getenv("VEHICLE_IMGSZ", "480"))
    VEHICLE_CONF: float = float(os.getenv("VEHICLE_CONF", "0.3"))
    PLATE_IMGSZ: int = int(os.getenv("PLATE_IMGSZ", "320"))
    PLATE_CONF: float = float(os.getenv("PLATE_CONF", "0.35"))

    # Upper bound for the plate detector's adaptive inference size. PLATE_IMGSZ
    # is the floor; a vehicle crop bigger than that is run nearer its own size
    # so a plate on a large vehicle is not squeezed out of existence.
    PLATE_IMGSZ_MAX: int = int(os.getenv("PLATE_IMGSZ_MAX", "480"))

    # Gates in front of OCR, the pipeline's most expensive step. A plate box
    # narrower than this has no readable characters left.
    MIN_PLATE_WIDTH: int = int(os.getenv("MIN_PLATE_WIDTH", "40"))

    # Variance-of-Laplacian floor. Below it the crop is motion-blurred past
    # reading — common on fast vehicles, and previously the source of garbage
    # strings that competed with the real reads in the vote. 0 disables.
    BLUR_MIN: float = float(os.getenv("BLUR_MIN", "45"))

    # --- Plate text structure ---------------------------------------------
    # Read OCR output as an Indian plate (LL-DD-LL-DDDD) rather than as a bare
    # string: repair characters the position rules out, strip the junk that
    # bolts and borders add to the ends, and reject what cannot be a plate.
    PLATE_FORMAT: bool = os.getenv(
        "PLATE_FORMAT", "true").strip().lower() in ("1", "true", "yes")

    # Drop reads that cannot be parsed as a plate at all. These are misreads —
    # XYS178, 158SEF8679, a truncated 26BT1607 — and letting them into the
    # vote lets them win it. Turn off if this site sees plates in a format
    # this module does not know.
    PLATE_FORMAT_STRICT: bool = os.getenv(
        "PLATE_FORMAT_STRICT", "true").strip().lower() in ("1", "true", "yes")

    # State codes this site actually sees, comma separated, e.g. "GJ,HR".
    # A reading whose code is outside the list loses to one inside it that is
    # a single confusion away — this is what turns AR26BT1607 into HR26BT1607.
    # Leave blank to accept every state equally.
    PLATE_STATES: set = {
        c.strip().upper()
        for c in os.getenv("PLATE_STATES", "").split(",")
        if c.strip()
    }

    # How much a plate from an unlisted state is disbelieved, in units of
    # character repairs. Above 1.0 a local plate one confusion away is
    # preferred to a distant one that needs no repair at all.
    PLATE_STATE_PENALTY: float = float(
        os.getenv("PLATE_STATE_PENALTY", "1.5"))

    MIN_VOTES: int = int(os.getenv("MIN_VOTES", "2"))

    # A single read this confident commits immediately, without waiting for
    # MIN_VOTES. This is the fast-vehicle path: a car that crosses the frame
    # in under a second only ever gets one or two looks. Set high — it decides
    # on one read. 0 disables.
    HIGH_CONF_COMMIT: float = float(os.getenv("HIGH_CONF_COMMIT", "0.9"))

    # Last chance for a vehicle leaving the frame below MIN_VOTES: emit it if
    # the best read has at least this many votes and this confidence, marked
    # provisional. Without this, every fast vehicle read exactly once was
    # thrown away.
    EXIT_MIN_VOTES: int = int(os.getenv("EXIT_MIN_VOTES", "1"))
    EXIT_MIN_CONF: float = float(os.getenv("EXIT_MIN_CONF", "0.75"))

    # Tracker settings passed to ultralytics. The bundled default is tuned for
    # a camera that moves; ours are bolted to walls. See the file itself.
    TRACKER_CONFIG: str = os.getenv(
        "TRACKER_CONFIG", str(Path(__file__).parent / "trackers"
                              / "fixed_camera.yaml"))

    # A track with no sighting for this long is finalized and dropped.
    TRACK_TTL: float = float(os.getenv("TRACK_TTL", "3.0"))

    # Same plate on the same camera inside this window is one visit, not two.
    # Covers the tracker losing a vehicle and re-acquiring it as a new ID.
    PLATE_COOLDOWN: int = int(os.getenv("PLATE_COOLDOWN", "60"))

    # Cross-camera duplicate window, held in a JSON file. A plate reported by
    # any camera inside this many seconds is logged locally but not uploaded
    # again. Unlike PLATE_COOLDOWN this ignores which camera saw it, which is
    # what catches one vehicle reported by two cameras at once.
    DEDUPE_WINDOW: float = float(os.getenv("DEDUPE_WINDOW", "5"))
    DEDUPE_FILE: str = os.getenv("DEDUPE_FILE", "./data/recent_plates.json")

    # Registered-plate lookup tolerance. Deliberately tighter than the
    # clustering distance in plate_scan — a wrong match here authorizes the
    # wrong vehicle, so it only forgives OCR-confusable characters.
    MATCH_MAX_DISTANCE: int = int(os.getenv("MATCH_MAX_DISTANCE", "1"))

    # Intra-op threads per YOLO call. 0 splits the cores between the running
    # cameras, which is almost always what you want; set it only to pin the
    # service to part of the box.
    TORCH_THREADS: int = int(os.getenv("TORCH_THREADS", "0"))

    SYNC_INTERVAL: int = int(os.getenv("SYNC_INTERVAL", "300"))
    HEARTBEAT_INTERVAL: int = int(os.getenv("HEARTBEAT_INTERVAL", "30"))

    LOCAL_DB_PATH: str = os.getenv("LOCAL_DB_PATH", "./data/plates.db")
    PLATE_DB_PATH: str = os.getenv("PLATE_DB_PATH", "./data/registered_plates.json")

    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8002"))


config = Config()

# The detections endpoint authenticates with the API key alone and takes no
# siteId, so a missing SITE_ID must not disable central reporting.
CENTRAL_ENABLED = bool(config.CENTRAL_API_URL and config.API_KEY)

# The camera list route needs no key at all, so it works whenever the URL is
# set — even if the API key is missing or wrong.
CAMERA_LIST_AVAILABLE = bool(config.CENTRAL_API_URL)

# plate_scan reads these as module globals inside plates_in_image(); its own
# main() steers it the same way.
plate_scan.PLATE_IMGSZ = config.PLATE_IMGSZ
plate_scan.PLATE_CONF = config.PLATE_CONF
plate_scan.USE_VEHICLE_CROP = True

VEHICLE_CLASSES = plate_scan.VEHICLE_CLASSES


def tune_cpu_threads(cameras: int = 1):
    """
    Split the cores between the camera threads.

    Every camera runs its own detection loop, and each YOLO call defaults to
    using every core for intra-op parallelism. With two cameras that means two
    passes each asking for all four cores, and they spend the time fighting
    over them: measurably slower than each taking half. OpenCV's own pool adds
    a third claimant, so it is pinned to one thread — the work here is model
    inference, not image ops.
    """
    cpus = os.cpu_count() or 1
    want = config.TORCH_THREADS or max(1, cpus // max(1, cameras))

    torch.set_num_threads(want)
    cv2.setNumThreads(1)

    logger.info(
        f"CPU: {cpus} core(s), {cameras} camera(s) "
        f"-> {want} torch thread(s) each"
    )
    return want


# =============================================================================
# Local Database (SQLite)
# =============================================================================

class LocalDatabase:
    """Local store for offline operation and the outbound sync queue."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()

        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=15)

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()

                cur.execute('''
                    CREATE TABLE IF NOT EXISTS plate_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        vehicle_type TEXT,
                        track_id INTEGER,
                        votes INTEGER,
                        confidence REAL,
                        registered INTEGER DEFAULT 0,
                        owner TEXT,
                        timestamp TEXT NOT NULL,
                        synced INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Vehicles seen but never read. Kept out of plate_log so that
                # table stays a clean record of actual plate events.
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS unread_vehicle (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT NOT NULL,
                        vehicle_type TEXT,
                        track_id INTEGER,
                        best_read TEXT,
                        votes INTEGER,
                        timestamp TEXT NOT NULL
                    )
                ''')

                cur.execute('''
                    CREATE TABLE IF NOT EXISTS sync_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data_type TEXT NOT NULL,
                        data TEXT NOT NULL,
                        attempts INTEGER DEFAULT 0,
                        synced INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_plate_lookup
                    ON plate_log (plate, camera_id, timestamp)
                ''')

                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_sync_pending
                    ON sync_queue (synced, created_at)
                ''')

                self._migrate(conn)

                conn.commit()
            finally:
                conn.close()

    def _migrate(self, conn):
        """
        Drop event_type from databases created before cameras stopped being
        entry/exit. Rows are preserved; only the column goes.

        Needs SQLite 3.35+ for DROP COLUMN, which no index here depends on.
        """

        for table in ("plate_log", "unread_vehicle"):
            columns = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]

            if "event_type" not in columns:
                continue

            if sqlite3.sqlite_version_info < (3, 35, 0):
                logger.warning(
                    f"{table} still has event_type and SQLite "
                    f"{sqlite3.sqlite_version} cannot drop it. Harmless — the "
                    f"column is simply left populated with its old values."
                )
                return

            backup = f"{self.db_path}.pre-migration"
            if not os.path.exists(backup):
                import shutil
                shutil.copy2(self.db_path, backup)
                logger.info(f"Backed up database to {backup}")

            conn.execute(f"ALTER TABLE {table} DROP COLUMN event_type")
            logger.info(f"Migrated {table}: dropped event_type")

    def log_plate(self, event: dict) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute('''
                    INSERT INTO plate_log
                        (plate, camera_id, vehicle_type, track_id,
                         votes, confidence, registered, owner, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event["plate"],
                    event["cameraId"],
                    event.get("vehicleType"),
                    event.get("trackId"),
                    event.get("votes"),
                    event.get("confidence"),
                    1 if event.get("registered") else 0,
                    event.get("owner"),
                    event["timestamp"],
                ))
                row_id = cur.lastrowid
                conn.commit()
                return row_id
            finally:
                conn.close()

    def log_unread(self, camera_id: str, vehicle_type: str,
                   track_id: int, best_read: str, votes: int):
        with self._lock:
            conn = self._connect()
            try:
                conn.execute('''
                    INSERT INTO unread_vehicle
                        (camera_id, vehicle_type, track_id,
                         best_read, votes, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    camera_id, vehicle_type, track_id,
                    best_read, votes, datetime.now(timezone.utc).isoformat(),
                ))
                conn.commit()
            finally:
                conn.close()

    def seen_recently(self, plate: str, camera_id: str, cooldown: int) -> bool:
        """True if this plate already logged on this camera inside the window."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=cooldown)
        ).isoformat()

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute('''
                    SELECT COUNT(*) FROM plate_log
                    WHERE plate = ? AND camera_id = ? AND timestamp > ?
                ''', (plate, camera_id, cutoff))
                return cur.fetchone()[0] > 0
            finally:
                conn.close()

    def recent_events(self, limit: int = 50) -> List[dict]:
        with self._lock:
            conn = self._connect()
            try:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute('''
                    SELECT plate, camera_id, vehicle_type, votes,
                           confidence, registered, owner, timestamp, synced
                    FROM plate_log
                    ORDER BY id DESC
                    LIMIT ?
                ''', (limit,))
                return [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()

    def add_to_sync_queue(self, data_type: str, data: dict) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO sync_queue (data_type, data) VALUES (?, ?)",
                    (data_type, json.dumps(data)),
                )
                queue_id = cur.lastrowid
                conn.commit()
                return queue_id
            finally:
                conn.close()

    def get_pending_sync(self, limit: int = 50) -> List[dict]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute('''
                    SELECT id, data_type, data FROM sync_queue
                    WHERE synced = 0
                    ORDER BY created_at ASC
                    LIMIT ?
                ''', (limit,))
                rows = cur.fetchall()
                return [
                    {"id": r[0], "type": r[1], "data": json.loads(r[2])}
                    for r in rows
                ]
            finally:
                conn.close()

    def mark_synced(self, ids: List[int]):
        if not ids:
            return
        with self._lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE sync_queue SET synced = 1 WHERE id IN ({placeholders})",
                    ids,
                )
                conn.commit()
            finally:
                conn.close()

    def bump_attempts(self, ids: List[int]):
        if not ids:
            return
        with self._lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE sync_queue SET attempts = attempts + 1 "
                    f"WHERE id IN ({placeholders})",
                    ids,
                )
                conn.commit()
            finally:
                conn.close()

    def stats(self) -> dict:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM plate_log")
                total = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM plate_log WHERE registered = 1")
                registered = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM unread_vehicle")
                unread = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM sync_queue WHERE synced = 0")
                pending = cur.fetchone()[0]
                return {
                    "events": total,
                    "registered_hits": registered,
                    "unread_vehicles": unread,
                    "pending_sync": pending,
                }
            finally:
                conn.close()


local_db = LocalDatabase(config.LOCAL_DB_PATH)


# =============================================================================
# Registered Plate Database
# =============================================================================

class PlateDatabase:
    """
    Plates authorized for this site, synced from central.

    Lookups go through plate_scan.normalize_for_match so that an OCR read of
    GJ01SK36O7 still resolves to the registered GJ01SK3607.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.plates: Dict[str, dict] = {}   # plate -> record
        self._index: Dict[str, str] = {}    # normalized -> plate
        self._lock = threading.Lock()
        self._load()

    def _reindex(self):
        self._index = {normalize_for_match(p): p for p in self.plates}

    def _load(self):
        if not os.path.exists(self.db_path):
            logger.info("No registered plate file yet; starting empty")
            return
        try:
            with open(self.db_path) as fh:
                self.plates = json.load(fh)
            self._reindex()
            logger.info(f"Loaded {len(self.plates)} registered plates")
        except Exception as e:
            logger.error(f"Failed to load plate database: {e}")

    def _save(self):
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Write-then-rename so a crash mid-write cannot truncate the database.
        tmp = f"{self.db_path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(self.plates, fh, indent=2)
        os.replace(tmp, self.db_path)

    def replace_all(self, records: List[dict]) -> int:
        """Swap in a fresh set from central."""
        fresh = {}
        for rec in records:
            raw = rec.get("plateNumber") or rec.get("plate") or ""
            plate = clean_plate_text(raw)
            if not plate:
                logger.warning(f"Skipping unusable registered plate: {raw!r}")
                continue
            fresh[plate] = {
                "plate": plate,
                "vehicleId": rec.get("_id") or rec.get("id", ""),
                "owner": rec.get("ownerName") or rec.get("owner", ""),
                "vehicleType": rec.get("vehicleType", ""),
                "active": rec.get("active", True),
            }

        with self._lock:
            self.plates = fresh
            self._reindex()
            self._save()

        return len(fresh)

    def lookup(self, plate: str) -> Optional[dict]:
        """Exact match first, then nearest confusable-equivalent plate."""
        with self._lock:
            if plate in self.plates:
                return self.plates[plate]

            norm = normalize_for_match(plate)

            hit = self._index.get(norm)
            if hit:
                return self.plates[hit]

            best, best_dist = None, config.MATCH_MAX_DISTANCE + 1

            for cand_norm, cand_plate in self._index.items():
                d = edit_distance(norm, cand_norm, config.MATCH_MAX_DISTANCE)
                if d < best_dist:
                    best, best_dist = cand_plate, d

            if best and best_dist <= config.MATCH_MAX_DISTANCE:
                return self.plates[best]

        return None

    def __len__(self):
        return len(self.plates)


plate_db = PlateDatabase(config.PLATE_DB_PATH)


# =============================================================================
# Central API Client
# =============================================================================


def _describe_payload(payload: dict) -> str:
    """
    One-line payload summary for the log.

    The image is base64 and tens of kilobytes long — its size is useful, its
    contents are not, and printing it would bury every other log line.
    """

    parts = []

    for key, value in payload.items():
        if key == "img":
            parts.append(f"img=<base64 {len(value) / 1024:.0f}KB>")
        elif isinstance(value, str) and len(value) > 48:
            parts.append(f"{key}={value[:45]!r}...")
        elif isinstance(value, (list, dict)):
            parts.append(f"{key}=<{type(value).__name__} len={len(value)}>")
        else:
            parts.append(f"{key}={value!r}")

    return "  ".join(parts)


def _call_chain(frame, depth: int = 3) -> str:
    """Where a call came from: 'caller() <- its caller()', with line numbers."""

    hops = []

    while frame is not None and len(hops) < depth:
        code = frame.f_code
        hops.append(f"{code.co_name}():{frame.f_lineno}")
        frame = frame.f_back

    return " <- ".join(hops)


class CentralAPIClient:
    def __init__(self):
        self.base_url = config.CENTRAL_API_URL.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {config.API_KEY}",
            "Content-Type": "application/json",
            "X-Site-ID": config.SITE_ID,
        }

        # Endpoints the server answered 404 for. A 404 means the route is not
        # implemented, which retrying cannot fix — unlike a timeout or a 5xx,
        # which are transient and stay worth retrying. Without this an
        # unimplemented heartbeat logs a warning every HEARTBEAT_INTERVAL
        # forever and buries real errors.
        self._missing: set = set()

        # The detections contract documents exactly two headers. The other
        # endpoints are ours to guess at and still get X-Site-ID; this one is
        # kept to the letter of the spec so it matches the reference curl.
        self.detection_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.API_KEY}",
        }

    def unavailable(self, path: str) -> bool:
        return path in self._missing

    def _mark_missing(self, path: str, verb: str):
        if path in self._missing:
            return

        self._missing.add(path)
        logger.warning(
            f"{verb} {path} -> 404. Central server has no such route; "
            f"disabling it for this run. Set the correct path in .env and "
            f"restart. Detections keep being logged locally either way."
        )

    async def _post(self, path: str, payload: dict, quiet: bool = False,
                    headers: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        origin = _call_chain(inspect.currentframe().f_back)
        say = logger.debug if quiet else logger.info

        if self.unavailable(path):
            say(f"API SKIP POST {url} (route known missing) [from {origin}]")
            return {"success": False, "missing": True}

        say(f"API ->  POST {url}")
        say(f"API ->  called from {origin}")
        say(f"API ->  payload: {_describe_payload(payload)}")

        started = time.perf_counter()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers or self.headers,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    took = (time.perf_counter() - started) * 1000
                    # The detections route answers 201 Created, not 200. Only
                    # accepting 200 would mark every stored detection as failed
                    # and re-queue it forever.
                    if resp.status in (200, 201):
                        try:
                            result = await resp.json()
                        except Exception:
                            result = {"success": True}

                        stored = (result.get("data") or {}).get("id")
                        say(f"API <-  {resp.status} OK in {took:.0f}ms"
                            + (f"  stored id={stored}" if stored else ""))
                        return result

                    body = await resp.text()

                    # 404 and 413 get their own, more specific lines below.
                    if resp.status not in (404, 413):
                        say(f"API <-  {resp.status} in {took:.0f}ms  "
                            f"{body[:160]}")

                    if resp.status == 404:
                        # Two very different things answer 404 here: a route
                        # that does not exist, and a cameraId that matches no
                        # camera. Disabling the endpoint for the second would
                        # silently stop all uploads over a config mistake.
                        if "route not found" in body.lower():
                            self._mark_missing(path, "POST")
                            return {"success": False, "missing": True}

                        logger.error(
                            f"API <-  404 from {url} in {took:.0f}ms: "
                            f"{body[:200]}  (a data problem, not a missing "
                            f"route — check the cameraId is registered on the "
                            f"server).  Called from {origin}"
                        )
                        return {"success": False, "status": 404}

                    if resp.status == 413:
                        logger.error(
                            f"API <-  413 from {url} in {took:.0f}ms: body "
                            f"over the server's 10MB limit. Lower "
                            f"SNAPSHOT_WIDTH in .env.  Called from {origin}"
                        )
                        return {"success": False, "status": 413}

                    logger.warning(f"POST {path} -> {resp.status}: {body[:200]}")
                    return {"success": False, "status": resp.status}
        except Exception as e:
            took = (time.perf_counter() - started) * 1000

            # Nothing answered at all: the server is down, the host is wrong,
            # or the network is cut. Distinguished from an HTTP error because
            # the caller's response differs — there is no point posting the
            # next 49 queued detections into the same silence.
            unreachable = isinstance(
                e, (aiohttp.ClientConnectorError, aiohttp.ServerTimeoutError,
                    asyncio.TimeoutError),
            )

            log = logger.debug if quiet else logger.warning
            log(
                f"API <-  POST {url} FAILED after {took:.0f}ms: "
                f"{type(e).__name__}: {e}  [from {origin}]"
            )
            return {
                "success": False,
                "error": str(e),
                "unreachable": unreachable,
            }

    async def fetch_registered_plates(self) -> List[dict]:
        if self.unavailable(config.EP_REGISTERED):
            return []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}{config.EP_REGISTERED}",
                    params={"siteId": config.SITE_ID},
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 404:
                        self._mark_missing(config.EP_REGISTERED, "GET")
                        return []

                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(
                            f"Plate sync failed: {resp.status} - {body[:200]}"
                        )
                        return []
                    result = await resp.json()
        except Exception as e:
            logger.error(f"Plate sync error: {e}")
            return []

        data = result.get("data", result)
        return data.get("vehicles") or data.get("plates") or []

    async def send_detection(self, event: dict, quiet: bool = False) -> dict:
        """
        Report one detection in the shape the backend expects:
        cameraId + number + base64 image. Everything else the edge tracks
        (votes, owner, vehicle type) stays local — the API does not take it.

        quiet suppresses the per-detection logging. The retry task passes it:
        with the server down and a queue of 50, the normal logging is three
        lines per detection per minute — enough noise to bury the one line
        that says what is actually wrong.
        """

        if not quiet:
            logger.info(
                f"DETECTION upload starting: plate={event['plate']!r} "
                f"cameraId={event['cameraId']!r} "
                f"format={config.DETECTION_FORMAT}"
            )

        img = event.get("capturedImage")

        if not img:
            # img is a required field; posting without it earns a 400.
            logger.warning(
                f"{event['plate']} has no snapshot, cannot upload "
                f"(kept in the local database)"
            )
            return {"success": False, "no_image": True}

        payload = {
            "cameraId": event["cameraId"],
            "number": event["plate"],
            "img": img,
            "format": config.DETECTION_FORMAT,
        }

        approx = len(img) + 200
        if approx > config.MAX_BODY_BYTES:
            logger.error(
                f"{event['plate']} snapshot is ~{approx // 1024}KB, over the "
                f"server limit; lower SNAPSHOT_WIDTH"
            )
            return {"success": False, "too_large": True}

        result = await self._post(
            config.EP_DETECTIONS, payload, headers=self.detection_headers,
            quiet=quiet)

        if result.get("success"):
            # Worth a line even when quiet: a queued detection finally landing
            # is the event you want in the log after an outage.
            data = result.get("data") or {}
            logger.info(
                f"DETECTION upload OK: plate={event['plate']!r} "
                f"serverId={data.get('id')} "
                f"updatedTime={data.get('updatedTime')}"
            )
        elif not quiet:
            logger.warning(
                f"DETECTION upload FAILED: plate={event['plate']!r} "
                f"reason={result.get('status') or result}"
            )

        return result

    async def fetch_camera_list(self) -> Optional[List[dict]]:
        """
        The camera list the plate service runs from.

        Deliberately sends no Authorization header: this route is
        unauthenticated by contract. Returns None on failure so the caller can
        tell "server unreachable" (keep running what we have) apart from
        "server says there are no cameras" (stop everything).
        """

        url = f"{self.base_url}{config.EP_CAMERA_LIST}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"Accept": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(
                            f"Camera list GET {url} -> {resp.status}: "
                            f"{body[:200]}")
                        return None

                    result = await resp.json()
        except Exception as e:
            logger.warning(
                f"Camera list GET {url} failed: {type(e).__name__}: {e}")
            return None

        cameras = (result.get("data") or {}).get("cameras")

        if cameras is None:
            logger.error(f"Camera list response had no data.cameras: "
                         f"{str(result)[:200]}")
            return None

        return cameras

    async def fetch_cameras(self) -> List[dict]:
        """Cameras the server knows about, used to validate our camera ids."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}{config.EP_CAMERAS}",
                    params={"siteId": config.SITE_ID},
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        return []
                    result = await resp.json()
        except Exception as e:
            logger.debug(f"Camera list fetch failed: {e}")
            return []

        return result.get("data", {}).get("cameras", []) or []

    async def send_heartbeat(self, cameras: List[dict]) -> dict:
        online = sum(1 for c in cameras if c.get("status") == "online")
        return await self._post(config.EP_HEARTBEAT, {
            "siteId": config.SITE_ID,
            "siteName": config.SITE_NAME,
            "service": "plate_recognition",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "online",
            "registeredPlates": len(plate_db),
            "cameraCount": len(cameras),
            "onlineCameras": online,
            "cameras": cameras,
        }, quiet=True)


central_api = CentralAPIClient()


# =============================================================================
# Per-Track Vote State
# =============================================================================

class TrackState:
    """Accumulated plate reads for one tracked vehicle."""

    __slots__ = (
        "track_id", "label", "counts", "top_conf", "best_conf", "best_area",
        "best_crop", "overlay", "first_seen", "last_seen", "committed",
    )

    def __init__(self, track_id: int, label: str, now: float):
        self.track_id = track_id
        self.label = label
        self.counts: Dict[str, int] = {}

        # Best OCR confidence seen for each spelling. The vote counts reads;
        # this is what lets one very confident read stand in for two ordinary
        # ones when a vehicle is through the frame too fast for a second look.
        self.top_conf: Dict[str, float] = {}
        self.best_conf = 0.0
        self.best_area = 0
        self.best_crop: Optional[np.ndarray] = None
        self.overlay: List[Tuple[tuple, str]] = []
        self.first_seen = now
        self.last_seen = now
        self.committed = False

    def winner(self) -> Optional[dict]:
        """Top cluster for this vehicle, or None if nothing has been read."""
        if not self.counts:
            return None
        clusters = cluster_reads(self.counts)
        return clusters[0] if clusters else None


# =============================================================================
# Shared model pool
# =============================================================================
#
# The vehicle detector cannot be shared: ultralytics stores tracker state on the
# model object, so two cameras using one instance would merge into a single
# tracker and shuffle each other's IDs.
#
# The plate detector and the OCR engine hold no state between calls, so one of
# each serves every camera. They are not documented as thread-safe, hence the
# lock. That serialises plate reads across cameras, which costs little here —
# on CPU the box is saturated by one camera's inference anyway, while a model
# set per camera costs well over a gigabyte of RAM.

_shared_lock = threading.Lock()
_shared_infer_lock = threading.Lock()
_shared_plate_model = None
_shared_ocr = None


def _new_ocr():
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def get_shared_models():
    """Load the plate model and OCR engine once, on first use."""
    global _shared_plate_model, _shared_ocr

    with _shared_lock:
        if _shared_plate_model is None:
            logger.info("Loading shared plate model + OCR (once for all cameras)")
            _shared_plate_model = YOLO(plate_scan.PLATE_MODEL_PATH)
            _shared_ocr = _new_ocr()

    return _shared_plate_model, _shared_ocr


# =============================================================================
# Recent-plate window (JSON)
# =============================================================================

class RecentPlates:
    """
    A short rolling window of plates already reported, kept in a JSON file.

    On each detection: drop entries older than the window, look for this plate,
    and add it. A plate already inside the window is not uploaded again.

    Why the key is the plate alone, and not (cameraId, plate)
    --------------------------------------------------------
    The SQLite cooldown in seen_recently() is already keyed per camera, and it
    works: over the whole local history there is not one same-camera repeat
    inside 60 seconds. Every duplicate that reached the server came from the
    *same plate under a different cameraId* — two camera records pointed at one
    physical stream, so both workers saw the same car and each was, correctly,
    the first sighting for its own camera id.

    A per-camera key here would therefore reproduce the existing behaviour and
    block none of them. The cameraId is still written to the file, as a record
    of which camera won the race, but the match is on the plate.

    The file is the interface — small, readable, and easy to inspect while the
    service runs. It is rewritten atomically so a reader never catches it
    half-written, and the in-memory copy is authoritative, so a corrupt or
    deleted file costs at most one window of deduplication rather than
    crashing the service.
    """

    def __init__(self, path: str, window: float):
        self.path = Path(path)
        self.window = window
        self._lock = threading.Lock()
        self._entries: List[dict] = []

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """Survive a restart: a service bounced mid-window must not re-report."""
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if isinstance(data, list):
                    self._entries = [
                        e for e in data
                        if isinstance(e, dict) and e.get("plate")
                    ]
        except Exception as e:
            logger.warning(
                f"Could not read {self.path} ({e}); starting with an empty "
                f"window"
            )
            self._entries = []

    def _save(self):
        # Write-then-rename: anything reading the file sees the old version or
        # the new one, never a truncated one.
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._entries, indent=2))
            tmp.replace(self.path)
        except Exception as e:
            logger.warning(f"Could not write {self.path}: {e}")

    @staticmethod
    def _age(entry: dict, now: float) -> float:
        """Seconds since this entry was written; huge if it cannot be read."""
        try:
            seen = datetime.fromisoformat(entry["time"])
            if seen.tzinfo is None:
                seen = seen.replace(tzinfo=timezone.utc)
            return now - seen.timestamp()
        except Exception:
            return float("inf")

    def should_send(self, plate: str, camera_id: str,
                    camera_name: str = "") -> bool:
        """
        True if this plate should go to the server.

        Prunes the window, then decides. The plate is recorded either way — a
        car still in front of the camera keeps refreshing its own entry, so the
        window measures time since it was last *seen*, not since it was first
        reported. Without that a vehicle sitting in frame for a minute would be
        re-sent every window.
        """
        now = datetime.now(timezone.utc).timestamp()

        with self._lock:
            self._entries = [
                e for e in self._entries if self._age(e, now) < self.window
            ]

            match = next(
                (e for e in self._entries if e["plate"] == plate), None)

            entry = {
                "plate": plate,
                "cameraId": camera_id,
                "cameraName": camera_name,
                "time": datetime.now(timezone.utc).isoformat(),
            }

            if match is not None:
                age = self._age(match, now)
                match.update(entry)
                self._save()
                logger.info(
                    f"DUPLICATE {plate} seen {age:.1f}s ago on "
                    f"{match.get('cameraName') or match.get('cameraId')} "
                    f"— logged locally, not sent to the server"
                )
                return False

            self._entries.append(entry)
            self._save()
            return True

    def snapshot(self) -> List[dict]:
        """Current window contents, for /status."""
        now = datetime.now(timezone.utc).timestamp()
        with self._lock:
            return [
                dict(e, ageSeconds=round(self._age(e, now), 1))
                for e in self._entries
                if self._age(e, now) < self.window
            ]


recent_plates = RecentPlates(config.DEDUPE_FILE, config.DEDUPE_WINDOW)


# =============================================================================
# Frame Grabber
# =============================================================================

class FrameGrabber(threading.Thread):
    """
    Owns the capture and keeps only the newest frame.

    The detection pass (YOLO + YOLO + OCR) is far slower than a frame interval,
    so when capture and detection share a thread the decoder queue fills while
    detection runs and the next read() returns video from a second ago. On a
    fast vehicle that is the difference between a plate in frame and a plate
    already past the camera — the reason fast traffic was being missed even
    though the models had time to look at it.

    Reading here in a thread of its own means the decoder is always drained at
    the stream's own rate and `latest()` is genuinely the newest frame. Frames
    the detector cannot keep up with are dropped, which is what we want: a
    stale frame is worth less than the current one.

    Files are the exception in pacing only — they are read at their own frame
    rate so a test video does not race past in seconds — but they go through
    the same newest-frame handoff, which makes a file test represent what a
    camera will actually do.
    """

    RECONNECT_DELAY = 5.0
    MAX_READ_ERRORS = 15

    def __init__(self, camera_id: str, source: str, is_file: bool,
                 loop_file: bool = True):
        super().__init__(name=f"grab-{camera_id}", daemon=True)

        self.camera_id = camera_id
        self.source = source
        self.is_file = is_file
        self.loop_file = loop_file

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_ts = 0.0

        # Bumped on every file rewind. The detector compares it against the
        # generation it last saw and drops its tracks when it changes: track
        # ids from the previous lap describe vehicles that no longer exist.
        self.generation = 0

        self.frame_delay = 0.0
        self.status = "starting"
        self.error_count = 0
        self.loops = 0
        self.frames_read = 0
        self.finished = False

    def stop(self):
        self._stop.set()

    def latest(self) -> Tuple[Optional[np.ndarray], float, int]:
        """The newest decoded frame, its capture time, and its generation."""
        with self._lock:
            return self._frame, self._frame_ts, self.generation

    def _publish(self, frame):
        with self._lock:
            self._frame = frame
            self._frame_ts = time.time()
        self.frames_read += 1

    # -- capture -----------------------------------------------------------

    def _open(self) -> Optional[cv2.VideoCapture]:
        kind = "video file" if self.is_file else "stream"
        logger.info(f"[{self.camera_id}] opening {kind}: {self.source}")
        self.status = "connecting"

        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            self.status = "error"
            logger.error(f"[{self.camera_id}] cannot open {kind}")
            cap.release()
            self._stop.wait(self.RECONNECT_DELAY)
            return None

        if self.is_file:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_delay = 1.0 / fps if fps > 0 else 0.04
            logger.info(
                f"[{self.camera_id}] video open: {total} frames @ {fps:.0f}fps"
                f"{', looping' if self.loop_file else ''}"
            )
        else:
            # One frame of decoder queue. Belt and braces alongside reading in
            # this thread — with both, a read never returns buffered video.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info(f"[{self.camera_id}] connected")

        return cap

    def run(self):
        cap = None

        try:
            while not self._stop.is_set():
                try:
                    if cap is None or not cap.isOpened():
                        cap = self._open()
                        if cap is None:
                            continue

                    started = time.time()
                    ok, frame = cap.read()

                    if not ok or frame is None:
                        if self.is_file:
                            if not self.loop_file:
                                logger.info(
                                    f"[{self.camera_id}] video finished")
                                self.status = "finished"
                                self.finished = True
                                break

                            self.loops += 1
                            logger.info(
                                f"[{self.camera_id}] video ended, "
                                f"restarting (lap {self.loops})")

                            # Rewind in place; reopen only if seeking fails.
                            if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                                cap.release()
                                cap = None

                            with self._lock:
                                self.generation += 1
                            continue

                        self.error_count += 1
                        self.status = "error"

                        # A stream blip must not take the camera down for the
                        # life of the process.
                        if self.error_count >= self.MAX_READ_ERRORS:
                            logger.warning(
                                f"[{self.camera_id}] {self.error_count} read "
                                f"errors, reconnecting"
                            )
                            cap.release()
                            cap = None
                            self.error_count = 0
                            self._stop.wait(self.RECONNECT_DELAY)
                        continue

                    self.error_count = 0
                    self.status = "online"
                    self._publish(frame)

                    # Play a file at its own frame rate; a stream is already
                    # paced by the camera and must be read as fast as it
                    # arrives or the decoder queue grows again.
                    if self.is_file and self.frame_delay:
                        spare = self.frame_delay - (time.time() - started)
                        if spare > 0:
                            self._stop.wait(spare)

                except Exception as e:
                    logger.error(f"[{self.camera_id}] capture error: {e}")
                    self._stop.wait(2.0)

        finally:
            if cap is not None:
                cap.release()
            if not self.finished:
                self.status = "stopped"
            logger.info(f"[{self.camera_id}] grabber exited")


# =============================================================================
# Camera Worker
# =============================================================================

class CameraWorker(threading.Thread):
    """
    One RTSP camera: track vehicles, vote on plates, emit events.

    Capture belongs to a FrameGrabber thread; this one only ever looks at the
    newest frame that thread has published, so a slow detection pass costs
    freshness of *sampling* and never freshness of the frame itself.

    Runs off the event loop. Completed events are handed to the loop through
    emit_cb, which is thread-safe.
    """

    def __init__(self, camera_id: str, rtsp_url: str,
                 name: str, emit_cb, upload: bool = True,
                 loop_file: bool = True):
        super().__init__(name=f"cam-{camera_id}", daemon=True)

        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.display_name = name or camera_id
        self.emit_cb = emit_cb
        self.upload = upload

        # A path with no scheme is a video file. Files are paced to their own
        # frame rate and loop at the end; a stream is read as fast as it
        # arrives and reconnects on failure. Both differences live in the
        # grabber.
        self.is_file = "://" not in str(rtsp_url)
        self.loop_file = loop_file

        self._stop = threading.Event()
        self.tracks: Dict[int, TrackState] = {}

        self.grabber: Optional[FrameGrabber] = None
        self.generation = 0

        self.latest_frame: Optional[np.ndarray] = None
        self.last_frame_time = 0.0
        self.status = "starting"
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.events_emitted = 0

        # Rolling cost of a detection pass, for /status and for the warning
        # that fires when the box cannot keep up with its cameras.
        self.last_pass_ms = 0.0
        self.avg_pass_ms = 0.0
        self.passes = 0

        # Plate boxes dropped before OCR. Visible in /status so the gates can
        # be tuned against real footage rather than guessed at.
        self.skipped_small = 0
        self.skipped_blur = 0

        # Reads the plate grammar repaired, and reads it threw out entirely.
        self.repaired_format = 0
        self.rejected_format = 0

        # Set while an MJPEG viewer is attached. Drawing the preview means a
        # full-frame copy plus text for every frame; with nobody watching that
        # is pure waste on a box that has no cycles to spare.
        self.preview_wanted = 0.0

        # Own models per worker: ultralytics stores tracker state on the model,
        # so a shared instance would merge both cameras into one tracker.
        self.vehicle_model = None
        self.plate_model = None
        self.ocr = None
        self.infer_lock: Optional[threading.Lock] = None

    # -- lifecycle ---------------------------------------------------------

    def _load_models(self):
        logger.info(f"[{self.camera_id}] loading models...")

        # Always its own: the tracker's state lives on this object.
        self.vehicle_model = YOLO(plate_scan.VEHICLE_MODEL_PATH)

        if config.SHARE_MODELS:
            self.plate_model, self.ocr = get_shared_models()
            self.infer_lock = _shared_infer_lock
        else:
            self.plate_model = YOLO(plate_scan.PLATE_MODEL_PATH)
            self.ocr = _new_ocr()
            # Private, so never contended — keeps the call site uniform.
            self.infer_lock = threading.Lock()

        logger.info(
            f"[{self.camera_id}] models ready"
            + ("  (plate model + OCR shared)" if config.SHARE_MODELS else "")
        )

    def stop(self):
        self._stop.set()
        if self.grabber is not None:
            self.grabber.stop()

    def run(self):
        try:
            self._load_models()
        except Exception as e:
            logger.error(f"[{self.camera_id}] model load failed: {e}")
            self.status = "error"
            return

        self.grabber = FrameGrabber(
            self.camera_id, self.rtsp_url, self.is_file, self.loop_file)
        self.grabber.start()

        last_ts = 0.0

        try:
            while not self._stop.is_set():
                try:
                    frame, ts, generation = self.grabber.latest()

                    self.status = self.grabber.status

                    if frame is None or ts == last_ts:
                        # Nothing new decoded yet. A short wait rather than a
                        # spin: at 25fps a frame lands every 40ms.
                        if self.grabber.finished:
                            break
                        self._stop.wait(0.005)
                        continue

                    last_ts = ts

                    # A file restarted. Track ids from the previous lap belong
                    # to vehicles that are gone.
                    if generation != self.generation:
                        self.generation = generation
                        self.tracks.clear()

                    self.last_frame_time = ts

                    pass_started = time.time()
                    self._detect(frame, pass_started)
                    self._record_pass(time.time() - pass_started)

                    if self._preview_active():
                        self.latest_frame = self._render(frame)

                    # DETECT_INTERVAL is a ceiling on how hard this camera may
                    # push the CPU, not a target. Left at 0 the loop runs
                    # back to back, which is what a fast vehicle needs; raise
                    # it to hand cycles to other cameras.
                    if config.DETECT_INTERVAL > 0:
                        spare = config.DETECT_INTERVAL - (
                            time.time() - pass_started)
                        if spare > 0:
                            self._stop.wait(spare)

                except Exception as e:
                    logger.error(f"[{self.camera_id}] loop error: {e}")
                    self._stop.wait(2.0)

        finally:
            self.grabber.stop()
            self.status = "finished" if self.grabber.finished else "stopped"
            logger.info(f"[{self.camera_id}] worker exited")

    def _record_pass(self, seconds: float):
        self.last_pass_ms = seconds * 1000.0
        self.passes += 1
        # Exponential average: one slow pass behind a lorry should not read as
        # a box that is permanently behind.
        self.avg_pass_ms = (
            self.last_pass_ms if self.passes == 1
            else 0.9 * self.avg_pass_ms + 0.1 * self.last_pass_ms
        )

    def _preview_active(self) -> bool:
        """True while an MJPEG viewer has asked for frames in the last 2s."""
        return (time.time() - self.preview_wanted) < 2.0

    # -- detection ---------------------------------------------------------

    def _tracked_vehicles(self, frame) -> List[Tuple[int, str, tuple]]:
        """Vehicle boxes with stable IDs. Untracked boxes are skipped."""
        results = self.vehicle_model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            imgsz=config.VEHICLE_IMGSZ,
            conf=config.VEHICLE_CONF,
            iou=plate_scan.VEHICLE_IOU,
            classes=VEHICLE_CLASSES,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None:
            return []

        out = []
        h, w = frame.shape[:2]

        for box in boxes:
            if box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Clamp: the tracker can predict a box past the frame edge.
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            label = self.vehicle_model.names[int(box.cls)]
            out.append((int(box.id.item()), label, (x1, y1, x2, y2)))

        return out

    def _detect(self, frame, now: float):
        seen = set()

        for track_id, label, (x1, y1, x2, y2) in self._tracked_vehicles(frame):
            seen.add(track_id)

            state = self.tracks.get(track_id)
            if state is None:
                state = TrackState(track_id, label, now)
                self.tracks[track_id] = state

            state.last_seen = now
            state.label = label

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Keep the largest crop of this vehicle for the event snapshot.
            area = (x2 - x1) * (y2 - y1)
            if area > state.best_area:
                state.best_area = area
                state.best_crop = roi.copy()

            if state.committed:
                continue

            with self.infer_lock:
                found = self._plates_in_vehicle(roi, offset=(x1, y1))

            state.overlay = [(box, text) for box, text, _ in found]

            for _, text, conf in found:
                if text:
                    state.counts[text] = state.counts.get(text, 0) + 1
                    state.best_conf = max(state.best_conf, conf)
                    state.top_conf[text] = max(
                        state.top_conf.get(text, 0.0), conf)

            self._maybe_commit(state)

        self._sweep(seen, now)

    # -- plate stage -------------------------------------------------------

    def _plate_imgsz(self, roi) -> int:
        """
        Inference size for the plate detector on one vehicle crop.

        A fixed 320 is right for a crop that is roughly that size, and wrong
        for a lorry filling a 1080p frame: the crop is squeezed by 4x and the
        plate lands on too few pixels to detect. Scaling with the crop, capped
        at PLATE_IMGSZ_MAX, keeps a plate at a workable size without paying
        640 for every scooter.
        """
        longest = max(roi.shape[:2])
        step = 32 * ((min(longest, config.PLATE_IMGSZ_MAX) + 31) // 32)
        return int(max(config.PLATE_IMGSZ, min(step, config.PLATE_IMGSZ_MAX)))

    @staticmethod
    def _sharpness(crop) -> float:
        """
        Variance of the Laplacian — low means motion blur.

        Cheap (well under a millisecond) next to the ~200ms OCR call it
        decides whether to make.
        """
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(grey, cv2.CV_64F).var())

    def _plates_in_vehicle(self, roi, offset=(0, 0)):
        """
        Detect and read plates inside one vehicle crop.

        Same contract as plate_scan.plates_in_image — a list of
        (box, text, confidence) in full-frame coordinates — with two gates in
        front of the OCR call, which is the most expensive step in the
        pipeline by a wide margin:

          * a plate narrower than MIN_PLATE_WIDTH has no readable characters
            left, so OCR can only return noise;
          * a plate below BLUR_MIN is motion-blurred past reading, which is
            exactly what a fast vehicle produces.

        Both used to cost a full OCR pass and then feed a garbage string into
        the vote, where it competed with the real reads.
        """
        out = []
        ox, oy = offset

        results = self.plate_model(
            roi,
            imgsz=self._plate_imgsz(roi),
            conf=config.PLATE_CONF,
            iou=plate_scan.PLATE_IOU,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None:
            return out

        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            crop = roi[max(0, y1):y2, max(0, x1):x2]

            if crop.size == 0:
                continue

            shifted = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)

            if crop.shape[1] < config.MIN_PLATE_WIDTH:
                self.skipped_small += 1
                out.append((shifted, "", 0.0))
                continue

            if config.BLUR_MIN > 0 and self._sharpness(crop) < config.BLUR_MIN:
                self.skipped_blur += 1
                out.append((shifted, "", 0.0))
                continue

            text, conf = plate_scan.read_plate(crop, self.ocr)
            text, conf = self._as_plate(text, conf)
            out.append((shifted, text, conf))

        return out

    def _as_plate(self, text: str, conf: float):
        """
        Read the OCR string as an Indian plate, repairing what the structure
        rules out and rejecting what cannot be one.

        This is where AR26BT1607 becomes HR26BT1607. OCR confuses characters
        that look alike; the plate's structure says which member of a pair can
        stand in a given slot, and the state-code list says which two-letter
        codes exist at all. Neither is knowable from the characters alone,
        which is why cleaning the string could never fix these.

        The confidence is cut for each repair made. A repaired read is a
        deduction, not a sighting, and it must not clear HIGH_CONF_COMMIT on
        one look the way a clean read of the same plate would.
        """
        if not text or not config.PLATE_FORMAT:
            return text, conf

        parsed = plate_format.parse(
            text,
            states=config.PLATE_STATES or None,
            state_penalty=config.PLATE_STATE_PENALTY,
        )

        if parsed is None:
            # Not a plate. In strict mode it never reaches the vote, where it
            # would otherwise compete with the real reads.
            if config.PLATE_FORMAT_STRICT:
                self.rejected_format += 1
                return "", 0.0
            return text, conf

        if parsed["plate"] != text:
            self.repaired_format += 1

        penalty = 1.0 - min(0.5, 0.15 * parsed["repairs"])
        return parsed["plate"], conf * penalty

    def _maybe_commit(self, state: TrackState):
        winner = state.winner()

        if winner is None:
            return

        votes = winner["votes"]
        conf = state.top_conf.get(winner["text"], 0.0)

        # The normal path: several reads agreeing.
        ready = votes >= config.MIN_VOTES

        # The fast-vehicle path. A car crossing the frame in under a second
        # gets one or two looks, so demanding two agreeing reads means it is
        # never logged at all. One read that OCR is very sure of is better
        # evidence than nothing, and HIGH_CONF_COMMIT is set high enough that
        # a blurred guess does not clear it.
        if not ready and config.HIGH_CONF_COMMIT > 0:
            ready = conf >= config.HIGH_CONF_COMMIT

        if not ready:
            return

        state.committed = True
        self._emit(state, winner)

    def _sweep(self, seen: set, now: float):
        """Finalize tracks that have left the frame."""
        for track_id in list(self.tracks):
            if track_id in seen:
                continue

            state = self.tracks[track_id]

            if now - state.last_seen < config.TRACK_TTL:
                continue

            if not state.committed:
                self._finalize(state, track_id)

            del self.tracks[track_id]

    def _finalize(self, state: TrackState, track_id: int):
        """
        Last call on a vehicle that is leaving without having cleared the vote.

        Discarding it outright was losing exactly the traffic this service
        exists for: a vehicle moving fast enough to be seen once or twice, but
        read clearly on one of those looks. If the best read is confident
        enough it is emitted here, marked so the record shows it rested on
        fewer reads than a normal commit. Anything weaker is still logged as
        unread, so a camera reading nothing stays visible rather than silent.
        """
        winner = state.winner()
        votes = winner["votes"] if winner else 0
        best = winner["text"] if winner else ""
        conf = state.top_conf.get(best, 0.0) if best else 0.0

        if (best
                and votes >= config.EXIT_MIN_VOTES
                and conf >= config.EXIT_MIN_CONF):
            state.committed = True
            logger.info(
                f"[{self.camera_id}] committing {best} on exit "
                f"(votes={votes} conf={conf:.2f})"
            )
            self._emit(state, winner, provisional=True)
            return

        local_db.log_unread(
            self.camera_id, state.label, track_id, best, votes,
        )
        logger.info(
            f"[{self.camera_id}] unread {state.label} "
            f"ID:{track_id} (best={best or '-'} votes={votes} "
            f"conf={conf:.2f})"
        )

    # -- event emission ----------------------------------------------------

    def _snapshot(self, state: TrackState) -> Optional[str]:
        if state.best_crop is None or state.best_crop.size == 0:
            return None
        try:
            crop = state.best_crop
            h, w = crop.shape[:2]

            target = config.SNAPSHOT_WIDTH
            if w > target:
                crop = cv2.resize(crop, (target, int(h * target / w)),
                                  interpolation=cv2.INTER_AREA)

            if config.DETECTION_FORMAT == "png":
                ok, buf = cv2.imencode(".png", crop)
            else:
                ok, buf = cv2.imencode(
                    ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # Raw base64, no data: URI prefix — the API rejects the prefix.
            return base64.b64encode(buf).decode() if ok else None
        except Exception as e:
            logger.warning(f"[{self.camera_id}] snapshot failed: {e}")
            return None

    def _emit(self, state: TrackState, winner: dict,
              provisional: bool = False):
        plate = winner["text"]

        if not camera_manager.claim(plate, self.camera_id):
            logger.info(f"[{self.camera_id}] {plate} within cooldown, skipped")
            return

        # Survives a restart, where the in-memory claims do not.
        if local_db.seen_recently(plate, self.camera_id, config.PLATE_COOLDOWN):
            logger.info(f"[{self.camera_id}] {plate} already logged, skipped")
            return

        record = plate_db.lookup(plate)

        event = {
            "plate": plate,
            "siteId": config.SITE_ID,
            "cameraId": self.camera_id,
            "cameraName": self.display_name,
            "vehicleType": state.label,
            "trackId": state.track_id,
            "votes": winner["votes"],
            "variants": winner["variants"],
            "confidence": round(state.best_conf, 3),

            # True when the vehicle left before reaching MIN_VOTES and was
            # committed on the strength of one confident read instead. The
            # plate is reported either way; this says how much to trust it.
            "provisional": provisional,
            "registered": record is not None,
            "vehicleId": record.get("vehicleId") if record else None,
            "owner": record.get("owner") if record else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "capturedImage": self._snapshot(state),

            # False for a test video: it is logged locally and shown in the
            # stream, but never posted to the central server.
            "upload": self.upload,
        }

        # Cross-camera duplicate check. Deliberately after the event is built
        # and before it is handed off: a duplicate is still worth a local
        # record — it is evidence the vehicle was there — it just must not be
        # reported to the server a second time.
        if event["upload"] and not recent_plates.should_send(
                plate, self.camera_id, self.display_name):
            event["upload"] = False
            event["duplicate"] = True

        self.events_emitted += 1

        tag = "REGISTERED" if record else "UNREGISTERED"
        if provisional:
            tag += "/provisional"
        who = f" ({record['owner']})" if record and record.get("owner") else ""
        logger.info(
            f"[{self.camera_id}] {tag} {plate}{who} "
            f"votes={winner['votes']} {state.label} on {self.display_name}"
        )

        self.emit_cb(event)

    # -- preview -----------------------------------------------------------

    def _render(self, frame):
        canvas = frame.copy()

        for state in self.tracks.values():
            winner = state.winner()
            text = winner["text"] if winner else ""

            for box, read in state.overlay:
                draw_plate(canvas, box, read)

            if text:
                colour = (30, 200, 30) if state.committed else (0, 200, 255)
                cv2.putText(
                    canvas,
                    f"{state.label} #{state.track_id} {text}",
                    (10, 30 + 26 * (state.track_id % 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    colour,
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            canvas,
            f"{self.display_name}  "
            f"tracks={len(self.tracks)} events={self.events_emitted}",
            (10, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return canvas

    # -- health ------------------------------------------------------------

    def health(self) -> dict:
        age = (
            time.time() - self.last_frame_time
            if self.last_frame_time else float("inf")
        )
        status = "online" if age < 10 else self.status

        return {
            "cameraId": self.camera_id,
            "name": self.display_name,
            "status": status,
            "isReachable": status == "online",
            "source": "file" if self.is_file else "stream",
            "uploads": self.upload,
            "loops": (self.grabber.loops
                      if self.is_file and self.grabber else None),
            "activeTracks": len(self.tracks),
            "eventsEmitted": self.events_emitted,
            "lastFrameAgeMs": int(age * 1000) if age != float("inf") else None,

            # How long one detection pass costs, and how often the camera is
            # therefore looked at. detectionFps is the number that matters for
            # fast traffic: below ~3 a vehicle can cross the frame unseen.
            "passMs": round(self.avg_pass_ms, 1),
            "detectionFps": (round(1000.0 / self.avg_pass_ms, 2)
                             if self.avg_pass_ms > 0 else None),
            "framesRead": self.grabber.frames_read if self.grabber else 0,
            "skippedSmall": self.skipped_small,
            "skippedBlur": self.skipped_blur,
            "formatRepaired": self.repaired_format,
            "formatRejected": self.rejected_format,
            "startedAt": self.started_at,
            "lastChecked": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# Camera Manager
# =============================================================================

class CameraManager:
    def __init__(self):
        self.workers: Dict[str, CameraWorker] = {}
        self.events: "asyncio.Queue[dict]" = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # (plate, camera_id) -> emit time. The SQLite cooldown cannot cover the
        # window between emitting an event and event_pump writing its row, so
        # claim the plate here, synchronously, on the emitting thread.
        self._claims: Dict[Tuple[str, str], float] = {}
        self._claim_lock = threading.Lock()

    def claim(self, plate: str, camera_id: str) -> bool:
        """Reserve a plate for logging. False if it is still in cooldown."""
        now = time.time()
        key = (plate, camera_id)

        with self._claim_lock:
            for stale, when in list(self._claims.items()):
                if now - when > config.PLATE_COOLDOWN:
                    del self._claims[stale]

            if key in self._claims:
                return False

            self._claims[key] = now
            return True

    def bind_loop(self, loop):
        self._loop = loop

    def _emit(self, event: dict):
        """Called from a camera thread; hands the event to the event loop."""
        if self._loop is None:
            logger.error("Event loop not bound; dropping event")
            return
        self._loop.call_soon_threadsafe(self.events.put_nowait, event)

    def start_camera(self, camera_id: str, rtsp_url: str, name: str = "",
                     upload: bool = True, loop_file: bool = True) -> bool:
        self.stop_camera(camera_id)

        worker = CameraWorker(camera_id, rtsp_url, name, self._emit,
                              upload=upload, loop_file=loop_file)
        self.workers[camera_id] = worker
        worker.start()

        logger.info(f"Started camera {camera_id} ({name or camera_id})")
        return True

    def stop_camera(self, camera_id: str) -> bool:
        worker = self.workers.pop(camera_id, None)
        if worker is None:
            return False

        worker.stop()
        worker.join(timeout=10)
        logger.info(f"Stopped camera {camera_id}")
        return True

    def stop_all(self):
        for camera_id in list(self.workers):
            self.stop_camera(camera_id)

    def health(self) -> List[dict]:
        return [w.health() for w in self.workers.values()]


camera_manager = CameraManager()


# =============================================================================
# Camera Directory
# =============================================================================

class CameraDirectory:
    """
    Keeps the running cameras matched to the server's list.

    The server is the source of truth, but an edge box must survive it being
    unreachable, so the last good list is cached on disk and used at startup
    when the fetch fails.
    """

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.known: Dict[str, dict] = {}     # id -> {name, url}

    # -- cache ------------------------------------------------------------

    def _load_cache(self) -> List[dict]:
        if not os.path.exists(self.cache_path):
            return []
        try:
            with open(self.cache_path) as fh:
                return json.load(fh)
        except Exception as e:
            logger.warning(f"Could not read camera cache: {e}")
            return []

    def _save_cache(self, cameras: List[dict]):
        parent = os.path.dirname(self.cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        tmp = f"{self.cache_path}.tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(cameras, fh, indent=2)
            os.replace(tmp, self.cache_path)
        except Exception as e:
            logger.warning(f"Could not write camera cache: {e}")

    # -- normalising ------------------------------------------------------

    @staticmethod
    def _clean(cameras: List[dict]) -> Dict[str, dict]:
        """Keep only usable entries, keyed by id."""

        out: Dict[str, dict] = {}

        for cam in cameras:
            cam_id = str(cam.get("id") or "").strip()
            url = str(cam.get("streamUrl") or "").strip()
            name = str(cam.get("name") or "").strip() or cam_id

            if not cam_id:
                logger.warning(f"Camera with no id, skipped: {cam}")
                continue

            if not url:
                logger.warning(
                    f"Camera {name!r} ({cam_id}) has no streamUrl, skipped")
                continue

            out[cam_id] = {"id": cam_id, "name": name, "url": url}

        # Two ids on one URL means two workers pulling the same feed: double
        # the decode and inference cost, and the same vehicle reported twice
        # under different camera ids.
        by_url: Dict[str, List[str]] = {}
        for cam in out.values():
            by_url.setdefault(cam["url"], []).append(cam["name"])

        for url, names in by_url.items():
            if len(names) > 1:
                logger.warning(
                    f"{len(names)} cameras share one stream URL "
                    f"({', '.join(names)}). Each gets its own worker, so the "
                    f"feed is decoded once per camera and a vehicle is "
                    f"reported once per camera id."
                )

        return out

    # -- reconciliation ---------------------------------------------------

    async def refresh(self, first_run: bool = False) -> int:
        """Fetch the list and make the running cameras match it."""

        fetched = None

        if CAMERA_LIST_AVAILABLE:
            fetched = await central_api.fetch_camera_list()

        if fetched is None:
            if not first_run:
                logger.info(
                    "Camera list unavailable; leaving running cameras alone")
                return len(self.known)

            cached = self._load_cache()
            if not cached:
                logger.error(
                    "Cannot reach the camera list and no cache to fall back "
                    "on. No cameras will start. "
                    f"Check {config.CENTRAL_API_URL}{config.EP_CAMERA_LIST}"
                )
                return 0

            logger.warning(
                f"Camera list unavailable; starting {len(cached)} camera(s) "
                f"from the cache written at "
                f"{datetime.fromtimestamp(os.path.getmtime(self.cache_path))}"
            )
            fetched = cached
        else:
            self._save_cache(fetched)

        desired = self._clean(fetched)

        added = [c for cid, c in desired.items() if cid not in self.known]
        removed = [self.known[cid] for cid in self.known if cid not in desired]
        changed = [
            c for cid, c in desired.items()
            if cid in self.known and (
                c["url"] != self.known[cid]["url"]
                or c["name"] != self.known[cid]["name"]
            )
        ]

        for cam in removed:
            logger.info(f"Camera removed on server: {cam['name']} ({cam['id']})")
            camera_manager.stop_camera(cam["id"])

        for cam in changed:
            logger.info(f"Camera changed on server: {cam['name']} ({cam['id']})")
            camera_manager.start_camera(cam["id"], cam["url"], cam["name"])

        if added:
            cpus = os.cpu_count() or 1
            total = len(desired)

            # Re-split the cores now the camera count is known.
            tune_cpu_threads(total)

            logger.info(
                f"Starting {len(added)} camera(s) "
                f"({total} total, {cpus} CPU(s))"
                + ("  [plate model + OCR shared]" if config.SHARE_MODELS
                   else "  [each camera loads its own models]")
            )

            if total > cpus:
                logger.warning(
                    f"{total} cameras on {cpus} CPU(s): detection passes will "
                    f"share cycles. Watch detectionFps in /status: below "
                    f"about 3 per camera, fast vehicles start crossing the "
                    f"frame between passes. Drop VEHICLE_IMGSZ/PLATE_IMGSZ, "
                    f"or use smaller models, before adding more cameras."
                )

        for i, cam in enumerate(added):
            camera_manager.start_camera(cam["id"], cam["url"], cam["name"])
            logger.info(f"  {cam['name']}  id={cam['id']}")

            # Several model loads at once spike memory and CPU.
            if config.CAMERA_START_STAGGER and i < len(added) - 1:
                await asyncio.sleep(config.CAMERA_START_STAGGER)

        self.known = desired
        return len(desired)


camera_directory = CameraDirectory(config.CAMERA_CACHE_PATH)


# =============================================================================
# MJPEG Streaming
# =============================================================================

def mjpeg_frames(camera_id: str):
    while True:
        worker = camera_manager.workers.get(camera_id)

        if worker is None:
            break

        # Tells the worker to keep rendering. Without a viewer it skips the
        # frame copy and the overlay entirely, which is a meaningful slice of
        # the per-frame cost on a CPU-only box.
        worker.preview_wanted = time.time()

        frame = worker.latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)))

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            continue

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )

        time.sleep(0.05)


# =============================================================================
# Background Tasks
# =============================================================================

async def event_pump():
    """
    Drain camera events: record locally first, then upload.

    Local write happens before the upload so a detection survives whatever the
    network does. A failed upload goes on the retry queue, not into the void.
    """

    while True:
        event = await camera_manager.events.get()

        try:
            local_db.log_plate(event)

            if not event.get("upload", True):
                if not event.get("duplicate"):
                    logger.info(
                        f"TEST SOURCE {event['plate']} logged locally only "
                        f"(upload disabled for {event['cameraId']})"
                    )
                # A duplicate has already said so, with the age that made it
                # one; repeating it here would only be noise.
                continue

            if not CENTRAL_ENABLED:
                local_db.add_to_sync_queue("detection", event)
                continue

            result = await central_api.send_detection(event)

            if result.get("success"):
                stored = (result.get("data") or {}).get("id")
                logger.info(
                    f"Uploaded {event['plate']}"
                    + (f" (server id {stored})" if stored else "")
                )
            elif result.get("no_image"):
                # Nothing to retry with; it is already in the local database.
                pass
            else:
                local_db.add_to_sync_queue("detection", event)
                logger.info(f"Queued {event['plate']} for retry")

        except Exception as e:
            logger.error(f"Event pump error: {e}")
            local_db.add_to_sync_queue("detection", event)


async def sync_registered_plates() -> int:
    """Pull the authorized plate list from central."""
    if not CENTRAL_ENABLED:
        logger.info("Standalone mode: skipping registered plate sync")
        return 0

    records = await central_api.fetch_registered_plates()

    if not records:
        logger.warning("No registered plates returned; keeping existing list")
        return len(plate_db)

    count = plate_db.replace_all(records)
    logger.info(f"Synced {count} registered plates")
    return count


async def plate_sync_task():
    """Unlike the face service, the list refreshes on a timer, not just boot."""
    while True:
        await asyncio.sleep(config.SYNC_INTERVAL)

        if central_api.unavailable(config.EP_REGISTERED):
            logger.info("Registered-plate route absent; stopping sync task")
            return

        try:
            await sync_registered_plates()
        except Exception as e:
            logger.error(f"Plate sync task error: {e}")


async def pending_sync_task():
    if not CENTRAL_ENABLED:
        logger.info("Standalone mode: retry task disabled")
        return

    while True:
        await asyncio.sleep(60)

        try:
            pending = local_db.get_pending_sync(50)
            if not pending:
                continue

            synced, failed = [], []
            reason = None

            for item in pending:
                result = await central_api.send_detection(
                    item["data"], quiet=True)

                if result.get("success") or result.get("no_image"):
                    # no_image can never succeed, so stop retrying it.
                    synced.append(item["id"])
                else:
                    failed.append(item["id"])
                    reason = reason or result.get("status") or result.get(
                        "error")

                    # The server is down, not rejecting this one detection.
                    # Trying the other 49 only produces 49 more of the same
                    # error; they keep their place in the queue either way.
                    if result.get("unreachable"):
                        failed.extend(
                            i["id"] for i in pending[len(synced) + len(failed):]
                        )
                        break

            local_db.mark_synced(synced)
            local_db.bump_attempts(failed)

            if synced:
                logger.info(f"Synced {len(synced)} queued detection(s)")

            if failed:
                # One line per cycle, not one per queued detection. Says
                # plainly that the data is safe, which the old per-item
                # warnings did not.
                logger.warning(
                    f"{len(failed)} detection(s) still queued for upload "
                    f"— {reason}. They are stored locally and will be sent "
                    f"when {config.CENTRAL_API_URL} answers again."
                )

        except Exception as e:
            logger.error(f"Pending sync error: {e}")


async def heartbeat_task():
    if not CENTRAL_ENABLED:
        logger.info("Standalone mode: heartbeat disabled")
        return

    await asyncio.sleep(15)

    while True:
        try:
            await central_api.send_heartbeat(camera_manager.health())
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")

        if central_api.unavailable(config.EP_HEARTBEAT):
            logger.info("Heartbeat route absent; stopping heartbeat task")
            return

        await asyncio.sleep(config.HEARTBEAT_INTERVAL)


async def camera_sync_task():
    """
    Re-read the server's camera list on a timer.

    Cameras added, removed or re-pointed on the server take effect without a
    restart. A failed fetch leaves the running cameras exactly as they are —
    a blip in the API must not take the road cameras down.
    """

    while True:
        await asyncio.sleep(config.CAMERA_SYNC_INTERVAL)

        try:
            await camera_directory.refresh()
        except Exception as e:
            logger.error(f"Camera sync error: {e}")


def start_test_video_if_configured():
    """TEST_VIDEO is the one camera still declared locally, for testing."""

    if not config.TEST_VIDEO:
        return

    if not os.path.exists(config.TEST_VIDEO):
        logger.error(f"TEST_VIDEO not found: {config.TEST_VIDEO}")
        return

    camera_manager.start_camera(
        config.TEST_VIDEO_ID,
        config.TEST_VIDEO,
        "test video",
        upload=config.TEST_VIDEO_UPLOAD,
        loop_file=config.TEST_VIDEO_LOOP,
    )

    logger.info(
        f"TEST VIDEO {config.TEST_VIDEO} as {config.TEST_VIDEO_ID!r} "
        f"— uploads "
        + ("ENABLED" if config.TEST_VIDEO_UPLOAD else "disabled")
        + f".  Watch: http://localhost:{config.PORT}"
          f"/stream/{config.TEST_VIDEO_ID}"
    )


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PlateScope Edge...")
    logger.info(f"Site: {config.SITE_NAME} ({config.SITE_ID or 'standalone'})")
    logger.info(f"Central: {config.CENTRAL_API_URL or 'disabled'}")

    camera_manager.bind_loop(asyncio.get_running_loop())

    tune_cpu_threads(1)

    await sync_registered_plates()

    tasks = [
        asyncio.create_task(event_pump()),
        asyncio.create_task(plate_sync_task()),
        asyncio.create_task(pending_sync_task()),
        asyncio.create_task(heartbeat_task()),
        asyncio.create_task(camera_sync_task()),
    ]

    # Cameras come from the server, not from .env.
    count = await camera_directory.refresh(first_run=True)
    logger.info(f"{count} camera(s) from the server")

    start_test_video_if_configured()

    yield

    logger.info("Shutting down...")
    camera_manager.stop_all()

    for task in tasks:
        task.cancel()

    logger.info("PlateScope Edge stopped")


app = FastAPI(title="PlateScope Edge", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CameraStartRequest(BaseModel):
    camera_id: str
    rtsp_url: str
    name: str = ""
    upload: bool = True
    loop: bool = True


class TestVideoRequest(BaseModel):
    """Play a video file through the pipeline instead of a camera."""

    path: str
    camera_id: str = "test-video"

    # Defaults to off so a looping test clip cannot fill the central server
    # with detections. Set true, with a real registered cameraId, to test
    # the upload path end to end.
    upload: bool = False
    loop: bool = True


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "plate_recognition",
        "site_id": config.SITE_ID,
        "site_name": config.SITE_NAME,
        "registered_plates": len(plate_db),
        "cameras_active": len(camera_manager.workers),
        "cameras_known": len(camera_directory.known),
        "central_enabled": CENTRAL_ENABLED,
    }


@app.get("/status")
async def status():
    return {
        "success": True,
        "data": {
            "site": {"id": config.SITE_ID, "name": config.SITE_NAME},
            "cameraSource": f"{config.CENTRAL_API_URL}{config.EP_CAMERA_LIST}",
            "registeredPlates": len(plate_db),
            "cameras": camera_manager.health(),
            "database": local_db.stats(),
            "config": {
                "minVotes": config.MIN_VOTES,
                "highConfCommit": config.HIGH_CONF_COMMIT,
                "exitMinVotes": config.EXIT_MIN_VOTES,
                "exitMinConf": config.EXIT_MIN_CONF,
                "detectInterval": config.DETECT_INTERVAL,
                "plateImgsz": config.PLATE_IMGSZ,
                "plateImgszMax": config.PLATE_IMGSZ_MAX,
                "minPlateWidth": config.MIN_PLATE_WIDTH,
                "blurMin": config.BLUR_MIN,
                "plateFormat": config.PLATE_FORMAT,
                "plateFormatStrict": config.PLATE_FORMAT_STRICT,
                "plateStates": sorted(config.PLATE_STATES) or None,
                "vehicleImgsz": config.VEHICLE_IMGSZ,
                "vehicleModel": plate_scan.VEHICLE_MODEL_PATH,
                "plateModel": plate_scan.PLATE_MODEL_PATH,
                "plateCooldown": config.PLATE_COOLDOWN,
                "dedupeWindow": config.DEDUPE_WINDOW,
                "dedupeFile": config.DEDUPE_FILE,
                "shareModels": config.SHARE_MODELS,
                "torchThreads": torch.get_num_threads(),
            },
        },
    }


@app.get("/recent-plates")
async def recent_plate_window():
    """
    What is currently inside the cross-camera duplicate window.

    The same content as DEDUPE_FILE, with each entry's age worked out, so the
    window can be watched while the service runs.
    """
    entries = recent_plates.snapshot()
    return {
        "success": True,
        "data": {
            "windowSeconds": config.DEDUPE_WINDOW,
            "file": config.DEDUPE_FILE,
            "count": len(entries),
            "plates": entries,
        },
    }


@app.get("/cameras")
async def list_cameras():
    return {"success": True, "data": {"cameras": camera_manager.health()}}


@app.post("/cameras/start")
async def start_camera(req: CameraStartRequest):
    camera_manager.start_camera(
        req.camera_id, req.rtsp_url, req.name,
        upload=req.upload, loop_file=req.loop,
    )
    return {"success": True, "message": f"Camera {req.camera_id} started"}


@app.post("/test/video")
async def start_test_video(req: TestVideoRequest):
    """
    Run a video file through the full pipeline, for testing.

    Detection, voting, snapshots, the local database and the MJPEG preview all
    behave exactly as they do for a camera. Uploading to the central server is
    the one thing that is off by default.
    """

    if not os.path.exists(req.path):
        raise HTTPException(404, f"No such file: {req.path}")

    camera_manager.start_camera(
        req.camera_id, req.path, "test video",
        upload=req.upload, loop_file=req.loop,
    )

    return {
        "success": True,
        "message": f"Playing {req.path}",
        "data": {
            "cameraId": req.camera_id,
            "watch": f"/stream/{req.camera_id}",
            "uploadsToCentral": req.upload,
            "looping": req.loop,
        },
    }


@app.post("/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str):
    if not camera_manager.stop_camera(camera_id):
        raise HTTPException(404, "Camera not found")
    return {"success": True, "message": f"Camera {camera_id} stopped"}


@app.get("/stream/{camera_id}")
async def stream(camera_id: str):
    if camera_id not in camera_manager.workers:
        raise HTTPException(404, "Camera not found")

    return StreamingResponse(
        mjpeg_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/events")
async def events(limit: int = 50):
    return {
        "success": True,
        "data": {"events": local_db.recent_events(min(limit, 500))},
    }


@app.get("/plates")
async def registered_plates():
    return {
        "success": True,
        "data": {
            "count": len(plate_db),
            "plates": list(plate_db.plates.values()),
        },
    }


@app.post("/sync/cameras")
async def trigger_camera_sync():
    """Re-read the camera list from the server now, without restarting."""
    count = await camera_directory.refresh()
    return {
        "success": True,
        "data": {
            "cameras": count,
            "running": len(camera_manager.workers),
        },
    }


@app.post("/sync/plates")
async def trigger_plate_sync():
    """Refresh the registered list and report the new count."""
    count = await sync_registered_plates()
    return {"success": True, "data": {"registeredPlates": count}}


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PLATESCOPE EDGE — Live Plate Recognition")
    print("=" * 60)
    print(f"  Site:    {config.SITE_NAME}")
    print(f"  Port:    {config.PORT}")
    print(f"  Central: {config.CENTRAL_API_URL or 'standalone'}")
    print("=" * 60 + "\n")

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
