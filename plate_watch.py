"""
PLATE WATCH — simple detection logger.

A stripped-down way to see what the model actually sees. No server, no
database, no central sync. It watches one camera (or a video file), and every
time it reads a plate it:

  1. prints a line to the terminal
  2. saves the image the plate was read from
  3. appends a row to a CSV

That is all it does.

  Standalone by design
  --------------------
  This imports only plate_scan.py, exactly as plate_edge.py does. It does not
  import plate_edge, does not touch data/plates.db, and does not talk to the
  central server. Running it changes nothing about the main service — you can
  run both at once, though on one machine they will compete for CPU.

Usage
-----
    # use the camera from .env
    python plate_watch.py

    # a specific camera
    python plate_watch.py --source rtsp://USER:PASSWORD@CAMERA-IP:554/Streaming/Channels/101

    # a video file, to check the model without a camera
    python plate_watch.py --source videos/bike_plates.mp4

    # name the camera in the log
    python plate_watch.py --camera-id gate-1

Stop with Ctrl+C. A summary is printed on exit.
"""

import os
import csv
import sys
import time
import argparse

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


from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

_env = Path(__file__).parent / ".env"
load_dotenv(_env if _env.exists() else Path(__file__).parent / ".env.example")

import cv2

# Imports that only exist inside the virtual environment. Running this with the
# system python is the easy mistake to make, and the bare ModuleNotFoundError
# does not hint at the cause, so say it plainly.
try:
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

# Same detection and voting code the real service uses.
import plate_scan
from plate_scan import cluster_reads, clean_plate_text, draw_plate


# ===========================
# Settings
# ===========================

OUT_DIR = os.getenv("WATCH_OUT_DIR", "detections")

# Seconds between detection passes, for a live camera. A stream delivers
# frames in real time, so pacing by the clock is what limits CPU.
DETECT_INTERVAL = float(os.getenv("DETECT_INTERVAL", "0.4"))

# Frames between detection passes, for a video file. A file is read as fast as
# the disk allows, so clock pacing would race through the whole clip in a few
# seconds and take only a handful of passes — never enough reads to reach
# MIN_VOTES. Count frames instead, the way plate_scan.py does.
DETECT_EVERY_N = int(os.getenv("DETECT_EVERY_N", "10"))

# Agreeing reads needed before a plate is accepted.
MIN_VOTES = int(os.getenv("MIN_VOTES", "2"))

# A vehicle unseen for this long is forgotten.
TRACK_TTL = float(os.getenv("TRACK_TTL", "3.0"))

VEHICLE_IMGSZ = int(os.getenv("VEHICLE_IMGSZ", "480"))
VEHICLE_CONF = float(os.getenv("VEHICLE_CONF", "0.3"))

plate_scan.PLATE_IMGSZ = int(os.getenv("PLATE_IMGSZ", "320"))
plate_scan.PLATE_CONF = float(os.getenv("PLATE_CONF", "0.35"))


# ===========================
# OCR with the raw read kept
# ===========================
#
# plate_scan.plates_in_image() returns text that has already been through
# clean_plate_text(), so a read that OCR made but validation threw away comes
# back as "" and you cannot tell whether the plate was never seen, never read,
# or read and rejected. That distinction is the whole question when something
# is not being detected, so these two functions do the same work while keeping
# the raw string.
#
# Validation itself still goes through plate_scan.clean_plate_text, so what
# counts as a valid plate here is identical to the real service.

def reject_reason(raw):
    """Why clean_plate_text() threw this read away. None if it is valid."""

    if not raw:
        return "nothing read"

    import re as _re
    stripped = _re.sub(r"[^A-Z0-9]", "", raw.upper())

    if not stripped:
        return "no letters or digits"

    if not _re.search(r"[A-Z]", stripped):
        return "no letters"

    if not _re.search(r"[0-9]", stripped):
        return "no digits"

    if len(stripped) < plate_scan.MIN_PLATE_LEN:
        return f"too short ({len(stripped)}, need {plate_scan.MIN_PLATE_LEN})"

    if len(stripped) > plate_scan.MAX_PLATE_LEN:
        return f"too long ({len(stripped)}, max {plate_scan.MAX_PLATE_LEN})"

    return None


def read_plate_verbose(crop, ocr_engine):
    """OCR one plate crop, returning (raw_text, cleaned_text, confidence)."""

    results = ocr_engine.ocr(crop)

    if not results:
        return "", "", 0.0

    raw = ""
    best_conf = 0.0

    for res in results:

        if res is None:
            continue

        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []

        for i, text in enumerate(texts):

            if not text:
                continue

            raw += text

            if i < len(scores):
                best_conf = max(best_conf, float(scores[i]))

    return raw, clean_plate_text(raw), best_conf


def plates_verbose(img, plate_model, ocr_engine, offset=(0, 0)):
    """
    Every plate box in one image, with what OCR made of it.

    Returns dicts: box (full-frame coords), raw, text, conf, reason.
    """

    out = []

    results = plate_model(
        img,
        imgsz=plate_scan.PLATE_IMGSZ,
        conf=plate_scan.PLATE_CONF,
        iou=plate_scan.PLATE_IOU,
        verbose=False,
    )

    boxes = results[0].boxes

    if boxes is None:
        return out

    ox, oy = offset

    for box in boxes.xyxy.cpu().numpy():

        x1, y1, x2, y2 = map(int, box)

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        raw, text, conf = read_plate_verbose(crop, ocr_engine)

        out.append({
            "box": (x1 + ox, y1 + oy, x2 + ox, y2 + oy),
            "raw": raw,
            "text": text,
            "conf": conf,
            "reason": reject_reason(raw),
        })

    return out


# ===========================
# Output
# ===========================

class DetectionLog:
    """Writes one CSV row and one JPEG per detection."""

    COLUMNS = [
        "timestamp", "camera_id", "plate", "votes",
        "confidence", "vehicle_type", "image", "variants",
    ]

    def __init__(self, out_dir):
        self.day = datetime.now().strftime("%Y-%m-%d")
        self.image_dir = Path(out_dir) / self.day
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = Path(out_dir) / "detections.csv"
        new_file = not self.csv_path.exists()

        self.fh = open(self.csv_path, "a", newline="")
        self.writer = csv.writer(self.fh)

        if new_file:
            self.writer.writerow(self.COLUMNS)
            self.fh.flush()

        self.count = 0

    def save(self, camera_id, plate, votes, confidence,
             vehicle_type, variants, image):

        now = datetime.now()
        stamp = now.strftime("%Y%m%d-%H%M%S")

        # Camera id and plate can both contain characters that are awkward in a
        # filename; keep only safe ones.
        safe_cam = "".join(c if c.isalnum() or c in "-_" else "_"
                           for c in camera_id)

        filename = f"{safe_cam}_{plate}_{stamp}.jpg"
        path = self.image_dir / filename

        saved = ""
        if image is not None and image.size > 0:
            if cv2.imwrite(str(path), image):
                saved = str(path)

        self.writer.writerow([
            now.isoformat(timespec="seconds"),
            camera_id,
            plate,
            votes,
            round(confidence, 3),
            vehicle_type,
            saved,
            " ".join(variants),
        ])
        self.fh.flush()   # flush every row so tail -f works and Ctrl+C is safe

        self.count += 1

        print(
            f"{now.strftime('%H:%M:%S')}  {camera_id}  {plate:<12} "
            f"votes={votes}  conf={confidence:.2f}  {vehicle_type}",
            flush=True,
        )

        if saved:
            print(f"                    saved {saved}", flush=True)

    def close(self):
        self.fh.close()


# ===========================
# Live view
# ===========================


def _screen_size():
    """
    Usable screen size, so the preview can be scaled to fit it.

    A window taller than the display does not just overflow — the Qt backend
    OpenCV uses collapses it to a stub showing only the toolbar. A 1280x960
    camera on a 1366x768 screen hits this exactly.
    """

    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        # No display, or no tk. Assume a small laptop screen; too small only
        # costs a bit of preview detail, too large loses the window entirely.
        return (1280, 720)


class LiveView:
    """
    Shows the camera with everything the detector is thinking drawn on top.

    Falls back to writing a video file when there is no display (SSH, headless
    box), because cv2.imshow raises rather than degrading on its own.
    """

    GREEN = (60, 200, 60)      # accepted
    RED = (60, 60, 230)        # read but rejected
    GREY = (150, 150, 150)     # plate box, nothing read
    BLUE = (230, 160, 40)      # vehicle
    WHITE = (255, 255, 255)

    def __init__(self, enabled, save_path, camera_id, fps, size,
                 max_width=None, topmost=True):
        self.enabled = enabled
        self.camera_id = camera_id

        # Leave room for the window frame, the OpenCV toolbar and the taskbar.
        screen_w, screen_h = _screen_size()
        self.max_w = max_width or int(screen_w * 0.92)
        self.max_h = int(screen_h * 0.86)
        self.window = f"plate_watch — {camera_id}   [q]uit [p]ause [s]ave"
        self.paused = False
        self.writer = None
        self.snaps = 0
        self.failed = False
        self.opened = False
        self.topmost = topmost

        if save_path:
            self.writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(fps, 1),
                size,
            )
            if not self.writer.isOpened():
                print(f"WARNING: cannot write {save_path}", file=sys.stderr)
                self.writer = None

    def draw(self, frame, vehicles, plates, stats):
        """
        vehicles : (box, label, track_id, votes, committed, plate)
        plates   : dicts from plates_verbose
        """

        canvas = frame.copy()

        for (x1, y1, x2, y2), label, tid, votes, done, plate in vehicles:

            colour = self.GREEN if done else self.BLUE
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

            caption = f"{label} #{tid}"

            if plate:
                caption += f"  {plate}"

            caption += f"  {votes}/{MIN_VOTES}"

            cv2.putText(canvas, caption, (x1, max(14, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2,
                        cv2.LINE_AA)

        for p in plates:

            x1, y1, x2, y2 = p["box"]

            if p["text"]:
                colour, note = self.GREEN, p["text"]
            elif p["raw"]:
                colour, note = self.RED, f'{p["raw"]}  <- {p["reason"]}'
            else:
                colour, note = self.GREY, "plate found, no text"

            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

            # Below the box, so it does not collide with the vehicle caption.
            cv2.putText(canvas, note, (x1, min(canvas.shape[0] - 6, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2,
                        cv2.LINE_AA)

        self._hud(canvas, stats)
        return canvas

    def _hud(self, canvas, stats):
        h, w = canvas.shape[:2]

        lines = [
            f"{self.camera_id}   {w}x{h}",
            f"frames {stats['frames']}   passes {stats['passes']}"
            f"   {stats['pass_ms']} ms/pass",
            f"vehicles {stats['tracked']}   plates logged {stats['logged']}",
            stats["hint"],
        ]

        cv2.rectangle(canvas, (0, 0), (min(w, 470), 20 + 22 * len(lines)),
                      (0, 0, 0), -1)

        for i, line in enumerate(lines):
            if not line:
                continue
            cv2.putText(canvas, line, (10, 26 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.WHITE, 1,
                        cv2.LINE_AA)

        if self.paused:
            cv2.putText(canvas, "PAUSED", (w // 2 - 70, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3,
                        cv2.LINE_AA)

    def show(self, canvas, out_dir):
        """Returns False when the user asks to quit."""

        if self.writer is not None:
            self.writer.write(canvas)

        if not self.enabled or self.failed:
            return True

        # Fit inside the screen on BOTH axes. Checking width alone is not
        # enough: a 1280x960 camera is within any width limit yet still too
        # tall for a 768px display.
        h, w = canvas.shape[:2]
        scale = min(self.max_w / w, self.max_h / h, 1.0)

        if scale < 1.0:
            canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA)

        try:
            if not self.opened:
                # GNOME (and most WMs) will not raise a window created by a
                # background process — it opens behind whatever is focused and
                # you get an "is ready" notification instead. On a full-screen
                # editor that looks exactly like nothing happened. Ask for
                # top-most so the preview is actually visible.
                cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
                if self.topmost:
                    try:
                        cv2.setWindowProperty(
                            self.window, cv2.WND_PROP_TOPMOST, 1)
                    except cv2.error:
                        pass

            cv2.imshow(self.window, canvas)

            if not self.opened:
                self.opened = True
                print(f"Window    : open ({canvas.shape[1]}x{canvas.shape[0]}), "
                      f"kept on top. If you still cannot see it, check the "
                      f"taskbar or Alt-Tab.", flush=True)
        except cv2.error as e:
            self.failed = True
            print("\nCannot open a window (no display?). Continuing without "
                  "the live view.", file=sys.stderr)
            print(f"  {e}", file=sys.stderr)
            print("  Over SSH, use --save-video out.mp4 instead.\n",
                  file=sys.stderr)
            return True

        while True:
            key = cv2.waitKey(1 if not self.paused else 100) & 0xFF

            if key in (ord("q"), 27):
                return False

            if key == ord("p"):
                self.paused = not self.paused
                if not self.paused:
                    break
                continue

            if key == ord("s"):
                self.snaps += 1
                path = Path(out_dir) / f"snapshot_{self.snaps:03d}.jpg"
                cv2.imwrite(str(path), canvas)
                print(f"          snapshot -> {path}", flush=True)

            if not self.paused:
                break

        return True

    def close(self):
        if self.writer is not None:
            self.writer.release()

        if self.enabled and not self.failed:
            cv2.destroyAllWindows()


# ===========================
# Per-vehicle vote tracking
# ===========================

class Track:
    """Reads collected for one tracked vehicle."""

    def __init__(self, label, now):
        self.label = label
        self.misses = 0
        self.rejects = []
        self.counts = {}
        self.best_conf = 0.0
        self.best_area = 0
        self.best_image = None
        self.last_seen = now
        self.done = False

    def winner(self):
        if not self.counts:
            return None
        clusters = cluster_reads(self.counts)
        return clusters[0] if clusters else None


# ===========================
# Main loop
# ===========================

def watch(source, camera_id, out_dir, show_misses,
          show=False, save_video=None, view_width=None,
          no_topmost=False):

    print("Loading models... (10-30 seconds)")

    vehicle_model = YOLO(plate_scan.VEHICLE_MODEL_PATH)
    plate_model = YOLO(plate_scan.PLATE_MODEL_PATH)
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print("Models ready.\n")

    log = DetectionLog(out_dir)

    print(f"Watching : {source}")
    print(f"Camera   : {camera_id}")
    print(f"Images   : {log.image_dir}/")
    print(f"CSV      : {log.csv_path}")
    print(f"Votes    : {MIN_VOTES} agreeing reads needed")
    print("\nCtrl+C to stop.\n")

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    # Keep only the newest frame. Detection is slower than 25 fps, so without
    # this the decoder queue grows and every read returns older, staler video.
    if "://" in str(source):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: cannot open {source}", file=sys.stderr)
        print("  - for a camera, test the URL first:  ffplay \"<url>\"",
              file=sys.stderr)
        print("  - for a file, check the path is right", file=sys.stderr)
        return 1

    view = LiveView(
        enabled=show,
        save_path=save_video,
        camera_id=camera_id,
        fps=cap.get(cv2.CAP_PROP_FPS) or 25,
        size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        max_width=view_width,
        topmost=not no_topmost,
    )

    if show:
        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fit = min(view.max_w / max(cam_w, 1), view.max_h / max(cam_h, 1), 1.0)
        print(f"Live view : on   {cam_w}x{cam_h}"
              + (f" shown at {int(cam_w * fit)}x{int(cam_h * fit)}"
                 if fit < 1.0 else " (full size)")
              + "   (q quit, p pause, s snapshot)")
    if save_video:
        print(f"Video out : {save_video}")

    overlay_vehicles = []
    overlay_plates = []
    passes = 0
    pass_ms = 0
    hint = ""
    quit_requested = False

    # rtsp:// or http:// is a live stream; anything else is a file on disk.
    is_stream = "://" in str(source)

    print("Pacing   : "
          + (f"every {DETECT_INTERVAL}s (live stream)" if is_stream
             else f"every {DETECT_EVERY_N} frames (file)"))
    print()

    tracks = {}
    last_detect = 0.0
    frames = 0
    vehicles_seen = set()
    started = time.time()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                # End of file, or the stream dropped.
                print("\nSource ended.")
                break

            frames += 1
            now = time.time()

            if frames == 1:
                print(f"First frame: {frame.shape[1]}x{frame.shape[0]} "
                      f"after {time.time() - started:.1f}s", flush=True)

            due = (now - last_detect >= DETECT_INTERVAL) if is_stream \
                else (frames % DETECT_EVERY_N == 0)

            if not due:
                # Not a detection frame. Still draw, so the view plays at full
                # rate with the most recent boxes rather than stuttering.
                if view.enabled or view.writer is not None:
                    canvas = view.draw(frame, overlay_vehicles, overlay_plates, {
                        "frames": frames, "passes": passes,
                        "pass_ms": pass_ms, "tracked": len(tracks),
                        "logged": log.count, "hint": hint,
                    })
                    if not view.show(canvas, out_dir):
                        quit_requested = True
                        break
                continue

            if is_stream:
                last_detect = now

            passes += 1
            pass_started = time.time()

            if passes == 1:
                print("First detection pass starting (models warm up here, "
                      "can take several seconds)...", flush=True)
            overlay_vehicles = []
            overlay_plates = []

            results = vehicle_model.track(
                frame,
                persist=True,
                imgsz=VEHICLE_IMGSZ,
                conf=VEHICLE_CONF,
                iou=plate_scan.VEHICLE_IOU,
                classes=plate_scan.VEHICLE_CLASSES,
                verbose=False,
            )

            boxes = results[0].boxes
            seen = set()
            h, w = frame.shape[:2]

            if boxes is not None:
                for box in boxes:

                    if box.id is None:
                        continue

                    track_id = int(box.id.item())
                    seen.add(track_id)
                    vehicles_seen.add(track_id)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    label = vehicle_model.names[int(box.cls)]

                    track = tracks.get(track_id)
                    if track is None:
                        track = Track(label, now)
                        tracks[track_id] = track

                    track.last_seen = now

                    if track.done:
                        _w = track.winner()
                        overlay_vehicles.append((
                            (x1, y1, x2, y2), label, track_id,
                            _w["votes"] if _w else 0, True,
                            _w["text"] if _w else "",
                        ))
                        continue

                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    found = plates_verbose(
                        roi, plate_model, ocr, offset=(x1, y1)
                    )

                    overlay_plates.extend(found)

                    for p in found:
                        if p["text"]:
                            track.counts[p["text"]] = \
                                track.counts.get(p["text"], 0) + 1
                            track.best_conf = max(track.best_conf, p["conf"])
                        elif p["raw"]:
                            track.rejects.append(f'{p["raw"]} ({p["reason"]})')

                    # Keep the biggest view of this vehicle, with the plate
                    # boxes drawn on it — this is the image that gets saved.
                    area = (x2 - x1) * (y2 - y1)
                    if area > track.best_area:
                        snap = roi.copy()
                        for p in found:
                            bx1, by1, bx2, by2 = p["box"]
                            draw_plate(
                                snap,
                                (bx1 - x1, by1 - y1, bx2 - x1, by2 - y1),
                                p["text"],
                            )
                        track.best_area = area
                        track.best_image = snap

                    winner = track.winner()

                    _w = track.winner()
                    overlay_vehicles.append((
                        (x1, y1, x2, y2), label, track_id,
                        _w["votes"] if _w else 0,
                        track.done,
                        _w["text"] if _w else "",
                    ))

                    if winner and winner["votes"] >= MIN_VOTES:
                        track.done = True
                        log.save(
                            camera_id,
                            winner["text"],
                            winner["votes"],
                            track.best_conf,
                            track.label,
                            winner["variants"],
                            track.best_image,
                        )

            pass_ms = int((time.time() - pass_started) * 1000)

            # One line telling you what to look at when nothing is detected.
            if not overlay_vehicles:
                hint = "no vehicle detected - lower VEHICLE_CONF / raise VEHICLE_IMGSZ"
            elif not overlay_plates:
                hint = "vehicle but no plate box - raise PLATE_IMGSZ"
            elif not any(p["text"] for p in overlay_plates):
                hint = "plate found but text rejected - see red labels"
            else:
                hint = ""

            if view.enabled or view.writer is not None:
                canvas = view.draw(frame, overlay_vehicles, overlay_plates, {
                    "frames": frames, "passes": passes,
                    "pass_ms": pass_ms, "tracked": len(tracks),
                    "logged": log.count, "hint": hint,
                })
                if not view.show(canvas, out_dir):
                    quit_requested = True
                    break

            # Forget vehicles that have left. On a file the wall clock runs
            # far ahead of video time, so age tracks by missed passes instead.
            if is_stream:
                expired = [t for t, tr in tracks.items()
                           if t not in seen and now - tr.last_seen > TRACK_TTL]
            else:
                for t, tr in tracks.items():
                    tr.misses = 0 if t in seen else getattr(tr, "misses", 0) + 1
                expired = [t for t, tr in tracks.items()
                           if getattr(tr, "misses", 0) > 5]

            for track_id in expired:
                track = tracks.pop(track_id)

                if not track.done and show_misses:
                    winner = track.winner()
                    best = winner["text"] if winner else "-"
                    votes = winner["votes"] if winner else 0
                    print(
                        f"          {track.label} left unread "
                        f"(best guess {best}, {votes} vote(s))",
                        flush=True,
                    )
                    for r in track.rejects[:4]:
                        print(f"                rejected: {r}", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        cap.release()
        view.close()

        if quit_requested:
            print("\nClosed by user.")

        elapsed = time.time() - started

        print("\n" + "=" * 52)
        print(f"  Frames read      : {frames}")
        print(f"  Vehicles tracked : {len(vehicles_seen)}")
        print(f"  Plates logged    : {log.count}")
        print(f"  Ran for          : {elapsed:.0f}s")
        print(f"  Images           : {log.image_dir}/")
        print(f"  CSV              : {log.csv_path}")
        print("=" * 52)

        log.close()

    return 0


def main():
    global MIN_VOTES, DETECT_EVERY_N

    parser = argparse.ArgumentParser(
        description="Watch a camera or video and log every plate it reads."
    )
    parser.add_argument(
        "--source", default=None,
        help="RTSP URL or video file (default: ENTRY_CAMERA_URL from .env)",
    )
    parser.add_argument(
        "--camera-id", default=None,
        help="name for the log (default: ENTRY_CAMERA_ID from .env)",
    )
    parser.add_argument(
        "--out", default=OUT_DIR,
        help=f"output folder (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--min-votes", type=int, default=None,
        help=f"agreeing reads needed (default: {MIN_VOTES})",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="open a window showing the camera with detections drawn on it",
    )
    parser.add_argument(
        "--no-topmost", action="store_true",
        help="do not keep the preview window above other windows",
    )
    parser.add_argument(
        "--view-width", type=int, default=None,
        help="force the preview window width in pixels",
    )
    parser.add_argument(
        "--save-video", default=None,
        help="write the annotated video to this file (use over SSH, where "
             "--show cannot open a window)",
    )
    parser.add_argument(
        "--every", type=int, default=None,
        help=f"file mode: detect every Nth frame (default: {DETECT_EVERY_N})",
    )
    parser.add_argument(
        "--show-misses", action="store_true",
        help="also print vehicles whose plate was never read",
    )

    args = parser.parse_args()

    if args.min_votes:
        MIN_VOTES = args.min_votes

    if args.every:
        DETECT_EVERY_N = args.every

    source = args.source or os.getenv("ENTRY_CAMERA_URL", "")
    camera_id = args.camera_id or os.getenv("ENTRY_CAMERA_ID") or "camera"

    if not source:
        print("No source. Pass --source, or set ENTRY_CAMERA_URL in .env",
              file=sys.stderr)
        print("\n  python plate_watch.py --source videos/bike_plates.mp4",
              file=sys.stderr)
        return 1

    # A local file path must exist; an rtsp:// or http:// URL is checked by
    # opening it.
    if "://" not in source and not os.path.exists(source):
        print(f"No such file: {source}", file=sys.stderr)
        return 1

    return watch(source, camera_id, args.out, args.show_misses,
                 show=args.show, save_video=args.save_video,
                 view_width=args.view_width,
                 no_topmost=args.no_topmost)


if __name__ == "__main__":
    sys.exit(main())
