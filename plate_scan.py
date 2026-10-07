"""
PLATESCOPE — Full-frame license plate scanner.

Architecture:
  - Vehicles are detected first, and plates are searched inside each vehicle
    crop. Cropping magnifies the plate, so distant vehicles whose plates are
    too small to survive full-frame downscaling are still readable.
  - Every plate is read on many frames, and the final text is decided by a vote
    across all reads rather than by whichever read happened to land first.

Measured on videos/bike_plates.mp4 (9 bikes, ground truth known):
  vehicle -> ROI, one read per track ID : 7 plates, several misread
  full-frame + voting                   : 8 plates, misses the distant bike
  vehicle -> ROI + voting (this script) : 9 plates, all correct, ~125s

Pass --full-frame to skip the vehicle stage. That is ~2x faster and fine when
every plate is large in frame, but it misses small ones: on bike_plates.mp4 it
drops GJ03CD5378, whose plate is ~90px wide against 130-190px for the rest.
"""

# ===========================
# Environment & Logging Setup
# ===========================

import os
import logging
import warnings

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_log_level"] = "3"

# PaddleX defaults to the oneDNN CPU backend, which crashes inside Paddle's PIR
# executor on PP-OCRv6_medium_det:
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#                        [pir::ArrayAttribute<pir::DoubleAttribute>]
# This is the flag PaddleX actually reads (paddlex/utils/flags.py).
# FLAGS_use_mkldnn / PADDLE_DISABLE_MKLDNN are ignored by PaddleX.
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"

logging.getLogger("ppocr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# ===========================
# Imports
# ===========================

import re
import sys
import json
import time
import argparse

import cv2
import numpy as np

from ultralytics import YOLO
from paddleocr import PaddleOCR


# ===========================
# Configuration
# ===========================

PLATE_MODEL_PATH = os.environ.get(
    "PLATE_MODEL_PATH",
    "models/license-plate-finetune-v1x.pt"
)

VEHICLE_MODEL_PATH = os.environ.get(
    "VEHICLE_MODEL_PATH",
    "yolo12m.pt"
)

# Search for plates inside vehicle crops rather than on the whole frame.
USE_VEHICLE_CROP = True

# Input resolution for the vehicle detector. 480 keeps every vehicle large
# enough to matter; the ones it drops at this size are too distant for their
# plates to be readable anyway.
VEHICLE_IMGSZ = 480
VEHICLE_CONF = 0.3
VEHICLE_IOU = 0.5

# COCO class IDs: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Input resolution for the plate detector, applied to the vehicle crop (or to
# the whole frame under --full-frame). Measured on bike_plates.mp4: 320 finds
# the same plates as 640 at a quarter the cost. Raise to 640 for traffic
# footage where plates stay small even after cropping.
PLATE_IMGSZ = 320

PLATE_CONF = 0.35
PLATE_IOU = 0.5

# Detect on every Nth frame. At ~30 fps, 10 gives ~3 scans per second, which
# still yields well over the votes needed per plate.
DETECT_EVERY_N = 10

# A plate cluster needs at least this many votes to be reported.
# Measured on bike_plates.mp4: all 9 real plates scored 2-13 votes and the only
# noise scored 1, so 2 separates them. Raise it if noise gets through, which is
# more likely on long videos where a bad read has more chances to repeat.
MIN_VOTES = 2

# Plate text validation
MIN_PLATE_LEN = 6
MAX_PLATE_LEN = 12

# Characters OCR routinely confuses. Used only for grouping variants of the
# same plate; the reported text is always a verbatim read, never normalized.
CONFUSABLE = str.maketrans({
    "O": "0", "Q": "0", "D": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "T": "7",
    "G": "6",
})

# Two reads within this edit distance (after normalization) are the same plate.
CLUSTER_MAX_DISTANCE = 2


# ===========================
# Model Loading
# ===========================

_plate_model = None
_vehicle_model = None
_ocr = None


def get_models():
    """
    Load the models once, on first use. The vehicle model is only loaded when
    the vehicle-crop stage is enabled.
    """

    global _plate_model, _vehicle_model, _ocr

    if _plate_model is None:
        _plate_model = YOLO(PLATE_MODEL_PATH)

    if _vehicle_model is None and USE_VEHICLE_CROP:
        _vehicle_model = YOLO(VEHICLE_MODEL_PATH)

    if _ocr is None:
        # Document orientation, unwarping and textline orientation each load and
        # run an extra model per call. Plate crops are small and upright.
        _ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    return _plate_model, _vehicle_model, _ocr


# ===========================
# Text Handling
# ===========================

def clean_plate_text(text):
    """
    Normalize and validate a raw OCR string. Returns "" if it is not a
    plausible plate.
    """

    if not text:
        return ""

    text = re.sub(r"[^A-Z0-9]", "", text.upper())

    if not re.search(r"[A-Z]", text):
        return ""

    if not re.search(r"[0-9]", text):
        return ""

    if len(text) < MIN_PLATE_LEN or len(text) > MAX_PLATE_LEN:
        return ""

    return text


def normalize_for_match(text):
    """
    Collapse commonly confused characters so that variants of one plate compare
    equal. For grouping only — never reported to the caller.
    """

    return text.translate(CONFUSABLE)


def edit_distance(a, b, cutoff=3):
    """
    Levenshtein distance, abandoning early when the strings differ too much in
    length to be within cutoff.
    """

    if abs(len(a) - len(b)) > cutoff:
        return cutoff + 1

    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):

        cur = [i] + [0] * len(b)

        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb)
            )

        prev = cur

    return prev[-1]


def cluster_reads(counts):
    """
    Group raw OCR strings that refer to the same physical plate.

    Reads are absorbed in descending frequency order, so the most-seen spelling
    becomes the cluster's reported text and rarer misreads fold into it.

    Returns a list of dicts sorted by vote count, descending.
    """

    clusters = []

    for text, n in sorted(counts.items(), key=lambda kv: -kv[1]):

        norm = normalize_for_match(text)

        for cl in clusters:

            rep = normalize_for_match(cl["text"])

            # Near-identical, or one is a truncated read of the other
            if edit_distance(norm, rep, CLUSTER_MAX_DISTANCE) <= CLUSTER_MAX_DISTANCE \
               or norm in rep or rep in norm:

                cl["votes"] += n
                cl["variants"].append(text)
                break

        else:

            clusters.append({
                "text": text,
                "votes": n,
                "variants": []
            })

    return sorted(clusters, key=lambda c: -c["votes"])


# ===========================
# Detection & OCR
# ===========================

def read_plate(crop, ocr_engine):
    """
    OCR one plate crop. Multi-line plates are concatenated top to bottom.

    Crops are passed at native size — upscaling was measured to cost 2-4x and
    never improved the read on this footage.
    """

    results = ocr_engine.ocr(crop)

    if not results:
        return "", 0.0

    raw_text = ""
    best_conf = 0.0

    for res in results:

        if res is None:
            continue

        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []

        for i, text in enumerate(texts):

            if not text:
                continue

            raw_text += text

            if i < len(scores):
                best_conf = max(best_conf, float(scores[i]))

    return clean_plate_text(raw_text), best_conf


def plates_in_image(img, plate_model, ocr_engine, offset=(0, 0)):
    """
    Detect and read every plate in one image.

    offset shifts the returned boxes back into full-frame coordinates when img
    is a crop, so the overlay draws in the right place.

    Returns a list of (box, text, confidence).
    """

    out = []

    results = plate_model(
        img,
        imgsz=PLATE_IMGSZ,
        conf=PLATE_CONF,
        iou=PLATE_IOU,
        verbose=False
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

        text, conf = read_plate(crop, ocr_engine)

        out.append(((x1 + ox, y1 + oy, x2 + ox, y2 + oy), text, conf))

    return out


def vehicle_rois(frame, vehicle_model):
    """
    Vehicle crops from one frame, each with its top-left offset.
    """

    rois = []

    boxes = vehicle_model(
        frame,
        imgsz=VEHICLE_IMGSZ,
        conf=VEHICLE_CONF,
        iou=VEHICLE_IOU,
        verbose=False
    )[0].boxes

    if boxes is None:
        return rois

    for box in boxes:

        if int(box.cls) not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        roi = frame[y1:y2, x1:x2]

        if roi.size:
            rois.append((roi, (x1, y1)))

    return rois


def detect_plates(frame, plate_model, vehicle_model, ocr_engine):
    """
    Find and read every plate in a frame, going through vehicle crops unless
    full-frame mode is selected.
    """

    if not USE_VEHICLE_CROP:
        return plates_in_image(frame, plate_model, ocr_engine)

    out = []

    for roi, offset in vehicle_rois(frame, vehicle_model):
        out.extend(plates_in_image(roi, plate_model, ocr_engine, offset))

    return out


# ===========================
# Drawing
# ===========================

def draw_plate(img, box, text):

    x1, y1, x2, y2 = box

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 30), 2)

    if not text:
        return

    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    # Keep the label on screen when the plate sits near the top edge
    top = y1 - h - 12

    if top < 0:
        top = y2 + 4

    cv2.rectangle(img, (x1, top), (x1 + w, top + h + 10), (0, 0, 0), -1)

    cv2.putText(
        img,
        text,
        (x1, top + h + 4),
        font,
        scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ===========================
# Video Processing
# ===========================

def process_video(input_path, output_path, json_path=None):

    plate_model, vehicle_model, ocr_engine = get_models()

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None

    if output_path:

        out_dir = os.path.dirname(output_path)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not out.isOpened():
            raise RuntimeError(f"Cannot open output video: {output_path}")

    print(f"Scanning {input_path} ({width}x{height}, {total} frames)")

    counts = {}          # raw OCR string -> times seen
    last_overlay = []    # boxes reused on skipped frames
    frames_with_plates = 0
    scanned = 0

    frame_idx = 0
    started = time.time()

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if frame_idx % DETECT_EVERY_N == 0:

            scanned += 1

            found = detect_plates(frame, plate_model, vehicle_model, ocr_engine)

            last_overlay = [(box, text) for box, text, _ in found]

            got_one = False

            for _, text, _ in found:

                if text:
                    counts[text] = counts.get(text, 0) + 1
                    got_one = True

            if got_one:
                frames_with_plates += 1

        if out is not None:

            for box, text in last_overlay:
                draw_plate(frame, box, text)

            out.write(frame)

        frame_idx += 1

    cap.release()

    if out is not None:
        out.release()

    elapsed = time.time() - started

    # ===== Vote =====

    clusters = cluster_reads(counts)

    threshold = MIN_VOTES

    confirmed = [c for c in clusters if c["votes"] >= threshold]
    rejected = [c for c in clusters if c["votes"] < threshold]

    # ===== Report =====

    print(f"\nScanned {scanned} of {frame_idx} frames in {elapsed:.0f}s")
    print(f"{len(counts)} raw reads -> {len(clusters)} plates "
          f"(vote threshold {threshold})\n")

    print(f"CONFIRMED PLATES ({len(confirmed)}):")

    for c in confirmed:

        print(f"  {c['text']:<14} votes={c['votes']}")

        if c["variants"]:
            shown = ", ".join(c["variants"][:6])
            more = "" if len(c["variants"]) <= 6 else f" (+{len(c['variants']) - 6} more)"
            print(f"      variants: {shown}{more}")

    if rejected:
        print(f"\nBELOW THRESHOLD ({len(rejected)}), likely noise:")
        for c in rejected:
            print(f"  {c['text']:<14} votes={c['votes']}")

    if output_path:
        print(f"\nAnnotated video: {output_path}")

    if json_path:

        with open(json_path, "w") as fh:
            json.dump({
                "input": input_path,
                "frames": frame_idx,
                "scanned_frames": scanned,
                "seconds": round(elapsed, 1),
                "threshold": threshold,
                "plates": [c["text"] for c in confirmed],
                "detail": confirmed,
                "rejected": rejected,
            }, fh, indent=2)

        print(f"Results JSON:    {json_path}")

    return confirmed


# ===========================
# Main Entry Point
# ===========================

def main():

    global DETECT_EVERY_N, PLATE_IMGSZ, USE_VEHICLE_CROP, MIN_VOTES

    parser = argparse.ArgumentParser(
        description="Scan a video for license plates using full-frame detection and voting."
    )

    parser.add_argument("input", nargs="?", default="videos/bike_plates.mp4",
                        help="input video path")
    parser.add_argument("-o", "--output", default=None,
                        help="annotated output video (default: outputs/<name>_scan.mp4)")
    parser.add_argument("--json", default=None,
                        help="write results to this JSON file")
    parser.add_argument("--no-video", action="store_true",
                        help="skip writing the annotated video (faster)")
    parser.add_argument("--every", type=int, default=None,
                        help=f"detect every Nth frame (default {DETECT_EVERY_N})")
    parser.add_argument("--imgsz", type=int, default=None,
                        help=f"plate detector input size (default {PLATE_IMGSZ})")
    parser.add_argument("--full-frame", action="store_true",
                        help="skip the vehicle stage; faster but misses small plates")
    parser.add_argument("--min-votes", type=int, default=None,
                        help=f"votes needed to confirm a plate (default {MIN_VOTES})")

    args = parser.parse_args()

    if args.every:
        DETECT_EVERY_N = args.every

    if args.imgsz:
        PLATE_IMGSZ = args.imgsz

    if args.full_frame:
        USE_VEHICLE_CROP = False

    if args.min_votes:
        MIN_VOTES = args.min_votes

    if not os.path.exists(args.input):
        print(f"No such video: {args.input}", file=sys.stderr)
        return 1

    output = None

    if not args.no_video:
        output = args.output or os.path.join(
            "outputs",
            f"{os.path.splitext(os.path.basename(args.input))[0]}_scan.mp4"
        )

    process_video(args.input, output, args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
