# ===========================
# Environment & Logging Setup
# ===========================

import os
import logging
import warnings

# Suppress Paddle & System Logs
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_log_level"] = "3"

# PaddleX picks the oneDNN CPU backend by default, which crashes inside
# Paddle's PIR executor on PP-OCRv6_medium_det:
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#                        [pir::ArrayAttribute<pir::DoubleAttribute>]
# This is the flag PaddleX actually reads (paddlex/utils/flags.py).
# FLAGS_use_mkldnn / PADDLE_DISABLE_MKLDNN are ignored by PaddleX and do nothing.
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"

logging.getLogger("ppocr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# ===========================
# Imports
# ===========================

import re
import cv2
import numpy as np

from ultralytics import YOLO
from paddleocr import PaddleOCR


# ===========================
# Model Paths
# ===========================

PLATE_MODEL_PATH = "models/license-plate-finetune-v1x.pt"

# Swap to "yolov8s.pt" or "yolo26s.pt" for ~3x faster vehicle detection
# at some cost in accuracy.
VEHICLE_MODEL_PATH = os.environ.get("VEHICLE_MODEL_PATH", "yolo12m.pt")

INPUT_VIDEO = "videos/bike_plates.mp4"
base_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
OUTPUT_VIDEO = f"outputs/{base_name}_output.mp4"


# ===========================
# Performance Tuning
# ===========================

# Vehicles are large objects; 1920 was paying full cost for no benefit.
VEHICLE_IMGSZ = 192*2

# The plate model runs on a cropped vehicle ROI, so it needs less resolution.
PLATE_IMGSZ = 640

# Run vehicle detection every Nth frame and reuse boxes in between.
# A vehicle barely moves in 1/30 s, so this is close to free.
DETECT_EVERY_N = 8

# Each tracked vehicle is OCR'd at most this many times, then left alone.
MAX_SCAN_ATTEMPTS = 3


# ===========================
# Initialize Models
# ===========================

# License Plate Detector
plate_model = YOLO(PLATE_MODEL_PATH)

# Vehicle Detector
vehicle_model = YOLO(VEHICLE_MODEL_PATH)

# OCR Engine
# Document orientation / unwarping / textline orientation each load and run an
# extra model per call. Plate crops are small and upright, so they are off.
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


# ===========================
# Constants
# ===========================

# Vehicle class IDs (COCO format)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, bus, truck, motorcycle


# ===========================
# Utility Functions
# ===========================

def clean_plate_text(text: str) -> str:
    """
    Clean and validate OCR output.
    """

    if not text:
        return ""

    # Convert to uppercase
    text = text.upper()

    # Keep only letters & numbers
    text = re.sub(r"[^A-Z0-9]", "", text)

    # Must contain letters & digits
    if not re.search(r"[A-Z]", text):
        return ""

    if not re.search(r"[0-9]", text):
        return ""

    # Length validation
    if len(text) < 6 or len(text) > 12:
        return ""

    return text


# ===========================
# OCR Function
# ===========================

def perform_ocr(plate_img: np.ndarray):
    """
    Perform OCR on cropped license plate image.

    PaddleOCR 3.x returns a list of dict-like OCRResult objects carrying
    "rec_texts" / "rec_scores", not the 2.x list-of-word-tuples.
    """

    raw_text = ""
    best_conf = 0.35

    results = ocr.ocr(plate_img)

    if not results:
        return "", best_conf

    for res in results:

        if res is None:
            continue

        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []

        for i, text in enumerate(texts):

            if not text:
                continue

            conf = float(scores[i]) if i < len(scores) else 0.0

            raw_text += text
            best_conf = max(best_conf, conf)

    return raw_text, best_conf


# ===========================
# Plate Detection
# ===========================

def detect_plates(img: np.ndarray):
    """
    Detect license plates and extract text.

    Returns:
        List of detected plates with bbox and confidence.
    """

    detections = []

    results = plate_model(
        img,
        imgsz=PLATE_IMGSZ,
        conf=0.35,
        iou=0.5,
        verbose=False
    )

    for r in results:

        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            plate_crop = img[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # OCR
            raw_text, conf = perform_ocr(plate_crop)

            # Clean text
            plate_text = clean_plate_text(raw_text)

            # Draw plate box
            draw_plate_box(img, x1, y1, x2, y2, plate_text)

            detections.append({
                "plate": plate_text,
                "box": (x1, y1, x2, y2),
                "confidence": conf
            })

    return detections


# ===========================
# Drawing Functions
# ===========================

def draw_plate_box(img, x1, y1, x2, y2, text):
    """
    Draw bounding box and text for plate.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2

    # Draw box
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if not text:
        return

    # Text size
    (w, h), b = cv2.getTextSize(text, font, scale, thickness)

    # Background
    cv2.rectangle(
        img,
        (x1, y1 - h - 12),
        (x1 + w, y1),
        (0, 0, 0),
        -1
    )

    # Text
    cv2.putText(
        img,
        text,
        (x1, y1 - 5),
        font,
        scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_vehicle_box(img, x1, y1, x2, y2, label):
    """
    Draw vehicle bounding box.
    """

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.putText(
        img,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        3
    )


# ===========================
# Video Processing
# ===========================

def process_video():

    out_dir = os.path.dirname(OUTPUT_VIDEO)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {INPUT_VIDEO}")

    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Output Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        raise RuntimeError(f"Cannot open output video for writing: {OUTPUT_VIDEO}")

    print("Processing video...")

    # vehicle_id -> confirmed plate text
    plates_by_id = {}

    # vehicle_id -> number of OCR attempts spent
    scan_attempts = {}

    # Boxes from the most recent detection frame, reused on skipped frames
    last_boxes = []

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % DETECT_EVERY_N == 0:

            # Track so each vehicle gets a stable ID and is scanned once
            results = vehicle_model.track(
                frame,
                persist=True,
                imgsz=VEHICLE_IMGSZ,
                conf=0.3,
                iou=0.5,
                verbose=False
            )

            last_boxes = []

            boxes = results[0].boxes

            if boxes is not None:

                for box in boxes:

                    cls_id = int(box.cls)

                    if cls_id not in VEHICLE_CLASSES:
                        continue

                    label = vehicle_model.names[cls_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    vehicle_id = int(box.id.item()) if box.id is not None else None

                    last_boxes.append((x1, y1, x2, y2, label, vehicle_id))

                    # Without an ID we cannot avoid re-scanning every frame
                    if vehicle_id is None:
                        continue

                    # Already read, or we gave up on this vehicle
                    if plates_by_id.get(vehicle_id):
                        continue

                    if scan_attempts.get(vehicle_id, 0) >= MAX_SCAN_ATTEMPTS:
                        continue

                    vehicle_roi = frame[y1:y2, x1:x2]

                    if vehicle_roi.size == 0:
                        continue

                    scan_attempts[vehicle_id] = scan_attempts.get(vehicle_id, 0) + 1

                    # Detect plates inside vehicle
                    plates = detect_plates(vehicle_roi)

                    # Keep the first valid read for this vehicle
                    for plate in plates:

                        if plate["plate"]:

                            plates_by_id[vehicle_id] = plate["plate"]

                            print(
                                f"Detected Plate: {plate['plate']} "
                                f"(vehicle {vehicle_id}, frame {frame_idx})"
                            )

                            break

        # Draw vehicles (on detection and skipped frames alike)
        for x1, y1, x2, y2, label, vehicle_id in last_boxes:

            caption = label

            if vehicle_id is not None:
                caption = f"{label} ID:{vehicle_id}"

            text = plates_by_id.get(vehicle_id)

            if text:
                caption = f"{caption} {text}"

            draw_vehicle_box(frame, x1, y1, x2, y2, caption)

        out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

    print("Processing completed!")
    print(f"Frames: {frame_idx}, unique plates: {len(set(plates_by_id.values()))}")
    print("Saved to:", OUTPUT_VIDEO)


# ===========================
# Main Entry Point
# ===========================

if __name__ == "__main__":

    process_video()
