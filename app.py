"""
PLATESCOPE — Flask Backend
License Plate Detection API with Server-Sent Events for live progress streaming.
Updated with:
- Vehicle Tracking
- Trigger Line
- One-Time Plate Scan per Vehicle
- Base64 Encoded Plate Images
"""

# ===========================
# Environment & Logging Setup
# ===========================

import os
import logging
import warnings
import json
import time
import uuid
import threading
import queue

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_log_level"] = "3"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["FLAGS_enable_pir"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"

logging.getLogger("ppocr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ===========================
# Imports
# ===========================

import re
import cv2
import numpy as np
import base64

from ultralytics import YOLO
from paddleocr import PaddleOCR
from flask import Flask, request, jsonify, Response, send_file, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename


# ===========================
# App Configuration
# ===========================

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ===========================
# Model Paths
# ===========================

PLATE_MODEL_PATH = os.environ.get(
    "PLATE_MODEL_PATH",
    "models/license-plate-finetune-v1x.pt"
)

VEHICLE_MODEL_PATH = os.environ.get(
    "VEHICLE_MODEL_PATH",
    "yolo12m.pt"
)


# ===========================
# Lazy Model Loading
# ===========================

_plate_model = None
_vehicle_model = None
_ocr = None
_model_lock = threading.Lock()


def get_models():

    global _plate_model, _vehicle_model, _ocr

    with _model_lock:

        if _plate_model is None:
            _plate_model = YOLO(PLATE_MODEL_PATH)

        if _vehicle_model is None:
            _vehicle_model = YOLO(VEHICLE_MODEL_PATH)

        if _ocr is None:
            _ocr = PaddleOCR(
                lang="en",
                use_textline_orientation=True,
                # show_log=False
            )

    return _plate_model, _vehicle_model, _ocr


# ===========================
# Constants
# ===========================

VEHICLE_CLASSES = [2, 3, 5, 7]

TRIGGER_LINE_Y = 900   # Adjust for your video


# ===========================
# In-Memory Job Store
# ===========================

jobs = {}


# ===========================
# Utility Functions
# ===========================

def allowed_file(filename):

    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_plate_text(text):

    if not text:
        return ""

    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    if not re.search(r"[A-Z]", text):
        return ""

    if not re.search(r"[0-9]", text):
        return ""

    if len(text) < 6 or len(text) > 12:
        return ""

    return text


def perform_ocr(img, ocr_engine):

    raw_text = ""
    best_conf = 0.35

    results = ocr_engine.ocr(img)

    if not results:
        return "", best_conf

    for line in results:

        if line is None:
            continue

        for word in line:

            if word is None or len(word) < 2:
                continue

            text, conf = word[1]

            raw_text += text
            best_conf = max(best_conf, conf)

    return raw_text, best_conf


def detect_plates(img, plate_model, ocr_engine, conf, iou):

    detections = []

    results = plate_model(
        img,
        imgsz=1280,
        conf=conf,
        iou=iou,
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

            raw_text, c = perform_ocr(plate_crop, ocr_engine)

            plate_text = clean_plate_text(raw_text)

            draw_plate_box(img, x1, y1, x2, y2, plate_text)

            detections.append({
                "plate": plate_text,
                "confidence": round(float(c), 3),
                "box": [x1, y1, x2, y2]
            })

    return detections


def draw_plate_box(img, x1, y1, x2, y2, text):

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 30), 2)

    if text:

        (w, h), _ = cv2.getTextSize(text, font, 1, 2)

        cv2.rectangle(
            img,
            (x1, y1 - h - 10),
            (x1 + w, y1),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            img,
            text,
            (x1, y1 - 5),
            font,
            1,
            (20, 80, 200),
            2
        )


def draw_vehicle_box(img, x1, y1, x2, y2, label):

    cv2.rectangle(img, (x1, y1), (x2, y2), (30, 160, 90), 3)

    cv2.putText(
        img,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 160, 90),
        3
    )


# ===========================
# Core Processing
# ===========================

def process_video_job(
    job_id,
    input_path,
    output_path,
    conf,
    iou,
    imgsz
):

    q = jobs[job_id]["queue"]
    cancel_event = jobs[job_id]["cancel"]

    def emit(event, data):

        q.put({"event": event, "data": data})

    try:

        emit("status", {"message": "Loading models...", "progress": 5})

        plate_model, vehicle_model, ocr = get_models()

        emit("status", {"message": "Models loaded", "progress": 10})

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():

            emit("error", {"message": "Cannot open video"})
            return


        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        scanned_vehicle_ids = set()

        frame_idx = 0
        vehicle_count = 0
        plate_count = 0

        unique_plates = set()
        all_detections = []

        emit("status", {"message": "Processing started", "progress": 15})


        while True:

            if cancel_event.is_set():

                emit("cancelled", {"message": "Cancelled"})
                break


            ret, frame = cap.read()

            if not ret:
                break


            frame_idx += 1


            results = vehicle_model.track(
                frame,
                persist=True,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False
            )


            if results[0].boxes is None:
                out.write(frame)
                continue


            for box in results[0].boxes:

                if box.id is None:
                    continue


                vehicle_id = int(box.id.item())
                cls_id = int(box.cls.item())

                if cls_id not in VEHICLE_CLASSES:
                    continue


                label = vehicle_model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_y = (y1 + y2) // 2

                roi = frame[y1:y2, x1:x2]

                if roi.size == 0:
                    continue


                draw_vehicle_box(
                    frame,
                    x1, y1, x2, y2,
                    f"{label} ID:{vehicle_id}"
                )

                # vehicle_count += 1


                # ===== Trigger Scan =====

                if center_y >= TRIGGER_LINE_Y and \
                   vehicle_id not in scanned_vehicle_ids:


                    scanned_vehicle_ids.add(vehicle_id)


                    plates = detect_plates(
                        roi,
                        plate_model,
                        ocr,
                        conf,
                        iou
                    )


                    for p in plates:

                        text = p["plate"]

                        plate_count += 1

                        if text:
                            unique_plates.add(text)

                        det = {
                            "frame": frame_idx,
                            "plate": text,
                            "confidence": p["confidence"],
                            "vehicle": label,
                            "vehicle_id": vehicle_id
                        }

                        all_detections.append(det)

                        emit("detection", det)


            # Draw trigger line
            cv2.line(
                frame,
                (0, TRIGGER_LINE_Y),
                (width, TRIGGER_LINE_Y),
                (0, 0, 255),
                3
            )


            out.write(frame)
            vehicle_count = len(scanned_vehicle_ids)

            if frame_idx % 5 == 0 or frame_idx == total:

                progress = int((frame_idx / max(total, 1)) * 85) + 15

                emit("progress", {
                    "frame": frame_idx,
                    "total": total,
                    "progress": progress,
                    "vehicles": vehicle_count,
                    "plates": plate_count,
                    "unique": len(unique_plates)
                })


        cap.release()
        out.release()


        if not cancel_event.is_set():

            jobs[job_id]["result"] = {
                "output_path": output_path,
                "total_frames": frame_idx,
                "vehicle_count": vehicle_count,
                "plate_count": plate_count,
                "unique_count": len(unique_plates),
                "detections": all_detections
            }

            jobs[job_id]["status"] = "done"

            emit("done", {
                "message": "Completed",
                "progress": 100
            })


    except Exception as e:

        emit("error", {"message": str(e)})

        jobs[job_id]["status"] = "error"


    finally:

        q.put(None)


# ===========================
# Routes
# ===========================

@app.route("/")
def index():
    return send_file("templates/index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400


    file = request.files["video"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400


    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid format"}), 400


    conf = float(request.form.get("conf", 0.35))
    iou = float(request.form.get("iou", 0.5))
    imgsz = int(request.form.get("imgsz", 1280))


    job_id = str(uuid.uuid4())

    filename = secure_filename(file.filename)


    input_path = os.path.join(
        UPLOAD_FOLDER,
        f"{job_id}_{filename}"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"{job_id}_output.mp4"
    )


    file.save(input_path)


    jobs[job_id] = {
        "status": "queued",
        "queue": queue.Queue(),
        "cancel": threading.Event(),
        "result": None,
        "filename": filename
    }


    t = threading.Thread(
        target=process_video_job,
        args=(job_id, input_path, output_path, conf, iou, imgsz),
        daemon=True
    )

    t.start()


    return jsonify({
        "job_id": job_id,
        "filename": filename
    })


@app.route("/stream/<job_id>")
def stream(job_id):

    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404


    def gen():

        q = jobs[job_id]["queue"]

        while True:

            item = q.get()

            if item is None:

                yield "event: close\ndata: {}\n\n"
                break


            yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"


    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream"
    )


@app.route("/output/<job_id>")
def output(job_id):

    if job_id not in jobs or not jobs[job_id]["result"]:
        return jsonify({"error": "Not ready"}), 404


    return send_file(jobs[job_id]["result"]["output_path"])


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):

    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404


    jobs[job_id]["cancel"].set()

    return jsonify({"message": "Cancelled"})


# ===========================
# Main
# ===========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )