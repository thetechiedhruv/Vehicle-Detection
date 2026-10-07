from ultralytics import YOLO
import cv2

model = YOLO("models/helmet_best.pt")
video_path = "videos/helmet2.mp4"

def detect_helmets_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = video_path.replace(".mp4", "_annotated.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(
            frame,
            imgsz=1280,
            conf=0.4,
            iou=0.5,
            verbose=False
        )

        annotated = frame.copy()
        total_vehicles = len(results[0].boxes)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = model.names[cls_id]

            if label != "helmet":
                continue

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            annotated,
            f"Total Vehicles: {total_vehicles}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        out.write(annotated)

        # display = cv2.resize(annotated, (1200, 700))
        # cv2.imshow("Helmet Detection", display)

        # if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
        #     break

    cap.release()
    out.release()
    # cv2.destroyAllWindows()

    print(f"Processed {video_path}, saved to {output_path}")


detect_helmets_in_video(video_path)
