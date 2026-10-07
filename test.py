from ultralytics import YOLO
model = YOLO('models/yolov8s.pt')
print(model.names)





# from ultralytics import YOLO
# import cv2
# import numpy as np

# model = YOLO('models/auto.pt')
# # The model returns a list of 'Results' objects
# results = model('images/auto.jpg', imgsz=1280, conf=0.65, iou=0.5, verbose=False)
# VEHICLE_CLASSES = [0, 1, 2, 3, 5, 7]

# # 1. Extract the actual numpy image array from the results
# img = results[0].orig_img.copy() 

# for box in results[0].boxes:
#     cls_id = int(box.cls)
#     if cls_id not in VEHICLE_CLASSES:
#         continue
#     # 2. Ensure coordinates are integers (OpenCV requirement)
#     x1, y1, x2, y2 = map(int, box.xyxy[0])
#     # cls_id = int(box.cls)
#     conf = float(box.conf)
#     label = model.names[cls_id]

#     print(f"Detected {label}")

#     # 3. Draw on the numpy array
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(
#         img,
#         f"{label} {conf:.2f}",
#         (x1, y1 - 10),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.9,
#         (0, 255, 0),
#         2
#     )

# # 4. Optional: Resize for display if the 1280px image is too large for your monitor
# cv2.imwrite('output_truck.jpg', img)
# display_img = cv2.resize(img, (1200, 700))
# cv2.imshow('YOLOv8 Detection', display_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
