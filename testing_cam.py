import os
import re
import cv2
import sys

from pathlib import Path
from dotenv import load_dotenv

# ==============================
# RTSP CAMERA URL
# ==============================
# Never hardcoded: the URL carries the camera's password, and this file is in
# version control. Give it on the command line, or set TEST_RTSP_URL in .env
# (which is gitignored).
#
#   python testing_cam.py rtsp://user:pass@192.168.1.106:554/Streaming/Channels/101

load_dotenv(Path(__file__).parent / ".env")

RTSP_URL = (sys.argv[1] if len(sys.argv) > 1
            else os.getenv("TEST_RTSP_URL", "")).strip()

if not RTSP_URL:
    print("No camera URL given.\n")
    print("  python testing_cam.py <rtsp-url>")
    print("\nor set TEST_RTSP_URL in .env")
    sys.exit(1)


def safe(url: str) -> str:
    """The URL with its password starred out, so logs stay shareable."""
    return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:****@", url)


print("Connecting to RTSP camera...")
print(safe(RTSP_URL))

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("ERROR: Could not connect to RTSP stream.")
    sys.exit(1)

print("Connected successfully!")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame. Retrying...")
        continue

    # Display live video
    cv2.imshow("RTSP Camera Live Stream", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()