import sys
from ultralytics import YOLO
import cv2

cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0

model = YOLO("yolo11n.pt")

results = model.track(
    source=cam,
    stream=True,
    conf=0.4,
    tracker="bytetrack.yaml",
)

for r in results:
    frame = r.plot()
    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

cv2.destroyAllWindows()
