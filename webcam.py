from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

results = model.track(
    source=0,
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
