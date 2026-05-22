from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.track(
    source=0,
    show=True,
    stream=True,
    save=True,
    project="runs",
    name="tracking_test",
    conf=0.4,
    tracker="bytetrack.yaml",
)

try:
    for r in results:
        ids = r.boxes.id
        n = 0 if ids is None else len(ids)
        print(f"tracking {n} object(s)")
except KeyboardInterrupt:
    print("Stopped.")
