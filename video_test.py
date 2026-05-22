from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source=0,
    show=True,
    stream=True,
    save=True,
    project="runs",
    name="video_test",
    conf=0.4,
)

try:
    for r in results:
        print(f"{len(r.boxes)} object(s) detected")
except KeyboardInterrupt:
    print("Stopped.")
