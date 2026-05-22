from ultralytics import YOLO
model = YOLO("yolo11n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

results[0].save("output.jpg")

for box in results[0].boxes:
    cls_id = int(box.cls)
    conf = float(box.conf)
    name = model.names[cls_id]
    print(f"{name}: {conf:.2f}")
    