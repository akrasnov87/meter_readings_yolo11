from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")
metrics = model.val()

print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)  # a list contains map50-95 of each category