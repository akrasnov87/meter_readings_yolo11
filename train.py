from ultralytics import YOLO

model = YOLO("yolo11n.pt")
#model = YOLO("runs/detect/train4/weights/last.pt")

#results = model.train(data="./datasets/vxcvxcv/data.yaml", epochs=10, device='cpu')
results = model.train(data="./datasets/electric_meter_1/data.yaml", epochs=100, device='cpu')