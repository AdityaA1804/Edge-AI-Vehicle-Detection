from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="data.yaml",epochs=500,imgsz=640,batch=32,device=0)
