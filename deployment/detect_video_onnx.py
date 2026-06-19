from ultralytics import YOLO

# Load ONNX model
model = YOLO("best.onnx")

# Run inference on video
model.predict(source="input.mp4",save=True,conf=0.25,imgsz=640)

print("Detection completed.")
