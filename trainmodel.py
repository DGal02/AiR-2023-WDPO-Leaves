from ultralytics import YOLO
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
model = YOLO("yolov8m.pt")

def start():
    model.train(data="data.yaml", epochs=600, patience=800, batch=16)

if __name__ == "__main__":
    start()