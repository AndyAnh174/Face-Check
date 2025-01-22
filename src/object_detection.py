import cv2
import os
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        """Khởi tạo model YOLO"""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(self.base_dir, "data", "models", "train", "weights", "best.pt")
        
        # Nếu có custom model thì dùng, không thì dùng model mặc định
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            self.model = YOLO('yolov8n.pt')
            
        self.classes = self.model.names
        self.is_enabled = True  # flag để bật/tắt nhận diện đồ vật
        
    def detect_objects(self, frame):
        """Nhận diện đồ vật trong frame"""
        if not self.is_enabled:
            return frame
            
        # Thực hiện dự đoán
        results = self.model(frame, conf=0.5)  # confidence threshold 0.5
        
        # Vẽ kết quả lên frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Lấy class và confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Tên class và độ tin cậy
                label = f"{self.classes[cls]} ({conf:.2f})"
                
                # Vẽ khung và label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
        return frame
        
    def toggle(self):
        """Bật/tắt nhận diện đồ vật"""
        self.is_enabled = not self.is_enabled
        return self.is_enabled
        
    def reload_model(self):
        """Tải lại model sau khi train"""
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            self.classes = self.model.names
            return True
        return False 