import cv2
import face_recognition
import numpy as np
from .data_manager import DataManager

class FaceRecognitionApp:
    def __init__(self):
        # Khởi tạo data manager
        self.data_manager = DataManager()
        
        # Load các khuôn mặt đã biết
        self.known_face_encodings, self.known_face_names = self.data_manager.get_all_faces()
        
        # Khởi tạo camera
        self.camera = cv2.VideoCapture(0)
        
    def add_new_face(self):
        """Thêm khuôn mặt mới vào cơ sở dữ liệu"""
        # Chụp ảnh từ camera
        ret, frame = self.camera.read()
        if not ret:
            print("Không thể chụp ảnh!")
            return
        
        # Phát hiện khuôn mặt trong ảnh
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            print("Không phát hiện khuôn mặt trong ảnh!")
            return
        
        # Yêu cầu thông tin người dùng
        name = input("Nhập tên người dùng: ")
        if not name:
            print("Tên không hợp lệ!")
            return
            
        # Thêm thông tin bổ sung
        additional_info = {
            "age": input("Nhập tuổi (để trống nếu không muốn): "),
            "note": input("Ghi chú thêm (để trống nếu không muốn): ")
        }
        
        # Lưu vào cơ sở dữ liệu
        user_id = self.data_manager.add_face(frame, name, additional_info)
        
        # Cập nhật danh sách khuôn mặt đã biết
        self.known_face_encodings, self.known_face_names = self.data_manager.get_all_faces()
        
        print(f"Đã thêm khuôn mặt mới cho {name} (ID: {user_id})")
    
    # ... phần còn lại của class giữ nguyên ... 