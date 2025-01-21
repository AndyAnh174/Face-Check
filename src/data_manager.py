import os
import cv2
import face_recognition
import json
from datetime import datetime

class DataManager:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.known_faces_dir = os.path.join(self.data_dir, "known_faces")
        self.training_data_dir = os.path.join(self.data_dir, "training_data")
        self.metadata_file = os.path.join(self.data_dir, "metadata.json")
        
        # Tạo các thư mục cần thiết
        self._create_directories()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _create_directories(self):
        """Tạo cấu trúc thư mục cần thiết"""
        os.makedirs(self.known_faces_dir, exist_ok=True)
        os.makedirs(self.training_data_dir, exist_ok=True)
        
    def _load_metadata(self):
        """Load metadata từ file JSON"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"users": {}}
    
    def _save_metadata(self):
        """Lưu metadata vào file JSON"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)
            
    def add_face(self, frame, name, additional_info=None):
        """Thêm khuôn mặt mới vào cơ sở dữ liệu"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = f"user_{timestamp}"
        
        # Tạo thư mục cho người dùng
        user_dir = os.path.join(self.known_faces_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Lưu ảnh
        image_path = os.path.join(user_dir, f"{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        
        # Cập nhật metadata
        self.metadata["users"][user_id] = {
            "name": name,
            "created_at": timestamp,
            "images": [image_path],
            "additional_info": additional_info or {}
        }
        
        self._save_metadata()
        return user_id
    
    def get_all_faces(self):
        """Lấy tất cả khuôn mặt đã biết và encoding"""
        known_face_encodings = []
        known_face_names = []
        
        for user_id, user_data in self.metadata["users"].items():
            for image_path in user_data["images"]:
                if os.path.exists(image_path):
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(user_data["name"])
                        
        return known_face_encodings, known_face_names
    
    def update_user_info(self, user_id, new_info):
        """Cập nhật thông tin người dùng"""
        if user_id in self.metadata["users"]:
            self.metadata["users"][user_id].update(new_info)
            self._save_metadata()
            return True
        return False 