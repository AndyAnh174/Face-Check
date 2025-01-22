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
                    try:
                        # Load ảnh
                        image = face_recognition.load_image_file(image_path)
                        
                        # Phát hiện vị trí khuôn mặt
                        face_locations = face_recognition.face_locations(image, model="hog")
                        
                        if face_locations:
                            # Lấy face landmarks
                            face_landmarks_list = face_recognition.face_landmarks(image, face_locations)
                            
                            if face_landmarks_list:
                                # Lấy encoding cho khuôn mặt đầu tiên
                                face_encodings = face_recognition.face_encodings(
                                    image,
                                    known_face_locations=face_locations,
                                    model="small"
                                )
                                
                                if face_encodings:
                                    known_face_encodings.append(face_encodings[0])
                                    known_face_names.append(user_data["name"])
                                    
                    except Exception as e:
                        print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
                        continue
                    
        return known_face_encodings, known_face_names
    
    def update_user_info(self, user_id, new_info):
        """Cập nhật thông tin người dùng"""
        if user_id in self.metadata["users"]:
            self.metadata["users"][user_id].update(new_info)
            self._save_metadata()
            return True
        return False

    def delete_user(self, user_id):
        """Xóa người dùng khỏi cơ sở dữ liệu"""
        if user_id in self.metadata["users"]:
            user_data = self.metadata["users"][user_id]
            
            # Xóa các ảnh
            for image_path in user_data["images"]:
                if os.path.exists(image_path):
                    os.remove(image_path)
            
            # Xóa thư mục người dùng
            user_dir = os.path.dirname(user_data["images"][0])
            if os.path.exists(user_dir):
                os.rmdir(user_dir)
            
            # Xóa metadata
            del self.metadata["users"][user_id]
            self._save_metadata()
            return True
        return False

    def add_face_image(self, user_id, frame):
        """Thêm ảnh mới cho người dùng đã tồn tại"""
        if user_id in self.metadata["users"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user_dir = os.path.join(self.known_faces_dir, user_id)
            
            # Lưu ảnh mới
            image_path = os.path.join(user_dir, f"{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            
            # Cập nhật metadata
            self.metadata["users"][user_id]["images"].append(image_path)
            self._save_metadata()
            return True
        return False

    def get_user_info(self, user_id):
        """Lấy thông tin chi tiết của người dùng"""
        return self.metadata["users"].get(user_id)

    def list_all_users(self):
        """Liệt kê tất cả người dùng"""
        return [(user_id, data["name"]) for user_id, data in self.metadata["users"].items()]

    def add_face_from_path(self, image_path, name, additional_info=None):
        """Thêm khuôn mặt mới từ đường dẫn ảnh"""
        if not os.path.exists(image_path):
            return None, "File ảnh không tồn tại"
            
        try:
            # Đọc ảnh
            frame = cv2.imread(image_path)
            if frame is None:
                return None, "Không thể đọc file ảnh"
                
            # Thêm khuôn mặt như bình thường
            user_id = self.add_face(frame, name, additional_info)
            return user_id, None
            
        except Exception as e:
            return None, f"Lỗi khi xử lý ảnh: {str(e)}" 