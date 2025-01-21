import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

class FaceRecognitionApp:
    def __init__(self):
        # Khởi tạo các biến
        self.known_face_encodings = []
        self.known_face_names = []
        self.dataset_dir = "dataset"
        self.known_faces_dir = "known_faces"
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.known_faces_dir, exist_ok=True)
        
        # Load các khuôn mặt đã biết
        self.load_known_faces()
        
        # Khởi tạo camera
        self.camera = cv2.VideoCapture(0)
        
    def load_known_faces(self):
        """Load các khuôn mặt đã biết từ thư mục known_faces"""
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith((".jpg", ".png")):
                # Lấy tên người từ tên file
                name = os.path.splitext(filename)[0]
                
                # Load và mã hóa khuôn mặt
                image_path = os.path.join(self.known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)
    
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
        
        # Yêu cầu người dùng nhập tên
        name = input("Nhập tên người dùng: ")
        if not name:
            print("Tên không hợp lệ!")
            return
        
        # Lưu ảnh vào thư mục known_faces
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jpg"
        file_path = os.path.join(self.known_faces_dir, filename)
        cv2.imwrite(file_path, frame)
        
        # Cập nhật danh sách khuôn mặt đã biết
        encoding = face_recognition.face_encodings(frame)[0]
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)
        
        print(f"Đã thêm khuôn mặt mới cho {name}")
    
    def run(self):
        """Chạy ứng dụng nhận diện khuôn mặt"""
        while True:
            # Đọc frame từ camera
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Giảm kích thước frame để tăng hiệu suất
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Chuyển từ BGR sang RGB
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Phát hiện khuôn mặt và mã hóa
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Xử lý từng khuôn mặt trong frame
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # So sánh với các khuôn mặt đã biết
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Không nhận dạng"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                
                # Scale lại vị trí khuôn mặt
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Vẽ khung và hiển thị tên
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Hiển thị frame
            cv2.imshow('Face Recognition', frame)
            
            # Xử lý phím nhấn
            key = cv2.waitKey(1)
            if key == ord('q'):  # Nhấn 'q' để thoát
                break
            elif key == ord('a'):  # Nhấn 'a' để thêm khuôn mặt mới
                self.add_new_face()
        
        # Giải phóng tài nguyên
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run() 