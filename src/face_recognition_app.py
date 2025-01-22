import cv2
import face_recognition
import numpy as np
from src.data_manager import DataManager
from src.object_detection import ObjectDetector
from src.object_trainer import ObjectTrainer
from utils.image_utils import resize_with_aspect_ratio, draw_text_with_background

class FaceRecognitionApp:
    def __init__(self):
        # Khởi tạo data manager
        self.data_manager = DataManager()
        
        # Khởi tạo object detector và trainer
        self.object_detector = ObjectDetector()
        self.object_trainer = ObjectTrainer()
        
        # Load các khuôn mặt đã biết
        self.known_face_encodings, self.known_face_names = self.data_manager.get_all_faces()
        
        # Khởi tạo camera
        self.camera = cv2.VideoCapture(0)
        
        self.is_running = True
        
    def show_menu(self):
        """Hiển thị menu chức năng"""
        print("\n=== MENU ===")
        print("1. Thêm khuôn mặt mới từ camera (a)")
        print("2. Thêm khuôn mặt mới từ file ảnh (f)")
        print("3. Xóa người dùng (d)")
        print("4. Thêm ảnh cho người dùng hiện có (i)")
        print("5. Xem thông tin người dùng (v)")
        print("6. Bật/tắt nhận diện đồ vật (o)")
        print("7. Thêm class đồ vật mới (c)")
        print("8. Training model nhận diện đồ vật (t)")
        print("9. Thoát (q)")
        print("============")
    
    def delete_user(self):
        """Xóa người dùng"""
        users = self.data_manager.list_all_users()
        if not users:
            print("Không có người dùng nào trong cơ sở dữ liệu!")
            return
            
        print("\nDanh sách người dùng:")
        for user_id, name in users:
            print(f"ID: {user_id} - Tên: {name}")
            
        user_id = input("\nNhập ID người dùng cần xóa: ")
        if self.data_manager.delete_user(user_id):
            print("Đã xóa người dùng thành công!")
            # Cập nhật lại danh sách khuôn mặt
            self.known_face_encodings, self.known_face_names = self.data_manager.get_all_faces()
        else:
            print("Không tìm thấy người dùng!")
    
    def process_frame(self, frame):
        """Xử lý frame để phù hợp với face_recognition"""
        # Chuyển BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame
        
    def add_new_face(self):
        """Thêm khuôn mặt mới vào cơ sở dữ liệu"""
        print("\nĐang chụp ảnh... Hãy nhìn vào camera và giữ yên.")
        
        # Đợi camera ổn định
        for _ in range(10):
            self.camera.read()
            cv2.waitKey(100)  # Đợi 100ms
            
        ret, frame = self.camera.read()
        if not ret:
            print("Không thể chụp ảnh!")
            return
            
        # Xử lý frame
        rgb_frame = self.process_frame(frame)
        
        try:
            # Phát hiện khuôn mặt với model HOG
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            if not face_locations:
                print("Không phát hiện khuôn mặt trong ảnh!")
                return
            
            # Lấy face landmarks
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
            
            if not face_landmarks_list:
                print("Không thể nhận dạng đặc điểm khuôn mặt!")
                return
            
            # Thử encode khuôn mặt
            face_encodings = face_recognition.face_encodings(
                rgb_frame,
                known_face_locations=face_locations,
                model="small"
            )
            
            if not face_encodings:
                print("Không thể mã hóa khuôn mặt!")
                return
            
            # Tiếp tục với việc nhập thông tin
            name = input("Nhập tên người dùng: ")
            if not name:
                print("Tên không hợp lệ!")
                return
            
            additional_info = {
                "age": input("Nhập tuổi (để trống nếu không muốn): "),
                "note": input("Ghi chú thêm (để trống nếu không muốn): ")
            }
            
            # Lưu vào cơ sở dữ liệu
            user_id = self.data_manager.add_face(frame, name, additional_info)
            
            # Cập nhật danh sách khuôn mặt
            self.known_face_encodings, self.known_face_names = self.data_manager.get_all_faces()
            
            print(f"Đã thêm khuôn mặt mới cho {name} (ID: {user_id})")
            
        except Exception as e:
            print(f"Lỗi khi xử lý khuôn mặt: {str(e)}")
            return
    
    def add_face_to_existing_user(self):
        """Thêm ảnh cho người dùng hiện có"""
        users = self.data_manager.list_all_users()
        if not users:
            print("Không có người dùng nào trong cơ sở dữ liệu!")
            return
            
        print("\nDanh sách người dùng:")
        for user_id, name in users:
            print(f"ID: {user_id} - Tên: {name}")
            
        user_id = input("\nNhập ID người dùng cần thêm ảnh: ")
        
        print("\nĐang chụp ảnh... Hãy nhìn vào camera và giữ yên.")
        # Đợi một chút để người dùng chuẩn bị
        for _ in range(10):
            self.camera.read()
            
        ret, frame = self.camera.read()
        if not ret:
            print("Không thể chụp ảnh!")
            return
            
        # Xử lý frame
        rgb_frame = self.process_frame(frame)
        
        # Phát hiện khuôn mặt
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            print("Không phát hiện khuôn mặt trong ảnh!")
            return
            
        if self.data_manager.add_face_image(user_id, frame):
            print("Đã thêm ảnh mới thành công!")
            self.known_face_encodings, self.known_face_names = self.data_manager.get_all_faces()
        else:
            print("Không tìm thấy người dùng!")
    
    def view_user_info(self):
        """Xem thông tin người dùng"""
        users = self.data_manager.list_all_users()
        if not users:
            print("Không có người dùng nào trong cơ sở dữ liệu!")
            return
            
        print("\nDanh sách người dùng:")
        for user_id, name in users:
            print(f"ID: {user_id} - Tên: {name}")
            
        user_id = input("\nNhập ID người dùng cần xem: ")
        user_info = self.data_manager.get_user_info(user_id)
        
        if user_info:
            print("\nThông tin người dùng:")
            print(f"Tên: {user_info['name']}")
            print(f"Ngày tạo: {user_info['created_at']}")
            print(f"Số ảnh: {len(user_info['images'])}")
            print("Thông tin bổ sung:")
            for key, value in user_info['additional_info'].items():
                if value:
                    print(f"- {key}: {value}")
        else:
            print("Không tìm thấy người dùng!")
    
    def add_face_from_file(self):
        """Thêm khuôn mặt mới từ file ảnh"""
        image_path = input("Nhập đường dẫn đến file ảnh: ")
        
        # Kiểm tra đường dẫn
        if not image_path:
            print("Đường dẫn không hợp lệ!")
            return
            
        # Nhập thông tin người dùng
        name = input("Nhập tên người dùng: ")
        if not name:
            print("Tên không hợp lệ!")
            return
            
        additional_info = {
            "age": input("Nhập tuổi (để trống nếu không muốn): "),
            "note": input("Ghi chú thêm (để trống nếu không muốn): ")
        }
        
        # Thêm khuôn mặt vào cơ sở dữ liệu
        user_id, error = self.data_manager.add_face_from_path(image_path, name, additional_info)
        
        if error:
            print(f"Lỗi: {error}")
        else:
            print(f"Đã thêm khuôn mặt mới cho {name} (ID: {user_id})")
            # Cập nhật danh sách khuôn mặt
            self.known_face_encodings, self.known_face_names = self.data_manager.get_all_faces()
    
    def add_object_class(self):
        """Thêm class đồ vật mới và chụp ảnh training"""
        class_name = input("Nhập tên loại đồ vật: ")
        if not class_name:
            print("Tên không hợp lệ!")
            return
            
        num_images = input("Số lượng ảnh muốn chụp (mặc định 30): ")
        try:
            num_images = int(num_images) if num_images else 30
        except:
            num_images = 30
            
        print("\nChuẩn bị chụp ảnh training...")
        print("- Đặt đồ vật ở các góc độ khác nhau")
        print("- Nhấn SPACE để chụp ảnh")
        print("- Nhấn Q để kết thúc")
        
        self.object_trainer.capture_training_images(class_name, num_images)
        print(f"\nĐã thêm class {class_name} với {num_images} ảnh")
        
    def train_object_detection(self):
        """Training model nhận diện đồ vật"""
        if not self.object_trainer.classes:
            print("Chưa có class nào được thêm vào! Hãy thêm class trước khi training.")
            return
            
        print("\nDanh sách classes hiện tại:")
        for i, cls in enumerate(self.object_trainer.classes):
            print(f"{i+1}. {cls}")
            
        epochs = input("\nSố epochs muốn train (mặc định 50): ")
        try:
            epochs = int(epochs) if epochs else 50
        except:
            epochs = 50
            
        print("\nBắt đầu training...")
        model_path = self.object_trainer.train_model(epochs)
        
        if model_path:
            print("Training hoàn tất!")
            print("Đang tải lại model...")
            if self.object_detector.reload_model():
                print("Đã tải model mới thành công!")
            else:
                print("Không thể tải model mới!")
        else:
            print("Training thất bại!")
    
    def run(self):
        """Chạy ứng dụng nhận diện khuôn mặt"""
        print("Khởi động ứng dụng nhận diện khuôn mặt...")
        print("Nhấn 'h' để hiển thị menu trợ giúp")
        print("Nhấn 'o' để bật/tắt nhận diện đồ vật")
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            # Xử lý frame để tăng hiệu suất
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = self.process_frame(small_frame)
            
            try:
                # Phát hiện và nhận dạng khuôn mặt
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                
                if face_locations and self.known_face_encodings:
                    face_encodings = face_recognition.face_encodings(
                        rgb_small_frame,
                        known_face_locations=face_locations,
                        model="small"
                    )
                    
                    # Xử lý từng khuôn mặt
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        name = "Không nhận dạng"
                        
                        if self.known_face_encodings:
                            # Tính toán khoảng cách Euclidean với tất cả khuôn mặt đã biết
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            
                            # Tìm khoảng cách nhỏ nhất và index tương ứng
                            min_distance_idx = np.argmin(face_distances)
                            min_distance = face_distances[min_distance_idx]
                            
                            # Chỉ nhận dạng nếu khoảng cách đủ nhỏ (tolerance thấp hơn để nghiêm ngặt hơn)
                            if min_distance < 0.4:  # Giảm tolerance từ 0.6 xuống 0.4
                                name = self.known_face_names[min_distance_idx]
                                # Thêm độ tin cậy vào tên hiển thị
                                confidence = round((1 - min_distance) * 100)
                                name = f"{name} ({confidence}%)"
                            else:
                                # Nếu khoảng cách quá lớn, đánh dấu là không nhận dạng
                                name = "Không nhận dạng"
                        
                        # Scale lại vị trí
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        # Vẽ khung và tên
                        color = (0, 255, 0) if name != "Không nhận dạng" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        draw_text_with_background(frame, name, (left + 6, bottom - 6))
                    
            except Exception as e:
                print(f"Lỗi khi xử lý frame: {str(e)}")
                continue
                
            # Nhận diện đồ vật
            frame = self.object_detector.detect_objects(frame)
            
            # Hiển thị frame
            cv2.imshow('Face Recognition', frame)
            
            # Xử lý phím nhấn
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('h'):
                self.show_menu()
            elif key == ord('a'):
                self.add_new_face()
            elif key == ord('f'):
                self.add_face_from_file()
            elif key == ord('d'):
                self.delete_user()
            elif key == ord('i'):
                self.add_face_to_existing_user()
            elif key == ord('v'):
                self.view_user_info()
            elif key == ord('o'):
                is_enabled = self.object_detector.toggle()
                print(f"Nhận diện đồ vật: {'Bật' if is_enabled else 'Tắt'}")
            elif key == ord('c'):
                self.add_object_class()
            elif key == ord('t'):
                self.train_object_detection()
        
        # Giải phóng tài nguyên
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run() 