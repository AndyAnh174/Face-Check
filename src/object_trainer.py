import os
import cv2
import yaml
import shutil
from ultralytics import YOLO

class ObjectTrainer:
    def __init__(self):
        """Khởi tạo trainer"""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.dataset_dir = os.path.join(self.data_dir, "object_dataset")
        self.images_dir = os.path.join(self.dataset_dir, "images")
        self.labels_dir = os.path.join(self.dataset_dir, "labels")
        self.model_dir = os.path.join(self.data_dir, "models")
        
        # Tạo các thư mục cần thiết
        for dir_path in [self.dataset_dir, self.images_dir, self.labels_dir, 
                        os.path.join(self.images_dir, "train"),
                        os.path.join(self.images_dir, "val"),
                        os.path.join(self.labels_dir, "train"),
                        os.path.join(self.labels_dir, "val"),
                        self.model_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Load hoặc tạo mới danh sách classes
        self.classes_file = os.path.join(self.dataset_dir, "classes.txt")
        self.classes = self._load_classes()
        
    def _load_classes(self):
        """Load danh sách classes từ file"""
        if os.path.exists(self.classes_file):
            with open(self.classes_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines()]
        return []
        
    def _save_classes(self):
        """Lưu danh sách classes vào file"""
        with open(self.classes_file, 'w', encoding='utf-8') as f:
            for cls in self.classes:
                f.write(f"{cls}\n")
                
    def add_class(self, class_name):
        """Thêm class mới"""
        if class_name not in self.classes:
            self.classes.append(class_name)
            self._save_classes()
            return len(self.classes) - 1  # trả về index của class mới
        return self.classes.index(class_name)
        
    def capture_training_images(self, class_name, num_images=30):
        """Chụp ảnh training cho một class"""
        class_idx = self.add_class(class_name)
        
        # Khởi tạo camera
        cap = cv2.VideoCapture(0)
        
        images_captured = 0
        while images_captured < num_images:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Hiển thị frame
            cv2.imshow('Capture Training Images', frame)
            cv2.putText(frame, f"Press SPACE to capture ({images_captured}/{num_images})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # space key
                # Lưu ảnh
                timestamp = f"{images_captured:04d}"
                image_path = os.path.join(self.images_dir, "train", 
                                        f"{class_name}_{timestamp}.jpg")
                cv2.imwrite(image_path, frame)
                
                # Tạo label file
                self._create_label_file(image_path, class_idx)
                
                images_captured += 1
                print(f"Đã chụp {images_captured}/{num_images} ảnh")
        
        cap.release()
        cv2.destroyAllWindows()
        
    def _create_label_file(self, image_path, class_idx):
        """Tạo file label cho ảnh training"""
        # Chuyển đổi từ images/train/image.jpg sang labels/train/image.txt
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        # Tạo label file với bbox mặc định ở giữa ảnh
        with open(label_path, 'w') as f:
            # Format: <class_idx> <x_center> <y_center> <width> <height>
            # Giá trị được chuẩn hóa từ 0-1
            f.write(f"{class_idx} 0.5 0.5 0.8 0.8\n")
            
    def create_data_yaml(self):
        """Tạo file data.yaml cho training"""
        data = {
            'path': self.dataset_dir,
            'train': os.path.join('images', 'train'),
            'val': os.path.join('images', 'val'),
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        
        yaml_path = os.path.join(self.dataset_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
            
        return yaml_path
        
    def train_model(self, epochs=50):
        """Training model"""
        if not self.classes:
            print("Chưa có classes nào được thêm vào!")
            return
            
        # Tạo file data.yaml
        data_yaml = self.create_data_yaml()
        
        # Khởi tạo model
        model = YOLO('yolov8n.pt')  # load model cơ bản
        
        # Training
        try:
            model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=640,
                batch=16,
                patience=10,
                project=self.model_dir
            )
            print("Training hoàn tất!")
            return os.path.join(self.model_dir, 'train', 'weights', 'best.pt')
        except Exception as e:
            print(f"Lỗi khi training: {str(e)}")
            return None 