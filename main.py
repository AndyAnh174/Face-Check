import cv2
from src.face_recognition_app import FaceRecognitionApp

def main():
    app = FaceRecognitionApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nĐóng ứng dụng...")
    except Exception as e:
        print(f"Lỗi không mong muốn: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 