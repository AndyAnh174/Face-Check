import cv2
import numpy as np

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Thay đổi kích thước ảnh giữ nguyên tỷ lệ"""
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
        
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)

def draw_text_with_background(image, text, position, font=cv2.FONT_HERSHEY_DUPLEX, 
                            font_scale=0.6, color=(255, 255, 255), thickness=1):
    """Vẽ text với nền"""
    # Lấy kích thước text
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Vẽ nền
    cv2.rectangle(image, 
                 (position[0], position[1] - text_height - baseline),
                 (position[0] + text_width, position[1] + baseline),
                 (0, 255, 0),
                 cv2.FILLED)
    
    # Vẽ text
    cv2.putText(image, text, position, font, font_scale, color, thickness) 