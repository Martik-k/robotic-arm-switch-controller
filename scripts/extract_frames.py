import cv2
import os


VIDEO_PATH = "videos/video1.mp4" 
OUTPUT_DIR = "images/train/video1"
FRAME_STEP = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0
saved_count = 0

print("Витягуємо кадри...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Зберігаємо лише кожен 10-й кадр
    if frame_id % FRAME_STEP == 0:
        # Формат імені точно повторює назви файлів розмітки з CVAT (6 цифр)
        filename = f"v1_frame_{frame_id:06d}.PNG"  # або .jpg
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, frame)
        saved_count += 1
        
    frame_id += 1

cap.release()
print(f"Готово! Збережено {saved_count} кадрів у папку '{OUTPUT_DIR}'.")