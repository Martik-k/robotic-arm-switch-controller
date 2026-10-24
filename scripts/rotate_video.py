import cv2
import glob
import os

video_files = glob.glob("*.mp4")
os.makedirs("fixed_videos", exist_ok=True)

for path in video_files:
    print(f"Обробка: {path}...")
    cap = cv2.VideoCapture(path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join("fixed_videos", os.path.basename(path))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Поворот на 180 градусів
        rotated = cv2.rotate(frame, cv2.ROTATE_180)
        out.write(rotated)
        
    cap.release()
    out.release()

print("Готово! Усі виправлені відео у папці 'fixed_videos'.")