"""Utility script to rotate video files 180 degrees to correct upside-down recordings."""

import glob
import os
import cv2

video_files = glob.glob("*.mp4")
os.makedirs("fixed_videos", exist_ok=True)

for path in video_files:
    print(f"Processing: {path}...")
    cap = cv2.VideoCapture(path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join("fixed_videos", os.path.basename(path))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Rotate 180 degrees
        rotated = cv2.rotate(frame, cv2.ROTATE_180)
        out.write(rotated)

    cap.release()
    out.release()

print("Done! All corrected videos are saved in the 'fixed_videos' directory.")