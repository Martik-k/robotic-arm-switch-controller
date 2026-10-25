"""Utility script to extract every N-th frame from video recordings for CVAT annotation."""

import os
import cv2

VIDEO_PATH = "videos/video6.mp4"
OUTPUT_DIR = "images/val/video6"
FRAME_STEP = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0
saved_count = 0

print(f"Extracting frames from '{VIDEO_PATH}' (step={FRAME_STEP})...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save only every 10th frame to reduce temporal correlation
    if frame_id % FRAME_STEP == 0:
        # Name format matches CVAT zero-padded 6-digit image indexing
        filename = f"v6_frame_{frame_id:06d}.PNG"
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, frame)
        saved_count += 1

    frame_id += 1

cap.release()
print(f"Done! Successfully saved {saved_count} frames to '{OUTPUT_DIR}'.")