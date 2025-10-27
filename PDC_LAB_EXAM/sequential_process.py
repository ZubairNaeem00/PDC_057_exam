import os
import time
import cv2
import numpy as np

input_dir = "Images"      
output_dir = "outputs/output_seq"

# Watermark text
WATERMARK_TEXT = "PDC"

def add_watermark_cv2(img, text, alpha=0.45):

    overlay = img.copy()
    h, w = img.shape[:2]

    # Choose font scale and thickness relative to image size
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(min(w, h) / 200))

    (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.8, thickness)
    x = int(w - text_width - 10)
    y = int(h - 10)

    # Put white text on overlay
    cv2.putText(overlay, text, (x, y), font, 0.8, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    # Blend overlay onto original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img



start_time = time.time()

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)

                # Read image with OpenCV
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: failed to read {input_path}")
                continue

                # Resize to 128x128 using Lanczos
            img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)

                # Add watermark
            img_watermarked = add_watermark_cv2(img_resized, WATERMARK_TEXT)

                # Save to output directory
            output_path = os.path.join(output_subdir, file)
            cv2.imwrite(output_path, img_watermarked)

total_time = time.time() - start_time
print(f"Sequential Processing Time: {total_time:.2f} seconds")



