import os
import time
import cv2
import numpy as np
from numba import jit, prange
from multiprocessing import Pool

input_dir = "Images"
output_dir = "outputs/output_parallel"

# Get number of CPU cores for OpenMP
NUM_CORES = os.cpu_count()

# Watermark text
WATERMARK_TEXT = "PDC"

def add_watermark(image, text, alpha=0.45):

    overlay = image.copy()
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(min(w, h) / 200))

    (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.8, thickness)
    x = int(w - text_width - 10)
    y = int(h - 10)

    cv2.putText(overlay, text, (x, y), font, 0.8, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def get_image_files(input_dir):
    
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_dir)
                image_files.append((input_path, rel_path))
    return image_files

def process_image(image_info):
    
    input_path, rel_path = image_info
    try:
        # Create output subdirectory
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        # Read image with OpenCV
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: failed to read {input_path}")
            return False

        # Resize to 128x128 using Lanczos
        img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)

        # Add watermark
        img_watermarked = add_watermark(img_resized, WATERMARK_TEXT)

        # Save to output directory
        output_path = os.path.join(output_subdir, os.path.basename(input_path))
        cv2.imwrite(output_path, img_watermarked)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_images_with_workers(num_workers):
    
    start_time = time.time()
    
    # Get list of all image files
    image_files = get_image_files(input_dir)
    total_images = len(image_files)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images using specified number of workers
    processed = 0
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_image, image_files)
        processed = sum(1 for x in results if x)
    
    # Calculate processing time
    total_time = time.time() - start_time
    return total_time, processed, total_images

def run_parallel_benchmarks():
    
    # Worker configurations to test
    worker_configs = [1, 2, 4, 8]
    results = []
    
    print("\nRunning parallel processing...")
    print("----------------------------------------")
    
    # Run benchmarks for each worker configuration
    for workers in worker_configs:
        time_taken, processed, total = process_images_with_workers(workers)
        results.append((workers, time_taken))
        print(f"\nCompleted run with {workers} workers:")
        print(f"Processed {processed}/{total} images in {time_taken:.2f} seconds")
    
    # Calculate speedups relative to single worker
    base_time = results[0][1]  # Time for single worker
    speedups = [(w, t, base_time/t) for w, t in results]
    
    # Print results table
    print("\nParallel Processing Results")
    print("==========================")
    print("\nWorkers | Time (s) | Speedup")
    print("---------|----------|--------")
    for workers, time_taken, speedup in speedups:
        print(f"{workers:7d} | {time_taken:8.2f} | {speedup:6.2f}x")


run_parallel_benchmarks()
