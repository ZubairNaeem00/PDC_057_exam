import os
import time
import cv2
import numpy as np
from mpi4py import MPI
import math

input_dir = "Images"
output_dir = "outputs/output_dis"

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
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: failed to read {input_path}")
            return False

        img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        img_watermarked = add_watermark(img_resized, WATERMARK_TEXT)
        output_path = os.path.join(output_subdir, os.path.basename(input_path))
        cv2.imwrite(output_path, img_watermarked)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def process_list(image_list):
    start = time.time()
    count = 0
    for info in image_list:
        if process_image(info):
            count += 1
    return count, time.time() - start


def get_sequential_time():
    files = get_image_files(input_dir)
    count, t = process_list(files)
    return count, t


def main_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("run with terminal")
        return

    # Only use first two ranks for this 2-node simulation
    if rank > 1:
        # Extra ranks idle and exit
        if rank == 2:
            print("Note: ranks >1 are idle in this 2-node simulation.")
        return

    if rank == 0:
        # Master (also node 1): gather file list and split
        all_files = get_image_files(input_dir)
        total_images = len(all_files)
        if total_images == 0:
            print("No images found in input directory. Nothing to do.")
            return

        mid = total_images // 2
        node0_files = all_files[:mid]
        node1_files = all_files[mid:]

        # Send node1_files to rank 1
        comm.send(node1_files, dest=1, tag=11)

        # Process node0 files locally
        processed0, time0 = process_list(node0_files)

        # Receive results from rank1
        res = comm.recv(source=1, tag=12)
        processed1 = res.get('processed', 0)
        time1 = res.get('time', 0.0)

        # Calculate distributed time (wall time = max of node times)
        total_distributed_time = max(time0, time1)

        # For comparison, run sequential processing (on master)
        seq_count, seq_time = get_sequential_time()

        # Print summary
        print(f"Node 1 processed {processed0} images in {time0:.1f}s")
        print(f"Node 2 processed {processed1} images in {time1:.1f}s")
        print(f"Total distributed time: {total_distributed_time:.1f}s")
        if total_distributed_time > 0:
            efficiency = seq_time / total_distributed_time
            print(f"Efficiency: {efficiency:.2f}x over sequential")
        else:
            print("Total distributed time is zero (unexpected)")

    elif rank == 1:
        # Worker node 2: receive file list and process
        node_files = comm.recv(source=0, tag=11)
        processed, t = process_list(node_files)
        comm.send({'processed': processed, 'time': t}, dest=0, tag=12)


main_mpi()
