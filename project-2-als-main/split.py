import os
import random
import shutil
from pathlib import Path

train_dir = "../data/training/"
val_dir = "../data/validation"

src_images = os.path.join(train_dir, "images")
src_groundtruth = os.path.join(train_dir, "groundtruth")

dst_images = os.path.join(val_dir, "images")
dst_groundtruth = os.path.join(val_dir, "groundtruth")

SPLIT_RATIO = 0.2 

def move_files():
    if not os.path.exists(src_images) or not os.path.exists(src_groundtruth):
        print("Error: Source directories not found. Check your paths.")
        return

    all_images = [f for f in os.listdir(src_images) if not f.startswith('.')]
    total_files = len(all_images)
    
    num_to_move = int(total_files * SPLIT_RATIO)
    print(f"Found {total_files} images. Moving {num_to_move} ({int(SPLIT_RATIO*100)}%) to validation...")

    files_to_move = random.sample(all_images, num_to_move)

    moved_count = 0
    for img_filename in files_to_move:
        current_img_path = os.path.join(src_images, img_filename)
        new_img_path = os.path.join(dst_images, img_filename)

        file_stem = Path(img_filename).stem

        gt_files = os.listdir(src_groundtruth)
        matching_gt = [f for f in gt_files if Path(f).stem == file_stem]

        if matching_gt:
            gt_filename = matching_gt[0]
            
            current_gt_path = os.path.join(src_groundtruth, gt_filename)
            new_gt_path = os.path.join(dst_groundtruth, gt_filename)

            shutil.move(current_img_path, new_img_path)
            shutil.move(current_gt_path, new_gt_path)
            
            moved_count += 1
        else:
            print(f"Skipped {img_filename}: No matching groundtruth found.")

    print(f"Success! Moved {moved_count} pairs to validation.")

if __name__ == "__main__":
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_groundtruth, exist_ok=True)
    
    move_files()