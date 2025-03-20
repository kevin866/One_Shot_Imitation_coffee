import os
import shutil
import random

# Define paths
dataset_path = "dataset"
train_img_dir = os.path.join(dataset_path, "images", "train")
train_lbl_dir = os.path.join(dataset_path, "labels", "train")
val_img_dir = os.path.join(dataset_path, "images", "val")
val_lbl_dir = os.path.join(dataset_path, "labels", "val")

# Ensure validation directories exist
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# Get list of images
train_images = [f for f in os.listdir(train_img_dir) if f.endswith(".jpg")]
num_val = int(0.1 * len(train_images))  # 20% split

# Randomly select validation images
val_images = random.sample(train_images, num_val)

# Move selected images and labels to validation folder
for img_name in val_images:
    img_path = os.path.join(train_img_dir, img_name)
    lbl_path = os.path.join(train_lbl_dir, img_name.replace(".jpg", ".txt"))

    # Move image
    shutil.move(img_path, os.path.join(val_img_dir, img_name))
    
    # Move label (if exists)
    if os.path.exists(lbl_path):
        shutil.move(lbl_path, os.path.join(val_lbl_dir, os.path.basename(lbl_path)))

print(f"Moved {num_val} images to validation set.")
