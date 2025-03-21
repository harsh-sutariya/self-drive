import os
import shutil
import random

# Set the root directory of your dataset
root_dir = 'dataset/project-1-at-2025-02-28-17-43-08123527'

# Set the split ratio (e.g., 0.8 for 80% train, 20% val)
train_ratio = 0.8

# Create train and val directories if they don't exist
for folder in ['images', 'labels']:
    for split in ['train', 'val']:
        os.makedirs(os.path.join(root_dir, folder, split), exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(os.path.join(root_dir, 'images')) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle the files randomly
random.shuffle(image_files)

# Calculate the split index
split_index = int(len(image_files) * train_ratio)

# Split and move files
for i, image_file in enumerate(image_files):
    source_image = os.path.join(root_dir, 'images', image_file)
    source_label = os.path.join(root_dir, 'labels', os.path.splitext(image_file)[0] + '.txt')
    
    if i < split_index:
        dest_folder = 'train'
    else:
        dest_folder = 'val'
    
    dest_image = os.path.join(root_dir, 'images', dest_folder, image_file)
    dest_label = os.path.join(root_dir, 'labels', dest_folder, os.path.splitext(image_file)[0] + '.txt')
    
    shutil.move(source_image, dest_image)
    if os.path.exists(source_label):
        shutil.move(source_label, dest_label)

print("Dataset split completed!")
