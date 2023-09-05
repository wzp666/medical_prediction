import glob
import os
import shutil
import random

# 设置路径和比例
source_folder = 'E:/Datasets/AI/diease_segment'
target_folder = 'E:/Datasets/AI/diease_segment_split'

os.makedirs(target_folder, exist_ok=True)

train_ratio = 0.8

image_dir_name = 'images'
mask_dir_name = 'masks'

source_image_path = os.path.join(source_folder, image_dir_name)
source_mask_path = os.path.join(source_folder, mask_dir_name)
target_image_path = os.path.join(target_folder, image_dir_name)
target_mask_path = os.path.join(target_folder, mask_dir_name)

image_paths = glob.glob(os.path.join(source_image_path, "*.jpg"))
mask_paths = glob.glob(os.path.join(source_mask_path, "*.png"))

assert len(image_paths) == len(mask_paths)

combine_paths = []
for i in range(len(image_paths)):
    combine_paths.append((image_paths[i], mask_paths[i]))

random.shuffle(combine_paths)

split_index = int(train_ratio * len(image_paths))

# 分割训练集和验证集
train_combines = combine_paths[:split_index]
val_combines = combine_paths[split_index:]

# 创建目标文件夹
train_image_folder = os.path.join(target_image_path, 'train')
train_mask_folder = os.path.join(target_mask_path, 'train')
val_image_folder = os.path.join(target_image_path, 'val')
val_mask_folder = os.path.join(target_mask_path, 'val')

os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(train_mask_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)
os.makedirs(val_mask_folder, exist_ok=True)

# 复制训练集图片到目标文件夹
for img, mask in train_combines:
    shutil.copy(img, os.path.join(train_image_folder, os.path.basename(img)))
    shutil.copy(mask, os.path.join(train_mask_folder, os.path.basename(mask)))

# 复制验证集图片到目标文件夹
for img, mask in val_combines:
    shutil.copy(img, os.path.join(val_image_folder, os.path.basename(img)))
    shutil.copy(mask, os.path.join(val_mask_folder, os.path.basename(mask)))
