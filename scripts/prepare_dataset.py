import os
import shutil
import random
from PIL import Image
from tqdm import tqdm

def create_dataset(input_folder, output_root="datasets/custom", val_split=0.1):
    train_hr = os.path.join(output_root, "train/hr")
    val_hr = os.path.join(output_root, "val/hr")
    val_lr = os.path.join(output_root, "val/lr")
    meta_info_path = os.path.join(output_root, "train/meta_info.txt")

    os.makedirs(train_hr, exist_ok=True)
    os.makedirs(val_hr, exist_ok=True)
    os.makedirs(val_lr, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(files)

    split = int(len(files) * val_split)
    val_files, train_files = files[:split], files[split:]

    print(f"Train images: {len(train_files)} | Val images: {len(val_files)}")

    # Copy files to appropriate folders
    for f in train_files:
        shutil.copy(os.path.join(input_folder, f), os.path.join(train_hr, f))
    for f in val_files:
        shutil.copy(os.path.join(input_folder, f), os.path.join(val_hr, f))

    # Generate LR images for val set
    for f in tqdm(val_files, desc="Generating LR val images"):
        img = Image.open(os.path.join(val_hr, f)).convert("RGB")
        img_lr = img.resize((img.width // 4, img.height // 4), Image.BICUBIC)
        img_lr.save(os.path.join(val_lr, f), quality=75)

    # Generate meta info
    with open(meta_info_path, 'w') as f:
        for name in train_files:
            f.write(f"{name}\n")

    print(f"Meta info written to {meta_info_path}")

if __name__ == "__main__":
    create_dataset("pret_images/cropped")  # Change this folder path
