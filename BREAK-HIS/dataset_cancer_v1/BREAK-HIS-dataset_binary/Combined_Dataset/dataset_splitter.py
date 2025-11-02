import os
import shutil
import random
from pathlib import Path

data_dir = Path("BREAK-HIS\\dataset_cancer_v1\\BREAK-HIS-dataset_binary\\Combined_Dataset\\400X_BALANCED")

base_dir = Path("Final-400X")
train_dir = base_dir / "train"
test_dir = base_dir / "test"
val_dir = base_dir / "val"

for folder in [train_dir, test_dir, val_dir]:
    for category in ["benign", "malignant"]:
        os.makedirs(folder / category, exist_ok=True)

train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1

for category in ["benign", "malignant"]:
    src_folder = data_dir / category
    images = os.listdir(src_folder)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_test = int(test_ratio * n_total)

    train_imgs = images[:n_train]
    test_imgs = images[n_train:n_train + n_test]
    val_imgs = images[n_train + n_test:]

    for img in train_imgs:
        shutil.copy(src_folder / img, train_dir / category / img)

    for img in test_imgs:
        shutil.copy(src_folder / img, test_dir / category / img)

    for img in val_imgs:
        shutil.copy(src_folder / img, val_dir / category / img)

print("Dataset split successfully!")