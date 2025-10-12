import os
from PIL import Image
import numpy as np
import albumentations as A
import shutil

# Paths
DATASET_DIR = os.path.join('breast_cancer_detection', 'datasetji')
BALANCED_DIR = os.path.join('breast_cancer_detection', 'datasetji_balanced')
BENIGN = 'Benign'
MALIGNANT = 'Malignant'

# Augmentation pipeline for malignant images using albumentations
malignant_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5)
])

# Utility to get image paths
def get_image_paths(folder):
    return [os.path.join(folder, fname) for fname in os.listdir(folder) if fname.lower().endswith('.png')]

# Create balanced directory structure
def setup_balanced_dirs():
    for cls in [BENIGN, MALIGNANT]:
        os.makedirs(os.path.join(BALANCED_DIR, cls), exist_ok=True)

# Copy benign images directly
def copy_benign_images():
    src = os.path.join(DATASET_DIR, BENIGN)
    dst = os.path.join(BALANCED_DIR, BENIGN)
    for img_path in get_image_paths(src):
        shutil.copy(img_path, dst)

# Augment malignant images to match benign count
def augment_malignant_images():
    src = os.path.join(DATASET_DIR, MALIGNANT)
    dst = os.path.join(BALANCED_DIR, MALIGNANT)
    benign_count = len(get_image_paths(os.path.join(DATASET_DIR, BENIGN)))
    malignant_paths = get_image_paths(src)
    malignant_count = len(malignant_paths)
    augment_needed = benign_count - malignant_count
    print(f"Benign: {benign_count}, Malignant: {malignant_count}, Augment needed: {augment_needed}")
    # Copy original malignant images
    for img_path in malignant_paths:
        shutil.copy(img_path, dst)
    # Augment
    i = 0
    while i < augment_needed:
        img_path = malignant_paths[i % malignant_count]
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        aug = malignant_aug(image=img_np)
        aug_img_np = aug['image']
        aug_img = Image.fromarray(aug_img_np)
        aug_name = f"aug_{i}_{os.path.basename(img_path)}"
        aug_img.save(os.path.join(dst, aug_name))
        i += 1
    print(f"Total malignant images after augmentation: {len(os.listdir(dst))}")

if __name__ == "__main__":
    setup_balanced_dirs()
    copy_benign_images()
    augment_malignant_images()
    print(f"Balanced dataset created at: {BALANCED_DIR}")