# Breast Cancer Detection (BCD)

## Overview
This repository provides a complete pipeline for breast cancer detection using deep learning. It includes data exploration, augmentation, balancing, splitting, model training (PyTorch and TensorFlow/Keras), and evaluation. The project is designed for researchers and practitioners in medical imaging and AI.

## Repository Structure
```
BCD/
├── breast_cancer_detection/
│   ├── datasetji/                # Original, imbalanced dataset
│   │   ├── Benign/
│   │   └── Malignant/
│   ├── datasetji_balanced/       # Balanced dataset after augmentation
│   ├── Results/                  # Model results and outputs
│   ├── *.pth                     # Saved PyTorch models
│   ├── *.ipynb                   # Jupyter notebooks for exploration and training
│   └── ...
├── splitted_dataset/             # Train/val/test split for Keras/TensorFlow
│   ├── train/
│   ├── val/
│   └── test/
├── data_splitter.py              # Script to split balanced data
├── Oversamplings.py              # Script for oversampling/augmentation
├── E-Net.ipynb                   # Keras/TensorFlow EfficientNet notebook
├── efficientNET.ipynb            # PyTorch EfficientNet notebook
├── data-exploration.ipynb        # Data analysis and visualization notebook
├── EDA.ipynb                     # Additional exploratory data analysis
├── dataset_comparison.ipynb      # Compare original and balanced datasets
├── README.md                     # Project documentation
└── ...
```

## Data Preparation
1. **Original Data**: Place raw images in `breast_cancer_detection/datasetji/Benign` and `.../Malignant`.
2. **Balancing**: Use `Oversamplings.py` to augment malignant images and create a balanced dataset in `datasetji_balanced`.
   - Uses augmentation to ensure equal samples for both classes.
3. **Splitting**: Run `data_splitter.py` to split the balanced dataset into train/val/test sets in `splitted_dataset/`.
   - Default split: 70% train, 20% test, 10% val.

## Model Training
### TensorFlow/Keras (E-Net.ipynb)
- Loads images from `splitted_dataset/`.
- Trains EfficientNetB0 with transfer learning and fine-tuning.
- Uses callbacks (`EarlyStopping`, `ModelCheckpoint`).
- Evaluates with accuracy, confusion matrix, classification report, ROC curve.

## Notebooks
- **data-exploration.ipynb**: Visualizes class distribution, pixel statistics, and imbalance analysis.
- **efficientNET.ipynb**: PyTorch pipeline for training and evaluation.
- **E-Net.ipynb**: Keras pipeline for training and evaluation.
- **EDA.ipynb**: Additional exploratory data analysis.
- **dataset_comparison.ipynb**: Compare original and balanced datasets.

## Scripts
- **Oversamplings.py**: Balances the dataset using augmentation.
- **data_splitter.py**: Splits balanced data into train/val/test folders.

## Requirements
- Python 3.8+
- PyTorch, torchvision
- TensorFlow, Keras
- albumentations (if used for augmentation)
- scikit-learn
- matplotlib, seaborn
- PIL, numpy

Install dependencies:
```bash
pip install torch torchvision tensorflow keras albumentations scikit-learn matplotlib seaborn pillow numpy
```

## Usage
1. **Balance the dataset**:
   ```bash
   python Oversamplings.py
   ```
2. **Split the dataset**:
   ```bash
   python data_splitter.py
   ```
3. **Train models**:
   - Open and run `E-Net.ipynb` (Keras).

## Results & Evaluation
- Confusion matrix, classification report, ROC/PR curves, and clinical metrics are generated in notebooks.
- Model weights are saved as `.pth` (PyTorch) or `.h5` (Keras).
- Results are stored in `breast_cancer_detection/Results/`.

## Researcher Notes
- The pipeline is designed for reproducibility and clinical relevance.
- Data augmentation is aggressive for the minority class (Malignant) to address imbalance.
- Evaluation focuses on sensitivity/recall for cancer detection.
- All scripts and notebooks are modular and can be adapted for other medical imaging tasks.

