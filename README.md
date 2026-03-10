# Mammogram Benign vs Malignant Classification (PyTorch EfficientNet)

## Overview

This project implements a **deep learning pipeline for classifying mammogram images as benign or malignant** using **PyTorch** and **transfer learning with EfficientNet-B0**. The program loads mammography images and metadata from CSV files, preprocesses the images, trains a neural network, and evaluates the model using multiple performance metrics.

The goal of the project is to demonstrate how **deep learning and medical imaging can be combined to assist in breast cancer detection** by automatically identifying suspicious patterns in mammogram images.

---

# Project Features

* Custom **PyTorch Dataset** for mammography data
* Image preprocessing and augmentation
* Transfer learning using **EfficientNet-B0**
* Training and validation pipelines
* Performance metrics including:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
* Automatic **checkpoint saving** for the best model
* GPU support if CUDA is available

---

# Project Structure

```
project/
│
├── data/
│   └── raw/
│       ├── jpeg/           # Mammogram image files (.jpg)
│       └── csv/            # Metadata CSV files
│
├── checkpoints/           # Saved trained models
│
├── train_model.py         # Main training script
│
└── README.md              # Project documentation
```

---

# How the Program Works

The program follows a **standard deep learning workflow**:

```
Dataset (Images + CSV)
        ↓
Custom PyTorch Dataset
        ↓
Image Preprocessing
        ↓
DataLoader (Batching)
        ↓
EfficientNet Model
        ↓
Training Loop
        ↓
Validation Metrics
        ↓
Best Model Saved
```

---

# Dataset Handling

The program uses a custom dataset class called:

```
CBIS-DDSM: Breast Cancer Image Dataset
```

This class:

1. Reads **CSV metadata files**
2. Extracts pathology labels
3. Locates the corresponding image files
4. Converts pathology descriptions into numeric labels

Label encoding:

```
Benign     → 0
Malignant  → 1
```

Each dataset sample contains:

```
(image_tensor, label)
```

---

# Image Preprocessing

Images are transformed before being passed to the neural network.

### Training Transformations

```
Resize image to 224×224
Random horizontal flip
Random rotation
Brightness/contrast adjustment
Convert to tensor
Normalize pixel values
```

Data augmentation improves model generalization by exposing the model to slightly different variations of each image.

### Validation Transformations

Validation images are only:

```
Resized
Converted to tensor
Normalized
```

No randomness is applied during validation to ensure consistent evaluation.

---

# Normalization

Images are normalized using the following values:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

These are the **ImageNet dataset statistics** used when training the pretrained EfficientNet model.

Normalization standardizes pixel values so that:

```
normalized_pixel = (pixel - mean) / std
```

This helps the neural network train more efficiently and ensures compatibility with the pretrained weights.

---

# Model Architecture

The program uses **EfficientNet-B0**, a convolutional neural network architecture designed for efficient image classification.

### Transfer Learning

Instead of training the network from scratch:

1. The pretrained EfficientNet model is loaded.
2. Early layers are **frozen** to preserve learned features.
3. The classifier layer is replaced with a new one for binary classification.

New classifier:

```
Dropout (0.2)
Fully Connected Layer (2 outputs)
```

Output classes:

```
0 → Benign
1 → Malignant
```

---

# Training Process

Training occurs over multiple epochs.

Each epoch consists of:

1. **Forward pass** – images are passed through the model
2. **Loss calculation** – prediction error is computed
3. **Backpropagation** – gradients are calculated
4. **Optimizer update** – model weights are updated

Loss function used:

```
CrossEntropyLoss
```

Optimizer used:

```
Adam Optimizer
```

A learning rate scheduler automatically reduces the learning rate when validation loss stops improving.

---

# Model Evaluation

After each epoch, the model is evaluated on the validation dataset.

The following metrics are calculated:

### Accuracy

Percentage of correct predictions.

### Precision

How many predicted malignant cases were actually malignant.

### Recall

How many actual malignant cases were correctly detected.

### F1 Score

Harmonic mean of precision and recall.

This metric is especially important for **medical classification problems** where false negatives are critical.

---

# Model Checkpoints

The program automatically saves the best performing model.

Checkpoint location:

```
checkpoints/best_model.pth
```

A checkpoint is saved whenever:

```
validation_loss < best_validation_loss
```

After training finishes, the best checkpoint is loaded for final evaluation.

---

# Hardware Support

The program automatically detects available hardware.

```
GPU (CUDA) if available
CPU otherwise
```

Device selection:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Using a GPU significantly speeds up training.

---

# How to Run the Project

### 1. Install Dependencies

```
pip install torch torchvision pandas numpy scikit-learn pillow
```

---

### 2. Organize the Dataset

```
data/raw/jpeg/   → image files
data/raw/csv/    → dataset metadata
```

---

### 3. Run Training

```
python train_model.py
```

The script will:

1. Load the dataset
2. Build the model
3. Train for multiple epochs
4. Evaluate validation performance
5. Save the best model

---

# Example Output

```
Epoch 5/20
Train Loss: 0.2841 Acc: 0.89
Val Loss: 0.3125 Acc: 0.86 Prec: 0.84 Rec: 0.82 F1: 0.83
```

Final results:

```
Final Validation Results:
Loss: 0.3012
Accuracy: 0.87
Precision: 0.85
Recall: 0.83
F1 Score: 0.84
```

---

# Future Improvements

Possible improvements to the project include:

* Fine-tuning deeper layers of EfficientNet
* Using larger EfficientNet variants
* Adding Grad-CAM visualizations for explainability
* Implementing k-fold cross-validation
* Handling class imbalance with weighted loss
* Deploying the model as a medical decision-support tool

---

# Disclaimer

This project is intended for **research and educational purposes only**.
It should **not be used as a medical diagnostic tool** without proper clinical validation.

---
