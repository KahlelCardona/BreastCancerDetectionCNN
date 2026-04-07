from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader   
from torchvision import transforms
    
class Config:
    DATA_DIR = Path("data/raw")
    JPEG_DIR = DATA_DIR / "jpeg"
    CSV_DIR = DATA_DIR / "csv"
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    IMAGE_SIZE = 224
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5           # added to reduce overfitting
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    PLOT_DIR = Path("plots")
    PLOT_DIR.mkdir(exist_ok=True)

class MammogramRawDataset(Dataset):
    def __init__(self, csv_types):
        self.samples = []
        self.num_benign = 0
        self.num_malignant = 0

        all_images = list(Config.JPEG_DIR.rglob("*.jpg"))
        print(f"Found {len(all_images)} total JPG files")

        for csv_type in csv_types:
            csv_path = self._get_csv_path(csv_type)
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping.")
                continue

            df = pd.read_csv(csv_path)
            df = df.dropna(subset=["pathology"])

            for _, row in df.iterrows():
                pathology = row["pathology"].strip().lower()
                label = 0 if "benign" in pathology else 1
                if label == 0:
                    self.num_benign += 1
                else:
                    self.num_malignant += 1

                img_path = self._find_image_path(row, all_images)
                if img_path is not None:
                    self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} labelled samples from {csv_types}")
        print(f"  - Benign: {self.num_benign}")
        print(f"  - Malignant: {self.num_malignant}")

    def _get_csv_path(self, csv_type):
        csv_files = {
            "mass_train": "mass_case_description_train_set.csv",
            "mass_test": "mass_case_description_test_set.csv",
            "calc_train": "calc_case_description_train_set.csv",
            "calc_test": "calc_case_description_test_set.csv",
        }
        return Config.CSV_DIR / csv_files[csv_type]

    def _find_image_path(self, row, all_images):
        for col in ["image file path", "cropped image file path"]:
            if col in row and isinstance(row[col], str):
                parts = row[col].split("/")
                if len(parts) >= 3:
                    uid = parts[2]   
                    for img_path in all_images:
                        if uid in str(img_path):
                            return img_path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return image, label  


#apply transforms
class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


#Transforms
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
