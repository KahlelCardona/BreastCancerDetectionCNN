import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

class Config:
    DATA_DIR = Path("data/raw")
    JPEG_DIR = DATA_DIR / "jpeg"
    CSV_DIR = DATA_DIR / "csv"
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    IMAGE_SIZE = 224
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    PLOT_DIR = Path("plots")
    PLOT_DIR.mkdir(exist_ok=True)

class MammogramDataset(Dataset):
    def __init__(self, csv_types, transform=None):
    # list of csv_types: ["mass_train", "mass_test", "calc_train", "calc_test"]
        self.transform = transform
        self.samples = []

       #scan image directory once
        all_images = list(Config.JPEG_DIR.rglob("*.jpg"))
        print(f" Found {len(all_images)} total JPG files")

        #process each csv file 
        for csv_type in csv_types:
            csv_path = self._get_csv_path(csv_type)
            if not csv_path.exists():
                print(f" Warning: {csv_path} not found, skipping.")
                continue

            df = pd.read_csv(csv_path)

            # Keep only rows with pathology info
            df = df.dropna(subset=["pathology"])

            for idx, row in df.iterrows():
                pathology = row["pathology"].strip().lower()
                # Map to binary benign if contains "benign", else malignant
                label = 0 if "benign" in pathology else 1

                # Try to find the corresponding image
                img_path = self._find_image_path(row, all_images)
                if img_path is not None:
                    self.samples.append((img_path, label))
    
        print(f" Loaded {len(self.samples)} labelled samples from {csv_types}")

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

        if self.transform:
            image = self.transform(image)

        return image, label
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


def create_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    #Classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, 2)  # 2 classes: benign(0), malignant(1)
    )

    # Unfreeze the classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(Config.DEVICE)
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, zero_division=0)
    epoch_rec = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1

def save_learning_curve(train_values, val_values, ylabel, filename):

    plt.figure(figsize=(8, 6))

    plt.plot(train_values, label="Train")
    plt.plot(val_values, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Learning Curve")
    plt.legend()

    plt.savefig(Config.PLOT_DIR / filename)

    plt.close()

def create_full_dataset():

    datasets = [
        MammogramDataset(["mass_train"], transform=get_train_transforms()),
        MammogramDataset(["calc_train"], transform=get_train_transforms())
    ]

    full_dataset = torch.utils.data.ConcatDataset(datasets)

    labels = []

    for dataset in datasets:
        labels.extend([label for _, label in dataset.samples])

    return full_dataset, np.array(labels)


def train_kfold():

    full_dataset, labels = create_full_dataset()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):

        print(f"\n========== Fold {fold+1}/5 ==========")

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        
        # Compute class weights for the training set
        train_labels = labels[train_idx]

        class_counts = np.bincount(train_labels)

        class_weights = 1.0 / class_counts

        class_weights = class_weights / class_weights.sum()

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32
        ).to(Config.DEVICE)

        train_loader = DataLoader(
            train_subset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS
        )

        model = create_model()

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=3,
            factor=0.1
        )

        best_val_loss = float('inf')

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        for epoch in range(Config.NUM_EPOCHS):

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )

            val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
                model,
                val_loader,
                criterion
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            scheduler.step(val_loss)

            print(f"Fold {fold+1} Epoch {epoch+1}/{Config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(
                f"Val Loss: {val_loss:.4f} "
                f"Acc: {val_acc:.4f} "
                f"Prec: {val_prec:.4f} "
                f"Rec: {val_rec:.4f} "
                f"F1: {val_f1:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    model.state_dict(),
                    Config.CHECKPOINT_DIR / f"best_model_fold_{fold+1}.pth"
                )
        save_learning_curve(
            train_losses,
            val_losses,
            "Loss",
            f"fold_{fold+1}_loss.png"
        )

        save_learning_curve(
            train_accs,
            val_accs,
            "Accuracy",
            f"fold_{fold+1}_accuracy.png"
        )

        fold_results.append(val_acc)
        
    print("\n========== Final Cross Validation ==========")
    print(f"Mean Accuracy: {np.mean(fold_results):.4f}")
    print(f"Std Accuracy : {np.std(fold_results):.4f}")

if __name__ == "__main__":
    print("PyTorch EfficientNet 5-Fold Cross Validation")
    print(f"Device: {Config.DEVICE}")

    train_kfold()
    # Build model
    model = create_model()
    print("Model created. Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
