import torch
from torch.utils.data import Dataset, DataLoader, Subset
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
from torch.utils.data import WeightedRandomSampler

from DataSetAugmentation import MammogramRawDataset, TransformDataset, Config, get_train_transforms, get_val_transforms

class Config:
    DATA_DIR = Path("data/raw")
    JPEG_DIR = DATA_DIR / "jpeg"
    CSV_DIR = DATA_DIR / "csv"
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    IMAGE_SIZE = 224
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5           # added to reduce overfitting
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    PLOT_DIR = Path("plots")
    PLOT_DIR.mkdir(exist_ok=True)


def create_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),   
        nn.Linear(num_features, 2)
    )

    # Unfreeze classifier only
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


def train_kfold():
    # Create raw dataset 
    raw_dataset = MammogramRawDataset(["mass_train", "calc_train"])
    labels = [label for _, label in raw_dataset.samples]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Store metrics for each fold
    all_fold_val_losses = []   # list of lists, one per fold
    all_fold_val_accs   = []
    all_fold_val_f1s     = []

    fold_results = []  # final validation accuracy per fold (last epoch)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n========== Fold {fold+1}/5 ==========")

        train_raw = Subset(raw_dataset, train_idx)
        val_raw   = Subset(raw_dataset, val_idx)

        train_dataset = TransformDataset(train_raw, transform=get_train_transforms())
        val_dataset   = TransformDataset(val_raw,   transform=get_val_transforms())

        # Weighted sampler for training
        train_labels = np.array(labels)[train_idx]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                                  sampler=sampler, num_workers=Config.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                                shuffle=False, num_workers=Config.NUM_WORKERS)

        model = create_model()
        criterion = nn.CrossEntropyLoss()

        # Unfreezing schedule: epoch
        unfreeze_schedule = {
            3: [7],        # last MBConv block
            8: [6, 7]      # second‑last and last
        }

        def get_optimizer():
            params = []
            params.append({'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': Config.WEIGHT_DECAY})
            for name, param in model.named_parameters():
                if param.requires_grad and 'classifier' not in name:
                    params.append({'params': param, 'lr': 1e-4, 'weight_decay': Config.WEIGHT_DECAY})
            return optim.AdamW(params, lr=1e-5)

        optimizer = get_optimizer()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

        best_val_loss = float('inf')
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        val_f1s = []   # store F1 per epoch for this fold

        for epoch in range(Config.NUM_EPOCHS):
            # Unfreeze blocks according to schedule
            if epoch in unfreeze_schedule:
                blocks = unfreeze_schedule[epoch]
                print(f"Unfreezing blocks: {blocks}")
                for blk in blocks:
                    for param in model.features[blk].parameters():
                        param.requires_grad = True
                optimizer = get_optimizer()
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            val_f1s.append(val_f1)

            scheduler.step(val_loss)

            print(f"Fold {fold+1} Epoch {epoch+1}/{Config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Prec: {val_prec:.4f} Rec: {val_rec:.4f} F1: {val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), Config.CHECKPOINT_DIR / f"best_model_fold_{fold+1}.pth")

        # Store this fold's metrics for later averaging
        all_fold_val_losses.append(val_losses)
        all_fold_val_accs.append(val_accs)
        all_fold_val_f1s.append(val_f1s)

        # Save per‑fold learning curves
        save_learning_curve(train_losses, val_losses, "Loss", f"fold_{fold+1}_loss.png")
        save_learning_curve(train_accs, val_accs, "Accuracy", f"fold_{fold+1}_accuracy.png")
        save_learning_curve(val_f1s, val_f1s, "F1 Score", f"fold_{fold+1}_f1.png")  # train vs val not needed, just val

        fold_results.append(val_accs[-1])   # final epoch validation accuracy

    # ------------------- Aggregate across folds -------------------
    # Convert to numpy arrays for easy averaging (epochs x folds)
    # All folds have same number of epochs (NUM_EPOCHS)
    val_losses_array = np.array(all_fold_val_losses)  # shape: (folds, epochs)
    val_accs_array   = np.array(all_fold_val_accs)
    val_f1s_array    = np.array(all_fold_val_f1s)

    mean_val_loss = np.mean(val_losses_array, axis=0)
    mean_val_acc  = np.mean(val_accs_array, axis=0)
    mean_val_f1   = np.mean(val_f1s_array, axis=0)

    # Plot mean curves
    epochs_range = range(1, Config.NUM_EPOCHS+1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, mean_val_loss, label="Mean Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mean Validation Loss Across 5 Folds")
    plt.legend()
    plt.savefig(Config.PLOT_DIR / "mean_val_loss.png")
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, mean_val_acc, label="Mean Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Mean Validation Accuracy Across 5 Folds")
    plt.legend()
    plt.savefig(Config.PLOT_DIR / "mean_val_accuracy.png")
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, mean_val_f1, label="Mean Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Mean Validation F1 Score Across 5 Folds")
    plt.legend()
    plt.savefig(Config.PLOT_DIR / "mean_val_f1.png")
    plt.close()

    # Final cross‑validation summary
    print("\n========== Final Cross Validation ==========")
    print(f"Mean Accuracy (last epoch): {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"Mean Validation Loss (last epoch): {np.mean([losses[-1] for losses in all_fold_val_losses]):.4f}")
    print(f"Mean F1 Score (last epoch): {np.mean([f1s[-1] for f1s in all_fold_val_f1s]):.4f}")
    print("\nAveraged learning curves saved in 'plots/' directory.")


if __name__ == "__main__":
    print("PyTorch EfficientNet 5-Fold Cross Validation")
    print(f"Device: {Config.DEVICE}")
    train_kfold()