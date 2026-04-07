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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

from DataSetAugmentation import MammogramRawDataset, TransformDataset, Config, get_train_transforms, get_val_transforms

class EfficientNetConfig:
    MODEL_NAME = "efficientnet"    # efficientnet_b0
    BATCH_SIZE = 16
    NUM_EPOCHS = 25
    BASE_LR = 1e-4                 # for unfrozen backbone blocks
    CLASSIFIER_LR = 1e-3           # higher LR for the classifier head
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = Path("checkpoints_efficientnet")
    PLOT_DIR = Path("plots_efficientnet")
    IMAGE_SIZE = 224
    NUM_WORKERS = 8
    DATA_DIR = Path("data/raw")
    JPEG_DIR = DATA_DIR / "jpeg"
    CSV_DIR = DATA_DIR / "csv"
 
    UNFREEZE_SCHEDULE = {4: [7], 10: [6, 7]}

# -------------------- Model creation --------------------
def create_efficientnet():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 2)
    )
    # Unfreeze the new classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model.to(EfficientNetConfig.DEVICE)


# -------------------- Training and validation --------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(EfficientNetConfig.DEVICE), labels.to(EfficientNetConfig.DEVICE)
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

def validate(model, loader):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(EfficientNetConfig.DEVICE), labels.to(EfficientNetConfig.DEVICE)
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


# -------------------- 5-Fold Cross-Validation for EfficientNet --------------------
def train_efficientnet_kfold():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled.")
    EfficientNetConfig.CHECKPOINT_DIR.mkdir(exist_ok=True)
    EfficientNetConfig.PLOT_DIR.mkdir(exist_ok=True)
    print(f"Using device: {EfficientNetConfig.DEVICE}")

    raw_dataset = MammogramRawDataset(["mass_train", "calc_train"])
    labels = [label for _, label in raw_dataset.samples]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_val_losses, all_val_accs, all_val_f1s = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n========== Fold {fold+1}/5 ==========")

        train_raw = Subset(raw_dataset, train_idx)
        val_raw   = Subset(raw_dataset, val_idx)
        train_dataset = TransformDataset(train_raw, transform=get_train_transforms())
        val_dataset   = TransformDataset(val_raw,   transform=get_val_transforms())

        train_loader = DataLoader(train_dataset, batch_size=EfficientNetConfig.BATCH_SIZE,
                                  shuffle=True, num_workers=EfficientNetConfig.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=EfficientNetConfig.BATCH_SIZE,
                                shuffle=False, num_workers=EfficientNetConfig.NUM_WORKERS)

        # Model with initial freezing (only classifier trainable)
        model = create_efficientnet()
        # Ensure backbone is frozen (already done in create_efficientnet, but double-check)
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

        # Class-weighted loss
        train_labels = np.array(labels)[train_idx]
        class_weights = compute_class_weight('balanced', classes=np.array([0,1]), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(EfficientNetConfig.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer with differential learning rates
        def get_optimizer():
            params = []
            # Classifier with high LR
            params.append({'params': model.classifier.parameters(), 'lr': EfficientNetConfig.CLASSIFIER_LR,
                           'weight_decay': EfficientNetConfig.WEIGHT_DECAY})
            # Any other trainable parameters (unfrozen blocks) with base LR
            for name, param in model.named_parameters():
                if param.requires_grad and 'classifier' not in name:
                    params.append({'params': param, 'lr': EfficientNetConfig.BASE_LR,
                                   'weight_decay': EfficientNetConfig.WEIGHT_DECAY})
            return optim.AdamW(params, lr=EfficientNetConfig.BASE_LR)

        optimizer = get_optimizer()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

        best_val_loss = float('inf')
        fold_val_losses, fold_val_accs, fold_val_f1s = [], [], []

        for epoch in range(EfficientNetConfig.NUM_EPOCHS):
            # Gradual unfreezing of EfficientNet blocks
            if epoch in EfficientNetConfig.UNFREEZE_SCHEDULE:
                blocks = EfficientNetConfig.UNFREEZE_SCHEDULE[epoch]
                print(f"  Unfreezing EfficientNet blocks: {blocks}")
                for blk in blocks:
                    for param in model.features[blk].parameters():
                        param.requires_grad = True
                optimizer = get_optimizer()
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader)
            scheduler.step(val_loss)

            fold_val_losses.append(val_loss)
            fold_val_accs.append(val_acc)
            fold_val_f1s.append(val_f1)

            print(f"  Epoch {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), EfficientNetConfig.CHECKPOINT_DIR / f"best_model_fold_{fold+1}.pth")

        all_val_losses.append(fold_val_losses)
        all_val_accs.append(fold_val_accs)
        all_val_f1s.append(fold_val_f1s)

        # Plot per-fold metrics
        epochs_range = range(1, EfficientNetConfig.NUM_EPOCHS+1)
        plt.figure()
        plt.plot(epochs_range, fold_val_losses, label='Val Loss')
        plt.plot(epochs_range, fold_val_accs, label='Val Acc')
        plt.plot(epochs_range, fold_val_f1s, label='Val F1')
        plt.xlabel('Epoch')
        plt.title(f'Fold {fold+1} EfficientNet Metrics')
        plt.legend()
        plt.savefig(EfficientNetConfig.PLOT_DIR / f"fold_{fold+1}_metrics.png")
        plt.close()

    # Aggregate across folds
    val_losses_array = np.array(all_val_losses)
    val_accs_array   = np.array(all_val_accs)
    val_f1s_array    = np.array(all_val_f1s)

    mean_val_loss = np.mean(val_losses_array, axis=0)
    mean_val_acc  = np.mean(val_accs_array, axis=0)
    mean_val_f1   = np.mean(val_f1s_array, axis=0)
    std_val_loss  = np.std(val_losses_array, axis=0)

    epochs = range(1, EfficientNetConfig.NUM_EPOCHS+1)
    plt.figure()
    plt.plot(epochs, mean_val_loss, label='Mean Val Loss')
    plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Mean Validation Loss Across Folds (EfficientNet)')
    plt.legend()
    plt.savefig(EfficientNetConfig.PLOT_DIR / "mean_val_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, mean_val_acc, label='Mean Val Accuracy')
    plt.fill_between(epochs, mean_val_acc - np.std(val_accs_array, axis=0),
                     mean_val_acc + np.std(val_accs_array, axis=0), alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Mean Validation Accuracy Across Folds (EfficientNet)')
    plt.legend()
    plt.savefig(EfficientNetConfig.PLOT_DIR / "mean_val_acc.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, mean_val_f1, label='Mean Val F1')
    plt.fill_between(epochs, mean_val_f1 - np.std(val_f1s_array, axis=0),
                     mean_val_f1 + np.std(val_f1s_array, axis=0), alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Mean Validation F1 Across Folds (EfficientNet)')
    plt.legend()
    plt.savefig(EfficientNetConfig.PLOT_DIR / "mean_val_f1.png")
    plt.close()

    print("\n========== Final Cross-Validation Results (EfficientNet) ==========")
    print(f"Mean Val Loss (last epoch): {mean_val_loss[-1]:.4f} ± {std_val_loss[-1]:.4f}")
    print(f"Mean Val Accuracy: {mean_val_acc[-1]:.4f} ± {np.std(val_accs_array[:,-1]):.4f}")
    print(f"Mean Val F1 Score: {mean_val_f1[-1]:.4f} ± {np.std(val_f1s_array[:,-1]):.4f}")


if __name__ == "__main__":
    train_efficientnet_kfold()