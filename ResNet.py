import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from DataSetAugmentation import MammogramRawDataset, TransformDataset, Config, get_train_transforms, get_val_transforms


# -------------------- Configuration --------------------
class ResNetConfig:
    MODEL_NAME = "resnet"           # resnet50
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    BASE_LR = 1e-4                  # base learning rate (for unfrozen backbone blocks)
    CLASSIFIER_LR = 1e-3            # higher LR for the final FC layer
    WEIGHT_DECAY = 1e-3             # FIX: increased from 1e-5 to reduce overfitting
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = Path("checkpoints_resnet")
    PLOT_DIR = Path("plots_resnet")
    IMAGE_SIZE = 224
    NUM_WORKERS = 4
    DATA_DIR = Path("data/raw")
    JPEG_DIR = DATA_DIR / "jpeg"
    CSV_DIR = DATA_DIR / "csv"

    UNFREEZE_SCHEDULE = {
        5:  ["layer4"],
        12: ["layer3"],
        18: ["layer2"],
    }

    # early stopping patience
    EARLY_STOPPING_PATIENCE = 5

    # tighter LR scheduler (was patience=4, factor=0.5)
    LR_SCHEDULER_PATIENCE = 2
    LR_SCHEDULER_FACTOR = 0.3


# -------------------- Model creation --------------------
def create_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # FIX: replace FC with a richer head — intermediate layer + BN + higher dropout
    # (was just Dropout(0.3) + Linear)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),          # FIX: increased from 0.3 to 0.5
        nn.Linear(512, 2)
    )

    # Unfreeze the new FC head
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(ResNetConfig.DEVICE)


# -------------------- Optimizer (fixed) --------------------
def get_optimizer(model):
    """
    FIX: backbone params are now collected as a single param group (list),
    not added one-by-one, avoiding duplicate optimizer state entries and
    accidental double-counting of FC parameters.
    """
    backbone_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and 'fc' not in name
    ]
    param_groups = [
        {
            'params': model.fc.parameters(),
            'lr': ResNetConfig.CLASSIFIER_LR,
            'weight_decay': ResNetConfig.WEIGHT_DECAY,
        },
    ]
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': ResNetConfig.BASE_LR,
            'weight_decay': ResNetConfig.WEIGHT_DECAY,
        })
    return optim.AdamW(param_groups)


# -------------------- Training and validation --------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(ResNetConfig.DEVICE), labels.to(ResNetConfig.DEVICE)
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
            images, labels = images.to(ResNetConfig.DEVICE), labels.to(ResNetConfig.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, zero_division=0)
    epoch_rec  = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1   = f1_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1


# -------------------- 5-Fold Cross-Validation --------------------
def train_resnet_kfold():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled.")
    ResNetConfig.CHECKPOINT_DIR.mkdir(exist_ok=True)
    ResNetConfig.PLOT_DIR.mkdir(exist_ok=True)
    print(f"Using device: {ResNetConfig.DEVICE}")

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

        train_loader = DataLoader(
            train_dataset, batch_size=ResNetConfig.BATCH_SIZE,
            shuffle=True, num_workers=ResNetConfig.NUM_WORKERS
        )
        val_loader = DataLoader(
            val_dataset, batch_size=ResNetConfig.BATCH_SIZE,
            shuffle=False, num_workers=ResNetConfig.NUM_WORKERS
        )

        # Model — backbone fully frozen initially
        model = create_resnet()

        # Class-weighted loss to handle imbalance
        train_labels = np.array(labels)[train_idx]
        class_weights = compute_class_weight(
            'balanced', classes=np.array([0, 1]), y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(ResNetConfig.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = get_optimizer(model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            patience=ResNetConfig.LR_SCHEDULER_PATIENCE,   # FIX: 2 (was 4)
            factor=ResNetConfig.LR_SCHEDULER_FACTOR        # FIX: 0.3 (was 0.5)
        )

        best_val_loss = float('inf')
        patience_counter = 0                               # FIX: early stopping counter
        fold_val_losses, fold_val_accs, fold_val_f1s = [], [], []
        best_epoch = 0

        for epoch in range(ResNetConfig.NUM_EPOCHS):

            # Gradual unfreezing
            if epoch + 1 in ResNetConfig.UNFREEZE_SCHEDULE:
                layers = ResNetConfig.UNFREEZE_SCHEDULE[epoch + 1]
                print(f"  Unfreezing ResNet layers: {layers}")
                for layer_name in layers:
                    layer = getattr(model, layer_name)
                    for param in layer.parameters():
                        param.requires_grad = True
                # FIX: rebuild optimizer as a proper grouped list after unfreezing
                optimizer = get_optimizer(model)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min',
                    patience=ResNetConfig.LR_SCHEDULER_PATIENCE,
                    factor=ResNetConfig.LR_SCHEDULER_FACTOR
                )

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader)
            scheduler.step(val_loss)

            fold_val_losses.append(val_loss)
            fold_val_accs.append(val_acc)
            fold_val_f1s.append(val_f1)

            print(
                f"  Epoch {epoch+1:>2}: "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} "
                f"Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}"
            )

            # FIX: save best checkpoint and track early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    ResNetConfig.CHECKPOINT_DIR / f"best_model_fold_{fold+1}.pth"
                )
            else:
                patience_counter += 1
                if patience_counter >= ResNetConfig.EARLY_STOPPING_PATIENCE:
                    print(
                        f"  Early stopping at epoch {epoch+1} "
                        f"(best was epoch {best_epoch}, val loss {best_val_loss:.4f})"
                    )
                    break

        all_val_losses.append(fold_val_losses)
        all_val_accs.append(fold_val_accs)
        all_val_f1s.append(fold_val_f1s)

        # Per-fold plot (handle variable length due to early stopping)
        ep = range(1, len(fold_val_losses) + 1)
        plt.figure()
        plt.plot(ep, fold_val_losses, label='Val Loss')
        plt.plot(ep, fold_val_accs,   label='Val Acc')
        plt.plot(ep, fold_val_f1s,    label='Val F1')
        plt.axvline(best_epoch, color='gray', linestyle='--', label=f'Best (ep {best_epoch})')
        plt.xlabel('Epoch')
        plt.title(f'Fold {fold+1} ResNet Metrics')
        plt.legend()
        plt.savefig(ResNetConfig.PLOT_DIR / f"fold_{fold+1}_metrics.png")
        plt.close()

    # ---- Aggregate across folds (pad shorter runs with NaN for mean/std) ----
    max_len = max(len(x) for x in all_val_losses)

    def pad(arr):
        return np.array([
            np.pad(x, (0, max_len - len(x)), constant_values=np.nan)
            for x in arr
        ])

    val_losses_array = pad(all_val_losses)
    val_accs_array   = pad(all_val_accs)
    val_f1s_array    = pad(all_val_f1s)

    mean_val_loss = np.nanmean(val_losses_array, axis=0)
    mean_val_acc  = np.nanmean(val_accs_array,   axis=0)
    mean_val_f1   = np.nanmean(val_f1s_array,    axis=0)
    std_val_loss  = np.nanstd(val_losses_array,  axis=0)
    std_val_acc   = np.nanstd(val_accs_array,    axis=0)
    std_val_f1    = np.nanstd(val_f1s_array,     axis=0)

    epochs = range(1, max_len + 1)

    for metric, mean, std, ylabel, filename in [
        ("Mean Validation Loss",     mean_val_loss, std_val_loss, "Loss",     "mean_val_loss.png"),
        ("Mean Validation Accuracy", mean_val_acc,  std_val_acc,  "Accuracy", "mean_val_acc.png"),
        ("Mean Validation F1",       mean_val_f1,   std_val_f1,   "F1 Score", "mean_val_f1.png"),
    ]:
        plt.figure()
        plt.plot(epochs, mean, label=metric)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.3)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{metric} Across Folds (ResNet)')
        plt.legend()
        plt.savefig(ResNetConfig.PLOT_DIR / filename)
        plt.close()

    # Final summary — report the best (minimum loss) epoch across mean curve
    best_mean_epoch = int(np.nanargmin(mean_val_loss))
    print("\n========== Final Cross-Validation Results (ResNet) ==========")
    print(f"Best mean val loss at epoch {best_mean_epoch+1}: "
          f"{mean_val_loss[best_mean_epoch]:.4f} ± {std_val_loss[best_mean_epoch]:.4f}")
    print(f"Val Accuracy at best epoch:  "
          f"{mean_val_acc[best_mean_epoch]:.4f} ± {std_val_acc[best_mean_epoch]:.4f}")
    print(f"Val F1 Score at best epoch:  "
          f"{mean_val_f1[best_mean_epoch]:.4f} ± {std_val_f1[best_mean_epoch]:.4f}")


if __name__ == "__main__":
    train_resnet_kfold()