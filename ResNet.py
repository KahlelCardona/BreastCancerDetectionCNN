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
    MODEL_NAME = "resnet"          # resnet50
    BATCH_SIZE = 16
    NUM_EPOCHS = 25
    BASE_LR = 1e-4                 # base learning rate (for unfrozen blocks)
    CLASSIFIER_LR = 1e-3           # higher LR for the final FC layer
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = Path("checkpoints_resnet")
    PLOT_DIR = Path("plots_resnet")
    IMAGE_SIZE = 224
    NUM_WORKERS = 4
    DATA_DIR = Path("data/raw")
    JPEG_DIR = DATA_DIR / "jpeg"
    CSV_DIR = DATA_DIR / "csv"
    # Unfreezing schedule: epoch -> list of layer names to unfreeze
    UNFREEZE_SCHEDULE = {4: ["layer4"], 10: ["layer3", "layer4"]}


# -------------------- Model creation --------------------
def create_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    # Replace FC layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 2)
    )
    # Unfreeze the new FC layer
    for param in model.fc.parameters():
        param.requires_grad = True
    return model.to(ResNetConfig.DEVICE)


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
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, zero_division=0)
    epoch_rec = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1


# -------------------- 5-Fold Cross-Validation for ResNet --------------------
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

        train_loader = DataLoader(train_dataset, batch_size=ResNetConfig.BATCH_SIZE,
                                  shuffle=True, num_workers=ResNetConfig.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=ResNetConfig.BATCH_SIZE,
                                shuffle=False, num_workers=ResNetConfig.NUM_WORKERS)

        # Model with initial freezing
        model = create_resnet()
        # Ensure backbone is frozen (already done in create_resnet, but double-check)
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

        # Class-weighted loss
        train_labels = np.array(labels)[train_idx]
        class_weights = compute_class_weight('balanced', classes=np.array([0,1]), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(ResNetConfig.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer with differential learning rates
        def get_optimizer():
            params = []
            # FC layer (classifier) with high LR
            params.append({'params': model.fc.parameters(), 'lr': ResNetConfig.CLASSIFIER_LR,
                           'weight_decay': ResNetConfig.WEIGHT_DECAY})
            # Any other trainable parameters (unfrozen layers) with base LR
            for name, param in model.named_parameters():
                if param.requires_grad and 'fc' not in name:
                    params.append({'params': param, 'lr': ResNetConfig.BASE_LR,
                                   'weight_decay': ResNetConfig.WEIGHT_DECAY})
            return optim.AdamW(params, lr=ResNetConfig.BASE_LR)  # default LR overridden

        optimizer = get_optimizer()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

        best_val_loss = float('inf')
        fold_val_losses, fold_val_accs, fold_val_f1s = [], [], []

        for epoch in range(ResNetConfig.NUM_EPOCHS):
            # Gradual unfreezing
            if epoch in ResNetConfig.UNFREEZE_SCHEDULE:
                layers = ResNetConfig.UNFREEZE_SCHEDULE[epoch]
                print(f"  Unfreezing ResNet layers: {layers}")
                for layer_name in layers:
                    layer = getattr(model, layer_name)
                    for param in layer.parameters():
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
                torch.save(model.state_dict(), ResNetConfig.CHECKPOINT_DIR / f"best_model_fold_{fold+1}.pth")

        all_val_losses.append(fold_val_losses)
        all_val_accs.append(fold_val_accs)
        all_val_f1s.append(fold_val_f1s)

        # Plot per-fold metrics
        epochs_range = range(1, ResNetConfig.NUM_EPOCHS+1)
        plt.figure()
        plt.plot(epochs_range, fold_val_losses, label='Val Loss')
        plt.plot(epochs_range, fold_val_accs, label='Val Acc')
        plt.plot(epochs_range, fold_val_f1s, label='Val F1')
        plt.xlabel('Epoch')
        plt.title(f'Fold {fold+1} ResNet Metrics')
        plt.legend()
        plt.savefig(ResNetConfig.PLOT_DIR / f"fold_{fold+1}_metrics.png")
        plt.close()

    # Aggregate across folds
    val_losses_array = np.array(all_val_losses)
    val_accs_array   = np.array(all_val_accs)
    val_f1s_array    = np.array(all_val_f1s)

    mean_val_loss = np.mean(val_losses_array, axis=0)
    mean_val_acc  = np.mean(val_accs_array, axis=0)
    mean_val_f1   = np.mean(val_f1s_array, axis=0)
    std_val_loss  = np.std(val_losses_array, axis=0)

    epochs = range(1, ResNetConfig.NUM_EPOCHS+1)
    plt.figure()
    plt.plot(epochs, mean_val_loss, label='Mean Val Loss')
    plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Mean Validation Loss Across Folds (ResNet)')
    plt.legend()
    plt.savefig(ResNetConfig.PLOT_DIR / "mean_val_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, mean_val_acc, label='Mean Val Accuracy')
    plt.fill_between(epochs, mean_val_acc - np.std(val_accs_array, axis=0),
                     mean_val_acc + np.std(val_accs_array, axis=0), alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Mean Validation Accuracy Across Folds (ResNet)')
    plt.legend()
    plt.savefig(ResNetConfig.PLOT_DIR / "mean_val_acc.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, mean_val_f1, label='Mean Val F1')
    plt.fill_between(epochs, mean_val_f1 - np.std(val_f1s_array, axis=0),
                     mean_val_f1 + np.std(val_f1s_array, axis=0), alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Mean Validation F1 Across Folds (ResNet)')
    plt.legend()
    plt.savefig(ResNetConfig.PLOT_DIR / "mean_val_f1.png")
    plt.close()

    print("\n========== Final Cross-Validation Results (ResNet) ==========")
    print(f"Mean Val Loss (last epoch): {mean_val_loss[-1]:.4f} ± {std_val_loss[-1]:.4f}")
    print(f"Mean Val Accuracy: {mean_val_acc[-1]:.4f} ± {np.std(val_accs_array[:,-1]):.4f}")
    print(f"Mean Val F1 Score: {mean_val_f1[-1]:.4f} ± {np.std(val_f1s_array[:,-1]):.4f}")


if __name__ == "__main__":
    train_resnet_kfold()