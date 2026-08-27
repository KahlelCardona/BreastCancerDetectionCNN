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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

from DataSetAugmentation import MammogramRawDataset, TransformDataset, Config, get_train_transforms, get_val_transforms
from losses import mixup_data, build_hybrid_criterion
from evaluate import evaluate_model, find_best_threshold
import training_log

class EfficientNetConfig:
    MODEL_NAME = "efficientnet"    # efficientnet_b0
    BATCH_SIZE = 16
    NUM_EPOCHS = 35
    BASE_LR = 1e-4                 # for unfrozen backbone blocks
    CLASSIFIER_LR = 1e-3           # higher LR for the classifier head
    NEW_UNFREEZE_LR = 1e-4         # blocks just unfrozen at the latest schedule epoch
    OLD_UNFREEZE_LR = 2e-5         # blocks unfrozen at an earlier schedule epoch
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = Path("checkpoints_efficientnet")
    PLOT_DIR = Path("plots_efficientnet")
    IMAGE_SIZE = 224
    NUM_WORKERS = 8
    DATA_DIR = Path("data/raw")
    JPEG_DIR = DATA_DIR / "jpeg"
    CSV_DIR = DATA_DIR / "csv"

    # Regularisation (mirrors ResNet.py's CFG, same dataset/problem)
    GRAD_CLIP_NORM      = 1.5
    LABEL_SMOOTHING     = 0.08
    FOCAL_GAMMA         = 1.5
    FOCAL_WEIGHT        = 0.4
    MIXUP_ALPHA         = 0.2
    EARLY_STOP_PATIENCE = 10
    MIN_EPOCH_FOR_BEST  = 6        # 0-indexed epoch loop: skip checkpointing before this epoch

    # Cosine annealing with warm restarts, rebuilt on each unfreeze event
    T_0      = 8
    T_MULT   = 1
    ETA_MIN  = 1e-7

    TTA_ENABLED = True

    UNFREEZE_SCHEDULE = {4: [8, 7], 10: [6], 16: [5]}

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
def train_one_epoch(model, loader, criterion, optimizer, mixup_alpha):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(EfficientNetConfig.DEVICE), labels.to(EfficientNetConfig.DEVICE)
        optimizer.zero_grad()

        if mixup_alpha > 0:
            mixed_images, soft_labels = mixup_data(images, labels, mixup_alpha)
            outputs = model(mixed_images)
            log_probs = nn.functional.log_softmax(outputs, dim=1)
            loss = -(soft_labels * log_probs).sum(dim=1).mean()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), EfficientNetConfig.GRAD_CLIP_NORM)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

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


# -------------------- Optimizer / scheduler with differentiated LR by unfreeze recency --------------------
def get_optimizer(model, current_epoch):
    fired  = [e for e in EfficientNetConfig.UNFREEZE_SCHEDULE if e <= current_epoch]
    latest = max(fired) if fired else 0

    param_groups = [
        {'params': model.classifier.parameters(), 'lr': EfficientNetConfig.CLASSIFIER_LR,
         'weight_decay': EfficientNetConfig.WEIGHT_DECAY}
    ]

    if latest > 0:
        unlock_epoch = {}
        for ep, blocks in EfficientNetConfig.UNFREEZE_SCHEDULE.items():
            for blk in blocks:
                unlock_epoch[f"features.{blk}."] = ep

        new_params, old_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad or 'classifier' in name:
                continue
            matched = None
            for prefix, ep in unlock_epoch.items():
                if name.startswith(prefix):
                    matched = ep
                    break
            if matched is None:
                continue
            if matched == latest:
                new_params.append(param)
            else:
                old_params.append(param)

        if new_params:
            param_groups.append({'params': new_params, 'lr': EfficientNetConfig.NEW_UNFREEZE_LR,
                                  'weight_decay': EfficientNetConfig.WEIGHT_DECAY})
        if old_params:
            param_groups.append({'params': old_params, 'lr': EfficientNetConfig.OLD_UNFREEZE_LR,
                                  'weight_decay': EfficientNetConfig.WEIGHT_DECAY})

    return optim.AdamW(param_groups)


def get_scheduler(optimizer, epoch_offset=0):
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=EfficientNetConfig.T_0,
        T_mult=EfficientNetConfig.T_MULT,
        eta_min=EfficientNetConfig.ETA_MIN,
        last_epoch=epoch_offset - 1,
    )


# -------------------- 5-Fold Cross-Validation for EfficientNet --------------------
def train_efficientnet_kfold():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled.")
    EfficientNetConfig.CHECKPOINT_DIR.mkdir(exist_ok=True)
    EfficientNetConfig.PLOT_DIR.mkdir(exist_ok=True)
    print(f"Using device: {EfficientNetConfig.DEVICE}")

    raw_dataset = MammogramRawDataset(["mass_train", "calc_train"], include_cropped_patches=True)
    labels = [label for _, label in raw_dataset.samples]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    log_wb = training_log.create_workbook("efficientnet")
    log_path = training_log.LOG_DIR / "efficientnet_training_log.xlsx"

    all_val_losses, all_val_accs, all_val_f1s = [], [], []
    fold_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels, groups=raw_dataset.groups)):
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

        # Class-weighted hybrid CE + Focal loss (mirrors ResNet.py)
        train_labels = np.array(labels)[train_idx]
        class_weights = compute_class_weight('balanced', classes=np.array([0,1]), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(EfficientNetConfig.DEVICE)
        criterion = build_hybrid_criterion(class_weights, EfficientNetConfig.LABEL_SMOOTHING,
                                            EfficientNetConfig.FOCAL_GAMMA, EfficientNetConfig.FOCAL_WEIGHT)

        optimizer = get_optimizer(model, 0)
        scheduler = get_scheduler(optimizer, epoch_offset=0)

        best_val_loss = float('inf')
        best_val_f1   = 0.0
        patience      = 0
        fold_val_losses, fold_val_accs, fold_val_f1s = [], [], []

        for epoch in range(EfficientNetConfig.NUM_EPOCHS):
            # Gradual unfreezing of EfficientNet blocks
            if epoch in EfficientNetConfig.UNFREEZE_SCHEDULE:
                blocks = EfficientNetConfig.UNFREEZE_SCHEDULE[epoch]
                print(f"  Unfreezing EfficientNet blocks: {blocks}")
                for blk in blocks:
                    for param in model.features[blk].parameters():
                        param.requires_grad = True
                optimizer = get_optimizer(model, epoch)
                scheduler = get_scheduler(optimizer, epoch_offset=0)
                patience  = 0

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                         EfficientNetConfig.MIXUP_ALPHA)
            val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader)
            scheduler.step()

            fold_val_losses.append(val_loss)
            fold_val_accs.append(val_acc)
            fold_val_f1s.append(val_f1)

            print(f"  Epoch {epoch+1}: Train Loss {train_loss:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

            training_log.log_epoch(log_wb, log_path, fold + 1, epoch + 1,
                                    train_loss, val_loss, val_acc, val_prec, val_rec, val_f1)

            # 0-indexed epoch loop: MIN_EPOCH_FOR_BEST=6 means "epoch 6" in the 1-indexed
            # print above, i.e. epoch >= 5 here.
            if val_loss < best_val_loss and epoch >= EfficientNetConfig.MIN_EPOCH_FOR_BEST - 1:
                best_val_loss = val_loss
                best_val_f1   = val_f1
                patience      = 0
                torch.save(model.state_dict(), EfficientNetConfig.CHECKPOINT_DIR / f"best_model_fold_{fold+1}.pth")
                print("  --> saved best model")
            else:
                patience += 1

            if patience >= EfficientNetConfig.EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch+1} (best F1 {best_val_f1:.4f})")
                break

        # Early stopping can end a fold before NUM_EPOCHS; pad with the last value so every
        # fold's per-epoch history is the same length for the cross-fold mean/std plots below.
        while len(fold_val_losses) < EfficientNetConfig.NUM_EPOCHS:
            fold_val_losses.append(fold_val_losses[-1])
            fold_val_accs.append(fold_val_accs[-1])
            fold_val_f1s.append(fold_val_f1s[-1])

        all_val_losses.append(fold_val_losses)
        all_val_accs.append(fold_val_accs)
        all_val_f1s.append(fold_val_f1s)

        # ---- Re-evaluate best checkpoint with TTA (reuses evaluate.py's evaluate_model) ----
        model.load_state_dict(torch.load(
            EfficientNetConfig.CHECKPOINT_DIR / f"best_model_fold_{fold+1}.pth",
            map_location=EfficientNetConfig.DEVICE,
        ))
        tta_metrics = evaluate_model(model, val_loader, EfficientNetConfig.DEVICE, use_tta=True)
        best_threshold, best_threshold_f1 = find_best_threshold(
            model, val_loader, EfficientNetConfig.DEVICE, use_tta=True
        )
        fold_thresholds.append(best_threshold)
        print(f"  Fold {fold+1} best-val F1  (no TTA): {best_val_f1:.4f}")
        print(f"  Fold {fold+1} best-val F1 (with TTA): {tta_metrics['f1']:.4f}")
        print(f"  Fold {fold+1} best threshold: {best_threshold:.2f} "
              f"(val F1 {best_threshold_f1:.4f} vs default-0.5 F1 {tta_metrics['f1']:.4f})\n")
        training_log.log_fold_summary(log_wb, log_path, fold + 1, best_val_f1, tta_metrics['f1'], best_threshold)

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
    print(f"Per-fold thresholds: {[f'{t:.2f}' for t in fold_thresholds]}")


if __name__ == "__main__":
    train_efficientnet_kfold()