import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
import torchvision.transforms.functional as TF
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pathlib import Path

from DataSetAugmentation import (
    MammogramRawDataset, TransformDataset,
    get_train_transforms, get_val_transforms,
)
from evaluate import find_best_threshold
import training_log

# ------------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------------
class CFG:
    MODEL_NAME      = "resnet50"
    BATCH_SIZE      = 16
    IMG_SIZE        = 224
    EPOCHS          = 55
    NUM_WORKERS     = 4
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR        = Path("data/raw")
    CHECKPOINT_DIR  = Path("checkpoints_resnet")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Learning rates
    HEAD_LR   = 2e-4
    BASE_LR   = 4e-5          # slightly lower for freshly unfrozen layers
    FROZEN_LR = 8e-6          # slightly lower for previously unfrozen
    WD        = 1e-4

    # Unfreeze schedule – moved earlier based on training logs showing
    # the model stabilises faster than expected
    UNFREEZE_SCHEDULE = {
        5:  ["layer4"],        # was epoch 7
        10: ["layer3"],        # was epoch 13
    }

    # Regularisation
    LABEL_SMOOTHING       = 0.08           # slightly reduced; focal loss handles hard negatives
    MIXUP_ALPHA           = 0.2            # increased from 0.1 for more regularisation
    FOCAL_GAMMA           = 1.5            # focal loss concentration parameter
    FOCAL_WEIGHT          = 0.4            # blend: 0.6 * CE + 0.4 * Focal
    GRAD_CLIP_NORM        = 1.5            # tighter clipping
    EARLY_STOP_PATIENCE   = 12
    MIN_EPOCH_FOR_BEST    = 8              # don't checkpoint before epoch 8 (head warmup noise)

    # Cosine annealing with warm restarts
    T_0                   = 15             # first restart period (epochs)
    T_MULT                = 1             # keep same period after each restart
    ETA_MIN               = 1e-7

    # Test-time augmentation
    TTA_ENABLED           = True
    TTA_FLIPS             = True           # horizontal + vertical flip ensemble

    # SAM optimizer
    USE_SAM               = True
    SAM_RHO               = 0.07          # SAM neighbourhood size


# ------------------------------------------------------------------
#  Focal Loss
# ------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss for binary (2-class) classification."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


# ------------------------------------------------------------------
#  SAM Optimizer wrapper
# ------------------------------------------------------------------
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimisation (Foret et al., 2021).
    Wraps any base optimizer. Requires two forward+backward passes per step.
    """
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)                   # climb to local max
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # descend from local max
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ------------------------------------------------------------------
#  MixUp helpers
# ------------------------------------------------------------------
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a = torch.zeros(batch_size, 2, device=x.device).scatter_(1, y.unsqueeze(1), 1)
    y_b = torch.zeros(batch_size, 2, device=x.device).scatter_(1, y[index].unsqueeze(1), 1)
    soft_labels = lam * y_a + (1 - lam) * y_b
    return mixed_x, soft_labels


# ------------------------------------------------------------------
#  Model – AttentionPool replaces flat global average pool
# ------------------------------------------------------------------
class AttentionPool(nn.Module):
    """
    Channel-wise attention over spatial feature map before the FC head.
    Replaces the plain adaptive average pool in ResNet.
    Adds minimal parameters but focuses the head on discriminative regions.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        weights = self.attn(x).view(x.size(0), x.size(1), 1, 1)
        pooled  = (x * weights).mean(dim=(2, 3))
        return pooled


class ResNetWithAttnPool(nn.Module):
    """ResNet-50 with attention pooling and a stronger FC head."""
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Keep backbone layers
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        in_feat = base.fc.in_features          # 2048
        self.attn_pool = AttentionPool(in_feat)

        self.fc = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.55),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 2),
        )

        # Freeze everything except fc and attn_pool
        for p in self.parameters():
            p.requires_grad = False
        for p in self.fc.parameters():
            p.requires_grad = True
        for p in self.attn_pool.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attn_pool(x)
        return self.fc(x)


def build_model():
    return ResNetWithAttnPool().to(CFG.DEVICE)


# ------------------------------------------------------------------
#  Optimizer builder with differentiated LR
# ------------------------------------------------------------------
def get_optimizer(model, current_epoch):
    fired  = [e for e in CFG.UNFREEZE_SCHEDULE if e <= current_epoch]
    latest = max(fired) if fired else 0
    wd     = 2e-4 if latest > 0 else 1e-4

    param_groups = [
        {"params": list(model.fc.parameters()) + list(model.attn_pool.parameters()),
         "lr": CFG.HEAD_LR, "weight_decay": wd}
    ]

    if latest > 0:
        unlock_epoch = {}
        for ep, layers in CFG.UNFREEZE_SCHEDULE.items():
            for ln in layers:
                unlock_epoch[ln] = ep

        base_params, frozen_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'fc' in name or 'attn_pool' in name:
                continue
            matched = None
            for ln, ep in unlock_epoch.items():
                if name.startswith(ln):
                    matched = ep
                    break
            if matched is None:
                continue
            if matched == latest:
                base_params.append(p)
            else:
                frozen_params.append(p)

        if base_params:
            param_groups.append({"params": base_params,
                                  "lr": CFG.BASE_LR, "weight_decay": wd})
        if frozen_params:
            param_groups.append({"params": frozen_params,
                                  "lr": CFG.FROZEN_LR, "weight_decay": wd})

    if CFG.USE_SAM:
        return SAM(param_groups, optim.AdamW, rho=CFG.SAM_RHO,
                   lr=CFG.HEAD_LR, weight_decay=wd)
    return optim.AdamW(param_groups)


def get_scheduler(optimizer, epoch_offset=0):
    """Cosine annealing with warm restarts."""
    base_opt = optimizer.base_optimizer if CFG.USE_SAM else optimizer
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(
        base_opt,
        T_0=CFG.T_0,
        T_mult=CFG.T_MULT,
        eta_min=CFG.ETA_MIN,
        last_epoch=epoch_offset - 1,
    )


# ------------------------------------------------------------------
#  Hybrid criterion: (1 - w) * CE  +  w * Focal
# ------------------------------------------------------------------
def build_criterion(class_weights):
    ce    = nn.CrossEntropyLoss(weight=class_weights,
                                label_smoothing=CFG.LABEL_SMOOTHING)
    focal = FocalLoss(gamma=CFG.FOCAL_GAMMA, weight=class_weights,
                      label_smoothing=CFG.LABEL_SMOOTHING)

    def hybrid(logits, labels):
        return (1 - CFG.FOCAL_WEIGHT) * ce(logits, labels) \
             + CFG.FOCAL_WEIGHT       * focal(logits, labels)

    return hybrid


# ------------------------------------------------------------------
#  Training epoch  (SAM-aware)
# ------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, mixup_alpha):
    model.train()
    running_loss = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(CFG.DEVICE), labels.to(CFG.DEVICE)

        if mixup_alpha > 0:
            imgs, soft_labels = mixup_data(imgs, labels, mixup_alpha)

        def forward_pass():
            logits = model(imgs)
            if mixup_alpha > 0:
                lp = nn.functional.log_softmax(logits, dim=1)
                return -(soft_labels * lp).sum(dim=1).mean()
            return criterion(logits, labels)

        if CFG.USE_SAM:
            # First forward-backward: find local max
            loss = forward_pass()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP_NORM)
            optimizer.first_step(zero_grad=True)

            # Second forward-backward: actual update from local max
            loss2 = forward_pass()
            loss2.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP_NORM)
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            loss = forward_pass()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP_NORM)
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


# ------------------------------------------------------------------
#  Validation  (with optional TTA)
# ------------------------------------------------------------------
def validate(model, loader, use_tta=False):
    model.eval()
    plain_criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(CFG.DEVICE), labels.to(CFG.DEVICE)

            if use_tta and CFG.TTA_ENABLED and CFG.TTA_FLIPS:
                # Ensemble: original + hflip + vflip
                logits_orig  = model(imgs)
                logits_hflip = model(TF.hflip(imgs))
                logits_vflip = model(TF.vflip(imgs))
                logits = (logits_orig + logits_hflip + logits_vflip) / 3.0
            else:
                logits = model(imgs)

            loss = plain_criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n           = len(loader.dataset)
    epoch_loss  = running_loss / n
    epoch_acc   = accuracy_score(all_labels, all_preds)
    epoch_prec  = precision_score(all_labels, all_preds, zero_division=0)
    epoch_rec   = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1    = f1_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1


# ------------------------------------------------------------------
#  One fold
# ------------------------------------------------------------------
WARMUP_EPOCHS_AFTER_UNFREEZE = 2   # ramp freshly-unfrozen LR over this many epochs


def train_fold(fold, raw_dataset, train_idx, val_idx, all_labels, log_wb, log_path):
    train_ds = TransformDataset(Subset(raw_dataset, train_idx), get_train_transforms())
    val_ds   = TransformDataset(Subset(raw_dataset, val_idx),   get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.DEVICE.type == "cuda",
    )

    model     = build_model()
    y_train   = np.array(all_labels)[train_idx]
    class_w   = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    w_tensor  = torch.tensor(class_w, dtype=torch.float).to(CFG.DEVICE)
    criterion = build_criterion(w_tensor)

    optimizer = get_optimizer(model, 0)
    scheduler = get_scheduler(optimizer, epoch_offset=0)

    best_val_loss            = float("inf")
    best_f1                  = 0.0
    best_epoch               = 0
    patience                 = 0
    unfreeze_warmup_remaining = 0   # countdown for post-unfreeze LR warmup

    for epoch in range(1, CFG.EPOCHS + 1):

        # ----------------------------------------------------------------
        # Unfreeze schedule
        # ----------------------------------------------------------------
        if epoch in CFG.UNFREEZE_SCHEDULE:
            layers = CFG.UNFREEZE_SCHEDULE[epoch]
            print(f"  [Fold {fold+1}] Epoch {epoch}: Unfreezing {layers}")
            for ln in layers:
                for p in getattr(model, ln).parameters():
                    p.requires_grad = True
            optimizer = get_optimizer(model, epoch)
            scheduler = get_scheduler(optimizer, epoch_offset=0)
            patience                  = 0
            unfreeze_warmup_remaining = WARMUP_EPOCHS_AFTER_UNFREEZE

        # ----------------------------------------------------------------
        # Post-unfreeze LR warmup for freshly unfrozen param group (index 1)
        # Ramps from BASE_LR * (1/W) up to BASE_LR over W epochs, which
        # prevents the large initial gradient from the new parameters from
        # destabilising the already-trained head and previously-unfrozen layers.
        # ----------------------------------------------------------------
        if unfreeze_warmup_remaining > 0:
            warmup_step  = WARMUP_EPOCHS_AFTER_UNFREEZE - unfreeze_warmup_remaining + 1
            warmup_scale = warmup_step / WARMUP_EPOCHS_AFTER_UNFREEZE   # e.g. 0.5, 1.0
            base_opt = optimizer.base_optimizer if CFG.USE_SAM else optimizer
            # param_group index 1 is always the freshly-unfrozen layer group
            if len(base_opt.param_groups) > 1:
                base_opt.param_groups[1]["lr"] = CFG.BASE_LR * warmup_scale
                print(f"  [Warmup] Freshly-unfrozen LR = "
                      f"{base_opt.param_groups[1]['lr']:.2e} "
                      f"(step {warmup_step}/{WARMUP_EPOCHS_AFTER_UNFREEZE})")
            unfreeze_warmup_remaining -= 1

        # ----------------------------------------------------------------
        # Train / validate
        # ----------------------------------------------------------------
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 CFG.MIXUP_ALPHA)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, use_tta=False
        )

        # Step cosine scheduler (operates on base optimizer internally)
        scheduler.step()

        print(f"  Epoch {epoch:2d} | Train Loss {train_loss:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} "
              f"Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

        training_log.log_epoch(log_wb, log_path, fold + 1, epoch,
                                train_loss, val_loss, val_acc, val_prec, val_rec, val_f1)

        # ----------------------------------------------------------------
        # Checkpoint on val loss, deferred until after head warmup noise
        # ----------------------------------------------------------------
        if val_loss < best_val_loss and epoch >= CFG.MIN_EPOCH_FOR_BEST:
            best_val_loss = val_loss
            best_f1       = val_f1
            best_epoch    = epoch
            patience      = 0
            torch.save(model.state_dict(),
                       CFG.CHECKPOINT_DIR / f"fold{fold+1}_best.pth")
            print("  --> saved best model")
        else:
            patience += 1

        if patience >= CFG.EARLY_STOP_PATIENCE:
            print(f"  Early stop at epoch {epoch} "
                  f"(best F1 {best_f1:.4f} at epoch {best_epoch})")
            break

    # ---- Re-evaluate best checkpoint with TTA ----
    model.load_state_dict(
        torch.load(CFG.CHECKPOINT_DIR / f"fold{fold+1}_best.pth",
                   map_location=CFG.DEVICE)
    )
    _, tta_acc, tta_prec, tta_rec, tta_f1 = validate(model, val_loader, use_tta=True)
    best_threshold, best_threshold_f1 = find_best_threshold(model, val_loader, CFG.DEVICE, use_tta=True)
    print(f"  Fold {fold+1} best-val F1  (no TTA): {best_f1:.4f}")
    print(f"  Fold {fold+1} best-val F1 (with TTA): {tta_f1:.4f}")
    print(f"  Fold {fold+1} best threshold: {best_threshold:.2f} "
          f"(val F1 {best_threshold_f1:.4f} vs default-0.5 F1 {tta_f1:.4f})\n")
    training_log.log_fold_summary(log_wb, log_path, fold + 1, best_f1, tta_f1, best_threshold)
    return best_f1, tta_f1, best_threshold


# ------------------------------------------------------------------
#  5‑fold cross‑validation
# ------------------------------------------------------------------
def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    raw_dataset = MammogramRawDataset(["mass_train", "calc_train"], include_cropped_patches=True)
    labels      = [lbl for _, lbl in raw_dataset.samples]
    skf         = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    log_wb   = training_log.create_workbook("resnet")
    log_path = training_log.LOG_DIR / "resnet_training_log.xlsx"

    fold_f1, fold_tta_f1, fold_threshold = [], [], []
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels, groups=raw_dataset.groups)
    ):
        print(f"\n{'='*50}")
        print(f"  Fold {fold+1} / 5")
        print(f"{'='*50}")
        f1, tta_f1, threshold = train_fold(fold, raw_dataset, train_idx, val_idx, labels, log_wb, log_path)
        fold_f1.append(f1)
        fold_tta_f1.append(tta_f1)
        fold_threshold.append(threshold)

    fold_f1     = np.array(fold_f1)
    fold_tta_f1 = np.array(fold_tta_f1)

    print(f"\n{'='*50}")
    print("  Final cross-validation results (ResNet-50 v2)")
    print(f"{'='*50}")
    print(f"  Val F1 no-TTA  (mean ± std): {fold_f1.mean():.4f} ± {fold_f1.std():.4f}")
    print(f"  Val F1 TTA     (mean ± std): {fold_tta_f1.mean():.4f} ± {fold_tta_f1.std():.4f}")
    print(f"  Per-fold thresholds: {[f'{t:.2f}' for t in fold_threshold]}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()