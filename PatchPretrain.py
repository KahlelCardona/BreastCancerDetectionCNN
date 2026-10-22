"""Pretrain a model's backbone on lesion/normal-tissue patches, once per cross-validation fold.

Reads data/patches_{H}x{W}/index.csv (written by make_patches.py). For each of the 5
patient-grouped folds used by ResNet.py and EfficientNet.py, trains on the patches of that
fold's training patients only (so whole-image validation stays unseen) and measures on its
validation patients' patches. Three classes: 0 normal tissue, 1 benign lesion, 2 malignant
lesion. Saves only the backbone weights, under the parameter names the whole-image model
uses, to checkpoints_patch_{model}_{H}x{W}/fold{k}_backbone.pth. Folds already saved are skipped.
"""
import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from DataSetAugmentation import MammogramRawDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 12
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
CROP = 224
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
BACKBONE_PREFIX = {"resnet": "layer", "efficientnet": "features."}


class PatchDataset(Dataset):
    def __init__(self, frame, transform):
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


def build_model(name):
    """The whole-image model with a 3-class head and every parameter trainable."""
    if name == "resnet":
        from ResNet import ResNetWithAttnPool
        model = ResNetWithAttnPool()
        model.fc = nn.Linear(2048, 3)
    else:
        from EfficientNet import create_efficientnet
        model = create_efficientnet()
        model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(1280, 3))
    for p in model.parameters():
        p.requires_grad = True
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass; trains if an optimizer is given. Returns (mean loss, labels, predictions, logits)."""
    model.train(optimizer is not None)
    total, all_y, all_logits = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits = logits.float()
            loss = criterion(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            all_y.append(y.cpu())
            all_logits.append(logits.detach().cpu())
    y = torch.cat(all_y).numpy()
    logits = torch.cat(all_logits)
    return total / len(y), y, logits.argmax(1).numpy(), logits.numpy()


def lesion_f1(y, logits):
    """Benign-vs-malignant F1 on lesion patches only, deciding between the two lesion scores."""
    lesion = y > 0
    pred_malignant = logits[lesion, 2] > logits[lesion, 1]
    return f1_score(y[lesion] == 2, pred_malignant, zero_division=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=["resnet", "efficientnet"])
    parser.add_argument("--image-hw", type=int, nargs=2, required=True, metavar=("H", "W"))
    args = parser.parse_args()
    H, W = args.image_hw
    patches = pd.read_csv(f"data/patches_{H}x{W}/index.csv")
    out_dir = Path(f"checkpoints_patch_{args.model}_{H}x{W}")
    out_dir.mkdir(exist_ok=True)

    # Rebuild the exact whole-image folds (same dataset, same splitter and seed).
    with contextlib.redirect_stdout(io.StringIO()):
        raw = MammogramRawDataset(["mass_train", "calc_train"])
    labels = [l for _, l in raw.samples]
    patient_ids = np.array(raw.patient_ids)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    train_tf = transforms.Compose([
        transforms.RandomCrop(CROP), transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10), transforms.ToTensor(), NORMALIZE,
    ])
    val_tf = transforms.Compose([transforms.CenterCrop(CROP), transforms.ToTensor(), NORMALIZE])

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels, groups=raw.patient_ids)):
        out_path = out_dir / f"fold{fold + 1}_backbone.pth"
        if out_path.exists():
            print(f"Fold {fold + 1}: {out_path} exists, skipping")
            continue
        torch.manual_seed(42 + fold)
        train_df = patches[patches.patient_id.isin(set(patient_ids[train_idx]))]
        val_df = patches[patches.patient_id.isin(set(patient_ids[val_idx]))]
        val_counts = val_df.label.value_counts().reindex([0, 1, 2], fill_value=0)
        print(f"\n===== Fold {fold + 1}/5 ({args.model}, {H}x{W}) =====")
        print(f"  train patches normal/benign/malignant: "
              f"{'/'.join(str(c) for c in train_df.label.value_counts().reindex([0, 1, 2], fill_value=0))}")
        print(f"  val patches   normal/benign/malignant: {'/'.join(str(c) for c in val_counts)}; "
              f"largest-class share {val_counts.max() / len(val_df):.3f}", flush=True)

        train_loader = DataLoader(PatchDataset(train_df, train_tf), batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(PatchDataset(val_df, val_tf), batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

        model = build_model(args.model)
        weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_df.label.values)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=DEVICE))
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0)

        best = None
        for epoch in range(1, EPOCHS + 1):
            train_loss, *_ = run_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()
            val_loss, y, pred, logits = run_epoch(model, val_loader, criterion)
            acc = (pred == y).mean()
            lf1 = lesion_f1(y, logits)
            print(f"  Epoch {epoch:2d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} "
                  f"3-class Acc {acc:.4f} Lesion F1 {lf1:.4f}", flush=True)
            if best is None or val_loss < best[1]:
                prefix = BACKBONE_PREFIX[args.model]
                backbone = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(prefix)}
                best = (epoch, val_loss, acc, lf1, backbone)

        epoch, val_loss, acc, lf1, backbone = best
        torch.save(backbone, out_path)
        print(f"Fold {fold + 1}: saved {out_path} (best epoch {epoch}, val loss {val_loss:.2f}, "
              f"3-class acc {acc:.2f}, lesion F1 {lf1:.2f})", flush=True)


if __name__ == "__main__":
    main()
