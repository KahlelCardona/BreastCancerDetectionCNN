import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from DataSetAugmentation import MammogramRawDataset, TransformDataset, get_val_transforms


def build_test_dataset():
    raw = MammogramRawDataset(["mass_test", "calc_test"])
    ds = TransformDataset(raw, get_val_transforms())
    return DataLoader(
        ds, batch_size=16, shuffle=False, num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(model_name, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if model_name == "resnet":
        from ResNet import ResNetWithAttnPool
        model = ResNetWithAttnPool().to(device)
    elif model_name == "efficientnet":
        from EfficientNet import create_efficientnet
        model = create_efficientnet().to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}, expected 'resnet' or 'efficientnet'")

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Failed to load checkpoint {checkpoint_path} into {model_name} model: {e}")
        raise

    model.eval()
    return model


def _batch_malignant_probs(model, imgs, use_tta):
    if use_tta:
        logits = (
            model(imgs) + model(TF.hflip(imgs)) + model(TF.vflip(imgs))
        ) / 3.0
    else:
        logits = model(imgs)
    return F.softmax(logits, dim=1)[:, 1]


def _collect_probs_and_labels(models, loader, device, use_tta):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            batch_probs = torch.stack(
                [_batch_malignant_probs(m, imgs, use_tta) for m in models]
            ).mean(dim=0)
            all_probs.extend(batch_probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def _metrics_from_probs(probs, labels, threshold):
    preds = probs > threshold
    return {
        "n_samples": len(labels),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def evaluate_model(model, loader, device, use_tta, threshold=0.5):
    model.eval()
    probs, labels = _collect_probs_and_labels([model], loader, device, use_tta)
    return _metrics_from_probs(probs, labels, threshold)


def evaluate_ensemble(model_name, checkpoint_paths, loader, device, use_tta, threshold=0.5):
    models = [load_model(model_name, p, device) for p in checkpoint_paths]
    probs, labels = _collect_probs_and_labels(models, loader, device, use_tta)
    return _metrics_from_probs(probs, labels, threshold)


def find_best_threshold(model, loader, device, use_tta):
    model.eval()
    probs, labels = _collect_probs_and_labels([model], loader, device, use_tta)

    best_threshold, best_f1 = 0.5, f1_score(labels, probs > 0.5, zero_division=0)
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(labels, probs > t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)

    return best_threshold, best_f1


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a ResNet or EfficientNet checkpoint (or ensemble of checkpoints) "
                    "on the held-out mammogram test set."
    )
    parser.add_argument("--model", required=True, choices=["resnet", "efficientnet"])
    parser.add_argument("--checkpoint", required=True, nargs="+")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = [Path(c) for c in args.checkpoint]

    loader = build_test_dataset()
    metrics = evaluate_ensemble(args.model, checkpoint_paths, loader, device, args.tta, args.threshold)

    print(
        f"[{args.model}] checkpoints={[str(c) for c in checkpoint_paths]} tta={args.tta} "
        f"threshold={args.threshold} n={metrics['n_samples']} acc={metrics['accuracy']:.4f} "
        f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    )

    if args.out:
        out_path = Path(args.out)
    else:
        stem = (
            checkpoint_paths[0].stem if len(checkpoint_paths) == 1
            else f"ensemble{len(checkpoint_paths)}fold"
        )
        out_path = Path("eval_results") / f"{args.model}_{stem}_test_metrics.json"

    out_path.parent.mkdir(exist_ok=True)
    payload = {
        "model": args.model,
        "checkpoints": [str(c) for c in checkpoint_paths],
        "tta": args.tta,
        "threshold": args.threshold,
        **metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
