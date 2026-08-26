import argparse
import json
from pathlib import Path

import torch
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


def evaluate_model(model, loader, device, use_tta):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            if use_tta:
                logits = (
                    model(imgs) + model(TF.hflip(imgs)) + model(TF.vflip(imgs))
                ) / 3.0
            else:
                logits = model(imgs)

            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "n_samples": len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a ResNet or EfficientNet checkpoint on the held-out mammogram test set."
    )
    parser.add_argument("--model", required=True, choices=["resnet", "efficientnet"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)

    model = load_model(args.model, checkpoint_path, device)
    loader = build_test_dataset()
    metrics = evaluate_model(model, loader, device, args.tta)

    print(
        f"[{args.model}] checkpoint={checkpoint_path} tta={args.tta} "
        f"n={metrics['n_samples']} acc={metrics['accuracy']:.4f} "
        f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    )

    out_path = (
        Path(args.out) if args.out
        else Path("eval_results") / f"{args.model}_{checkpoint_path.stem}_test_metrics.json"
    )
    out_path.parent.mkdir(exist_ok=True)
    payload = {"model": args.model, "checkpoint": str(checkpoint_path), "tta": args.tta, **metrics}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
