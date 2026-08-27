import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from DataSetAugmentation import get_val_transforms
from evaluate import load_model

SELECTED_MODELS_PATH = Path("eval_results/selected_models.json")


def load_selected_models(device):
    with open(SELECTED_MODELS_PATH) as f:
        selected = json.load(f)

    models = {}
    for name in ("resnet", "efficientnet"):
        entry = selected[name]
        checkpoint = entry["checkpoint"]
        if isinstance(checkpoint, list):
            loaded = [load_model(name, c, device) for c in checkpoint]
        else:
            loaded = load_model(name, checkpoint, device)
        threshold = float(entry.get("threshold", 0.5))
        models[name] = (loaded, float(entry["accuracy"]), threshold)
    return models


def _predict_single(model_or_models, image_tensor, device, threshold):
    models = model_or_models if isinstance(model_or_models, list) else [model_or_models]
    imgs = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        probs_per_model = []
        for model in models:
            logits = (
                model(imgs) + model(TF.hflip(imgs)) + model(TF.vflip(imgs))
            ) / 3.0
            probs_per_model.append(F.softmax(logits, dim=1)[0])
        probs = torch.stack(probs_per_model).mean(dim=0)

    malignant_prob = float(probs[1])
    label = "Malignant" if malignant_prob > threshold else "Benign"
    confidence = malignant_prob if label == "Malignant" else 1.0 - malignant_prob
    return label, malignant_prob, confidence


def predict(image, models):
    resnet_first = models["resnet"][0]
    resnet_first_model = resnet_first[0] if isinstance(resnet_first, list) else resnet_first
    device = next(resnet_first_model.parameters()).device
    image_tensor = get_val_transforms()(image)

    resnet_models, resnet_weight, resnet_threshold = models["resnet"]
    efficientnet_models, efficientnet_weight, efficientnet_threshold = models["efficientnet"]

    resnet_label, resnet_prob, resnet_conf = _predict_single(
        resnet_models, image_tensor, device, resnet_threshold
    )
    eff_label, eff_prob, eff_conf = _predict_single(
        efficientnet_models, image_tensor, device, efficientnet_threshold
    )

    blended_prob = (
        resnet_prob * resnet_weight + eff_prob * efficientnet_weight
    ) / (resnet_weight + efficientnet_weight)
    final_label = "Malignant" if blended_prob > 0.5 else "Benign"
    final_conf = blended_prob if final_label == "Malignant" else 1.0 - blended_prob

    return {
        "resnet": {
            "label": resnet_label,
            "malignant_probability": resnet_prob,
            "confidence": resnet_conf,
            "reported_accuracy": resnet_weight,
        },
        "efficientnet": {
            "label": eff_label,
            "malignant_probability": eff_prob,
            "confidence": eff_conf,
            "reported_accuracy": efficientnet_weight,
        },
        "final": {
            "label": final_label,
            "malignant_probability": blended_prob,
            "confidence": final_conf,
            "models_agreed": resnet_label == eff_label,
        },
    }
