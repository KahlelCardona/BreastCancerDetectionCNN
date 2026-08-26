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
        model = load_model(name, entry["checkpoint"], device)
        models[name] = (model, float(entry["accuracy"]))
    return models


def _predict_single(model, image_tensor, device):
    imgs = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = (
            model(imgs) + model(TF.hflip(imgs)) + model(TF.vflip(imgs))
        ) / 3.0
        probs = F.softmax(logits, dim=1)[0]

    malignant_prob = float(probs[1])
    label = "Malignant" if malignant_prob > 0.5 else "Benign"
    confidence = malignant_prob if label == "Malignant" else 1.0 - malignant_prob
    return label, malignant_prob, confidence


def predict(image, models):
    device = next(models["resnet"][0].parameters()).device
    image_tensor = get_val_transforms()(image)

    resnet_model, resnet_weight = models["resnet"]
    efficientnet_model, efficientnet_weight = models["efficientnet"]

    resnet_label, resnet_prob, resnet_conf = _predict_single(resnet_model, image_tensor, device)
    eff_label, eff_prob, eff_conf = _predict_single(efficientnet_model, image_tensor, device)

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
