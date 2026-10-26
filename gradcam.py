"""
gradcam.py

Grad-CAM heatmap overlays for the ResNet and EfficientNet classifiers
(Milestone 4, Task 2).

Target layers:
    ResNet.ResNetWithAttnPool        -> layer4 (last conv block, before attn_pool)
    EfficientNet.create_efficientnet -> features[-1] (last conv block)

predict_with_heatmaps() is a wrapper around inference.predict(). It returns the
same result dict with a "heatmap" entry (JPEG data URI) added under
result["resnet"] and result["efficientnet"]. Model and inference code are
unchanged.

Notes:
    - The heatmap is computed for the class the model voted for
      (Malignant = 1, Benign = 0), so it matches the verdict shown.
    - The EfficientNet entry in selected_models.json is a 5-checkpoint
      ensemble, so its heatmap is the mean of the five Grad-CAMs.
    - Backbone weights are frozen (requires_grad=False), so the input tensor
      is set to require grad; otherwise no gradient reaches the hooked layer.
    - Heatmaps use the plain forward pass, without flip TTA.

Standalone check (untrained heads, no checkpoints required):
    python gradcam.py path/to/image.jpg
"""

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

MAX_OVERLAY_SIDE = 512   # longest side of the overlay image, in pixels
OVERLAY_ALPHA = 0.45     # maximum heatmap opacity


# ------------------------------------------------------------------
#  Core Grad-CAM
# ------------------------------------------------------------------
def _target_layer(model):
    """Return the last convolutional block of a ResNet or EfficientNet model."""
    if hasattr(model, "layer4"):        # ResNetWithAttnPool
        return model.layer4
    return model.features[-1]           # torchvision EfficientNet-B0


def compute_cam(model, image_tensor, class_idx):
    """Compute the Grad-CAM map for a single model.

    Args:
        model: ResNetWithAttnPool or EfficientNet-B0 model.
        image_tensor: normalized (3, H, W) tensor from get_val_transforms().
        class_idx: class whose score is backpropagated (0 = Benign, 1 = Malignant).

    Returns:
        (H, W) float32 numpy array scaled to [0, 1].
    """
    model.eval()
    device = next(model.parameters()).device
    store = {}
    handle = _target_layer(model).register_forward_hook(
        lambda module, inputs, output: store.__setitem__("act", output)
    )
    try:
        # enable_grad keeps this working when called inside torch.no_grad()
        with torch.enable_grad():
            x = image_tensor.unsqueeze(0).to(device).clone().requires_grad_(True)
            logits = model(x)
            act = store["act"]
            grad = torch.autograd.grad(logits[0, class_idx], act)[0]
    finally:
        handle.remove()

    weights = grad.mean(dim=(2, 3), keepdim=True)             # per-channel importance
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))    # (1, 1, h, w)
    cam = F.interpolate(cam, size=tuple(image_tensor.shape[-2:]),
                        mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy().astype(np.float32)

    cam -= cam.min()
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def compute_cam_ensemble(model_or_models, image_tensor, class_idx):
    """Compute the mean Grad-CAM over one model or a list of fold models.

    Returns an (H, W) float32 array scaled to [0, 1].
    """
    models = model_or_models if isinstance(model_or_models, list) else [model_or_models]
    cam = np.mean([compute_cam(m, image_tensor, class_idx) for m in models], axis=0)
    cam -= cam.min()
    peak = cam.max()
    return cam / peak if peak > 0 else cam


# ------------------------------------------------------------------
#  Overlay rendering
# ------------------------------------------------------------------
def make_overlay(cam, original_image):
    """Blend a [0, 1] Grad-CAM map onto the original mammogram.

    The models take the image resized to a square, so the square map is
    resized back to the original aspect ratio to keep regions aligned.
    Heatmap opacity scales with attention, so low-attention areas show the
    original image unchanged.

    Returns a PIL RGB image (longest side capped at MAX_OVERLAY_SIDE).
    """
    img = original_image.convert("RGB")
    img.thumbnail((MAX_OVERLAY_SIDE, MAX_OVERLAY_SIDE))

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        img.size, Image.Resampling.BILINEAR
    )
    cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0

    heat = colormaps["jet"](cam_arr)[..., :3]
    base = np.asarray(img, dtype=np.float32) / 255.0
    alpha = OVERLAY_ALPHA * cam_arr[..., None]
    blended = base * (1 - alpha) + heat * alpha
    return Image.fromarray((blended * 255).astype(np.uint8))


def to_data_uri(pil_image):
    """Encode a PIL image as a base64 JPEG data URI for use in an <img> tag."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def heatmap_overlay(model_or_models, image_tensor, original_image, class_idx):
    """Return the Grad-CAM overlay for a model (or fold ensemble) as a data URI."""
    cam = compute_cam_ensemble(model_or_models, image_tensor, class_idx)
    return to_data_uri(make_overlay(cam, original_image))


# ------------------------------------------------------------------
#  Wrapper around inference.predict()
# ------------------------------------------------------------------
def predict_with_heatmaps(image, models):
    """Run inference.predict() and attach a Grad-CAM heatmap for each model.

    Returns the same dict as predict(), with result[name]["heatmap"] added for
    "resnet" and "efficientnet". If heatmap generation fails, the entry is
    None and the verdicts are still returned.

    Each model gets its OWN input resolution (models[name][3], the image_hw
    from selected_models.json) instead of one shared tensor: as of the
    2026-09-28 checkpoints the ResNet ensemble trains on 512x288 while
    EfficientNet stays at 224x224 (image_hw absent -> get_val_transforms
    default). Computing the ResNet heatmap on the wrong resolution would
    silently produce a mismatched map, the same failure mode the checkpoint
    README warns about for predictions.
    """
    from DataSetAugmentation import get_val_transforms
    from inference import predict

    image = image.convert("RGB")
    result = predict(image, models)

    for name in ("resnet", "efficientnet"):
        loaded_models, _weight, _threshold, image_hw = models[name]
        tensor = get_val_transforms(image_hw)(image)
        class_idx = 1 if result[name]["label"] == "Malignant" else 0
        try:
            result[name]["heatmap"] = heatmap_overlay(
                loaded_models, tensor, image, class_idx
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[gradcam] {name} heatmap failed: {exc!r}")
            result[name]["heatmap"] = None
    return result


# ------------------------------------------------------------------
#  Standalone check: python gradcam.py path/to/image.jpg
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from DataSetAugmentation import get_val_transforms
    from EfficientNet import create_efficientnet
    from ResNet import ResNetWithAttnPool

    if len(sys.argv) != 2:
        sys.exit("usage: python gradcam.py path/to/image.jpg")

    img = Image.open(sys.argv[1]).convert("RGB")
    tensor = get_val_transforms()(img)

    # Untrained heads: output is not meaningful, but it exercises the hooks
    # and gradient flow on both architectures.
    for name, model in (("resnet", ResNetWithAttnPool()),
                        ("efficientnet", create_efficientnet())):
        model.eval()
        cam = compute_cam(model, tensor, class_idx=1)
        out = f"gradcam_selftest_{name}.png"
        make_overlay(cam, img).save(out)
        print(f"{name}: cam shape {cam.shape}, min {cam.min():.2f}, max {cam.max():.2f} -> {out}")