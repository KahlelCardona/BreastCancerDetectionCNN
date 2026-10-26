from dataclasses import dataclass, field

import numpy as np
from PIL import Image

MIN_SIDE_PX = 64
MIN_ASPECT, MAX_ASPECT = 0.35, 2.85   # height/width
MAX_MEAN_CHROMA = 12.0                # grayscale ~0, colour photos 20-60+
MAX_P95_CHROMA = 40.0
MIN_INTENSITY_STD = 8.0               # rejects blank images


@dataclass
class ValidationResult:
    ok: bool
    reasons: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate_mammogram_like(image: Image.Image) -> ValidationResult:
    w, h = image.size
    reasons = []

    if min(w, h) < MIN_SIDE_PX:
        reasons.append(f"The image is too small ({w}x{h} px).")

    aspect = h / w if w else 0.0
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        reasons.append(f"The image shape ({w}x{h}) is not supported.")

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    mean_chroma, p95_chroma = float(chroma.mean()), float(np.percentile(chroma, 95))
    if mean_chroma > MAX_MEAN_CHROMA or p95_chroma > MAX_P95_CHROMA:
        reasons.append("The image is in colour; grayscale images are required.")

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if float(gray.std()) < MIN_INTENSITY_STD:
        reasons.append("The image is blank or nearly uniform.")

    stats = dict(width=w, height=h, aspect=round(aspect, 3),
                 mean_chroma=round(mean_chroma, 2), p95_chroma=round(p95_chroma, 2))
    return ValidationResult(ok=not reasons, reasons=reasons, stats=stats)