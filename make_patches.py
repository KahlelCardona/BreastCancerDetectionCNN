"""Cut lesion and normal-tissue patches from the training mammograms for patch pretraining.

Each mammogram is first resized to the whole-image training size (--image-hw H W), exactly
as the training transform does, so a lesion in a patch appears at the same scale as in
whole-image training. For every lesion row in the two training CSVs, a 288x288 patch is
cut around the centre of its ROI mask (label 1 benign, 2 malignant); up to 3 random
288x288 normal-tissue patches per mammogram that touch no mask are added (label 0).

Writes data/patches_{H}x{W}/{normal,benign,malignant}/*.jpg and data/patches_{H}x{W}/index.csv
(columns path,label,patient_id). Refuses to overwrite an existing patch set.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from DataSetAugmentation import _count_distinct_grayscale_values

CSV_DIR = Path("data/raw/csv")
JPEG_DIR = Path("data/raw/jpeg")
TRAIN_CSVS = ["mass_case_description_train_set.csv", "calc_case_description_train_set.csv"]
PATCH = 288
NORMALS_PER_IMAGE = 3
MAX_ATTEMPTS = 200
CLASS_DIRS = {0: "normal", 1: "benign", 2: "malignant"}


def folder_jpgs(path_value):
    """JPEGs in the data/raw/jpeg/<uid> folder named by a CSV path column."""
    if not isinstance(path_value, str):
        return []
    parts = path_value.strip().split("/")
    if len(parts) < 3:
        return []
    return sorted((JPEG_DIR / parts[2]).glob("*.jpg"))


def find_mask(row, size):
    """The row's ROI mask: the near-binary image (<20 grey values) of the full image's size."""
    candidates = folder_jpgs(row["ROI mask file path"]) + folder_jpgs(row["cropped image file path"])
    for p in dict.fromkeys(candidates):
        if _count_distinct_grayscale_values(p) < 20 and Image.open(p).size == size:
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image-hw", type=int, nargs=2, required=True, metavar=("H", "W"))
    args = parser.parse_args()
    H, W = args.image_hw
    out_dir = Path(f"data/patches_{H}x{W}")
    index_path = out_dir / "index.csv"
    if index_path.exists():
        print(f"{index_path} already exists; delete {out_dir} to rebuild. Nothing written.")
        return
    for d in CLASS_DIRS.values():
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    rows = pd.concat([pd.read_csv(CSV_DIR / f) for f in TRAIN_CSVS], ignore_index=True)
    rows = rows.dropna(subset=["pathology"])

    # Group lesion rows by their full mammogram so each image is decoded once.
    by_image, skipped = defaultdict(list), 0
    for _, row in rows.iterrows():
        full = folder_jpgs(row["image file path"])
        if len(full) != 1:
            skipped += 1
            continue
        by_image[full[0]].append(row)

    rng = np.random.default_rng(42)
    index, counts = [], {0: 0, 1: 0, 2: 0}
    for n_img, (img_path, img_rows) in enumerate(by_image.items(), start=1):
        image = Image.open(img_path).convert("L")
        size = image.size
        small = np.array(image.resize((W, H), Image.BILINEAR))
        union = np.zeros((H, W), dtype=bool)
        lesions = []
        for row in img_rows:
            mask_path = find_mask(row, size)
            if mask_path is None:
                skipped += 1
                continue
            mask = np.array(Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)) > 127
            if not mask.any():
                skipped += 1
                continue
            union |= mask
            ys, xs = np.nonzero(mask)
            centre = ((xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2)
            label = 2 if "MALIGNANT" in str(row["pathology"]).upper() else 1
            lesions.append((centre, label, row["patient_id"]))
        if not lesions:
            continue

        patient_id, uid = lesions[0][2], img_path.parent.name
        small_img = Image.fromarray(small)
        n = 0

        def save(box, label):
            nonlocal n
            name = f"{patient_id}_{uid}_{n}.jpg"
            rel = out_dir / CLASS_DIRS[label] / name
            small_img.crop(box).save(rel, quality=95)   # PIL pads with black outside the image
            index.append((rel.as_posix(), label, patient_id))
            counts[label] += 1
            n += 1

        for (cx, cy), label, _ in lesions:
            save((cx - PATCH // 2, cy - PATCH // 2, cx + PATCH // 2, cy + PATCH // 2), label)

        found = attempts = 0
        while found < NORMALS_PER_IMAGE and attempts < MAX_ATTEMPTS and H >= PATCH and W >= PATCH:
            attempts += 1
            x, y = int(rng.integers(0, W - PATCH + 1)), int(rng.integers(0, H - PATCH + 1))
            if union[y:y + PATCH, x:x + PATCH].any():
                continue
            if (small[y:y + PATCH, x:x + PATCH] > 30).mean() < 0.5:
                continue
            save((x, y, x + PATCH, y + PATCH), 0)
            found += 1

        if n_img % 250 == 0:
            print(f"  {n_img}/{len(by_image)} mammograms processed", flush=True)

    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "patient_id"])
        writer.writerows(index)

    print(f"Read {len(rows)} lesion rows from {len(TRAIN_CSVS)} CSVs ({len(by_image)} mammograms); "
          f"skipped {skipped} (missing file, no matching mask, or empty mask)")
    print(f"Wrote patches: normal {counts[0]}, benign {counts[1]}, malignant {counts[2]}  ({index_path})")


if __name__ == "__main__":
    main()
