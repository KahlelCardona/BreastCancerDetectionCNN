# Split cross-validation by patient and train EfficientNet-B0 at a higher, aspect-preserving resolution

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not check in a copy of PLANS.md. The authoring and execution rules for this document live at `~/.agents/PLANS.md` on the machine of whoever is running this plan (a global, agent-agnostic file, not part of this git repository). This document must be maintained in accordance with that file.

This plan builds on `specs/efficientnet-light-augmentation-and-sam.md`, which is checked in and committed (commit `f8e2840`). That plan changed `EfficientNet.py` to train with light image augmentation (horizontal flip and a 10 degree rotation only), no MixUp, and the Sharpness-Aware Minimisation optimiser (SAM, an optimiser that steps towards flat regions of the loss surface using two forward/backward passes per batch; it lives in `optimizers.py`). It produced the current best model, whose held-out numbers are the baseline here. Everything needed from that plan is restated below.

## Approval

Status: Approved

Approved by: Repository owner (chat instruction: "approved")

Date: 2026-09-26

Requested scope: All five milestones below, including the two multi-hour training runs in Milestones 3 and 4.

## Purpose / Big Picture

This project trains EfficientNet-B0 (a small image-classification neural network) to label a mammogram benign or malignant, and judges each change by its F1 score (the harmonic mean of precision and recall) on a 704-image held-out test set that is never trained on. The latest run reached held-out F1 0.6552 (single best fold) and 0.6460 (five-fold ensemble). Two problems limit how far it can go, and this plan fixes both.

First, the cross-validation folds leak patients. Cross-validation cuts the 2,864 training images into five groups, trains five models, and lets each model choose its checkpoint and decision threshold using its own validation group. The images come from 1,248 patients, several images per patient (up to 24), and the current splitter, scikit-learn's `StratifiedKFold`, knows nothing about patients. Measured on 2026-09-26 with the exact splitter settings used by `EfficientNet.py`, 78 to 82 percent of the validation images in every fold belong to a patient who also has images in that fold's training set. The model is therefore being validated partly on people it has already seen, so validation scores are inflated (mean cross-validation F1 was 0.72 while held-out F1 was only 0.65 to 0.66), and checkpoint selection, early stopping and threshold choice are being steered by an over-optimistic signal. Splitting by patient removes the leak.

Second, the images are downsampled far too aggressively. The training images are full-field mammograms of about 3,000 by 5,100 pixels (measured medians: width 3,048, height 5,191; every one is taller than 3,200 pixels). The code squashes each one to 224 by 224, about 23 times smaller along the long side, and also distorts its shape by roughly 1.7 times. Small lesions and calcification clusters are the diagnostic signal and are largely destroyed at that size.

After this plan, a person can run `py EfficientNet.py --folds 5` (patient-grouped folds, original 224 by 224 size) and `py EfficientNet.py --folds 5 --image-hw 512 288 --batch-size 8` (patient-grouped folds, 512 tall by 288 wide, which keeps the images' natural shape) and read both runs' held-out F1 next to the current baseline. Comparing the two runs separates the two changes: run A isolates the effect of honest, leak-free validation on the final model; run B adds resolution on top. The observable result is three held-out F1 numbers (baseline, A, B) measured identically on the same 704 images.

## Progress

- [x] (2026-09-26) Milestone 1: record each sample's patient id in `MammogramRawDataset`, switch `EfficientNet.py` to patient-grouped folds, add a test proving no patient appears on both sides of any split.
- [x] (2026-09-26) Milestone 2: make image size and batch size configurable (`--image-hw`, `--batch-size`), including in `evaluate.py`, and prove a training step fits in GPU memory at 512 by 288, batch 8.
- [x] (2026-09-26) Milestone 3: run A, the grouped 5-fold run at 224 by 224; evaluate on the held-out set.
- [x] (2026-09-27) Milestone 4: run B, the grouped 5-fold run at 512 by 288; evaluate on the held-out set.
- [x] (2026-09-27) Milestone 5: write the three-way comparison and verdict.

## Surprises & Discoveries

- (2026-09-26, research) Patient leakage measured with `StratifiedKFold(5, shuffle=True, random_state=42)` over the 2,864 training rows: validation images sharing a patient with training images, fold by fold: 78.5, 78.2, 81.2, 79.6, 81.5 percent. Validation patients per fold about 450 to 470, of which 366 to 393 also appear in that fold's training data.
- (2026-09-26, research) The held-out test set also shares patients with the training set: 31 of its patients appear in the training CSVs. This slightly favours every model equally and is out of scope here (changing the test set would break comparability with every earlier result). It is recorded so nobody is surprised by it.
- (2026-09-26, research) `git log` shows an earlier commit titled "add cropped images and stratified group k fold" (`20e1ff2`), and `MammogramRawDataset` has a `groups` list, but the grouping is by abnormality row number, not by patient, and the current `EfficientNet.py` does not use it. No patient-level grouping exists in the code today.
- (2026-09-26, M1) Verified with a scratchpad script on the real dataset (2,864 samples, 1,248 patients): for 3, 5 and 10 folds no patient appears in both train and validation, every sample is validated exactly once, and each fold's malignant fraction is 0.409 to 0.415. 5-fold validation sizes: 573, 574, 573, 571, 573.
- (2026-09-26, M2) Shape checks pass: default transforms give 3x224x224, `--image-hw 512 288` gives 3x512x288 for both training and validation transforms and for `evaluate.build_test_dataset((512, 288))`. `py EfficientNet.py --help` and `py evaluate.py --help` list the new options.
- (2026-09-26, M2) Memory: three real SAM training steps at batch 8 and 512x288 (with all four unfreeze stages trainable, the worst case) peaked at only 552 MB allocated. The frozen early backbone blocks store no activations, so memory is not the constraint the Decision Log assumed. Batch 8 is therefore not needed for memory; batch 16 would fit. The plan's batch-8 choice for run B is left as written pending the repository owner's decision, because changing it is a material change.
- (2026-09-26, M2) Timing: a real `--image-hw 512 288 --batch-size 8` run logged epoch 1 at 12:13:50 and epoch 3 at 12:18:13, about 2.2 minutes per epoch, which is not slower than the 224x224 run (about 2.5 minutes per epoch). Training is limited by decoding the 3,000x5,100-pixel JPEGs on the CPU, not by GPU compute, so resolution is nearly free at this size. Projected run B: about 6.5 to 9 hours (later epochs with more unfrozen blocks cost more), not the 15 to 22 hours budgeted. The timing run was stopped after epoch 3 and its partial outputs (new directories, log, workbook, all created that day) were deleted.
- (2026-09-26, M3) Run A (grouped folds, 224x224, batch 16) finished cleanly at about 18:00 after roughly 5h 40m (start 12:19): empty error log, five 16,342,523-byte checkpoints, five "Fold Summary" rows. Folds 1, 2 and 4 early-stopped (epochs 27, 32, 31); 160 epochs logged. Validation TTA F1 by fold: 0.6474, 0.6375, 0.6680, 0.6114, 0.6710. Script summary: Mean Val Loss 0.5679 +/- 0.0305, Mean Val Accuracy 0.6837 +/- 0.0294, Mean Val F1 0.6470 +/- 0.0348. Mean CV F1 fell from 0.7200 (leaky folds) to 0.6470 (patient-grouped), which is the leak-inflation effect the plan predicted, now measured: honest validation is about 0.07 lower and much closer to the held-out level.
- (2026-09-26, M3) Held-out result for run A, unexpectedly BELOW the leaky-fold baseline: single best fold (fold 5, chosen by validation TTA F1) acc 0.6662, prec 0.5804, rec 0.5362, F1 0.5574; five-fold ensemble acc 0.6705, prec 0.5655, rec 0.6884, F1 0.6209 (baseline: 0.6552 and 0.6460). Both printed n=704. Cautions when reading this: (1) a 704-image test set with 276 malignant cases gives an F1 sampling error of very roughly 0.02 to 0.03, so differences of a few hundredths are within noise; (2) the single-best-fold number depends on which one fold validation picks and swung widely (0.5574 here), while the ensemble is steadier; (3) the baseline's validation was leaky, so its checkpoint selection (lowest validation loss on data resembling the training set) tended to favour more heavily trained checkpoints, which may have suited the held-out set, but that is a hypothesis, not something measured here.
- (2026-09-27, M4) Run B (grouped folds, 512x288, batch 8) finished cleanly at 00:29 after about 6h 18m (start 18:12), far below the 15 to 22 hours budgeted and in line with the Milestone 2 timing: empty error log, five 16,342,523-byte checkpoints. Folds 4 and 5 early-stopped at epoch 32. Validation TTA F1 by fold: 0.6723, 0.6468, 0.7010, 0.6523, 0.7210. Script summary: Mean Val Loss 0.5060 +/- 0.0182, Mean Val Accuracy 0.7340 +/- 0.0211, Mean Val F1 0.6899 +/- 0.0334 (run A on the same grouped folds: 0.6470). Per-fold thresholds 0.40, 0.26, 0.39, 0.35, 0.43 (recorded only; evaluation uses 0.5).
- (2026-09-27, M4) Held-out result for run B, evaluated with `--image-hw 512 288 --tta`, both printed n=704: single best fold (fold 5, validation TTA F1 0.7210) acc 0.7131, prec 0.6312, rec 0.6449, F1 0.6380; five-fold ensemble acc 0.7088, prec 0.6164, rec 0.6812, F1 0.6472.

## Decision Log

- Decision: group the folds by patient using scikit-learn's `StratifiedGroupKFold(n_splits=N, shuffle=True, random_state=42)`, and make this the unconditional behaviour of `EfficientNet.py`, with no flag to turn it off.
  Rationale: `StratifiedGroupKFold` keeps every patient's images entirely on one side of each split while keeping the benign/malignant ratio approximately balanced, which is exactly what is needed. The old leaky splitter gave inflated, misleading numbers; there is no reason anyone should choose it again, and a switch would be speculative configurability. The previous behaviour stays reproducible from git history (commit `f8e2840`) and from the existing `checkpoints_efficientnet_sam_5fold/` results. Because the folds now contain different images, per-fold and mean cross-validation numbers of the new runs are NOT comparable with earlier runs' cross-validation numbers; only the held-out test numbers are, and the plan compares only those.
  Date/Author: 2026-09-26, drafted by agent.
- Decision: identify a patient by the `patient_id` column of the CBIS-DDSM CSV files (for example `P_00001`), stored per sample in a new `patient_ids` list on `MammogramRawDataset`, appended in lock-step with `samples`.
  Rationale: `patient_id` is present in both the mass and calcification CSVs. It is the correct unit: one person's two breasts and multiple views are strongly correlated images and must not straddle a split.
  Date/Author: 2026-09-26, drafted by agent.
- Decision: use a non-square target of 512 pixels tall by 288 wide (`--image-hw 512 288`) for the high-resolution run, with batch size 8.
  Rationale: the source images are about 1.7 times taller than wide, so 512 by 288 preserves their shape, which the current square squash does not. It also has 147,456 pixels, the same as a 384 by 384 square, so cost is the same either way (about 2.9 times the 50,176 pixels of 224 by 224), but avoiding the distortion is free. It is an intermediate step (long side about 10 times smaller than native, versus about 23 times now) chosen because it fits the 8 GB GPU shared with the desktop (about 3 GB is in use by other programs) and keeps run time tolerable; going higher is a follow-up if this helps. EfficientNet-B0 is fully convolutional with global average pooling, so it accepts non-square input with no architecture change and the pretrained weights still apply. Batch size drops from 16 to 8 only to fit memory; learning rates are left unchanged.
  Date/Author: 2026-09-26, drafted by agent.
- Decision: run the experiment as two sequential runs, A (grouped, 224 by 224, batch 16) then B (grouped, 512 by 288, batch 8), rather than only B.
  Rationale: changing the splitter and the resolution together in one run would make it impossible to say which helped. Run A costs about 6.5 hours (measured last time) and cleanly isolates the leak fix; B minus A isolates resolution. Both are compared against the current committed baseline.
  Date/Author: 2026-09-26, drafted by agent.
- Decision: add exactly two command-line options to `EfficientNet.py`, `--image-hw H W` (default 224 224) and `--batch-size B` (default 16), and one option to `evaluate.py`, `--image-hw H W` (default: unchanged 224 by 224 behaviour). Output directories for run A are `checkpoints_efficientnet_sam_grouped_5fold/`, `plots_efficientnet_sam_grouped_5fold/` and `training_logs/efficientnet_sam_grouped_5fold_training_log.xlsx`; when `--image-hw` is not the default, the marker `_grouped_{H}x{W}` replaces `_grouped` (run B: `checkpoints_efficientnet_sam_grouped_512x288_5fold/` and so on).
  Rationale: the two runs must be launched with different sizes and must not overwrite each other or any earlier output, and `evaluate.py` must build the held-out test images at exactly the size the checkpoint was trained on or the numbers are meaningless. No other switches are added. `get_light_train_transforms()` and `get_val_transforms()` each gain one optional argument `size` defaulting to the current square `Config.IMAGE_SIZE`, so `ResNet.py` and every other caller behave exactly as before. When `--folds` is omitted the script still runs 5 folds and now uses the grouped output names too.
  Date/Author: 2026-09-26, drafted by agent.
- Decision: do not change the held-out test set, the decision threshold (0.5), test-time augmentation (flips), the loss, the augmentation, the optimiser settings, the unfreeze schedule or the epoch count; do not touch `eval_results/selected_models.json` or the live app.
  Rationale: one-variable-at-a-time. Every listed setting stays exactly as committed in `f8e2840`. Repointing the live app is the repository owner's decision after seeing results.
  Date/Author: 2026-09-26, drafted by agent.
- Decision: budget about 6.5 hours for run A and an estimated 15 to 22 hours for run B, and require a one-epoch timing measurement in Milestone 2 before committing to B.
  Rationale: B processes about 2.9 times the pixels per image, and the previous 224 by 224 run took about 2.5 minutes per epoch, so about 7 minutes per epoch is expected, times 175 epochs. That is an estimate, not a measurement, so Milestone 2 measures it and this plan records the real figure before the long run starts. If the measured time is unacceptable, stop and ask for approval to reduce scope (for example fewer folds) rather than quietly cutting it.
  Date/Author: 2026-09-26, drafted by agent.
- Decision: launch each long run as a background process managed by the coding agent's tooling, one at a time, never two at once.
  Rationale: established project practice (a terminal-window close once killed a multi-hour run) and the 8 GB GPU cannot hold two trainings. The previous plan's run confirmed the background launch survives well past the tool's nominal timeout.
  Date/Author: 2026-09-26, drafted by agent.

## Outcomes & Retrospective

(2026-09-27) All numbers on the same 704 held-out images, TTA on, threshold 0.5. Mean CV F1 is not comparable between the baseline (leaky folds) and A/B (patient-grouped folds); A and B share the same folds and are comparable with each other.

| Configuration | Mean CV F1 | Single-best acc / prec / rec / F1 | Ensemble acc / prec / rec / F1 | Wall-clock |
|---|---|---|---|---|
| Baseline `f8e2840` (leaky folds, 224x224) | 0.7200 | 0.6861 / 0.5753 / 0.7609 / 0.6552 | 0.6761 / 0.5652 / 0.7536 / 0.6460 | ~6h 36m |
| Run A (grouped, 224x224, batch 16) | 0.6470 | 0.6662 / 0.5804 / 0.5362 / 0.5574 | 0.6705 / 0.5655 / 0.6884 / 0.6209 | ~5h 50m |
| Run B (grouped, 512x288, batch 8) | 0.6899 | 0.7131 / 0.6312 / 0.6449 / 0.6380 | 0.7088 / 0.6164 / 0.6812 / 0.6472 | ~6h 18m |

Verdict.

A minus baseline (effect of honest validation): held-out ensemble F1 -0.025, accuracy -0.006; single-best F1 -0.098. Patient grouping did not improve the final model at 224x224; its value is that cross-validation now tells the truth (CV F1 0.647 versus held-out 0.62 to 0.56, instead of 0.72 versus 0.65).

B minus A (effect of resolution, same folds): ensemble F1 +0.026, accuracy +0.038, precision +0.051; single-best F1 +0.081, accuracy +0.047. Mean CV F1 also rose +0.043. Resolution helped on every measure, and it came at no time cost because training is bound by JPEG decoding on the CPU.

B versus baseline (net result of the plan): accuracy is about 0.03 higher (0.709 to 0.713 versus 0.676 to 0.686) and precision about 0.05 higher, but recall is about 0.07 to 0.12 lower, so F1 is essentially tied (ensemble 0.6472 versus 0.6460; single-best 0.6380 versus 0.6552). Run B makes fewer false alarms and misses more cancers at the fixed 0.5 threshold. With an F1 sampling error of roughly 0.02 to 0.03 on 704 images, only the accuracy and precision/recall shifts are clearly larger than noise; the F1 difference is not.

Recommendation: run B, preferably the ensemble, is the better-validated model and the best direction to pursue. Higher resolution is cheap here, so the natural next step is a larger size (for example 768x432 or 1024x576). Whether to repoint `eval_results/selected_models.json` to it is the repository owner's decision, and it depends on how much recall is worth against precision. The lower recall should be weighed before deploying anything. A threshold chosen on validation data (not the test set) may recover it.

## Context and Orientation

All commands run from the repository root, `A:\Projects\BreastCancerDetectionCNN`, using `py` (the Windows Python launcher; plain `python` is not on the PATH). Python source files here use Windows (CRLF) line endings; edits must preserve them. The machine has an NVIDIA RTX 3070 Ti with 8 GB, PyTorch 2.10 and TorchVision 0.25.

`DataSetAugmentation.py` holds the data layer. `MammogramRawDataset(csv_types)` reads the CBIS-DDSM CSVs in `data/raw/csv/`, and for each row with a `pathology` value it finds the matching JPEG under `data/raw/jpeg/`, appending `(image_path, label)` to `self.samples` (label 0 benign, 1 malignant) and a running abnormality number to `self.groups`. Rows whose image is not found are skipped, so `samples` can be shorter than the CSV; any per-sample list must be appended in exactly the same place as `samples`. There is a second append block for optional cropped patches (`include_cropped_patches`, off in this project's runs) which must also record the patient id if it appends to `samples`. `TransformDataset` applies a transform to each image. `get_light_train_transforms()` (flip, rotate 10 degrees, resize, normalise) and `get_val_transforms()` (resize, normalise) both resize to `(Config.IMAGE_SIZE, Config.IMAGE_SIZE)` with `IMAGE_SIZE = 224`.

`EfficientNet.py` is the training script. `EfficientNetConfig` holds the settings (`BATCH_SIZE = 16`, `IMAGE_SIZE = 224`, `NUM_EPOCHS = 35`, `USE_SAM = True`, and so on). `train_efficientnet_kfold(n_folds, checkpoint_dir, plot_dir, log_name)` builds the dataset, calls `StratifiedKFold(...).split(...)`, and for each fold trains a model, saves `best_model_fold_{k}.pth` when validation loss improves, and logs to `training_logs/`. The `__main__` block parses `--folds`. `IMAGE_SIZE` in `EfficientNetConfig` is not actually read by the transforms; they read `DataSetAugmentation.Config.IMAGE_SIZE`.

`evaluate.py` measures checkpoints on the held-out test set. `build_test_dataset()` creates the 704-image loader using `get_val_transforms()`. Run as a script it takes `--model`, `--checkpoint` (one or many), `--tta`, `--threshold`, `--out`. It also exports `evaluate_model` and `find_best_threshold`, which `EfficientNet.py` uses at the end of each fold with the fold's own validation loader (so they need no change).

The baseline, committed in `f8e2840`, files `eval_results/sam_efficientnet_5fold_singlebest_test_metrics.json` and `eval_results/sam_efficientnet_5fold_ensemble_test_metrics.json`, all with test-time augmentation on and threshold 0.5 on the same 704 images: single best fold accuracy 0.6861, F1 0.6552; five-fold ensemble accuracy 0.6761, F1 0.6460; training time about 6h 36m. "Single best fold" means the fold with the highest validation F1, chosen without looking at the test set.

## Plan of Work

### Milestone 1: patient-grouped folds

In `DataSetAugmentation.py`, add `self.patient_ids = []` next to `self.samples = []` in `MammogramRawDataset.__init__`. In the main per-row loop, immediately after `self.samples.append((img_path, label))`, append `row["patient_id"]` to `self.patient_ids`; in the cropped-patch block, immediately after that block's own `self.samples.append(...)`, append the same patient id. Nothing else in the file changes.

In `EfficientNet.py`, in `train_efficientnet_kfold`, change the import line to add `StratifiedGroupKFold` (import it from `sklearn.model_selection` alongside the existing `StratifiedKFold`; remove `StratifiedKFold` from the import only if it is no longer used anywhere in the file), and replace the splitter construction and its `.split` call so that it is `StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)` and the split call passes `groups=raw_dataset.patient_ids`. Nothing else about the fold loop changes.

Verification. Write a throwaway script to the agent's scratchpad (not the repository), `check_groups.py`, that builds `MammogramRawDataset(["mass_train", "calc_train"])`, runs the same splitter for `n_folds` in 3, 5 and 10, and asserts for every fold that the set of patient ids in the validation indices and the set in the training indices are disjoint, that the union of all validation indices covers every sample exactly once, and prints each fold's validation size and malignant fraction. Expected: no assertion fails; validation sizes near 2,864 divided by the fold count (a few percent of drift is normal because whole patients move together); malignant fraction near 0.41 in every fold. Then run `py EfficientNet.py --help` and confirm it still prints usage without training.

### Milestone 2: configurable image size and batch size, and a memory and timing check

In `DataSetAugmentation.py`, change `get_light_train_transforms()` and `get_val_transforms()` to take an optional argument `size=None`, and use `size` as the `Resize` target when given, otherwise the existing `(Config.IMAGE_SIZE, Config.IMAGE_SIZE)`. The heavy `get_train_transforms()` is untouched.

In `EfficientNet.py`, add two attributes to `EfficientNetConfig`, `IMAGE_HW = (224, 224)` (replacing the unused `IMAGE_SIZE = 224` line only if grep shows nothing reads it; otherwise leave it) and keep `BATCH_SIZE = 16`. Pass `size=EfficientNetConfig.IMAGE_HW` to `get_light_train_transforms` and `get_val_transforms` in the fold loop. In the `__main__` block, add `--image-hw` (two integers, default 224 224) and `--batch-size` (integer, default 16) and copy them into `EfficientNetConfig.IMAGE_HW` and `EfficientNetConfig.BATCH_SIZE` before training starts. Build the output names as decided: with default size, `checkpoints_efficientnet_sam_grouped_{N}fold`, `plots_efficientnet_sam_grouped_{N}fold` and log name `efficientnet_sam_grouped_{N}fold`; with a non-default size, `_grouped_{H}x{W}` in place of `_grouped`. Update the `--folds` help text so it names these directories. When `--folds` is omitted, use `N = 5` with the same grouped names (so the default path never writes to any pre-existing directory), and update `EfficientNetConfig.CHECKPOINT_DIR` and `PLOT_DIR` to the matching `_grouped` names.

In `evaluate.py`, add `--image-hw H W` (default: none, meaning today's behaviour). `build_test_dataset()` gains an optional `size` argument passed to `get_val_transforms(size)`, and `main()` passes it through. `EfficientNet.py`'s calls to `evaluate_model` and `find_best_threshold` take a loader and need no change.

Verification. `py EfficientNet.py --help` shows the new options and the grouped directory names. Then a throwaway smoke script in the scratchpad builds the model, the SAM optimiser and a batch of 8 random tensors of shape `3 x 512 x 288`, runs three real `train_one_epoch` steps on a fake loader, and prints `torch.cuda.max_memory_allocated()` in megabytes. Expected: it completes, the model output shape is `(8, 2)`, and peak allocated memory is below about 4,500 MB (the GPU has 8 GB with about 3 GB used by other programs). If it exceeds that or raises an out-of-memory error, lower the batch size in the plan (record it in the Decision Log) rather than continuing. Then time one real epoch: run `py -u EfficientNet.py --folds 5 --image-hw 512 288 --batch-size 8` in the background for about ten minutes, read the first epoch's wall-clock time from the log file's modification times, and stop the process. Delete the partial output directories it created (they are new and contain nothing worth keeping; confirm the names before deleting). Record the measured minutes per epoch and the resulting projected total for run B in Surprises & Discoveries and revisit the Decision Log time budget. Finally confirm the light transform still yields `3 x 224 x 224` by default, and that `evaluate.py --image-hw 512 288` builds a loader producing `3 x 512 x 288` tensors (a one-line check on the first batch).

### Milestone 3: run A (grouped, 224 by 224)

Confirm the GPU is free of other training and that `checkpoints_efficientnet_sam_grouped_5fold/` does not exist. Launch as a background process from the repository root:

    py -u EfficientNet.py --folds 5 > efficientnet_sam_grouped_5fold.log 2> efficientnet_sam_grouped_5fold.err.log

Expect about 6.5 hours. Accept when the error log is empty, the checkpoint directory has five files of 16,342,523 bytes, the "Fold Summary" sheet of `training_logs/efficientnet_sam_grouped_5fold_training_log.xlsx` has five data rows, and the log ends with the "Final Cross-Validation Results" block. Note that the cross-validation numbers here should come out LOWER than the previous run's 0.72 mean F1 if the leak was inflating them; record them. Then evaluate on the held-out set, picking the single best fold from validation TTA F1 only:

    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_sam_grouped_5fold/best_model_fold_K.pth --tta --out eval_results/samgrouped_efficientnet_5fold_singlebest_test_metrics.json

and the five-checkpoint ensemble to `eval_results/samgrouped_efficientnet_5fold_ensemble_test_metrics.json`. Both printed lines must show `n=704`.

### Milestone 4: run B (grouped, 512 by 288)

Same procedure, once Milestone 3 has finished and the GPU is free, with the checkpoints in `checkpoints_efficientnet_sam_grouped_512x288_5fold/` and logs `efficientnet_sam_grouped_512x288_5fold.log` and `.err.log`:

    py -u EfficientNet.py --folds 5 --image-hw 512 288 --batch-size 8 > efficientnet_sam_grouped_512x288_5fold.log 2> efficientnet_sam_grouped_512x288_5fold.err.log

Evaluate exactly as before but with `--image-hw 512 288` added to both `evaluate.py` commands (this is essential: without it the test images would be resized to 224 by 224 and the numbers would be wrong) and outputs `eval_results/samgrouped512x288_efficientnet_5fold_singlebest_test_metrics.json` and `..._ensemble_test_metrics.json`. Both must show `n=704`.

### Milestone 5: comparison and verdict

Write into Outcomes & Retrospective a table of the three configurations (baseline `f8e2840`, run A, run B) giving mean cross-validation F1 (noting it is not comparable across the baseline and A/B because the folds differ), held-out single-best F1 and accuracy, held-out ensemble F1 and accuracy, and wall-clock time. State plainly which change helped and by how much: A minus baseline is the effect of honest validation on the final model, B minus A is the effect of resolution. Also record precision and recall for each, since F1 can hide a shift between them. Do not tune the threshold on the test set, and do not repoint `eval_results/selected_models.json`.

## Validation and Acceptance

The plan is complete when: the patient-disjointness script passes for 3, 5 and 10 folds; `py EfficientNet.py --help` prints the new options and grouped directory names; the memory smoke test passes and a real epoch time at 512 by 288 is recorded; both long runs finish with empty error logs, five 16,342,523-byte checkpoints each, and five-row fold summaries; all four held-out evaluations print `n=704`; and Outcomes contains the three-way table and verdict. Non-regression: only `DataSetAugmentation.py`, `EfficientNet.py` and `evaluate.py` may change among tracked files (plus the plan itself and new outputs); `ResNet.py`, `losses.py`, `optimizers.py`, `inference.py`, `app.py` and `eval_results/selected_models.json` must be unchanged, and the modification times of `checkpoints_efficientnet/`, `checkpoints_efficientnet_sam_5fold/` and `checkpoints_efficientnet_5fold/` must not change. `ResNet.py` and `training_logs/resnet_training_log.xlsx` already show as modified from before this plan and must not gain further changes from it.

## Idempotence and Recovery

Source edits are safe to repeat; `git diff` shows exactly what changed and `git checkout -- <file>` restores a file (no history-rewriting git commands). The training script cannot resume, so a killed run must restart from fold 1; move any partial checkpoint directory aside to `<name>.aborted-<date>/` instead of deleting it, and never evaluate a partial run's checkpoints as this plan's result. Evaluation is deterministic and safe to re-run.

## Artifacts and Notes

New outputs: `checkpoints_efficientnet_sam_grouped_5fold/`, `checkpoints_efficientnet_sam_grouped_512x288_5fold/`, matching `plots_...` directories, two Excel logs in `training_logs/`, four console log files, and four JSON files in `eval_results/`. Model weight files are ignored by git. Open follow-ups, not in scope: higher resolutions than 512 by 288; ensembling different architectures; the 31 patients shared between the training and test sets; the SAM BatchNorm refinement.

## Interfaces and Dependencies

No new dependency (`StratifiedGroupKFold` is in the already-required scikit-learn). Signatures: `get_light_train_transforms(size=None)` and `get_val_transforms(size=None)` gain one optional argument with unchanged default behaviour; `build_test_dataset(size=None)` in `evaluate.py` likewise; `MammogramRawDataset` gains a `patient_ids` attribute. `create_efficientnet()` is unchanged, so old checkpoints and `inference.py` keep working.
