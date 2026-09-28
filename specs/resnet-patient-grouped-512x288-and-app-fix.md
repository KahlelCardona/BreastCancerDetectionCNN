# Retrain ResNet-50 with patient-grouped folds at 512 by 288 and repair the web app's ResNet model

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not check in a copy of PLANS.md. The authoring and execution rules for this document live at `~/.agents/PLANS.md` on the machine of whoever is running this plan (a global, agent-agnostic file, not part of this git repository). This document must be maintained in accordance with that file.

This plan builds on two checked-in ExecPlans: `specs/efficientnet-light-augmentation-and-sam.md` (lighter augmentation and the SAM optimiser for EfficientNet-B0) and `specs/patient-grouped-cv-and-higher-resolution.md` (patient-grouped folds and 512 by 288 images for EfficientNet-B0). It supersedes the unfinished ResNet half of `specs/synthetic-augmentation-f1-improvement.md`. Everything needed from those documents is restated below.

## Approval

Status: Approved

Approved by: Repository owner (chat instruction: "approved")

Date: 2026-09-27

Requested scope: Milestones 1 to 4 below, including one multi-hour ResNet-50 training run (Milestone 2), and repointing the web app's ResNet entry in `eval_results/selected_models.json` (Milestone 4) under the rule stated in the Decision Log.

## Purpose / Big Picture

This project classifies a mammogram as benign or malignant with two neural networks, a ResNet-50 and an EfficientNet-B0. A Flask web app (`py app.py`, served at http://127.0.0.1:5000) shows each model's verdict and a blended "Final Answer". Each change is judged by F1 score, the harmonic mean of precision and recall, measured on a 704-image held-out test set that is never trained on.

The ResNet side has two problems, and this plan fixes both.

First, the web app is serving a half-trained ResNet. `eval_results/selected_models.json` tells the app to load `checkpoints_resnet/fold3_best.pth` and labels it with accuracy 0.6534. A ResNet retrain started on 2026-08-27 wrote its checkpoints into that same directory. It was killed at 11:44 that morning when its terminal window was closed, in epoch 12 of fold 3. `fold3_best.pth` was last written at exactly 11:44:32, matching that run's last "saved best model" log line. Measured on 2026-09-27 on the held-out set with test-time augmentation and the app's threshold of 0.45, the served checkpoint scores accuracy 0.5866, precision 0.4801, recall 0.6558 and F1 0.5544. The app also uses the stale 0.6534 as this model's weight when blending the two models' answers. Model weight files (`*.pth`) are ignored by git, so the checkpoints behind the recorded 0.6534 baseline cannot be recovered.

Second, ResNet-50 is still trained the old way. Its folds are not split by patient, so validation scores are inflated. It uses heavy image distortion and MixUp. It sees 224 by 224 images, about 23 times smaller than the source mammograms along the long side, squashed out of their natural shape. The same three problems were fixed for EfficientNet-B0 in September. There, patient grouping brought validation in line with held-out results, lighter augmentation raised held-out F1 by about 0.07, and 512 by 288 images raised held-out accuracy by about 0.04 at no extra training time.

After this plan, `py ResNet.py --image-hw 512 288` trains ResNet-50 the same way as the best EfficientNet, writing only to new directories. The web app serves the resulting five-model ResNet ensemble, feeding it 512 by 288 images. The observable proof is threefold. The new ResNet's held-out F1 is printed next to the served checkpoint's 0.5544 and the historical 0.6534. Uploading a mammogram in the web app returns both verdicts. And `eval_results/selected_models.json` no longer points into `checkpoints_resnet/`.

## Progress

- [x] (2026-09-27) Research: confirmed the served ResNet checkpoint was overwritten by the aborted 2026-08-27 run, and measured it on the held-out set (acc 0.5866, F1 0.5544 at threshold 0.45).
- [x] (2026-09-27) Research: memory and speed probe of ResNet-50 at 512 by 288 with SAM and layers 3 and 4 unfrozen. Batch 16 peaks at 1,526 MB and takes 0.20 s per step, about 0.5 minutes of GPU time per epoch.
- [x] (2026-09-27) Milestone 1: edit `ResNet.py` (patient-grouped folds, light transforms, MixUp off, `--image-hw` and `--batch-size`, new output names) and verify without a full run.
- [x] (2026-09-28 04:21) Milestone 2: run the full 5-fold ResNet-50 training at 512 by 288 in the background.
- [x] (2026-09-28) Milestone 3: evaluate the new ResNet on the held-out set (single best fold and five-fold ensemble).
- [x] (2026-09-28) Milestone 4: teach `inference.py` per-model image sizes, repoint the ResNet entry of `eval_results/selected_models.json`, verify the web app end to end, and mark the old synthetic-augmentation plan superseded.

## Surprises & Discoveries

- (2026-09-27, research) The live app's ResNet checkpoint is a partial checkpoint from an aborted run. Evidence: `resnet_retrain_aug.log` was created 2026-08-27 07:45:08 and last written 11:44:32. Its final lines are "Epoch 12 | ... Val ... F1 0.4870 / --> saved best model" in fold 3. `checkpoints_resnet/fold3_best.pth` has LastWriteTime 2026-08-27 11:44:32. `resnet_retrain_aug.err.log` holds five "forrtl: error (200): program aborting due to window-CLOSE event" traces.
- (2026-09-27, research) Held-out measurement of the served model: `py evaluate.py --model resnet --checkpoint checkpoints_resnet/fold3_best.pth --tta --threshold 0.45` printed `n=704 acc=0.5866 prec=0.4801 rec=0.6558 f1=0.5544`.
- (2026-09-27, research) `checkpoints_resnet/` now mixes three runs. `fold1_best.pth` to `fold3_best.pth` (107,226,339 bytes) come from the aborted 2026-08-27 run. `fold4_best.pth` and `fold5_best.pth` come from the 2026-08-26 cropped-patch run (`resnet_retrain.log`). `best_model_fold_1.pth` to `best_model_fold_5.pth` (98,828,267 bytes, May 2026) come from an older model definition. None is the 0.6534 baseline.
- (2026-09-27, research) Probe at 512 by 288 with SAM and layers 3 and 4 trainable (the worst case), from a scratchpad script calling `ResNet.train_epoch` on random tensors: `bs=16 peak_MB=1526 sec_per_step=0.201` and `bs=8 peak_MB=996 sec_per_step=0.106`. The model output shape is (N, 2), so the attention pooling and head accept non-square input. The GPU has 8,192 MiB with about 3,100 MiB used by the desktop, so batch 16 fits comfortably. GPU compute is about 0.5 minutes per epoch. The EfficientNet runs showed that decoding the roughly 3,000 by 5,100-pixel JPEGs on the CPU costs about 2.2 to 2.5 minutes per epoch, so loading, not the GPU, will set the pace.
- (2026-09-27, M1) Verification passed. `py ResNet.py --help` prints `--image-hw H W` and `--batch-size` without training. The patient-grouped split of the 2,864 training images gives validation sizes 573, 574, 573, 571 and 573, with 0 shared patients in every fold, every sample validated exactly once, and a malignant fraction of 0.412 to 0.414. Both transforms give `torch.Size([3, 512, 288])`. `CFG.MIXUP_ALPHA == 0.0`. The diff against the pre-plan copy is +35/-11 lines, exactly the Plan of Work edits, and CRLF line endings are preserved (0 LF-only lines).
- (2026-09-27, M2) Run launched 21:03:24 as a background process. Epoch 2 was logged by 21:07:42, about 2.1 minutes per epoch including dataset loading, and the error log is empty. Fold 1, epoch 1: train loss 0.5251, val F1 0.6165. The projected total is about 5 to 10 hours, depending on early stopping and on the slower epochs after layer3 and layer4 unfreeze.
- (2026-09-27, M4 code, done early while training runs) `inference.py` edited as planned. With the unchanged JSON (no `image_hw` fields) it loads both models with `image_hw=None`, predicts on the first `mass_test` image, and averages 0.225 s per `predict` call, confirming today's behaviour is preserved. A pre-existing quirk was noticed but is out of scope: the blended "final" label uses a fixed 0.5 cutoff, while each model uses its own threshold. On that image both models said "Malignant" (EfficientNet at probability 0.461, above its 0.302 threshold), `models_agreed` was True, yet the final label was "Benign" (blended 0.496).

## Decision Log

- Decision: train ResNet-50 in one run that changes three things at once: patient-grouped folds, light augmentation with MixUp off, and 512 by 288 images. There is no separate 224 by 224 run.
  Rationale: the goal is to put the best-known recipe onto ResNet-50 and to replace a broken live model, not to measure each change separately. Each change was already measured on EfficientNet-B0 in `specs/efficientnet-light-augmentation-and-sam.md` and `specs/patient-grouped-cv-and-higher-resolution.md`. A second multi-hour run to separate them would delay the app fix. The cost is that this plan cannot say which change helped ResNet, and it says so in its outcome.
  Date/Author: 2026-09-27, drafted by agent.
- Decision: keep every other ResNet setting unchanged. That includes SAM (already on, rho 0.07), the hybrid focal plus cross-entropy loss, label smoothing 0.08, dropout, learning rates, the unfreeze schedule (layer4 at epoch 5, layer3 at epoch 10), cosine restarts, 55 maximum epochs, early-stopping patience 12, and saving the checkpoint with the lowest validation loss from epoch 8 on.
  Rationale: one recipe change at a time, matching the EfficientNet plans. ResNet already uses SAM, so only the augmentation, MixUp, fold splitting and image size change.
  Date/Author: 2026-09-27, drafted by agent.
- Decision: use batch size 16 (ResNet's existing value) at 512 by 288.
  Rationale: measured peak memory at batch 16 is 1,526 MB, far below the roughly 5 GB free. There is no reason to change it.
  Date/Author: 2026-09-27, drafted by agent.
- Decision: write to brand-new locations named with a `sam_grouped` marker. With the default 224 by 224 size the names are `checkpoints_resnet_sam_grouped_5fold/` and `training_logs/resnet_sam_grouped_5fold_training_log.xlsx`. With `--image-hw H W` they become `checkpoints_resnet_sam_grouped_{H}x{W}_5fold/` and `training_logs/resnet_sam_grouped_{H}x{W}_5fold_training_log.xlsx`. `ResNet.py` must never write to `checkpoints_resnet/` or `training_logs/resnet_training_log.xlsx` again.
  Rationale: the August overwrite is the reason the app is broken now. The naming matches `EfficientNet.py`. No `--folds` flag is added, because only 5 folds are needed.
  Date/Author: 2026-09-27, drafted by agent.
- Decision: the app will serve the five-fold ResNet ensemble (all five `fold{k}_best.pth` files averaged) at threshold 0.5. This choice is fixed now, before any test-set number exists.
  Rationale: choosing between the single best fold and the ensemble, or tuning the threshold, by looking at the test set would make the test number optimistic. In the EfficientNet runs the ensemble was steadier than the single best fold, which swung between F1 0.557 and 0.655. The single best fold is still evaluated and reported, but it is not a candidate for serving.
  Date/Author: 2026-09-27, drafted by agent.
- Decision: repoint the ResNet entry of `eval_results/selected_models.json` only if the new ensemble's held-out F1 and accuracy both exceed the served checkpoint's (F1 0.5544, accuracy 0.5866). Otherwise stop and ask the repository owner. The entry's `accuracy` field (used by the app as the blending weight) is set to the ensemble's measured held-out accuracy. The EfficientNet entry is not changed by this plan.
  Rationale: the served model is known to be broken, so the bar for replacing it is only that the new one is measurably better. Repointing EfficientNet to its own 512 by 288 ensemble is a separate decision for the repository owner, because it trades recall for precision. With the per-model image size support added in Milestone 4, it becomes a one-entry JSON edit.
  Date/Author: 2026-09-27, drafted by agent.
- Decision: add an optional `image_hw` field to each entry of `eval_results/selected_models.json`, and make `inference.py` resize the uploaded image separately for each model. An entry without the field keeps today's 224 by 224 behaviour.
  Rationale: today `inference.predict` builds one 224 by 224 tensor and feeds it to both models. A model trained at 512 by 288 must see 512 by 288 input, or its predictions are meaningless. An optional field keeps the current EfficientNet entry working unchanged.
  Date/Author: 2026-09-27, drafted by agent.
- Decision: do not rerun the strengthened-augmentation ResNet retrain from `specs/synthetic-augmentation-f1-improvement.md`. Mark that plan superseded instead.
  Rationale: that configuration (random affine warp, 25 percent brightness and contrast jitter, random erasing, MixUp) lowered EfficientNet's held-out F1 to 0.586 (single) and 0.574 (ensemble), from 0.6375. Spending about 6 GPU-hours to confirm it on ResNet would not improve the app.
  Date/Author: 2026-09-27, drafted by agent.
- Decision: launch the long run as a background process managed by the coding agent's tooling, never in a manually opened terminal window, and never alongside another training process.
  Rationale: closing a terminal window destroyed the 2026-08-27 run. The 8 GB GPU cannot hold two trainings.
  Date/Author: 2026-09-27, drafted by agent.

## Outcomes & Retrospective

(2026-09-28) Milestone 2. The run ended cleanly at 04:20:57, after 7h 17m (start 21:03:24). The error log is 0 bytes. `checkpoints_resnet_sam_grouped_512x288_5fold/` holds `fold1_best.pth` to `fold5_best.pth`, each 107,226,339 bytes. The workbook has 5 "Fold Summary" rows and 226 epoch rows: folds 1 and 3 ran all 55 epochs, and folds 2, 4 and 5 early-stopped at epochs 33, 32 and 51. Validation F1 with TTA by fold: 0.7129, 0.6996, 0.7319, 0.6835, 0.7265. Script summary: mean validation F1 0.6968 +/- 0.0291 without TTA and 0.7109 +/- 0.0177 with TTA. Per-fold thresholds 0.42, 0.43, 0.49, 0.43, 0.42 (recorded only; not used).

Milestone 3. All numbers are on the same 704 held-out images, with TTA. The single best fold is fold 3, chosen by validation TTA F1 (0.7319) only.

    Model                                              Threshold  Acc     Prec    Rec     F1
    Served before this plan (checkpoints_resnet/fold3)   0.45     0.5866  0.4801  0.6558  0.5544
    Historical ResNet baseline (Aug, 224, leaky folds)   0.50     0.6776    -       -     0.6534   (checkpoints lost)
    New ResNet, single best fold 3                       0.50     0.7145  0.6183  0.7101  0.6610
    New ResNet, 5-fold ensemble (now served)             0.50     0.7131  0.6164  0.7101  0.6599
    Best EfficientNet (512x288 ensemble, for reference)  0.50     0.7088  0.6164  0.6812  0.6472

Verdict: the new ResNet beats all three reference points. Against the model the app was serving, the ensemble gains +0.105 F1 and +0.127 accuracy. Against the historical ResNet baseline it gains about +0.007 F1, which is within the roughly 0.02 to 0.03 sampling noise of a 704-image test set, and +0.036 accuracy. The honest reading is "at least as good, with better accuracy", achieved with leak-free validation. Against the best EfficientNet it is +0.013 F1 at equal precision and +0.029 recall, so ResNet-50 is now the stronger of the two architectures. Unlike EfficientNet at 512 by 288, it did not give up recall. Mean validation F1 (0.711 with TTA) is close to held-out F1 (0.660), so patient grouping kept validation honest here too. This plan changed three things at once, so it cannot say which of them helped ResNet (see Decision Log).

Milestone 4. The ResNet entry of `eval_results/selected_models.json` now lists the five new checkpoints with `"accuracy": 0.7131, "threshold": 0.5, "image_hw": [512, 288]`. `git diff` shows the `efficientnet` line unchanged. The scratchpad check loads `resnet` as `list[5]` with `(512, 288)` and `efficientnet` as `list[5]` with `None`. It predicts on the first `mass_test` image (ResNet Malignant 0.668, EfficientNet Malignant 0.461, final Malignant 0.572), with an average `predict` time of 0.250 s. Flask test client: the image upload returns 200 with a verdict, and both no file and a `.txt` file return 200 with "Please upload a .jpg, .jpeg, or .png image.". `specs/synthetic-augmentation-f1-improvement.md` now carries a dated "Superseded" outcome.

Non-regression. Tracked files changed by this plan: `ResNet.py`, `inference.py`, `eval_results/selected_models.json`, `specs/synthetic-augmentation-f1-improvement.md` and this plan. `DataSetAugmentation.py`, `EfficientNet.py`, `evaluate.py` and `training_logs/resnet_training_log.xlsx` show as modified only because of uncommitted changes that predate this plan. `checkpoints_resnet/` and every `checkpoints_efficientnet*` directory keep their previous modification times.

Remaining and follow-ups (not done here). The app's EfficientNet entry still serves the old `checkpoints_efficientnet/` ensemble at 224 by 224 (acc 0.6179 as recorded). Repointing it to `checkpoints_efficientnet_sam_grouped_512x288_5fold/` is now a one-entry JSON edit and the repository owner's decision. The blended "final" label uses a fixed 0.5 cutoff regardless of each model's threshold (see Surprises & Discoveries). Nothing has been committed.

## Context and Orientation

All commands run from the repository root, `A:\Projects\BreastCancerDetectionCNN`, in PowerShell, using `py` (the Windows Python launcher; plain `python` is not on the PATH). Python source files here use Windows (CRLF) line endings; edits must preserve them. The machine has an NVIDIA RTX 3070 Ti with 8 GB, PyTorch 2.10 and TorchVision 0.25. Model weights (`*.pth`) are ignored by git.

The data is CBIS-DDSM, a public mammography dataset. `DataSetAugmentation.py` holds the data layer. `MammogramRawDataset(["mass_train", "calc_train"])` loads the 2,864 training images (1,248 patients, up to 24 images each). It exposes `samples` (a list of `(image_path, label)`, label 0 benign, 1 malignant) and a parallel list `patient_ids` (for example `P_00001`). `TransformDataset(dataset, transform)` applies an image transform. Three transform builders exist. `get_train_transforms()` is the heavy pipeline: flip, affine warp, colour jitter, random erasing, always 224 by 224. `get_light_train_transforms(size=None)` applies a horizontal flip and a 10 degree rotation. `get_val_transforms(size=None)` only resizes and normalises. For the last two, `size` is a `(height, width)` tuple and `None` means 224 by 224. None of these are changed by this plan.

`ResNet.py` is the ResNet-50 training script. Its `CFG` class holds settings, including `BATCH_SIZE = 16`, an unused `IMG_SIZE = 224`, `EPOCHS = 55`, `MIXUP_ALPHA = 0.2`, `USE_SAM = True` and `CHECKPOINT_DIR = Path("checkpoints_resnet")`. That directory is created when the module is imported. `ResNetWithAttnPool` is the model: a pretrained ResNet-50 backbone, an attention-pooling layer, and a small classifier head. `train_fold(...)` builds the training and validation loaders from `get_train_transforms()` and `get_val_transforms()`, trains one fold, saves `CHECKPOINT_DIR / f"fold{k}_best.pth"` whenever validation loss improves (from epoch 8), and finally re-scores that checkpoint with test-time augmentation and logs a fold summary. `main()` builds the dataset, splits it with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, loops over folds, and writes an Excel log through `training_log.create_workbook("resnet")` to `training_logs/resnet_training_log.xlsx`. `ResNet.py` produces no plots. Before this plan it already has uncommitted edits (a revert from cropped patches and `StratifiedGroupKFold` back to plain `StratifiedKFold`), and `training_logs/resnet_training_log.xlsx` is already modified. Both are left as the starting point.

Some terms used below. A "fold" is one of five train/validation splits of the training set used for cross-validation. "Patient-grouped" folds use scikit-learn's `StratifiedGroupKFold`, which keeps all of one patient's images on the same side of every split while keeping the benign/malignant ratio balanced. "MixUp" blends pairs of training images and their labels; it is switched off by setting `MIXUP_ALPHA = 0.0`, which `train_epoch` already handles. "SAM" (Sharpness-Aware Minimisation) is an optimiser that takes two forward and backward passes per batch to prefer flat, better-generalising solutions; ResNet already uses it. "Test-time augmentation" (TTA) averages the model's output over the original image and its horizontal and vertical flips. An "ensemble" averages the malignant probabilities of several checkpoints.

`evaluate.py` measures checkpoints on the 704-image held-out test set (428 benign, 276 malignant). It needs no change. It takes `--model resnet|efficientnet`, `--checkpoint` (one or more paths, averaged as an ensemble), `--tta`, `--threshold` (default 0.5), `--image-hw H W` (the size the checkpoint was trained at) and `--out`.

`inference.py` serves the web app. `load_selected_models(device)` reads `eval_results/selected_models.json`. Each of its `resnet` and `efficientnet` entries has a `checkpoint` (a path or a list of paths), an `accuracy` (used as that model's weight when blending) and a `threshold`. It returns a dict mapping each name to `(loaded_model_or_list, accuracy, threshold)`. `predict(image, models)` converts the PIL image once with `get_val_transforms()` (224 by 224), runs both models with TTA, and blends the two probabilities weighted by accuracy. `app.py` calls `load_selected_models` at startup and `predict` for each upload. The JSON currently contains, for resnet, `{"checkpoint": "checkpoints_resnet/fold3_best.pth", "accuracy": 0.6534, "threshold": 0.45}`. For efficientnet it lists `checkpoints_efficientnet/best_model_fold_1.pth` to `_5.pth` with accuracy 0.6179 and threshold 0.302. The file is tracked by git.

The held-out test set shares 31 patients with the training set. This slightly favours every model equally and is not changed here, so all numbers stay comparable with earlier results.

## Plan of Work

Milestone 1 edits `ResNet.py` only. Change the import `from sklearn.model_selection import StratifiedKFold` to `StratifiedGroupKFold`. In the `from DataSetAugmentation import (...)` block, replace `get_train_transforms` with `get_light_train_transforms`, since the former becomes unused in this file. Add `import argparse` at the top. In `CFG`, add `IMAGE_HW = (224, 224)` directly below `IMG_SIZE`, and change `MIXUP_ALPHA` from `0.2` to `0.0`. In `train_fold`, build the training set with `get_light_train_transforms(CFG.IMAGE_HW)` and the validation set with `get_val_transforms(CFG.IMAGE_HW)`. Change `main()` to `main(log_name="resnet")`. At its start call `CFG.CHECKPOINT_DIR.mkdir(exist_ok=True)`, pass `log_name` to `training_log.create_workbook`, and set `log_path = training_log.LOG_DIR / f"{log_name}_training_log.xlsx"`. Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` and call `skf.split(np.zeros(len(labels)), labels, groups=raw_dataset.patient_ids)`. Replace the `if __name__ == "__main__":` block with an argparse parser that has two options. `--image-hw H W` takes two ints, default `[224, 224]`. `--batch-size` takes an int, default 16. The help text should name the output locations. The block sets `CFG.IMAGE_HW` and `CFG.BATCH_SIZE`. It computes `marker = "grouped"` when the size is `(224, 224)`, otherwise `f"grouped_{H}x{W}"`. It sets `CFG.CHECKPOINT_DIR = Path(f"checkpoints_resnet_sam_{marker}_5fold")` and calls `main(log_name=f"resnet_sam_{marker}_5fold")`. Nothing else in `ResNet.py` changes. The class-level `CHECKPOINT_DIR.mkdir` stays; it only ensures the old directory exists on import, as today.

Milestone 2 launches the run. Milestone 3 evaluates it. Neither edits code.

Milestone 4 edits `inference.py` and `eval_results/selected_models.json` and appends a note to `specs/synthetic-augmentation-f1-improvement.md`. In `load_selected_models`, read `image_hw = tuple(entry["image_hw"]) if "image_hw" in entry else None` and store a four-element tuple `(loaded, accuracy, threshold, image_hw)`. In `predict`, remove the single shared `image_tensor` line. Unpack the four-element tuples. Build `get_val_transforms(resnet_image_hw)(image)` for ResNet and `get_val_transforms(efficientnet_image_hw)(image)` for EfficientNet, and pass each tensor to its own `_predict_single` call. Then, if the Decision Log's repointing rule is met, replace the `resnet` entry of the JSON with the five new checkpoint paths, the ensemble's measured held-out accuracy (four decimals), `"threshold": 0.5` and `"image_hw": [512, 288]`. Leave the `efficientnet` entry byte-for-byte unchanged. Finally, in `specs/synthetic-augmentation-f1-improvement.md`, write a short dated Outcomes entry. It should say the ResNet retrain was aborted on 2026-08-27, that the EfficientNet evidence made rerunning it pointless, and that this plan supersedes it. Leave its Progress checkboxes unticked.

## Concrete Steps

Milestone 1 verification, from the repository root:

    py ResNet.py --help

Expect usage text listing `--image-hw H W` and `--batch-size`, and no training. Then, in a scratchpad script, import `ResNet` and `DataSetAugmentation` with the repository root on `sys.path` and the working directory set to the repository root. Build `MammogramRawDataset(["mass_train", "calc_train"])`, split it exactly as `main()` does, and assert that for each of the 5 folds the sets of `patient_ids` at `train_idx` and `val_idx` do not intersect and every sample is validated exactly once. Check that `get_light_train_transforms((512, 288))` and `get_val_transforms((512, 288))` turn a real training image into a tensor of shape `[3, 512, 288]`. Check that `ResNet.CFG.MIXUP_ALPHA == 0.0`. Expected output resembles:

    fold 1: train 2291 val 573 shared patients 0
    ... (5 lines, each with shared patients 0)
    shapes ok: torch.Size([3, 512, 288]) torch.Size([3, 512, 288])

Then run `git diff ResNet.py` and confirm that only the lines listed in Plan of Work changed relative to the pre-plan working copy.

Milestone 2, from the repository root, as a background process. First confirm with `nvidia-smi` that no Python training process holds the GPU, and that `checkpoints_resnet_sam_grouped_512x288_5fold/` does not exist:

    py -u ResNet.py --image-hw 512 288 > resnet_sam_grouped_512x288_5fold.log 2> resnet_sam_grouped_512x288_5fold.err.log

Expect roughly 2.5 minutes per epoch and up to 55 epochs per fold with early stopping, so about 6 to 11 hours in total. Record the first epoch's time in Surprises & Discoveries. The run is done when the log ends with the "Final cross-validation results (ResNet-50 v2)" block.

Milestone 3, from the repository root. Pick the single best fold `K` as the fold with the highest "best-val F1 (with TTA)" in the log, using validation numbers only. Then:

    py evaluate.py --model resnet --checkpoint checkpoints_resnet_sam_grouped_512x288_5fold/fold{K}_best.pth --tta --image-hw 512 288 --out eval_results/resnetsamgrouped512x288_5fold_singlebest_test_metrics.json
    py evaluate.py --model resnet --checkpoint checkpoints_resnet_sam_grouped_512x288_5fold/fold1_best.pth checkpoints_resnet_sam_grouped_512x288_5fold/fold2_best.pth checkpoints_resnet_sam_grouped_512x288_5fold/fold3_best.pth checkpoints_resnet_sam_grouped_512x288_5fold/fold4_best.pth checkpoints_resnet_sam_grouped_512x288_5fold/fold5_best.pth --tta --image-hw 512 288 --out eval_results/resnetsamgrouped512x288_5fold_ensemble_test_metrics.json

Both printed lines must show `n=704`. Omitting `--image-hw 512 288` would silently evaluate at 224 by 224 and give wrong numbers.

Milestone 4, from the repository root, after the edits. Run a scratchpad script that sets the working directory to the repository root, calls `inference.load_selected_models(device)`, opens one real held-out mammogram (the first path in `MammogramRawDataset(["mass_test"]).samples`), calls `inference.predict` on it, prints the returned dict, and prints the wall-clock time of `predict` averaged over 10 calls after one warm-up call. Expect a dict with `resnet`, `efficientnet` and `final` keys, each with a `label` of "Benign" or "Malignant". Then use Flask's test client (`app.app.test_client()`) to POST that image as the `mammogram` form field to `/`. Expect status 200 and the text "Benign" or "Malignant" in the body. POST with no file, and again with a `.txt` file, and expect status 200 with the message "Please upload a .jpg, .jpeg, or .png image." in both cases.

## Validation and Acceptance

Milestone 1 is accepted when `py ResNet.py --help` prints both options without training, the patient-disjointness check reports 0 shared patients for all 5 folds, the transform shape check prints `[3, 512, 288]`, and `git diff` shows only the planned edits.

Milestone 2 is accepted when `resnet_sam_grouped_512x288_5fold.err.log` is empty and `checkpoints_resnet_sam_grouped_512x288_5fold/` holds exactly five files `fold1_best.pth` to `fold5_best.pth`, each 107,226,339 bytes (the size of every current-architecture ResNet checkpoint). In addition, the "Fold Summary" sheet of `training_logs/resnet_sam_grouped_512x288_5fold_training_log.xlsx` must have five data rows, and the log must end with the final results block. Record the per-fold TTA F1s, mean validation F1 and wall-clock time.

Milestone 3 is accepted when both evaluations print `n=704` and their JSON files exist. Record accuracy, precision, recall and F1 for both, next to three reference points: the served checkpoint (acc 0.5866, F1 0.5544 at threshold 0.45); the historical ResNet baseline (acc 0.6776, F1 0.6534, single fold, 224 by 224, leaky folds, no longer reproducible); and the best EfficientNet (512 by 288 ensemble acc 0.7088, F1 0.6472). State plainly whether the new ResNet beats each one.

Milestone 4 is accepted when the prediction script prints a well-formed result dict with an average `predict` time below 2 seconds, and the Flask test-client checks behave as described. In addition, `eval_results/selected_models.json` must point `resnet` at the five new checkpoints with `"image_hw": [512, 288]`, while `git diff eval_results/selected_models.json` shows the `efficientnet` entry unchanged.

Non-regression: among tracked files, only `ResNet.py`, `inference.py`, `eval_results/selected_models.json`, `specs/synthetic-augmentation-f1-improvement.md` and this plan may change, plus new output files. `DataSetAugmentation.py`, `EfficientNet.py`, `evaluate.py`, `losses.py`, `optimizers.py`, `app.py` and `training_log.py` must be unchanged. The contents and modification times of `checkpoints_resnet/`, `checkpoints_efficientnet/` and every `checkpoints_efficientnet_*` directory must not change.

## Idempotence and Recovery

Source edits are safe to repeat. `git diff` shows exactly what changed, and `git checkout -- inference.py` or `git checkout -- eval_results/selected_models.json` restores a committed file. `ResNet.py` has pre-plan uncommitted edits, so copy it to the scratchpad before Milestone 1; restore it from that copy rather than from git. Use no history-rewriting git commands. The training script cannot resume. A killed run restarts from fold 1: move its partial directory, log and workbook aside to names ending `.aborted-<date>` instead of deleting them, and never evaluate a partial run as this plan's result. Evaluation is deterministic and safe to rerun. If the web app misbehaves after Milestone 4, `git checkout -- eval_results/selected_models.json inference.py` returns it to its pre-plan state (which serves the broken checkpoint, but runs).

## Artifacts and Notes

Served checkpoint, measured 2026-09-27:

    [resnet] checkpoints=['checkpoints_resnet\\fold3_best.pth'] tta=True threshold=0.45 n=704 acc=0.5866 prec=0.4801 rec=0.6558 f1=0.5544

Memory and speed probe, 2026-09-27 (random tensors 3 x 512 x 288, SAM, layer3 and layer4 trainable):

    bs=16 out=(2, 2) peak_MB=1526 sec_per_step=0.201 est_train_min_per_epoch=0.5
    bs=8 out=(2, 2) peak_MB=996 sec_per_step=0.106 est_train_min_per_epoch=0.5

## Interfaces and Dependencies

No new dependency. `StratifiedGroupKFold` is part of the already-required scikit-learn. At the end of the plan, `ResNet.main(log_name="resnet")` takes one optional argument. `ResNet.CFG` gains `IMAGE_HW` (a `(height, width)` tuple, default `(224, 224)`). `ResNet.ResNetWithAttnPool` is unchanged, so `evaluate.load_model("resnet", ...)` loads old and new checkpoints alike. `inference.load_selected_models(device)` returns `{name: (model_or_list, accuracy, threshold, image_hw_or_None)}`. `inference.predict(image, models)` keeps its signature and return shape. An entry in `eval_results/selected_models.json` may carry an optional `"image_hw": [H, W]`; without it, 224 by 224 is used.
