# Train ResNet-50 and EfficientNet-B0 at 1024 by 576 and serve the better resolution

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not check in a copy of PLANS.md. The authoring and execution rules for this document live at `~/.agents/PLANS.md` on the machine of whoever is running this plan (a global, agent-agnostic file, not part of this git repository). This document must be maintained in accordance with that file.

This plan builds on two checked-in ExecPlans: `specs/patient-grouped-cv-and-higher-resolution.md` (EfficientNet-B0 at 512 by 288) and `specs/resnet-patient-grouped-512x288-and-app-fix.md` (ResNet-50 at 512 by 288, per-model image sizes in the web app). Everything needed from them is restated below.

## Approval

Status: Pending approval

Approved by: pending

Date: pending

Requested scope: Milestone 1 (measurements that confirm or lower the resolution), two sequential multi-hour training runs (Milestone 2), held-out evaluation (Milestone 3), and updating the web app's ResNet entry under the rule in the Decision Log (Milestone 4). The app's EfficientNet entry is changed only if the repository owner says so when approving; the default is to leave it.

## Purpose / Big Picture

This project classifies mammograms as benign or malignant with a ResNet-50 and an EfficientNet-B0, and serves both through a Flask web app (`py app.py`). Raising the training image size from 224 by 224 to 512 by 288 (a shape that keeps the mammograms' natural 1.7-to-1 height-to-width ratio) raised EfficientNet's held-out accuracy by about 0.04. It also helped make the new ResNet the project's best model: held-out accuracy 0.7131, F1 0.6599, recall 0.7101. It cost no extra training time, because training was limited by decoding the large JPEGs on the CPU, not by the GPU.

The source images are about 3,000 by 5,100 pixels. At 512 by 288, each pixel of the network's input still covers about ten source pixels along each side, roughly 0.5 mm of breast tissue. That is coarser than many microcalcifications (roughly 0.1 to 0.5 mm), which are one of the two lesion types in this dataset. The recommended next size is 1024 by 576: twice as fine along each side (about 0.25 mm per pixel), with the same shape, and four times the pixels of 512 by 288. It is the largest size likely to fit this project's 8 GB GPU with both models, and it is in the range published whole-mammogram classifiers use (roughly 1,000 pixels on the long side and up). Going higher (1536 by 864 and up) would need much smaller batches or gradient accumulation, which is a code change and out of scope.

After this plan, both models will have been trained and measured at 1024 by 576 with the same folds and recipe as at 512 by 288. Anyone can compare the two sizes on identical terms. The web app will serve whichever ResNet size did better on validation data. The observable result is a table of held-out accuracy, precision, recall and F1 for both models at both sizes, plus a working app.

## Progress

- [ ] Milestone 1: measure GPU memory, GPU time and CPU image-loading cost at 768 by 432 and 1024 by 576 for both models, and fix the final size and batch size by the rule in the Decision Log.
- [ ] Milestone 2a: train EfficientNet-B0 at the chosen size (background run).
- [ ] Milestone 2b: train ResNet-50 at the chosen size (background run, after 2a finishes).
- [ ] Milestone 3: evaluate both on the held-out set (single best fold and five-fold ensemble).
- [ ] Milestone 4: apply the serving rule to the app's ResNet entry, verify the app end to end, and write the comparison.

## Surprises & Discoveries

- (2026-09-28, planning) The permission checker for shell commands returned transient errors while this plan was written, so the memory and speed probes planned below could not be run beforehand. They are Milestone 1 instead. The estimates in this plan are scaled from measured 512 by 288 numbers. ResNet-50 at batch 16 peaked at 1,526 MB with about 0.5 minutes of GPU time per epoch. EfficientNet-B0 at batch 8 peaked at 552 MB. Both runs took about 2.1 to 2.5 minutes per epoch in practice, limited by JPEG decoding. Scaling by four times the pixels suggests ResNet-50 at 1024 by 576 needs about 6 GB at batch 16 (too much, since about 5 GB is free) and about 3 GB at batch 8, while EfficientNet-B0 needs about 2.2 GB at batch 8.

## Decision Log

- Decision: target 1024 by 576 (`--image-hw 1024 576`) for both models, falling back to 768 by 432 if Milestone 1 shows 1024 by 576 does not fit or is too slow.
  Rationale: see Purpose. Doubling the linear resolution is the natural next step after a change that helped. It keeps the aspect ratio and the same pretrained backbones (both networks are fully convolutional with global pooling, so any input size works without architecture changes). 768 by 432 (2.25 times the pixels of 512 by 288) is the safe intermediate.
  Date/Author: 2026-09-28, drafted by agent.
- Decision: batch size 8 for both models at 1024 by 576.
  Rationale: the scaled estimate of about 6 GB for ResNet-50 at batch 16 would not fit next to the roughly 3 GB the desktop uses. Batch 8 is what EfficientNet already used at 512 by 288. For ResNet this is a second change alongside resolution (it used batch 16 at 512 by 288), accepted because there is no memory-safe alternative at this size. Learning rates are left unchanged, as they were for EfficientNet's move to batch 8.
  Date/Author: 2026-09-28, drafted by agent.
- Decision (Milestone 1 rule): for each model, use 1024 by 576 at batch 8 if the probe completes without an out-of-memory error, peak allocated memory is at most 4,500 MB, and the projected run time is at most 16 hours. The projection is the larger of GPU minutes per epoch and measured loading minutes per epoch, times 35 epochs (EfficientNet) or 55 epochs (ResNet), times 5 folds. If any condition fails, use 768 by 432 at batch 16 under the same three conditions. If that also fails, stop and ask the repository owner.
  Rationale: 4,500 MB leaves headroom on the 8 GB card shared with the desktop. The time cap keeps each run overnight-sized. The epoch counts are each script's maximum, so the projection is an upper bound (early stopping usually ends folds sooner).
  Date/Author: 2026-09-28, drafted by agent.
- Decision: change nothing else. Keep the patient-grouped folds (`StratifiedGroupKFold`, seed 42), light augmentation, MixUp off, SAM, losses, learning rates, unfreeze schedules and epoch limits. Make no code changes; both training scripts already accept `--image-hw` and `--batch-size`, `evaluate.py` accepts `--image-hw`, and `inference.py` reads a per-model `image_hw` from `eval_results/selected_models.json`.
  Rationale: one variable (resolution) plus the forced batch-size change for ResNet. Because the folds are identical to the 512 by 288 runs, validation scores are directly comparable across sizes.
  Date/Author: 2026-09-28, drafted by agent.
- Decision (serving rule): the app's ResNet entry switches to the new five-fold ensemble only if the new run's mean validation F1 with test-time augmentation (the mean of the five "best-val F1 (with TTA)" lines in its log) is higher than the 512 by 288 run's 0.7109. Otherwise it stays on the 512 by 288 ensemble. Held-out numbers are reported either way but do not decide. The EfficientNet entry is only changed if the repository owner asks at approval time, by the same rule against its 512 by 288 run's mean of 0.6787.
  Rationale: choosing between sizes by test-set score would make the reported test score optimistic. The validation folds are identical across sizes and patient-grouped, so they are a fair judge. EfficientNet is left to the owner because at 512 by 288 it traded recall for precision.
  Date/Author: 2026-09-28, drafted by agent.
- Decision: run the two trainings one after the other, EfficientNet first, each as a background process managed by the coding agent's tooling.
  Rationale: the 8 GB GPU cannot hold both. EfficientNet goes first because it is the cheaper, lower-risk run, and its timing refines the ResNet projection. A closed terminal destroyed an earlier run, so no run is started in a manual window.
  Date/Author: 2026-09-28, drafted by agent.

## Outcomes & Retrospective

(Nothing yet.)

## Context and Orientation

All commands run from the repository root, `A:\Projects\BreastCancerDetectionCNN`, in PowerShell, using `py` (the Windows Python launcher). The machine has an NVIDIA RTX 3070 Ti with 8 GB, of which the desktop uses about 3 GB. Model weights (`*.pth`) are ignored by git.

`EfficientNet.py` trains EfficientNet-B0 with 5-fold patient-grouped cross-validation. With `--image-hw H W --batch-size B` it writes to `checkpoints_efficientnet_sam_grouped_{H}x{W}_5fold/best_model_fold_{k}.pth`, `plots_efficientnet_sam_grouped_{H}x{W}_5fold/` and `training_logs/efficientnet_sam_grouped_{H}x{W}_5fold_training_log.xlsx`. Maximum 35 epochs per fold. `ResNet.py` does the same for ResNet-50, writing `checkpoints_resnet_sam_grouped_{H}x{W}_5fold/fold{k}_best.pth` and `training_logs/resnet_sam_grouped_{H}x{W}_5fold_training_log.xlsx`, with a maximum of 55 epochs per fold. Both print, per fold, lines of the form "Fold k best-val F1 (with TTA): x". TTA (test-time augmentation) means averaging predictions over the image and its horizontal and vertical flips.

`evaluate.py` scores checkpoints on the 704-image held-out test set (428 benign, 276 malignant): `py evaluate.py --model resnet|efficientnet --checkpoint <one or more .pth> --tta --image-hw H W --out <json>`. With several checkpoints it averages them as an ensemble. `--image-hw` must equal the training size.

`eval_results/selected_models.json` tells the web app what to serve. Currently `resnet` lists the five `checkpoints_resnet_sam_grouped_512x288_5fold/fold{k}_best.pth` with `"accuracy": 0.7131, "threshold": 0.5, "image_hw": [512, 288]`, and `efficientnet` lists the old `checkpoints_efficientnet/best_model_fold_{1..5}.pth` (224 by 224, no `image_hw`). `inference.py` resizes each upload to each model's `image_hw`.

Reference results at 512 by 288 on the held-out set (ensembles, TTA, threshold 0.5): ResNet-50 acc 0.7131, prec 0.6164, rec 0.7101, F1 0.6599 (mean validation TTA F1 0.7109, run time 7h 17m). EfficientNet-B0 acc 0.7088, prec 0.6164, rec 0.6812, F1 0.6472 (mean validation TTA F1 0.6787, run time about 6h 18m). A 704-image test set gives an F1 sampling error of very roughly 0.02 to 0.03.

## Plan of Work

There are no source edits. Milestone 1 writes two throwaway scripts in the scratchpad. The first builds each model with its latest unfreeze stage trainable (the most memory-hungry point of training: ResNet `layer3` and `layer4`; EfficientNet `features.5` to `features.8`), runs a few real SAM training steps on random tensors of the candidate size, and prints peak allocated memory and GPU minutes per epoch (seconds per step times 2,291 training images divided by the batch size). The second decodes 40 real training JPEGs and times `get_light_train_transforms((H, W))` on them, to estimate loading minutes per epoch. Milestone 2 launches the two trainings. Milestone 3 runs `evaluate.py`. Milestone 4 edits only the `resnet` entry of `eval_results/selected_models.json` (and the `efficientnet` entry only if the owner asked), then checks the app.

## Concrete Steps

Milestone 1, from the scratchpad: for each model, and for 1024 by 576 at batch 8 (and, if needed, 768 by 432 at batch 16), run the probe script. Expected output lines look like:

    resnet 1024x576 bs=8 peak_MB=3100 gpu_min_per_epoch=2.3

Apply the Milestone 1 rule from the Decision Log and record the chosen size and batch size per model in the Decision Log. Then, from the repository root (confirm with `nvidia-smi` that no training process is running and that the target checkpoint directories do not exist), start Milestone 2a in the background:

    py -u EfficientNet.py --folds 5 --image-hw 1024 576 --batch-size 8 > efficientnet_sam_grouped_1024x576_5fold.log 2> efficientnet_sam_grouped_1024x576_5fold.err.log

When it ends with the "Final Cross-Validation Results" block, start Milestone 2b:

    py -u ResNet.py --image-hw 1024 576 --batch-size 8 > resnet_sam_grouped_1024x576_5fold.log 2> resnet_sam_grouped_1024x576_5fold.err.log

(Substitute 768 432 and batch 16 where Milestone 1 chose the fallback.)

Milestone 3, for each model: pick the single best fold K as the fold with the highest "best-val F1 (with TTA)" in its log, then run

    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_sam_grouped_1024x576_5fold/best_model_fold_K.pth --tta --image-hw 1024 576 --out eval_results/samgrouped1024x576_efficientnet_5fold_singlebest_test_metrics.json
    py evaluate.py --model efficientnet --checkpoint (Get-ChildItem checkpoints_efficientnet_sam_grouped_1024x576_5fold\*.pth).FullName --tta --image-hw 1024 576 --out eval_results/samgrouped1024x576_efficientnet_5fold_ensemble_test_metrics.json
    py evaluate.py --model resnet --checkpoint checkpoints_resnet_sam_grouped_1024x576_5fold/foldK_best.pth --tta --image-hw 1024 576 --out eval_results/resnetsamgrouped1024x576_5fold_singlebest_test_metrics.json
    py evaluate.py --model resnet --checkpoint (Get-ChildItem checkpoints_resnet_sam_grouped_1024x576_5fold\*.pth).FullName --tta --image-hw 1024 576 --out eval_results/resnetsamgrouped1024x576_5fold_ensemble_test_metrics.json

Every printed line must show `n=704`.

Milestone 4: apply the serving rule. If the ResNet switches, set its entry to the five new checkpoints, the new ensemble's held-out accuracy (four decimals), `"threshold": 0.5` and `"image_hw": [1024, 576]`. Then rerun the scratchpad app check from the previous plan (`m4_app_check.py --flask`). It loads the selected models, predicts on the first `mass_test` image 10 times, and uses Flask's test client to post an image, no file, and a `.txt` file.

## Validation and Acceptance

Milestone 1 is accepted when the chosen size and batch size for each model, with the measured peak memory and projected hours behind them, are written into the Decision Log. Milestone 2 is accepted per run when its error log is empty and its checkpoint directory has exactly five files, of 16,342,523 bytes (EfficientNet) or 107,226,339 bytes (ResNet). In addition, its workbook's "Fold Summary" sheet must have five rows, and its log must end with the final results block. Milestone 3 is accepted when all four evaluations print `n=704`. Milestone 4 is accepted when the app check prints a well-formed result, the average `predict` time stays under 2 seconds, the upload returns a verdict, and the no-file and `.txt` cases return the "Please upload a .jpg, .jpeg, or .png image." message. `git diff eval_results/selected_models.json` must show only the intended entry changed. The Outcomes section must hold a table of both models at 512 by 288 and at the new size, giving mean validation TTA F1, held-out accuracy, precision, recall and F1 for single best and ensemble, and wall-clock time, with a plain statement of whether resolution helped each model.

Non-regression: no tracked source file changes. The 512 by 288 checkpoint directories, `checkpoints_resnet/` and `checkpoints_efficientnet/` keep their contents and modification times.

## Idempotence and Recovery

Probes and evaluations are safe to rerun. A training run cannot resume. If one is killed, move its partial checkpoint directory, plots directory, log files and workbook aside to names ending `.aborted-<date>`, and restart it from fold 1. Never evaluate a partial run as this plan's result. If the app misbehaves after Milestone 4, `git checkout -- eval_results/selected_models.json` returns it to the committed version. Note that the currently served 512 by 288 ResNet entry is not yet committed, so first copy the file to the scratchpad before editing it and restore from that copy.

## Artifacts and Notes

(Probe output, run timings and evaluation lines go here as they are produced.)

## Interfaces and Dependencies

No new dependencies and no interface changes. This plan uses the existing `--image-hw` and `--batch-size` options of `EfficientNet.py` and `ResNet.py`, the `--image-hw` option of `evaluate.py`, and the optional `image_hw` field read by `inference.load_selected_models`.
