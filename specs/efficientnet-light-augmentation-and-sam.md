# Retrain EfficientNet-B0 with lighter image augmentation and a Sharpness-Aware Minimisation optimiser

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not check in a copy of PLANS.md. The authoring and execution rules for this document live at `~/.agents/PLANS.md` on the machine of whoever is running this plan (a global, agent-agnostic file, not part of this git repository). This document must be maintained in accordance with that file.

This plan builds on four prior ExecPlans already checked into this repository under `specs/`: `model-improvement-and-comparison-app.md` (built the shared training, evaluation and inference infrastructure and the Flask web app), `expanded-training-data-and-f1-improvements.md` (added per-model decision-threshold tuning and checkpoint ensembling), `synthetic-augmentation-f1-improvement.md` (strengthened the training-image transform pipeline that this plan now partially reverses), and `efficientnet-kfold-comparison.md` (added the `--folds` flag to `EfficientNet.py` and measured 3-fold, 5-fold and 10-fold cross-validation against the held-out test set; it supplies the baseline numbers this plan compares against). This plan does not repeat those documents' contents. Every fact from them that is needed to follow this plan is restated below, so this document can be followed on its own.

## Approval

Status: Approved

Approved by: Repository owner (chat instruction: "approved")

Date: 2026-09-26

Requested scope: All four milestones as written below, including the full multi-hour 5-fold training run in Milestone 3 and the held-out test-set evaluation in Milestone 4.

## Purpose / Big Picture

Right now, `EfficientNet.py` trains an EfficientNet-B0 image classifier that tells benign mammograms apart from malignant ones, and it trains that classifier under a heavy stack of random image distortions plus a plain AdamW optimiser. Two things are wrong with that, and both are visible in this repository's own recorded results rather than being a matter of opinion.

First, the image distortions are too strong. The training pipeline currently applies, to every training image, a random horizontal flip, a random affine warp (up to 15 degrees of rotation, 10 percent translation, 0.9x-1.1x scaling and 5 degrees of shear), a random brightness and contrast jitter of plus or minus 25 percent, and a random erasing of a rectangular patch covering up to 15 percent of the image with probability 0.3 — and on top of all that, MixUp, which blends each training image with another random training image and blends their labels to match. The training logs from the most recent runs show the model never manages to fit even its own training data: in `efficientnet_kfoldsweep_10fold.log` the training loss plateaus around 0.56 to 0.62 and never falls below roughly 0.55 across a full 35-epoch fold. A model that cannot drive its training loss down is underfitting, and the standard cause of underfitting in a pipeline like this is too much regularisation, of which random image distortion is the largest single component here. The held-out numbers agree: the previous plan (`specs/efficientnet-kfold-comparison.md`) records that after this heavy augmentation was introduced, held-out F1 on the untouched test set fell to 0.5862 (single best fold) and 0.5735 (five-fold ensemble), down from the 0.6375 this project measured before the augmentation pipeline was strengthened.

Second, the optimiser leaves an improvement on the table that this repository has already implemented and proven elsewhere. `ResNet.py` in this same repository trains its ResNet-50 with Sharpness-Aware Minimisation, abbreviated SAM. `EfficientNet.py` does not use it at all; it uses plain AdamW. SAM is explained in full in the Context section below, but in one sentence: instead of taking the step that most reduces the loss at the current weights, SAM takes the step that most reduces the loss in the worst direction within a small neighbourhood around the current weights, which biases training towards flat regions of the loss surface and, empirically, generalises better on small datasets. This dataset is small — 2,864 training images — which is exactly the regime where SAM tends to help most.

After this plan, a person will be able to run one command from the repository root,

    py EfficientNet.py --folds 5

and get a complete five-fold cross-validation training run of EfficientNet-B0 in which the training images receive only a horizontal flip and a small rotation, MixUp is switched off, and every weight update is a SAM update. That run writes to brand-new directories (`checkpoints_efficientnet_sam_5fold/`, `plots_efficientnet_sam_5fold/`, `training_logs/efficientnet_sam_5fold_training_log.xlsx`) so that not one existing file, checkpoint, plot or log is overwritten — importantly including `checkpoints_efficientnet/`, which the live Flask web application currently serves predictions from. Then a second command will measure that run honestly against this project's untouched 704-image held-out test set, and this document will record the resulting F1 score next to the recorded baseline of 0.5862 single-fold / 0.5735 ensemble, stating plainly whether the change helped, hurt, or made no difference.

The observable proof of success is therefore two-part and concrete: a person can see the training loss in the new run's log fall clearly below the roughly 0.55 floor the current configuration is stuck at (evidence that the underfitting was caused by over-regularisation), and they can see a held-out F1 number printed by `evaluate.py` that can be compared, like for like, against 0.5862 and 0.5735.

## Progress

- [x] (2026-09-26) Milestone 1: add `get_light_train_transforms()` to `DataSetAugmentation.py` and create `optimizers.py` containing the `SAM` optimiser class. Verify both import cleanly and that the new transform produces a correctly shaped, correctly normalised tensor.
- [x] (2026-09-26) Milestone 2: wire the light transforms, SAM, `MIXUP_ALPHA = 0.0`, and the new `_sam` output directories into `EfficientNet.py`. Verify `py EfficientNet.py --help` prints the updated text without training, and verify with a synthetic-data smoke test that a SAM training step actually runs two forward/backward passes and changes the model's weights.
- [x] (2026-09-26) Milestone 3: run the full five-fold training run in the background and confirm it completes cleanly, with five checkpoints written and the training loss falling below 0.55.
- [x] (2026-09-26) Milestone 4: evaluate the new run against the 704-image held-out test set (single best fold and full five-fold ensemble), record the numbers in Outcomes & Retrospective, and state plainly whether the change helped.

## Surprises & Discoveries

- (2026-09-26, M1) `DataSetAugmentation.py`, `EfficientNet.py` and `ResNet.py` use CRLF line endings, and `python` is not on PATH (only `py`). Edits were made preserving CRLF. `git diff --stat` on `DataSetAugmentation.py` shows 3 deletions because that file already had uncommitted changes before this plan; diffing against a pre-edit copy confirmed this plan added exactly 9 lines and removed none. `optimizers.py` was written with LF endings.
- (2026-09-26, M1) Verification: `imports ok`; light transform on a grey 600x800 image gave `torch.Size([3, 224, 224]) torch.float32 -2.1179 0.4265`.
- (2026-09-26, M2) `py EfficientNet.py --help` prints the updated `_sam` help without training. SAM smoke test: `optimizer type: SAM`, `epoch loss 0.4950 | forward passes 4 | max weight change 0.002001`, `SAM smoke test passed`.
- (2026-09-26, M3) Milestone 3 launched 04:28 as a background job: `py -u EfficientNet.py --folds 5`. Fold 1 epochs 1-3 train loss: 0.5207, 0.5067, 0.5000 (about 2.5 minutes per epoch). Caveat for interpreting the "train loss falls below 0.55" signal: the baseline's train loss was computed on MixUp-blended soft labels, which inflates it, while this run has MixUp off. Part of any drop below 0.55 is therefore a change in what is being measured, not only better fit; the held-out F1 in Milestone 4 is the deciding evidence.

## Decision Log

- Decision: cut the training augmentation down to `Resize(224, 224)`, `RandomHorizontalFlip(p=0.5)`, `RandomAffine(degrees=10)`, `ToTensor()`, `Normalize(ImageNet statistics)`, and additionally set `MIXUP_ALPHA` to 0.0 — dropping `ColorJitter`, the translate/scale/shear components of `RandomAffine`, `RandomErasing`, and MixUp entirely.
  Rationale: the repository owner chose this level explicitly when offered a minimal, a moderate, and a no-augmentation-at-all option. Beyond that preference, each dropped item has a specific reason. `ColorJitter(brightness=0.25, contrast=0.25)` randomly rescales pixel intensity, but in a mammogram the absolute intensity and the local contrast of a calcification cluster against surrounding tissue *is* the diagnostic signal, so jittering it by up to 25 percent destroys information rather than simulating a real-world variation. `RandomErasing(p=0.3, scale=(0.02, 0.15))` blanks out a rectangle of up to 15 percent of the image; because these are whole 224x224 mammograms in which the lesion is small, that rectangle can land on the lesion and leave an image labelled "malignant" with no visible malignancy in it, which is label noise, not augmentation. Translation, scaling and shear are dropped because a horizontal flip plus a small rotation already covers the realistic variation in how a breast is positioned for imaging, and each additional random warp costs training signal. MixUp is switched off because blending two mammograms produces an image that is not a mammogram, and because SAM (added by this same plan) is itself a regulariser, so keeping a second strong regulariser risks reproducing the very underfitting this plan is trying to fix.
  Date/Author: 2026-09-25, drafted by agent per the repository owner's direct answer.
- Decision: implement the lighter pipeline as a new function, `get_light_train_transforms()`, in `DataSetAugmentation.py`, and leave the existing `get_train_transforms()` completely untouched. Only `EfficientNet.py` switches to the new function; `ResNet.py` continues to call `get_train_transforms()`.
  Rationale: the repository owner chose this explicitly. `get_train_transforms()` is shared by `ResNet.py`, which this plan does not retrain. Editing it in place would silently change ResNet-50's behaviour the next time anybody trains it, and would make its existing checkpoints in `checkpoints_resnet/` and its existing log `training_logs/resnet_training_log.xlsx` incomparable with any future run. This also satisfies this machine's standing instruction to preserve existing behaviour unless a plan explicitly calls for changing it.
  Date/Author: 2026-09-25, drafted by agent per the repository owner's direct answer.
- Decision: place the `SAM` class in a new file, `optimizers.py`, as a copy of the working implementation currently at `ResNet.py` lines 96 to 147, and do not modify `ResNet.py` to import from it.
  Rationale: `EfficientNet.py` needs SAM, and importing it from `ResNet.py` would make the EfficientNet training script depend on the ResNet training script, which also drags in ResNet's module-level side effect of creating `checkpoints_resnet/` on import. A new single-purpose module avoids that. `losses.py` was considered and rejected as the home for it, because an optimiser is not a loss function. Deliberately not refactoring `ResNet.py` to use the shared copy is a direct application of the surgical-change rule in `~/.agents/skills/code-generation-and-refactor-skill/SKILL.md`: `ResNet.py` is not broken, is not being retrained here, and touching it would risk a model this plan has no mandate over. The cost is one duplicated class, which is acknowledged here rather than hidden; note that `ResNet.py` already duplicates `FocalLoss` and `mixup_data` from `losses.py`, so a duplicate is the pre-existing pattern in this repository rather than a new deviation.
  Date/Author: 2026-09-25, drafted by agent.
- Decision: copy `ResNet.py`'s SAM implementation as it stands, including the fact that it does not freeze BatchNorm running statistics during SAM's second forward pass.
  Rationale: the canonical SAM reference implementation disables BatchNorm running-statistic updates during the second of its two forward passes, because those statistics would otherwise be measured at the deliberately-perturbed weights rather than the real ones. `ResNet.py`'s implementation omits that refinement, and EfficientNet-B0 does contain BatchNorm layers, so the omission is real here too. It is nevertheless being kept, for two reasons: it is the implementation that produced this project's only existing SAM results, so keeping it makes this run comparable with `ResNet.py`'s; and adding the refinement would be an unrequested change to behaviour, which this repository's rules forbid without approval. This is recorded as a known, deliberate deviation from the canonical implementation and an obvious candidate for a separate follow-up experiment. It is explicitly out of scope for this plan.
  Date/Author: 2026-09-25, drafted by agent.
- Decision: write all output from this plan's training run to new directories carrying a `_sam` marker, never to the existing ones. With no `--folds` flag the destinations are `checkpoints_efficientnet_sam/`, `plots_efficientnet_sam/` and `training_logs/efficientnet_sam_training_log.xlsx`; with `--folds N` they are `checkpoints_efficientnet_sam_{N}fold/`, `plots_efficientnet_sam_{N}fold/` and `training_logs/efficientnet_sam_{N}fold_training_log.xlsx`.
  Rationale: the repository owner chose this explicitly. Two concrete hazards make it necessary. `eval_results/selected_models.json` currently names the five files `checkpoints_efficientnet/best_model_fold_1.pth` through `best_model_fold_5.pth`, and `inference.py` loads exactly those on behalf of the live Flask app in `app.py`; overwriting them mid-run would leave the running app serving predictions from a half-written or differently-configured model. Separately, `checkpoints_efficientnet_5fold/` holds the non-SAM, heavy-augmentation five-fold run whose held-out F1 of 0.5862 and 0.5735 are the baseline this plan is measured against — overwriting it would destroy the comparison this plan exists to make.
  Date/Author: 2026-09-25, drafted by agent per the repository owner's direct answer.
- Decision: make the lighter augmentation and SAM the unconditional new behaviour of `EfficientNet.py`, rather than adding command-line switches to toggle them.
  Rationale: the request was to change how this model trains, not to build a configuration matrix. Adding `--sam` / `--no-sam` and `--light-aug` / `--heavy-aug` switches would be speculative configurability that nobody asked for, which the code-generation guidance at `~/.agents/skills/code-generation-and-refactor-skill/SKILL.md` specifically warns against. The previous configuration remains fully reproducible regardless, because `get_train_transforms()` is left in place unmodified and git history holds the previous `EfficientNet.py`. For the same reason, the two new settings are exposed as plain constants (`USE_SAM`, `SAM_RHO`) on the existing `EfficientNetConfig` class, matching how every other knob in that file is already expressed and how `ResNet.py`'s `CFG` class expresses the identical pair of settings.
  Date/Author: 2026-09-25, drafted by agent.
- Decision: leave every other regularisation and training setting in `EfficientNetConfig` exactly as it is — the classifier's `Dropout(p=0.3)`, `LABEL_SMOOTHING = 0.08`, `FOCAL_GAMMA = 1.5`, `FOCAL_WEIGHT = 0.4`, `GRAD_CLIP_NORM = 1.5`, `WEIGHT_DECAY = 1e-5`, all four learning rates, the unfreeze schedule, the cosine-annealing schedule, `NUM_EPOCHS = 35`, `EARLY_STOP_PATIENCE = 10` and `MIN_EPOCH_FOR_BEST = 6`.
  Rationale: the request named two changes, lighter image transforms and SAM. Changing further knobs in the same run would make the result uninterpretable — if held-out F1 moves, nobody could say which change moved it. Dropout and label smoothing are regularisers too and are plausible follow-up targets if this run still underfits, but they are not image transforms and they were not requested, so they stay.
  Date/Author: 2026-09-25, drafted by agent.
- Decision: verify with a full five-fold run followed by held-out test-set evaluation, and budget roughly 11 to 14 hours of wall-clock time for it.
  Rationale: the repository owner chose the full run over a quick smoke test. The budget follows from measurement rather than guesswork: `specs/efficientnet-kfold-comparison.md` records that the non-SAM five-fold run took 6 hours 18 minutes on this machine's RTX 3070 Ti, and SAM performs two forward and two backward passes per batch instead of one, so roughly double is the honest expectation. The lighter transform pipeline removes some CPU-side work per image, which may claw a little of that back, and early stopping may end some folds before all 35 epochs.
  Date/Author: 2026-09-25, drafted by agent per the repository owner's direct answer.
- Decision: launch the long training run as a background process managed by the coding agent's own tooling, never in a manually-opened terminal window, and never two training processes at once.
  Rationale: this is pre-existing, hard-won project practice. `specs/synthetic-augmentation-f1-improvement.md` records that a multi-hour ResNet-50 retrain was destroyed on 2026-08-27 when the terminal window running it was closed, aborting the process mid-fold-3 with a Fortran runtime error and losing every fold of progress. Additionally, the single available GPU has 8 GB of memory and cannot comfortably hold two of these training processes simultaneously.
  Date/Author: 2026-09-25, drafted by agent.

## Outcomes & Retrospective

### Milestone 3 outcome (2026-09-26): 5-fold training run

`py -u EfficientNet.py --folds 5` ran cleanly from 04:28 to about 11:03, roughly 6h 36m, which is much closer to the baseline's 6h 18m than the 11-14h estimate (the light transforms and SAM's second pass largely offset each other). `efficientnet_sam_5fold.err.log` is 0 bytes. No fold early-stopped (175 epochs logged, 5 x 35). `checkpoints_efficientnet_sam_5fold/` holds exactly five checkpoints of 16,342,523 bytes each, and the Excel "Fold Summary" sheet has 5 data rows. Per-fold validation F1 with TTA: 0.7269, 0.7112, 0.7320, 0.6753, 0.7186. The script's own summary: Mean Val Loss 0.4953 +/- 0.0069, Mean Val Accuracy 0.7507 +/- 0.0141, Mean Val F1 0.7200 +/- 0.0337. Train loss by the late epochs of fold 5 was about 0.32, well below the old 0.55 floor (with the MixUp caveat recorded in Surprises & Discoveries).

### Milestone 4 outcome (2026-09-26): held-out evaluation and verdict

Both evaluations ran on the same 704 images (n=704), with TTA and the default 0.5 threshold. The single best fold was chosen by validation TTA F1 only (fold 3, 0.7320), not by test data.

    Configuration                          Mean CV F1         Held-out single-best F1 (acc)   Held-out ensemble F1 (acc)   Wall-clock
    Heavy augmentation + AdamW (baseline)  0.5444 +/- 0.0643  0.5862  (0.6349)                0.5735  (0.6577)             6h 18m
    Light augmentation + SAM (this plan)   0.7200 +/- 0.0337  0.6552  (0.6861)                0.6460  (0.6761)             ~6h 36m

Single best fold (fold 3): precision 0.5753, recall 0.7609. Five-fold ensemble: precision 0.5652, recall 0.7536.

Verdict: the change helped. Held-out F1 rose by about 0.069 for the single best fold and 0.073 for the ensemble, and both now exceed the project's earlier best EfficientNet held-out F1 of 0.6375. Training time was essentially unchanged. Two cautions. First, mean CV F1 (0.72) is much higher than held-out F1 (0.65 to 0.66), so cross-validation still overstates real-world performance; the held-out number is the one to trust. Second, this plan changed two things at once (light transforms and SAM), so it cannot say how much each contributed; the ensemble also did not beat the single best fold, as in the baseline. A follow-up isolating the two changes (SAM with heavy augmentation, or light augmentation with AdamW) would answer that. Also unchanged and out of scope: `eval_results/selected_models.json` still points the live app at the old `checkpoints_efficientnet/` checkpoints; repointing it is the repository owner's decision. Non-regression: only the already-modified tracked files show as modified, and the modification times of `checkpoints_efficientnet/`, `checkpoints_efficientnet_5fold/` and `selected_models.json` are unchanged.

## Context and Orientation

This section explains everything about the repository and the techniques that a newcomer needs in order to carry out the plan. Read it before editing anything.

### What this project is, and where its pieces live

This is a breast-cancer screening research project. It trains image classifiers to label a mammogram as either benign (class 0) or malignant (class 1). Everything runs from the repository root, which on the machine this plan was written on is `A:\Projects\BreastCancerDetectionCNN`. Python is invoked as `py` on this machine (the Windows Python launcher); the installed environment has PyTorch 2.10.0 with CUDA 12.6, TorchVision 0.25.0, and a working NVIDIA GeForce RTX 3070 Ti, which `torch.cuda.is_available()` reports as `True`.

The files that matter to this plan are these.

`DataSetAugmentation.py` holds the data layer. A `Config` class holds shared constants, notably `IMAGE_SIZE = 224`. `MammogramRawDataset` is a PyTorch `Dataset` that reads the CBIS-DDSM metadata CSV files from `data/raw/csv/`, turns each row's `pathology` text field into a label (0 if the word "benign" appears in it, otherwise 1), locates the matching JPEG under `data/raw/jpeg/`, and returns `(PIL.Image, label)` pairs with no transformation applied. `TransformDataset` wraps any such dataset and applies a TorchVision transform pipeline to each image as it comes out. Finally, two functions build those pipelines: `get_train_transforms()` at line 139, which is the heavy pipeline this plan replaces for EfficientNet, and `get_val_transforms()`, which resizes to 224x224, converts to a tensor, and normalises — with no randomness at all, so that evaluation is repeatable.

`EfficientNet.py` is the EfficientNet-B0 training script and the main file this plan edits. Its `EfficientNetConfig` class holds every setting. `create_efficientnet()` loads ImageNet-pretrained EfficientNet-B0, freezes all of its weights, and replaces its classifier head with `nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, 2))`, leaving only that head trainable. `train_one_epoch()` runs one pass over the training data. `validate()` runs one pass over the validation data and returns loss, accuracy, precision, recall and F1. `get_optimizer()` builds an AdamW optimiser with different learning rates for different parts of the network, and `get_scheduler()` wraps it in a cosine-annealing-with-warm-restarts learning-rate schedule. `train_efficientnet_kfold()` ties it all together, and an `argparse` block at the bottom exposes the `--folds` flag.

`ResNet.py` is the equivalent script for ResNet-50. This plan does not modify it, but it is the source of the SAM implementation being copied: the `SAM` class occupies lines 96 to 147, and lines 320 to 360 show how a SAM-aware training epoch is written. Reading those two regions before starting Milestone 1 is strongly recommended.

`losses.py` holds shared training pieces: a `FocalLoss` class, a `mixup_data()` function, and `build_hybrid_criterion()`, which returns a callable computing `(1 - focal_weight) * CrossEntropy + focal_weight * Focal`. `EfficientNet.py` imports from it.

`evaluate.py` is the honest-measurement tool. Run as a script it loads one or more checkpoints, runs them over the held-out test set, and prints and saves accuracy, precision, recall and F1. It also exports two functions that `EfficientNet.py` imports and calls at the end of each fold: `evaluate_model()` and `find_best_threshold()`.

`training_log.py` writes an Excel workbook per run into `training_logs/`, with an "Epochs" sheet and a "Fold Summary" sheet.

`app.py`, `inference.py` and `eval_results/selected_models.json` are the live web application and the pointer file naming which checkpoints it serves. This plan must not disturb them, and does not.

### The data, and why the test set is sacred

`MammogramRawDataset(["mass_train", "calc_train"])` yields 2,864 labelled training images. `MammogramRawDataset(["mass_test", "calc_test"])` yields 704 held-out test images, of which 428 are benign and 276 malignant. Cross-validation splits only the 2,864. The 704 are never trained on, never validated on, and never used to pick a threshold or a checkpoint; they exist solely so that a final number can be trusted. This project treats the held-out F1 as the only trustworthy figure, and this plan keeps that discipline.

### What cross-validation means here, concretely

`train_efficientnet_kfold(n_folds=5, ...)` uses scikit-learn's `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` to cut the 2,864 training images into five equal groups with the benign/malignant ratio preserved in each. It then trains five completely independent models. Model number one trains on groups two through five (about 2,291 images) and is validated on group one (about 573 images); model number two trains on groups one, three, four, five and validates on group two; and so on. Each model saves its own best checkpoint, named `best_model_fold_{k}.pth`. The fixed `random_state=42` means the split is identical between runs, which is what makes this plan's run comparable with the recorded baseline.

### What "best checkpoint" means here

Within a fold, after every epoch, the code compares that epoch's validation loss against the lowest validation loss seen so far in that fold. If it is lower, and the epoch index is at least `MIN_EPOCH_FOR_BEST - 1`, the model's weights are written to `best_model_fold_{k}.pth`, overwriting any earlier save. `MIN_EPOCH_FOR_BEST = 6` combined with Python's zero-based loop means no checkpoint is written before the sixth epoch as printed in the log, which exists to avoid saving a lucky-looking model from the noisy early epochs while only the classifier head is training. If `EARLY_STOP_PATIENCE = 10` epochs pass with no new best, that fold stops early.

### What gradual unfreezing means here

`UNFREEZE_SCHEDULE = {4: [8, 7], 10: [6], 16: [5]}` says: for the first four epochs, train only the new classifier head while the whole pretrained backbone stays frozen; at epoch index 4, also start training backbone blocks 8 and 7 (`model.features[8]` and `model.features[7]`, the layers closest to the output); at epoch index 10 add block 6; at epoch index 16 add block 5. Each time this happens, the optimiser and the learning-rate schedule are rebuilt from scratch, and freshly-unfrozen blocks get a higher learning rate (`NEW_UNFREEZE_LR = 1e-4`) than blocks unfrozen at an earlier point in the schedule (`OLD_UNFREEZE_LR = 2e-5`), the idea being that layers which have already had time to adapt need smaller nudges. This plan does not change any of that, but the optimiser-rebuilding is why the SAM change has to be made in `get_optimizer()` and `get_scheduler()` rather than at one single construction site.

### What Sharpness-Aware Minimisation (SAM) is

Ordinary gradient descent asks: which small change to my weights most reduces the loss on this batch? It then makes that change. SAM, published by Foret and colleagues in 2021, asks a harder question: within a small ball of radius rho around my current weights, consider the single worst point — the weights in that neighbourhood with the highest loss — and make the change that most reduces the loss *there*. The motivation is that a set of weights sitting in a narrow, steep valley of the loss surface can score a very low training loss while being fragile, because the surface for unseen data is shifted slightly and a narrow valley shifted slightly is no longer a valley. Weights sitting in a broad, flat basin keep scoring well under such a shift. Optimising against the worst neighbour pushes training towards those flat basins.

Mechanically this costs two forward-and-backward passes per batch instead of one, and it works like this. Pass one: compute the loss and its gradients at the current weights. Then step the weights a distance rho *uphill*, in the direction the gradient points, and remember that displacement (`e_w`); this is `first_step()`, and it lands the weights on an approximation of the worst nearby point. Pass two: with the weights still perturbed, recompute the loss and its gradients — these are the gradients *at the worst neighbour*. Then subtract the remembered displacement to put the weights back exactly where they started, and finally let the wrapped base optimiser (AdamW here) apply its update using those second-pass gradients; this is `second_step()`. `SAM_RHO = 0.07` is the neighbourhood radius, copied from `ResNet.py`'s proven value, where it is documented as the "SAM neighbourhood size".

Two consequences matter for implementation. First, because SAM is a wrapper holding a real optimiser inside itself as `self.base_optimizer`, a learning-rate scheduler must be attached to that inner object, not to the wrapper. Second, `SAM` exposes `first_step()` and `second_step()` instead of a single `step()`, so any training loop using it has to be restructured, which is why `train_one_epoch()` is rewritten in Milestone 2.

### The baseline this plan is measured against

The non-SAM, heavy-augmentation five-fold run is already recorded, and its raw output is checked in. Its per-epoch log is `efficientnet_kfoldsweep_5fold.log`, its checkpoints are in `checkpoints_efficientnet_5fold/`, and its held-out numbers are in `eval_results/kfoldsweep_efficientnet_5fold_singlebest_test_metrics.json` and `eval_results/kfoldsweep_efficientnet_5fold_ensemble_test_metrics.json`. The figures, all measured on the same 704 images with test-time augmentation enabled and the default 0.5 decision threshold, are these.

    Single best fold (fold 5):   accuracy 0.6349   precision 0.5275   recall 0.6594   F1 0.5862
    Full five-fold ensemble:     accuracy 0.6577   precision 0.5606   recall 0.5870   F1 0.5735
    Script's own mean CV F1:     0.5444 +/- 0.0643
    Wall-clock training time:    6h 18m

Milestone 4 must produce exactly these four kinds of number for the new run and set them side by side.

## Plan of Work

### Milestone 1 — the two new building blocks

The goal of this milestone is that the repository contains a lighter training-transform pipeline and a SAM optimiser class, both importable, with nothing yet using them. Nothing about how any model trains changes in this milestone, which is deliberate: it can be completed and verified in a couple of minutes, and if something is wrong with either piece it surfaces before any multi-hour run is at stake.

First, add a new function to `DataSetAugmentation.py`. Place it immediately after the existing `get_train_transforms()` function and before `get_val_transforms()`, so that the three transform builders sit together. Do not modify `get_train_transforms()` in any way. The new function is this, written in the same style as its neighbours.

    def get_light_train_transforms():
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

Three details are worth stating explicitly. The normalisation mean and standard deviation must be copied character-for-character from the two existing functions in this same file, whatever they happen to be, so that a model trained with the new pipeline is normalised identically to how `get_val_transforms()` normalises its validation and test images; mismatching them would silently wreck evaluation while raising no error. `RandomAffine(degrees=10)` deliberately passes only `degrees`, leaving `translate`, `scale` and `shear` at their `None` defaults, which means rotation only. And there is no `RandomErasing` entry, which is why this list ends at `Normalize` rather than continuing past it — `RandomErasing` is the one transform in the old pipeline that operates on tensors rather than PIL images and therefore had to come last.

Second, create a new file, `optimizers.py`, at the repository root. It contains a module docstring, the imports it needs, and the `SAM` class copied verbatim from `ResNet.py` lines 96 to 147 — that is, `__init__`, `first_step`, `second_step`, `_grad_norm` and `load_state_dict`, unchanged in every respect. Do not retype the class from memory; copy it out of `ResNet.py`, so that no subtle difference creeps in. The only additions are a short module docstring noting where it came from and that `ResNet.py` retains its own copy. Read the class before copying and confirm exactly which names it uses at module scope; as written it needs only `import torch`.

Do not modify `ResNet.py` in this milestone or any other.

Verification. From the repository root, run

    py -c "from DataSetAugmentation import get_light_train_transforms, get_train_transforms, get_val_transforms; from optimizers import SAM; print('imports ok')"

and observe that it prints `imports ok` with no traceback. Then prove the new pipeline produces exactly what the model expects by running

    py -c "import torch; from PIL import Image; from DataSetAugmentation import get_light_train_transforms; t = get_light_train_transforms(); x = t(Image.new('RGB', (600, 800), color=(128, 128, 128))); print(x.shape, x.dtype, float(x.min()), float(x.max()))"

The expected output is a tensor of shape `torch.Size([3, 224, 224])` and dtype `torch.float32`. The minimum and maximum will be finite numbers roughly in the range -2.2 to 2.7 rather than 0 to 1, because normalisation has been applied; seeing values outside 0 to 1 is the confirmation that `Normalize` ran. A grey input rotated by up to 10 degrees will have black corners filled in, so the minimum will typically sit near the normalised value of black, around -2.1, which is expected and correct.

Finally run `git diff --stat` and confirm that exactly one tracked file changed, `DataSetAugmentation.py`, with roughly nine added lines and zero deleted, plus one new untracked file, `optimizers.py`. Any deletion in `DataSetAugmentation.py` means the existing `get_train_transforms()` was disturbed and must be restored.

### Milestone 2 — wire SAM, the light transforms, and the new output directories into `EfficientNet.py`

At the end of this milestone, `EfficientNet.py` trains with SAM and light transforms and writes to new directories, and that has been proven on synthetic data without committing to the long run. Everything in this milestone is confined to `EfficientNet.py`.

There are six edits, and they are described here in the order they appear in the file.

The import block at the top currently reads, on one line, `from DataSetAugmentation import MammogramRawDataset, TransformDataset, Config, get_train_transforms, get_val_transforms`. Change `get_train_transforms` to `get_light_train_transforms` in that list. Because `get_train_transforms` is referenced in exactly one place in this file (inside the fold loop, changed below), removing it from the import leaves no dangling reference; leaving it imported but unused would be an orphan, which the repository's code-generation guidance says to clean up. Then add one new import line, `from optimizers import SAM`, placed with the other local-module imports — that is, in the group that already contains the `DataSetAugmentation`, `losses`, `evaluate` and `training_log` imports.

In `EfficientNetConfig`, change `MIXUP_ALPHA` from `0.2` to `0.0`, and update its inline comment if it carries one, so the file does not claim a value it no longer uses. Then add two new settings. Place them in their own short block, after the `TTA_ENABLED = True` line and before `UNFREEZE_SCHEDULE`, formatted to match the surrounding style.

    # SAM (Sharpness-Aware Minimisation) optimizer - two forward/backward passes per step
    USE_SAM = True
    SAM_RHO = 0.07                 # SAM neighbourhood size

Also change `CHECKPOINT_DIR` from `Path("checkpoints_efficientnet")` to `Path("checkpoints_efficientnet_sam")` and `PLOT_DIR` from `Path("plots_efficientnet")` to `Path("plots_efficientnet_sam")`. These two constants are the defaults used when `train_efficientnet_kfold()` is called with no explicit directories, which is what the no-flag invocation does. Changing them here is what guarantees that a bare `py EfficientNet.py` can never overwrite the checkpoints the live web app is serving. Note carefully that `evaluate.py` imports `create_efficientnet` from this module but does not read `EfficientNetConfig.CHECKPOINT_DIR`, and neither does `inference.py` — they take checkpoint paths as arguments — so this change cannot affect them. Confirm that by searching: `grep -rn "CHECKPOINT_DIR\|checkpoints_efficientnet" *.py` should show `CHECKPOINT_DIR` referenced only inside `EfficientNet.py` and `ResNet.py` (which has its own, separate constant), and the string `checkpoints_efficientnet` appearing in `eval_results/selected_models.json`, which this plan does not touch.

In `get_optimizer(model, current_epoch)`, the function currently ends with `return optim.AdamW(param_groups)`. Replace that single return with the SAM-aware form, copied in structure from `ResNet.py` lines 285 to 288.

    if EfficientNetConfig.USE_SAM:
        return SAM(param_groups, optim.AdamW, rho=EfficientNetConfig.SAM_RHO,
                   lr=EfficientNetConfig.CLASSIFIER_LR,
                   weight_decay=EfficientNetConfig.WEIGHT_DECAY)
    return optim.AdamW(param_groups)

Why this works, since it is not obvious: every dictionary in `param_groups` already carries its own explicit `lr` and `weight_decay`, set per group by the differentiated-learning-rate logic above it. The `lr` and `weight_decay` keyword arguments passed to `SAM` become the *defaults* for any group that omits them, and since no group omits them, they never override the per-group values. They exist because PyTorch's base `Optimizer` requires a value to be available for every defaulted key, and because `SAM` forwards these same keyword arguments when it constructs the inner AdamW. The per-group learning rates are therefore preserved exactly as they are today. This is the identical pattern `ResNet.py` already uses successfully.

In `get_scheduler(optimizer, epoch_offset=0)`, the scheduler must be attached to the inner optimiser, not the wrapper, or the learning rate it computes will never reach AdamW. Add one line at the top of the function body and change what is passed to the scheduler, mirroring `ResNet.py` lines 291 to 300.

    base_opt = optimizer.base_optimizer if EfficientNetConfig.USE_SAM else optimizer

then pass `base_opt` as the scheduler's first argument in place of `optimizer`. Note that `SAM.__init__` sets `self.param_groups = self.base_optimizer.param_groups`, meaning the wrapper and the inner optimiser share one and the same list of group dictionaries; when the scheduler writes a new learning rate into those dictionaries, both objects see it. That sharing is why attaching to the inner optimiser is correct rather than merely a workaround.

Rewrite `train_one_epoch(model, loader, criterion, optimizer, mixup_alpha)` so that it performs SAM's two passes. The current body computes the loss inline, once. The new body must compute the loss twice, at two different sets of weights, which means the loss computation has to become a small reusable closure. Follow `ResNet.py` lines 322 to 360 closely; the resulting body is this.

    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(EfficientNetConfig.DEVICE), labels.to(EfficientNetConfig.DEVICE)

        if mixup_alpha > 0:
            images, soft_labels = mixup_data(images, labels, mixup_alpha)

        def forward_pass():
            outputs = model(images)
            if mixup_alpha > 0:
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                return -(soft_labels * log_probs).sum(dim=1).mean()
            return criterion(outputs, labels)

        if EfficientNetConfig.USE_SAM:
            loss = forward_pass()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), EfficientNetConfig.GRAD_CLIP_NORM)
            optimizer.first_step(zero_grad=True)

            loss2 = forward_pass()
            loss2.backward()
            nn.utils.clip_grad_norm_(model.parameters(), EfficientNetConfig.GRAD_CLIP_NORM)
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            loss = forward_pass()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), EfficientNetConfig.GRAD_CLIP_NORM)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

Four points of care. The MixUp branch is kept intact even though `MIXUP_ALPHA` is now 0.0; the `if mixup_alpha > 0` guard simply never fires, and keeping the branch means the capability is not destroyed and the diff stays small. The reported `running_loss` accumulates the *first* pass's loss, which is the loss at the real weights and therefore the meaningful number to log — this matches `ResNet.py` and keeps the log column comparable with the baseline log. The mixed images are assigned back into `images` (rather than into a separate `mixed_images` variable as the current code does) so that the closure reads one variable regardless of which branch ran; this is also what makes `images.size(0)` correct on the last line either way. And the closure is defined inside the batch loop on purpose, so that it closes over the current batch; moving it outside would silently train on one batch forever.

In the fold loop inside `train_efficientnet_kfold()`, find the line constructing the training dataset, `train_dataset = TransformDataset(train_raw, transform=get_train_transforms())`, and change the call to `get_light_train_transforms()`. Leave the very next line, which builds `val_dataset` from `get_val_transforms()`, exactly as it is — validation and test images must stay un-augmented or the measurement stops meaning anything.

Finally, in the `if __name__ == "__main__":` block, the `--folds` branch currently passes `Path(f"checkpoints_efficientnet_{args.folds}fold")`, `Path(f"plots_efficientnet_{args.folds}fold")` and `log_name=f"efficientnet_{args.folds}fold"`. Change all three to carry the `_sam` marker: `checkpoints_efficientnet_sam_{args.folds}fold`, `plots_efficientnet_sam_{args.folds}fold`, and `efficientnet_sam_{args.folds}fold`. Then update the `--folds` argument's `help` text, which currently describes the old directory names and claims the no-flag invocation "reproduces the original behavior exactly". That claim is no longer true, and leaving it would be a lie in the program's own user interface. Rewrite it to say that omitting the flag runs 5 folds writing to `checkpoints_efficientnet_sam/` and `plots_efficientnet_sam/`, and that supplying `N` writes instead to `checkpoints_efficientnet_sam_{N}fold/`, `plots_efficientnet_sam_{N}fold/` and `training_logs/efficientnet_sam_{N}fold_training_log.xlsx`, so that no run ever overwrites the pre-existing non-SAM directories or the live app's `checkpoints_efficientnet/`.

Verification. First confirm the file still parses and the help text is right, without starting any training.

    py EfficientNet.py --help

Expect the usage block to print and the process to exit immediately. Read the printed `--folds` help text and confirm it mentions `checkpoints_efficientnet_sam` and no longer claims to reproduce the original behaviour. If instead training starts, or a `SyntaxError` or `ImportError` appears, fix it before going further.

Then prove SAM genuinely works end to end, on synthetic data, in a few seconds rather than a few hours. Write this script to the agent's scratchpad directory rather than into the repository, since it is a throwaway diagnostic and not a project artifact. Name it `sam_smoke.py`.

    import torch
    import EfficientNet as E
    from losses import build_hybrid_criterion

    torch.manual_seed(0)
    dev = E.EfficientNetConfig.DEVICE
    model = E.create_efficientnet()
    opt = E.get_optimizer(model, 0)
    print("optimizer type:", type(opt).__name__)
    assert type(opt).__name__ == "SAM", "SAM was not used"
    sched = E.get_scheduler(opt, epoch_offset=0)

    crit = build_hybrid_criterion(torch.tensor([1.0, 1.0], device=dev), 0.08, 1.5, 0.4)
    before = model.classifier[1].weight.detach().clone()

    calls = {"n": 0}
    real_forward = model.forward
    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)
    model.forward = counting_forward

    class FakeLoader:
        def __init__(self): self.dataset = range(8)
        def __iter__(self):
            for _ in range(2):
                yield torch.randn(4, 3, 224, 224), torch.randint(0, 2, (4,))

    loss = E.train_one_epoch(model, FakeLoader(), crit, opt, E.EfficientNetConfig.MIXUP_ALPHA)
    moved = float((model.classifier[1].weight.detach() - before).abs().max())
    print(f"epoch loss {loss:.4f} | forward passes {calls['n']} | max weight change {moved:.6f}")
    assert calls["n"] == 4, f"expected 4 forward passes for 2 batches under SAM, got {calls['n']}"
    assert loss > 0 and loss == loss, "loss is not a finite positive number"
    assert moved > 0, "weights did not change"
    print("SAM smoke test passed")

Run it from the repository root so that its relative imports resolve, substituting the real scratchpad path:

    py <scratchpad>/sam_smoke.py

Expect output of roughly this shape, with the exact loss and weight-change figures varying:

    optimizer type: SAM
    epoch loss 0.7123 | forward passes 4 | max weight change 0.000843
    SAM smoke test passed

The four forward passes for two batches is the specific thing being proven: it shows `first_step` and `second_step` are both being exercised and that SAM is doing its two-pass work rather than silently behaving like plain AdamW. If the count is 2, `USE_SAM` is not reaching the training loop. If `get_scheduler` raises `AttributeError: 'AdamW' object has no attribute 'base_optimizer'`, the `USE_SAM` guard in `get_scheduler` is inverted or missing.

Close the milestone by running `git diff --stat` and confirming that `EfficientNet.py` is the only tracked file changed since Milestone 1.

### Milestone 3 — the full five-fold training run

This milestone produces the actual trained models. At the end of it, `checkpoints_efficientnet_sam_5fold/` contains five checkpoint files, `plots_efficientnet_sam_5fold/` contains the per-fold and cross-fold plots, and `training_logs/efficientnet_sam_5fold_training_log.xlsx` contains every epoch's metrics. Nothing is being decided or measured yet; this is the long wait.

Before launching, confirm the GPU is free — no other training process is running — and confirm that `checkpoints_efficientnet_sam_5fold/` does not already exist from an aborted earlier attempt. If it does, and the attempt is genuinely being restarted rather than resumed (this script has no resume capability; a killed run must start over), move the old directory aside to `checkpoints_efficientnet_sam_5fold.aborted-<date>/` rather than deleting it, so that nothing is destroyed irreversibly.

Launch it as a background process owned by the agent's tooling, never in a hand-opened terminal window, for the reason recorded in the Decision Log. From the repository root:

    py -u EfficientNet.py --folds 5 > efficientnet_sam_5fold.log 2> efficientnet_sam_5fold.err.log

The `-u` flag makes Python's output unbuffered, so the log file fills in as training proceeds rather than in large delayed chunks; this is what makes progress observable. Expect 11 to 14 hours. While it runs, check progress by reading the tail of `efficientnet_sam_5fold.log`.

What healthy output looks like. Early in the log, before the fold loop, expect the dataset summary — `Loaded 2864 labelled samples`, with benign and malignant counts, and `Using device: cuda`. Then per-epoch lines of the form

    Epoch 7: Train Loss 0.5210 | Val Loss 0.5533 Acc 0.7102 Prec 0.6612 Rec 0.6103 F1 0.6348

interleaved with `--> saved best model` whenever a new best validation loss is found (never before epoch 6), and `Unfreezing EfficientNet blocks: [8, 7]` at epoch 5 as printed, with `[6]` at epoch 11 and `[5]` at epoch 17. Each fold ends with three summary lines giving its best validation F1 without test-time augmentation, with it, and its best decision threshold.

The single most informative thing to watch is the train loss. Under the previous heavy-augmentation configuration it never fell below roughly 0.55; the whole premise of this plan is that the augmentation, not the model's capacity, was the ceiling. If train loss in this run drops clearly below 0.55 — say into the 0.35 to 0.50 range by the later epochs — that premise is confirmed, and it must be recorded in Surprises & Discoveries with the specific log lines as evidence. If it stubbornly stays at 0.55 or above, the premise is wrong and that is an important negative finding which must be recorded just as plainly rather than quietly omitted.

Accept the milestone when four things hold. `efficientnet_sam_5fold.err.log` is empty — any content in it, particularly the Fortran "program aborting due to window-CLOSE event" signature, means the run died and must be relaunched from scratch. `checkpoints_efficientnet_sam_5fold/` contains exactly five files named `best_model_fold_1.pth` through `best_model_fold_5.pth`, each about 16,342,523 bytes, which is this project's standard EfficientNet-B0 checkpoint size and therefore a quick integrity check. The "Fold Summary" sheet of `training_logs/efficientnet_sam_5fold_training_log.xlsx` has exactly five data rows. And the log ends with the `========== Final Cross-Validation Results (EfficientNet) ==========` block giving mean validation loss, accuracy and F1 with standard deviations, plus the per-fold thresholds. Record that block's numbers, the wall-clock duration, and which folds early-stopped, in Progress and Outcomes.

### Milestone 4 — honest measurement against the held-out test set, and the verdict

This milestone answers the question the plan exists to answer. It changes no code.

Two measurements are needed, both on the untouched 704 held-out images, both with test-time augmentation on and the default 0.5 decision threshold, because those are exactly the conditions under which the baseline figures of 0.5862 and 0.5735 were measured. Deviating from them would produce a number that cannot be compared with anything.

First, the single best fold. Read the "Fold Summary" sheet of `training_logs/efficientnet_sam_5fold_training_log.xlsx`, or equivalently the `best-val F1 (with TTA)` lines in `efficientnet_sam_5fold.log`, and identify which fold scored highest on its own validation data. Note that this selection uses validation data only, never the test set, which is what keeps the resulting test number honest. Then, substituting that fold's number for `K`:

    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_sam_5fold/best_model_fold_K.pth --tta --out eval_results/sam_efficientnet_5fold_singlebest_test_metrics.json

Second, the ensemble of all five folds, which averages the five models' malignancy probabilities before thresholding:

    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_sam_5fold/best_model_fold_1.pth checkpoints_efficientnet_sam_5fold/best_model_fold_2.pth checkpoints_efficientnet_sam_5fold/best_model_fold_3.pth checkpoints_efficientnet_sam_5fold/best_model_fold_4.pth checkpoints_efficientnet_sam_5fold/best_model_fold_5.pth --tta --out eval_results/sam_efficientnet_5fold_ensemble_test_metrics.json

Each command prints one line and writes one JSON file. The printed line looks like this, with the numbers being whatever they turn out to be:

    [efficientnet] checkpoints=['checkpoints_efficientnet_sam_5fold/best_model_fold_3.pth'] tta=True threshold=0.5 n=704 acc=0.6634 prec=0.5512 rec=0.7029 f1=0.6179
    Wrote eval_results/sam_efficientnet_5fold_singlebest_test_metrics.json

Check `n=704` in both lines. Any other sample count means the test dataset did not load as expected and the numbers are not comparable. Note also that `evaluate.py` loads these checkpoints through `create_efficientnet()`, which this plan did not change, so the checkpoints load into the identical architecture the baseline checkpoints load into — the comparison is genuinely like for like, and `evaluate.py` needs no modification at all.

Then write the verdict into Outcomes & Retrospective as a table setting the new numbers beside the recorded baseline, in this shape:

    Configuration                          Mean CV F1         Held-out single-best F1 (acc)   Held-out ensemble F1 (acc)   Wall-clock
    Heavy augmentation + AdamW (baseline)  0.5444 +/- 0.0643  0.5862  (0.6349)                0.5735  (0.6577)             6h 18m
    Light augmentation + SAM (this plan)   <measured>         <measured>                      <measured>                   <measured>

and follow it with a short paragraph of plain prose stating whether the change helped, by how much, and what it cost in training time — and, if the numbers are worse, saying so directly and noting which of the two changes is the more likely culprit and what a follow-up experiment would isolate. For reference in that discussion, this project's best-ever recorded EfficientNet held-out F1, from before the augmentation pipeline was strengthened, is 0.6375.

Two things this milestone must deliberately *not* do. It must not tune the decision threshold on the test set and report the resulting F1 as the headline number; threshold tuning was investigated as its own separate question in `specs/expanded-training-data-and-f1-improvements.md`, and mixing it in here would blur what is being measured. And it must not update `eval_results/selected_models.json` to point the live Flask app at the new checkpoints. Repointing the production app is a separate decision for the repository owner to make once these numbers exist, and it is explicitly out of scope for this plan.

## Validation and Acceptance

The plan is complete when all of the following are true, in order.

`py -c "from DataSetAugmentation import get_light_train_transforms; from optimizers import SAM; print('ok')"` prints `ok`. `py EfficientNet.py --help` prints usage text describing the `_sam` output directories and exits without training. The SAM smoke test from Milestone 2 prints `SAM smoke test passed`, having observed exactly four forward passes across two batches. `efficientnet_sam_5fold.err.log` is zero bytes. `checkpoints_efficientnet_sam_5fold/` holds exactly five checkpoints of about 16.3 MB each. Both `evaluate.py` invocations print a line containing `n=704` and write their JSON file. And this document's Outcomes & Retrospective section contains the filled-in comparison table and the verdict paragraph.

Two non-regression checks must also pass, because this plan's central safety claim is that it destroys nothing. `git status --short` must show exactly two modified tracked files attributable to this plan, `DataSetAugmentation.py` and `EfficientNet.py`. Note that `DataSetAugmentation.py`, `EfficientNet.py`, `ResNet.py` and `training_logs/resnet_training_log.xlsx` were all already showing as modified before this plan began, on 2026-09-25; `ResNet.py` and that workbook must show no *additional* change beyond what was already there. And the directories `checkpoints_efficientnet/`, `checkpoints_efficientnet_3fold/`, `checkpoints_efficientnet_5fold/`, `checkpoints_efficientnet_10fold/`, `checkpoints_resnet/`, `plots_efficientnet/` and the file `eval_results/selected_models.json` must all have modification timestamps predating the start of this plan's work. Verify with

    py -c "from pathlib import Path; [print(p, Path(p).stat().st_mtime) for p in ['checkpoints_efficientnet','checkpoints_efficientnet_5fold','eval_results/selected_models.json']]"

before and after, and compare.

## Idempotence and Recovery

Milestones 1 and 2 are pure source edits and are safe to repeat; if an edit goes wrong, `git diff` shows exactly what changed and `git checkout -- <file>` restores it. Do not use any git command that rewrites history or moves commits.

Milestone 3 is the fragile step, because this training script has no resume capability. If the run dies partway — closed window, power loss, out-of-memory — the only recovery is to start it over from fold 1. Partial output is not silently reused: each fold overwrites its own checkpoint file as it improves, so a directory holding three checkpoints from a run that died in fold 4 is a partial run, identifiable by its Excel "Fold Summary" sheet having fewer than five rows and its log lacking the final cross-validation block. When restarting, move the partial directory aside rather than deleting it, as described in Milestone 3. Never evaluate a partial run's checkpoints and report the result as this plan's outcome.

Milestone 4 is read-only with respect to models and fully repeatable; re-running either `evaluate.py` command simply overwrites its own JSON output file with the same numbers, since `get_val_transforms()` contains no randomness and the test-time augmentation is a fixed set of flips.

## Artifacts and Notes

New files created by this plan: `optimizers.py`; `checkpoints_efficientnet_sam_5fold/best_model_fold_1.pth` through `_5.pth`; the plots in `plots_efficientnet_sam_5fold/` (`fold_1_metrics.png` through `fold_5_metrics.png`, `mean_val_loss.png`, `mean_val_acc.png`, `mean_val_f1.png`); `training_logs/efficientnet_sam_5fold_training_log.xlsx`; `efficientnet_sam_5fold.log` and `efficientnet_sam_5fold.err.log`; and `eval_results/sam_efficientnet_5fold_singlebest_test_metrics.json` and `..._ensemble_test_metrics.json`.

Existing files modified: `DataSetAugmentation.py` (one function added, nothing removed) and `EfficientNet.py` (the six edits of Milestone 2).

Existing files this plan must not touch: `ResNet.py`, `losses.py`, `evaluate.py`, `inference.py`, `app.py`, `training_log.py`, `CompareModels.py`, `eval_results/selected_models.json`, and every pre-existing `checkpoints_*`, `plots_*` and `training_logs/*` entry.

The plan deliberately leaves three questions open for possible follow-up plans, none of them in scope here: whether freezing BatchNorm running statistics during SAM's second forward pass helps; whether the classifier's `Dropout(p=0.3)` and `LABEL_SMOOTHING = 0.08` are also contributing to underfitting; and whether the live app should be repointed at the new checkpoints.

## Interfaces and Dependencies

No new third-party dependency is introduced. `optimizers.py` needs only `torch`, which `requirements.txt` already lists, so `requirements.txt` needs no change.

Three function-level facts are worth stating. `DataSetAugmentation.get_light_train_transforms()` is new and takes no arguments, returning a `torchvision.transforms.Compose`. `EfficientNet.get_optimizer(model, current_epoch)` keeps its signature but now returns a `SAM` instance rather than an `optim.AdamW` instance whenever `EfficientNetConfig.USE_SAM` is true; its only callers are inside `EfficientNet.py` itself, and both pass the result to `get_scheduler()` and to `train_one_epoch()`, which this plan updates to match. `EfficientNet.train_one_epoch()` keeps its signature and its return type, a single float, and `EfficientNet.create_efficientnet()` is unchanged, which is what lets `evaluate.py` and `inference.py` continue to work untouched.
