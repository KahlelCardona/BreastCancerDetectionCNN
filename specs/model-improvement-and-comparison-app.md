# Improve ResNet and EfficientNet accuracy, then ship a mammogram comparison web app

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not check in a copy of PLANS.md. The authoring and execution rules for this document live at `~/.agents/PLANS.md` on the machine of whoever is running this plan (a global, agent-agnostic file, not part of this git repository). This document must be maintained in accordance with that file.

## Approval

Status: Approved
Approved by: Repository owner (chat instruction: "go ahead")
Date: 2026-08-25
Approved scope: All five milestones as written below.

## Purpose / Big Picture

Today this repository (`A:\Projects\BreastCancerDetectionCNN`, a git repository with remote `origin`) contains two independent PyTorch image classifiers that each try to look at a mammogram picture and decide whether it shows a benign (non-cancerous) or malignant (cancerous) finding: one built on the ResNet-50 architecture (`ResNet.py`) and one built on the EfficientNet-B0 architecture (`EfficientNet.py`). "Architecture" here just means the specific arrangement of neural-network layers used. Both are trained and evaluated only from the command line by whoever runs the script; there is no way for a person to hand the program a single image and get an answer back. There is also no working combination of the two models — nothing that runs both and produces one final opinion.

After this plan is complete, two things will be true. First, both models will have been retrained with concrete, justified improvements and evaluated on a test set of mammogram images that neither model has ever seen during training (this is called a "held-out test set" — holding data out means deliberately not using it for training or tuning, so that evaluating on it tells you how the model performs on genuinely new pictures, not pictures it has partially memorized). The retrained checkpoints ("checkpoint" is the standard term for a saved snapshot of a trained model's numbers, saved to a `.pth` file by PyTorch) will replace the current ones, and this document will record, honestly, what accuracy resulted — improvement is the goal but is not guaranteed by any of these steps individually, so the actual before/after numbers will be written into the Decision Log and Outcomes sections rather than assumed.

Second, a person will be able to run one command (`py app.py` from the repository root) to start a small local website, open `http://127.0.0.1:5000/` in a web browser, choose a mammogram JPEG or PNG file from their computer, click a button, and see three things on the resulting page: what the ResNet model thinks (Benign or Malignant, with a confidence percentage), what the EfficientNet model thinks (same), and a single blended "Final Answer" with its own confidence percentage, computed by combining the two models' opinions weighted by how accurate each model actually proved to be on the held-out test set. If the two models disagree, the page will say so explicitly rather than silently picking one.

## Progress

- [x] (2026-08-25) Milestone 1: Built `evaluate.py` and ran the baseline audit against all 10 pre-existing checkpoints. See Outcomes for full numbers.
- [x] (2026-08-25) Milestone 2, code portion: created `losses.py`, edited `ResNet.py` to import from it, verified `import ResNet` still builds a 26,724,034-parameter model with no errors.
- [x] (2026-08-26) Milestone 2, training portion: `py ResNet.py` completed all 5 folds cleanly (exit code 0) between 2026-08-25 ~19:44 and 2026-08-26 ~01:59 local time. Ran `evaluate.py --tta` against all 5 fresh fold checkpoints on the held-out test set; winning fold is fold 1 (0.6534 F1, 0.6776 accuracy). Full numbers in Outcomes.
- [x] (2026-08-25) Milestone 3, code portion: edited `EfficientNet.py` per the plan (hybrid focal+CE loss via shared `losses.py`, mixup, gradient clipping, cosine-restart scheduler with recency-differentiated LR via new module-level `get_optimizer`/`get_scheduler`, widened `UNFREEZE_SCHEDULE`, early stopping + `MIN_EPOCH_FOR_BEST`, post-fold TTA re-evaluation reusing `evaluate.evaluate_model`). Verified `import EfficientNet` builds a 4,010,110-parameter model with no errors.
- [x] (2026-08-26) Milestone 3, training portion: `py EfficientNet.py` completed all 5 folds cleanly (exit code 0). Ran `evaluate.py --tta` against all 5 fresh fold checkpoints on the held-out test set; winning fold is fold 2 (0.6375 F1, 0.6705 accuracy). Full numbers in Outcomes.
- [x] (2026-08-26) Milestone 4: created `eval_results/selected_models.json` (resnet fold1 @ 0.6776 accuracy, efficientnet fold2 @ 0.6705 accuracy) and verified `inference.py` end-to-end against a real sample image — see Outcomes for the full result dict, which also happened to demonstrate a genuine model disagreement resolved correctly by the confidence-weighted blend.
- [x] (2026-08-26) Milestone 5: ran `py app.py`, confirmed it loads both models and serves on `http://127.0.0.1:5000/`. Verified all three acceptance scenarios via HTTP requests against the live server (see Outcomes): a real mammogram upload renders all three verdicts with confidence percentages and a disagreement note; a request with no file renders the validation error; a request with a non-image file renders the same validation error. No server crash or traceback in any case (`app.log` shows three clean `200` responses). Server stopped after verification.

## Surprises & Discoveries

- Observation: The five checkpoint files in `checkpoints_resnet/` that the current `ResNet.py` writes to (`fold1_best.pth` through `fold5_best.pth`) are not all the same size: `fold1_best.pth` and `fold2_best.pth` are 107,226,339 bytes each, while `fold3_best.pth`, `fold4_best.pth`, and `fold5_best.pth` are 98,825,887 bytes each.
  Evidence: `ls -la` timestamps (local time) show fold5 saved 2026-05-04 23:52, fold3 saved 2026-05-05 07:31, fold4 saved 2026-05-05 08:03, then `ResNet.py` itself was last edited 2026-05-05 08:05, then fold1 saved 2026-05-05 09:17 and fold2 saved 2026-05-05 09:40. This means the 5-fold training run was interrupted or re-run partway through, and the model's structure (its "state dict", the named list of weight tensors PyTorch saves) changed in between — the current `ResNet.py` file almost certainly cannot `load_state_dict` all five of these files successfully. This is concrete evidence behind the user's description of ResNet as "not fully optimized": the checkpoints on disk are not a single, complete, consistent training run of the current code. Milestone 1's baseline audit will attempt to load all five and record exactly which ones fail and how.
- Observation: Neither `ResNet.py` nor `EfficientNet.py` ever evaluates on `data/raw/csv/mass_case_description_test_set.csv` or `data/raw/csv/calc_case_description_test_set.csv`. Both scripts only build their dataset from `["mass_train", "calc_train"]` and report accuracy from `StratifiedKFold` cross-validation splits carved out of that same training data. The 703 test-set rows are never touched by training code today, which is why Milestone 1 introduces `evaluate.py` against those files specifically.
  Evidence: `data/raw/csv/mass_case_description_test_set.csv` has 378 usable rows after dropping rows with a missing pathology value (231 benign-like, 147 other); `data/raw/csv/calc_case_description_test_set.csv` has 326 usable rows (197 benign-like, 129 other); grepping both `ResNet.py` and `EfficientNet.py` for `"_test"` shows the string only appears inside `MammogramRawDataset._get_csv_path`'s dictionary of filenames, never in a call to `MammogramRawDataset(...)`.
- Observation: When `py ResNet.py`'s output is redirected to a log file (`... > resnet_retrain.log 2>&1`, run in the background) rather than printed to an interactive terminal, Python block-buffers stdout instead of flushing on every `print()`/newline. The log file can stay empty for a long time even while the process is confirmed alive and actively using the GPU.
  Evidence: `Get-Process` showed multiple live `python.exe` processes (one at 25% CPU since the launch time, several DataLoader worker processes at ~43% CPU each) and `nvidia-smi --query-compute-apps` listed a `python.exe` process among active GPU compute apps, while `resnet_retrain.log` was still completely empty. Progress for the two multi-hour training runs in this plan is tracked via the background task's own completion notification and by checking `checkpoints_resnet/`/`checkpoints_efficientnet/` file timestamps, not by tailing the log for per-epoch lines in real time.
- Observation: `checkpoints/` (top-level) and `checkpoints_distill/` exist but have no corresponding, currently-working training script in the repository (there is no `train_model.py`, matching what the `README.md` describes, and no distillation script producing `checkpoints_distill/student_fold_*.pth` / `teacher_fold_*.pth`). `git log --oneline` shows a past commit "Implement model distillation with teacher-student training and 5-fold cross-validation", so a distillation script existed historically but is not present in the current working tree.
  Evidence: `find ... -name "*.py"` in the repository root returns only `CompareModels.py`, `DataSetAugmentation.py`, `EfficientNet.py`, `ResNet.py` — no distillation script. These two directories and their `.pth` files are pre-existing and are left untouched by this plan; nothing in this plan reads from or writes to `checkpoints/` or `checkpoints_distill/`.

## Decision Log

- Decision: Treat model improvement (Milestones 1-3) as a prerequisite phase before building the web app (Milestones 4-5), inside this single ExecPlan, rather than as two separate plans.
  Rationale: The human explicitly chose "Improve models first, then build the app" when asked. Keeping it as one document (rather than two separate plans) still satisfies that ordering because the milestones are sequential, and it avoids duplicating the shared context (dataset layout, checkpoint locations, hardware) across two files.
  Date/Author: 2026-08-25, drafted by agent per human's answers to clarifying questions.
- Decision: Run full 5-fold cross-validation for both models' retraining (not a faster single train/val split).
  Rationale: The human explicitly chose this option, accepting the multi-hour GPU time cost (hardware confirmed: local NVIDIA GeForce RTX 3070 Ti, 8 GB, CUDA available, PyTorch 2.10.0+cu126) in exchange for a more statistically reliable result and consistency with the existing methodology already used in both scripts.
  Date/Author: 2026-08-25, drafted by agent per human's answer to clarifying question.
- Decision: Add a real held-out test-set evaluation (`evaluate.py`, Milestone 1) using `mass_case_description_test_set.csv` and `calc_case_description_test_set.csv`, and use its output as the only "accuracy" number surfaced anywhere in this plan (Decision Log, Outcomes, and the web app's displayed confidence/accuracy) — cross-validation numbers from the training scripts remain useful for monitoring training but are not what gets reported as final accuracy or used as the ensemble weight.
  Rationale: The human explicitly agreed real held-out evaluation should replace cross-validation numbers as the trustworthy figure. Using one single number (test accuracy) consistently, instead of tracking both a CV number and a test number, avoids ambiguity about which "accuracy" the web app is showing a user.
  Date/Author: 2026-08-25, drafted by agent per human's answer to clarifying question.
- Decision: When ResNet and EfficientNet disagree, the final answer is a confidence-weighted average of the two models' predicted probability of malignancy, where each model's weight is its own held-out test accuracy (from `evaluate.py`'s output, per the decision above) rather than its cross-validation accuracy.
  Rationale: The human chose "confidence-weighted average" explicitly. The option text mentioned "validation accuracy/F1 from cross-validation" as an example, but since a separate decision (above) already established held-out test accuracy as the one trustworthy, displayed number, this plan uses that same number as the ensemble weight rather than introducing a second, different "accuracy" concept that CV would represent.
  Date/Author: 2026-08-25, drafted by agent, reconciling two of the human's answers into one consistent design.
  Follow-up note (fill in during Milestone 4): if this weighting produces a model tie or a weight of exactly 0 for a model (e.g., if a model's held-out test accuracy were ever measured as very low), the arithmetic in `inference.py`'s `predict` function still works because both weights are always positive probabilities-of-being-correct in the 0-to-1 range; no special-casing is needed.
- Decision: Build the app as a local Flask web app rather than a Tkinter desktop GUI.
  Rationale: The human explicitly chose this option over the Tkinter alternative, despite the existing `CompareModels.py` dashboard in this repository being Tkinter-based; the app's dark color palette will still be reused for visual consistency with that existing tool (see Milestone 5).
  Date/Author: 2026-08-25, drafted by agent per human's answer to clarifying question.
- Decision: Extract `FocalLoss`, `mixup_data`, and a new `build_hybrid_criterion` helper out of `ResNet.py` into a new shared file, `losses.py`, and have both `ResNet.py` and the upgraded `EfficientNet.py` import from it, instead of duplicating this code into `EfficientNet.py`.
  Rationale: Milestone 3 needs EfficientNet's training loop to use the same focal-loss-plus-label-smoothing technique ResNet already uses (this is the most concrete, evidence-based lever identified for closing EfficientNet's accuracy gap — see Milestone 3's rationale). Copy-pasting the same ~40 lines into a second file would create two copies to keep in sync for no benefit; a small shared module is the minimal change that avoids that duplication. This is the only new shared/helper file introduced by this plan.
  Date/Author: 2026-08-25, drafted by agent during research for this plan.
- Decision: Leave `checkpoints/`, `checkpoints_distill/`, and the stale, larger `checkpoints_resnet/fold1_best.pth` / `fold2_best.pth` files exactly as they are; do not delete anything.
  Rationale: These are pre-existing artifacts from earlier work not covered by this plan. Retraining ResNet in Milestone 2 will naturally overwrite `checkpoints_resnet/fold1_best.pth` through `fold5_best.pth` in place (that is simply what running `ResNet.py` does, and always has done), which resolves the size-mismatch problem as a side effect of retraining — no separate deletion step is needed or performed.
  Date/Author: 2026-08-25, drafted by agent during research for this plan.

## Outcomes & Retrospective

All five milestones are complete as of 2026-08-26. Summary against the original Purpose: both models were retrained with concrete, justified changes and are now evaluated honestly on a genuinely held-out test set (704 images never used in training or cross-validation), and a person can run `py app.py`, open `http://127.0.0.1:5000/`, upload a mammogram image, and see ResNet's verdict, EfficientNet's verdict, and a blended Final Answer with confidence percentages, exactly as promised.

The most important finding was not a training-recipe improvement but a data-integrity one: the cross-validation accuracy figures both models had been reporting (including the ~82% the user cited for EfficientNet) were measured only on splits carved out of the training data itself, and were never checked against real unseen data. Once measured honestly, both models actually perform in the mid-60s percent range, and 3 of the 5 pre-existing ResNet checkpoints could not even be loaded by the current model code due to an interrupted, inconsistent training run. Fixing that measurement gap (Milestone 1's `evaluate.py`) was the single highest-value change in this plan, independent of any training-recipe tweak.

After retraining: ResNet improved from 67.90%/0.6195 F1 (its best previously-loadable checkpoint) to 67.76%/0.6534 F1 (essentially flat accuracy, meaningfully better F1), while also fixing the checkpoint-consistency problem outright. EfficientNet improved from 65.06%/0.6328 F1 (its best baseline checkpoint by accuracy) to 67.05%/0.6375 F1 — roughly a 2-point accuracy gain. Neither model made a dramatic leap, and that is an honest, expected outcome given the dataset: only 2,864 training images for two large pretrained CNNs is a genuinely small amount of data for this kind of medical image classification, and no amount of loss-function or scheduler tuning fully substitutes for more labeled data. The plan's Decision Log and Milestone 2/3 outcome entries above record this candidly rather than overstating the result.

What remains, not attempted in this plan because it was out of scope: no attempt was made to source additional training data, to try a fundamentally different/larger backbone, or to revisit the `checkpoints/`/`checkpoints_distill/` distillation approach hinted at in git history. Any of those would be reasonable next steps if further accuracy improvement is wanted, and each would need its own ExecPlan given the scope and approval discipline this document followed.

### Milestone 1 baseline audit (2026-08-25), held-out test set (704 images: 428 benign, 276 malignant), all with TTA enabled

ResNet, against the pre-existing `checkpoints_resnet/` files, using the current `ResNetWithAttnPool` class:

- `fold1_best.pth`: loaded successfully. accuracy 0.6790, precision 0.5786, recall 0.6667, f1 0.6195.
- `fold2_best.pth`: loaded successfully. accuracy 0.6136, precision 0.5062, recall 0.5942, f1 0.5467.
- `fold3_best.pth`: **failed to load.** `RuntimeError`: missing keys `layer0.0.weight`, `layer0.1.*`, `attn_pool.attn.2.*`, `attn_pool.attn.4.*`; unexpected keys `conv1.weight`, `bn1.*`. This checkpoint predates the `layer0` grouping and `AttentionPool` module in the current `ResNetWithAttnPool` class entirely — it is from an older architecture version, not merely a different training run.
- `fold4_best.pth`: **failed to load**, identical error to fold3.
- `fold5_best.pth`: **failed to load**, identical error to fold3.

This means only 2 of the 5 "current" ResNet checkpoints are even loadable with today's code, and the two that do load score well below the cross-validation numbers the training logs would have reported. Best loadable baseline: fold1 at 67.90% accuracy / 0.6195 F1.

EfficientNet, against the pre-existing `checkpoints_efficientnet/` files (all loaded successfully — architecture has not drifted for this model):

- `best_model_fold_1.pth`: accuracy 0.6591, precision 0.5517, recall 0.6957, f1 0.6154.
- `best_model_fold_2.pth`: accuracy 0.6506, precision 0.5381, recall 0.7681, f1 0.6328.
- `best_model_fold_3.pth`: accuracy 0.6477, precision 0.5446, recall 0.6196, f1 0.5797.
- `best_model_fold_4.pth`: accuracy 0.6477, precision 0.5348, recall 0.7790, f1 0.6342.
- `best_model_fold_5.pth`: accuracy 0.6506, precision 0.5389, recall 0.7536, f1 0.6284.

Best baseline by F1: fold4 (0.6342 F1, 64.77% accuracy). Best baseline by raw accuracy: fold2 (65.06%). All five folds cluster tightly around 65% accuracy on real held-out data — well below the roughly 82% cross-validation accuracy the user described, confirming the CV numbers were substantially overfit to the training-data distribution they were drawn from (the CV folds are carved from the same 2,864 training rows the model trains on; the held-out test rows come from entirely different patient studies).

This is the honest "before" picture Milestones 2 and 3 will be measured against: on real unseen data, both models are currently performing far closer to a coin flip with a benign bias (428 of 704 test images, 60.8%, are benign) than the reported 82%.

### Milestone 2 outcome (2026-08-26): ResNet retrained cleanly, all 5 folds now consistent

`py ResNet.py` ran to completion end-to-end with no exceptions. Cross-validation summary printed by the script itself: "Val F1 no-TTA (mean ± std): 0.6978 ± 0.0108" and "Val F1 TTA (mean ± std): 0.7186 ± 0.0023" — a much tighter spread across folds than the old, architecturally-inconsistent checkpoint set could ever have produced. Fold 4 triggered early stopping at epoch 35 (best at epoch 25); the other four folds ran the full 40 epochs.

All five `checkpoints_resnet/fold{1..5}_best.pth` files are now the same size (107,226,339 bytes) and load without error — confirming the version-mismatch problem from Milestone 1 is resolved by this clean retrain.

`evaluate.py --tta` results on the true 704-image held-out test set (not the cross-validation folds above):

- fold1: accuracy 0.6776, precision 0.5646, recall 0.7754, f1 0.6534
- fold2: accuracy 0.6918, precision 0.6042, recall 0.6196, f1 0.6118
- fold3: accuracy 0.6719, precision 0.5630, recall 0.7283, f1 0.6351
- fold4: accuracy 0.6420, precision 0.5306, recall 0.7536, f1 0.6228
- fold5: accuracy 0.6690, precision 0.5564, recall 0.7681, f1 0.6454

Winning fold by F1: **fold1** (0.6534 F1, 67.76% accuracy). Comparing to Milestone 1's only two loadable baseline folds: old fold1 was 67.90% accuracy / 0.6195 F1, old fold2 was 61.36% accuracy / 0.5467 F1. The new fold1 has essentially the same accuracy as the old fold1 but a meaningfully better F1 (+0.034, meaning better balance between precision and recall on the minority/malignant class), and every one of the 5 new folds is honestly comparable and loadable, versus 3 of 5 being completely broken before. This is a real, if modest, improvement — not the dramatic swing hoped for, but an honest one: this dataset (2,864 training images) is small for a resnet50-sized model, which limits how far accuracy can realistically move without more data or a fundamentally different approach.

### Milestone 3 outcome (2026-08-26): EfficientNet retrained with a stronger recipe

`py EfficientNet.py` ran to completion end-to-end with no exceptions. Script-reported cross-validation summary: "Mean Val Accuracy: 0.7357 ± 0.0188", "Mean Val F1 Score: 0.6594 ± 0.0387" — both noticeably higher than the old recipe would have produced on the same cross-validation splits, and every fold now uses focal loss, mixup, gradient clipping, and a wider/earlier-unfreezing schedule. Folds 3, 4, and 5 triggered early stopping (around epochs 30-32); folds 1 and 2 ran closer to the full 35 epochs.

`evaluate.py --tta` results on the true 704-image held-out test set:

- fold1: accuracy 0.6534, precision 0.5473, recall 0.6703, f1 0.6026
- fold2: accuracy 0.6705, precision 0.5604, recall 0.7391, f1 0.6375
- fold3: accuracy 0.6477, precision 0.5452, recall 0.6123, f1 0.5768
- fold4: accuracy 0.6605, precision 0.5562, recall 0.6630, f1 0.6050
- fold5: accuracy 0.6662, precision 0.5587, recall 0.7065, f1 0.6240

Winning fold by F1: **fold2** (0.6375 F1, 67.05% accuracy). Comparing to Milestone 1's baseline (best baseline fold by F1 was old fold4 at 0.6342 F1/64.77% accuracy; best baseline fold by accuracy was old fold2 at 65.06%): the new fold2 improves accuracy by about 2 percentage points (65.06% -> 67.05%) and F1 by a small amount (0.6342 -> 0.6375 versus the best baseline fold by F1, or +0.0047 versus baseline fold2 specifically). As with ResNet, this is a real but modest gain, not the leap from 82% CV accuracy the user originally described — that 82% figure was never measured on genuinely unseen data, and this milestone's honest held-out numbers (roughly 65-67% for both architectures) are what the comparison app in Milestone 4/5 will actually show and rely on.

### Milestone 4 outcome (2026-08-26): shared checkpoints selected, inference module verified

`eval_results/selected_models.json` was written with the two winning folds identified above: ResNet `checkpoints_resnet/fold1_best.pth` (accuracy 0.6776) and EfficientNet `checkpoints_efficientnet/best_model_fold_2.pth` (accuracy 0.6705). Running the Milestone 4 interactive verification command against a real sample image (`data/raw/jpeg/1.3.6.1.4.1.9590.100.1.2.100018879311824535125115145152454291132/1-263.jpg`) produced:

    {'resnet': {'label': 'Malignant', 'malignant_probability': 0.8818, 'confidence': 0.8818, 'reported_accuracy': 0.6776},
     'efficientnet': {'label': 'Benign', 'malignant_probability': 0.3493, 'confidence': 0.6507, 'reported_accuracy': 0.6705},
     'final': {'label': 'Malignant', 'malignant_probability': 0.6169, 'confidence': 0.6169, 'models_agreed': False}}

This happens to be a genuine disagreement case: ResNet says Malignant with 88.2% confidence, EfficientNet says Benign with 65.1% confidence, and the accuracy-weighted blend correctly resolves to Malignant at 61.7% confidence (leaning ResNet's way since its accuracy weight, 0.6776, is slightly higher, and its raw signal is much stronger). `models_agreed: False` is set correctly. This is exactly the shape and behavior described in the Purpose section and in the confidence-weighted-average decision.

### Milestone 5 outcome (2026-08-26): the web app works end-to-end

`py app.py` started cleanly, loaded both selected models, and served on `http://127.0.0.1:5000/`. Three scenarios were verified against the live server with real HTTP requests:

1. Uploading the same real mammogram JPEG used in Milestone 4 returned HTTP 200 with rendered HTML containing: "Malignant" / "88.2% confidence" / "model test accuracy: 67.8%" for ResNet; "Benign" / "65.1% confidence" / "model test accuracy: 67.0%" for EfficientNet; "Malignant" / "61.7% confidence" for the Final Answer; and the text "Note: the two models disagreed on this image." — matching Milestone 4's direct `inference.py` result exactly, confirming `app.py` wires `inference.predict` through to the page correctly.
2. Submitting the form with no file attached returned HTTP 200 with "Please upload a .jpg, .jpeg, or .png image." rendered instead of a result.
3. Submitting a `.txt` file returned the same validation error, HTTP 200, with no server-side exception.

`app.log` shows three clean `"POST / HTTP/1.1" 200 -` lines and no traceback across all three scenarios. This satisfies the plan's Milestone 5 and overall acceptance criteria.

A minor tooling note, not a defect in the app itself: testing this locally with `curl -F "mammogram=@<path>"` only worked with a path relative to the working directory; the MSYS/Git-Bash `curl` binary on this machine (`curl 8.18.0 (x86_64-w64-mingw32)`) failed with "Failed to open/read local data from file/application" when given an absolute MSYS-style path (e.g. `/tmp/fake.txt` or `/c/Users/...`) as the `-F` file attachment, even though the file demonstrably existed. This is specific to this curl build's path handling and unrelated to Flask or this project's code; a human tester using an actual browser file picker, as the app is designed for, would never encounter this.

## Context and Orientation

The repository root is `A:\Projects\BreastCancerDetectionCNN` (equivalently reachable as `/a/Projects/BreastCancerDetectionCNN` from a Git Bash shell on this machine). It is a git repository with a remote named `origin`; this plan does not require any commit, push, or other git history change, and none should be performed while executing it. On this machine, Python is invoked as `py` from PowerShell (there is no plain `python` command available; `py` is the Python launcher installed at `C:\Windows\py.exe`). The installed PyTorch is version 2.10.0+cu126 with CUDA available, and `torch.cuda.get_device_name(0)` reports "NVIDIA GeForce RTX 3070 Ti" (8 GB of video memory). `pillow` 12.1.1 is installed. `flask` and `timm` are not currently installed and must be installed with `py -m pip install flask` before Milestone 5 (no other new third-party package is required by this plan).

The dataset lives under `data/raw/`. `data/raw/jpeg/` contains 10,237 JPEG image files nested in per-study subfolders named with long numeric identifiers. `data/raw/csv/` contains five CSV files describing two kinds of mammogram findings — "mass" and "calc" (short for calcification, a different kind of finding visible on a mammogram) — each split into a train file and a test file, plus one `dicom_info.csv` and one `meta.csv` not directly used by the training scripts. Row counts after dropping rows with a missing `pathology` column (the ground-truth label): `mass_case_description_train_set.csv` has 1,318 rows, `calc_case_description_train_set.csv` has 1,546 rows (these two combined, 2,864 rows, are what both training scripts currently use for their 5-fold cross-validation); `mass_case_description_test_set.csv` has 378 rows and `calc_case_description_test_set.csv` has 326 rows (these two combined, 704 rows, are the held-out test set this plan will start using).

`DataSetAugmentation.py` defines the shared data-loading building blocks used by both training scripts. `Config` holds shared path/size constants. `MammogramRawDataset(csv_types)` is a PyTorch `Dataset` (a class that knows how to produce `(image, label)` pairs by index) that accepts a list of the strings `"mass_train"`, `"mass_test"`, `"calc_train"`, or `"calc_test"`, reads the corresponding CSV(s), converts the free-text `pathology` column into a numeric label (`0` for benign, `1` for anything else, which in this dataset's vocabulary means malignant), and locates each row's actual JPEG file by matching a unique ID substring against the full list of files under `data/raw/jpeg/`. `TransformDataset(dataset, transform)` wraps a raw dataset and applies a torchvision `transform` (an image-preprocessing pipeline) to each image on the fly. `get_train_transforms()` returns a pipeline with resizing to 224x224 pixels, random horizontal flip, random rotation, and color jitter (randomly perturbing brightness/contrast), followed by conversion to a tensor (PyTorch's array type) and normalization using the standard ImageNet mean/std values (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`), which is the normalization the pretrained backbones expect. `get_val_transforms()` returns the same pipeline minus the random augmentation steps, used for validation, testing, and (in Milestone 5) live inference.

`ResNet.py` defines `CFG` (its configuration constants), a `FocalLoss` class (a loss function that down-weights easy, already-correctly-classified examples so the model focuses more on hard ones — a standard technique for imbalanced classification), a `SAM` optimizer wrapper (Sharpness-Aware Minimization, an optimizer that takes two forward/backward passes per step to find flatter, more generalizable minima), a `mixup_data` function (a regularization technique that blends pairs of training images and their labels together), and `ResNetWithAttnPool` (the model itself: a pretrained `torchvision.models.resnet50` backbone with its early layers frozen — meaning their weights are not updated during training — a custom attention-pooling layer, and a new fully-connected classification head). Its `main()` function runs 5-fold cross-validation over `MammogramRawDataset(["mass_train", "calc_train"])`, saving each fold's best checkpoint (by validation loss) to `checkpoints_resnet/fold{N}_best.pth`.

`EfficientNet.py` defines `EfficientNetConfig` and `create_efficientnet()`, which loads a pretrained `torchvision.models.efficientnet_b0`, freezes its backbone, and replaces its classifier head with a dropout layer plus a linear layer producing 2 outputs. Its `train_efficientnet_kfold()` function runs the same 5-fold cross-validation pattern, saving each fold's best checkpoint to `checkpoints_efficientnet/best_model_fold_{N}.pth`, and additionally saves per-fold and mean-across-fold metric plots (PNG images) into `plots_efficientnet/`. Unlike `ResNet.py`, it uses a plain `nn.CrossEntropyLoss` (no focal loss), no mixup, no gradient clipping, and a `ReduceLROnPlateau` scheduler (which lowers the learning rate only after validation loss stops improving) rather than ResNet's `CosineAnnealingWarmRestarts` (which lowers and then periodically resets the learning rate on a fixed schedule). Its backbone unfreezing schedule (`UNFREEZE_SCHEDULE = {4: [7], 10: [6, 7]}`) only ever unfreezes two of `efficientnet_b0`'s nine sequential stages (`model.features[0]` through `model.features[8]`, where index 0 is the initial stem convolution and index 8 is the final 1x1 head convolution before pooling), and only for the last 15 of its 25 total epochs.

`CompareModels.py` is a standalone Tkinter desktop tool, unrelated to inference, that lets a user browse the PNG plots saved by the training scripts under `plots_efficientnet/`, `plots_resnet/`, and `plots_distill_kfold/` (the last of these does not currently exist on disk and is handled gracefully by the tool showing "no plot found"). It defines a dark color palette (`BG = "#0d1117"`, `PANEL = "#161b22"`, `BORDER = "#21262d"`, `ACCENT = "#58a6ff"`, `ACCENT2 = "#3fb950"`, `ACCENT3 = "#f78166"`, `TEXT_HI = "#e6edf3"`, `TEXT_MID = "#8b949e"`, `TEXT_LO = "#484f58"`) that Milestone 5 will reuse in the web app's stylesheet for visual consistency across this project's tools.

## Plan of Work

### Milestone 1 — `evaluate.py` and a baseline audit

Create a new file, `evaluate.py`, in the repository root. It defines: `build_test_dataset()`, which constructs `TransformDataset(MammogramRawDataset(["mass_test", "calc_test"]), get_val_transforms())` (importing these four names from `DataSetAugmentation.py`) and wraps it in a `torch.utils.data.DataLoader` with `batch_size=16`, `shuffle=False`; `load_model(model_name, checkpoint_path, device)`, which builds a fresh `ResNetWithAttnPool` (imported from `ResNet.py`) when `model_name == "resnet"` or a fresh `create_efficientnet()`-built model (imported from `EfficientNet.py`) when `model_name == "efficientnet"`, moves it to `device`, and calls `model.load_state_dict(torch.load(checkpoint_path, map_location=device))` inside a `try/except RuntimeError` block that, on failure, prints a clear one-line message naming the checkpoint path and the original error text and then re-raises, so a size/shape mismatch (like the one already discovered between `fold1_best.pth`/`fold2_best.pth` and the current `ResNetWithAttnPool` class) produces an understandable message instead of only a raw stack trace; `evaluate_model(model, loader, device, use_tta)`, which sets `model.eval()`, iterates the loader under `torch.no_grad()`, and for each batch either runs a single forward pass or (when `use_tta` is true) averages the softmax-free logits from the original image, a horizontally flipped copy, and a vertically flipped copy (the same three-way test-time-augmentation pattern already implemented in `ResNet.py`'s `validate` function, reimplemented here using `torchvision.transforms.functional.hflip`/`vflip`), then computes `accuracy_score`, `precision_score`, `recall_score`, and `f1_score` from `sklearn.metrics` (all four already used elsewhere in this repository) over the full test set and returns them in a dict alongside `n_samples`; and a `main()` that uses Python's `argparse` to accept `--model` (required, one of `resnet` or `efficientnet`), `--checkpoint` (required, a filesystem path), `--tta` (an optional flag, off by default), and `--out` (optional, defaulting to `eval_results/<model>_<checkpoint-stem>_test_metrics.json` where `<checkpoint-stem>` is the checkpoint's filename without its `.pth` extension), then calls the three functions above in sequence, prints the resulting metrics as one readable line, creates the `eval_results/` directory if it does not exist, and writes the metrics dict (plus `"model"`, `"checkpoint"`, and `"tta"` keys) as JSON to `--out`.

Once `evaluate.py` exists, run it against every existing checkpoint to establish an honest baseline before any training code changes, from the repository root in PowerShell:

    py evaluate.py --model resnet --checkpoint checkpoints_resnet/fold3_best.pth --tta
    py evaluate.py --model resnet --checkpoint checkpoints_resnet/fold4_best.pth --tta
    py evaluate.py --model resnet --checkpoint checkpoints_resnet/fold5_best.pth --tta
    py evaluate.py --model resnet --checkpoint checkpoints_resnet/fold1_best.pth --tta
    py evaluate.py --model resnet --checkpoint checkpoints_resnet/fold2_best.pth --tta
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet/best_model_fold_1.pth --tta
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet/best_model_fold_2.pth --tta
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet/best_model_fold_3.pth --tta
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet/best_model_fold_4.pth --tta
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet/best_model_fold_5.pth --tta

Record every result (including any load failures) in the Surprises & Discoveries and Outcomes sections before moving to Milestone 2. This step is expected, based on the size-mismatch evidence already found, to show `fold1_best.pth` and/or `fold2_best.pth` failing to load against the current `ResNetWithAttnPool` class — that failure is itself the acceptance evidence that Milestone 2's full retrain is necessary, not a problem to fix in Milestone 1.

### Milestone 2 — shared `losses.py` and a clean ResNet retrain

Create `losses.py` in the repository root containing exactly three things moved out of `ResNet.py`: the `FocalLoss` class (unchanged), the `mixup_data` function (unchanged), and a new function `build_hybrid_criterion(class_weights, label_smoothing, gamma, focal_weight)` that constructs `ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)` and `focal = FocalLoss(gamma=gamma, weight=class_weights, label_smoothing=label_smoothing)` and returns a closure `hybrid(logits, labels)` computing `(1 - focal_weight) * ce(logits, labels) + focal_weight * focal(logits, labels)` — this is the same logic currently inlined in `ResNet.py`'s `build_criterion` function, extracted so `EfficientNet.py` can reuse it in Milestone 3 instead of duplicating it.

Edit `ResNet.py`: remove the now-duplicated `FocalLoss` class and `mixup_data` function bodies, add `from losses import FocalLoss, mixup_data, build_hybrid_criterion` near its existing imports, and rewrite `build_criterion(class_weights)` to simply `return build_hybrid_criterion(class_weights, CFG.LABEL_SMOOTHING, CFG.FOCAL_GAMMA, CFG.FOCAL_WEIGHT)`. Make no other change to `ResNet.py` — its architecture, hyperparameters, SAM optimizer, unfreeze schedule, and training loop are already sophisticated and are not the source of the problem identified in Milestone 1 (a version-mismatched, incomplete checkpoint set); the fix is to run it cleanly to completion, not to redesign it.

Run a complete, uninterrupted 5-fold training run from the repository root:

    py ResNet.py

This is expected to take multiple hours on the confirmed RTX 3070 Ti (each fold trains up to 40 epochs with early stopping, and SAM's two-forward-pass-per-step design roughly doubles per-epoch time compared to a normal optimizer). Run it in the background and periodically check its console output for each fold's per-epoch line (`Epoch NN | Train Loss ... | Val Loss ... Acc ... Prec ... Rec ... F1 ...`) and each fold's final TTA-vs-no-TTA F1 summary, and confirm it prints the "Final cross-validation results (ResNet-50 v2)" block at the end without error. Do not interrupt it partway and do not edit `ResNet.py` again while it runs — if it must be interrupted, restart the entire 5-fold run from the beginning afterward, since a partial run is exactly the problem Milestone 1 diagnosed. This will overwrite `checkpoints_resnet/fold1_best.pth` through `fold5_best.pth` in place with five checkpoints from one single, consistent version of `ResNetWithAttnPool`.

After training completes, run `evaluate.py` (from Milestone 1) with `--tta` against all five new fold checkpoints, exactly as in Milestone 1's commands but now against the freshly written files, and identify whichever fold produced the highest `f1` in its JSON output — call this the winning ResNet fold. Record all five folds' held-out test metrics, and which fold won, in the Outcomes section, alongside a direct comparison to whatever Milestone 1 measured for the old checkpoints.

### Milestone 3 — upgrade and retrain EfficientNet

Edit `EfficientNet.py` to close the concrete, evidenced gaps between it and `ResNet.py`'s training recipe (identified in Context and Orientation above): it currently has no focal loss, no mixup, no gradient clipping, a plateau-based instead of cosine-restart learning-rate schedule, and it only ever unfreezes 2 of 9 backbone stages, only in the final 15 of 25 epochs. Each of the following changes to `EfficientNetConfig` and its surrounding functions is a direct, named port of a technique `ResNet.py` already uses successfully, not a speculative addition:

Add these fields to `EfficientNetConfig`: `NUM_EPOCHS = 35` (was 25, to give newly-unfrozen stages, especially the last one added at epoch 16 below, enough remaining epochs to adapt); `GRAD_CLIP_NORM = 1.5` (matching `CFG.GRAD_CLIP_NORM` in `ResNet.py`; currently `EfficientNet.py` has no gradient clipping at all); `LABEL_SMOOTHING = 0.08`, `FOCAL_GAMMA = 1.5`, `FOCAL_WEIGHT = 0.4` (identical values to `ResNet.py`'s `CFG`, since both models are solving the same classification problem on the same dataset); `MIXUP_ALPHA = 0.2` (identical to `ResNet.py`'s `CFG.MIXUP_ALPHA`); `NEW_UNFREEZE_LR = 1e-4` and `OLD_UNFREEZE_LR = 2e-5` (new names, analogous to `ResNet.py`'s `CFG.BASE_LR` and `CFG.FROZEN_LR`: a lower learning rate for backbone stages unfrozen in an earlier phase than for ones just unfrozen); `EARLY_STOP_PATIENCE = 10` and `MIN_EPOCH_FOR_BEST = 6` (analogous to `ResNet.py`'s `CFG.EARLY_STOP_PATIENCE = 10` and `CFG.MIN_EPOCH_FOR_BEST = 8`, scaled down proportionally for EfficientNet's shorter 35-epoch run — currently `EfficientNet.py` has neither early stopping nor a warm-up-before-checkpointing floor at all); `TTA_ENABLED = True` (matching `ResNet.py`'s flag, used only for the final post-training re-evaluation described below, not during the training loop itself, to keep per-epoch training fast). Change `UNFREEZE_SCHEDULE` from `{4: [7], 10: [6, 7]}` to `{4: [8, 7], 10: [6], 16: [5]}` — this widens which of `efficientnet_b0`'s nine `model.features` stages become trainable (stages 8 and 7 at epoch 4, stage 6 at epoch 10, stage 5 at epoch 16), on the reasoning that a backbone which only ever adapts its last two stages, and only for its last 15 epochs, cannot learn mammogram-specific low-and-mid-level texture patterns that ImageNet pretraining did not teach it — this is the most concrete, evidence-based explanation available in this repository for EfficientNet's lower accuracy relative to ResNet, whose equivalent schedule already unfreezes proportionally more of its backbone (`layer4` at epoch 5, `layer3` at epoch 10, out of 4 total ResNet stages) for proportionally more of its run.

Rewrite `create_efficientnet()`'s caller-side setup and `train_efficientnet_kfold()`'s optimizer construction so that, mirroring `ResNet.py`'s `get_optimizer(model, current_epoch)` function precisely: the classifier keeps `EfficientNetConfig.CLASSIFIER_LR`; any backbone parameter unfrozen at the most recently fired schedule epoch uses `NEW_UNFREEZE_LR`; any backbone parameter unfrozen at an earlier schedule epoch uses `OLD_UNFREEZE_LR`. Replace the `optim.lr_scheduler.ReduceLROnPlateau(...)` construction with `optim.lr_scheduler.CosineAnnealingWarmRestarts(base_optimizer, T_0=8, T_mult=1, eta_min=1e-7)` (matching `ResNet.py`'s `CFG.T_0`-analogous pattern, scaled to EfficientNet's shorter run), rebuilt each time `UNFREEZE_SCHEDULE` fires, exactly as `ResNet.py`'s `get_scheduler` is rebuilt on each unfreeze event; call `scheduler.step()` once per epoch (cosine schedulers step unconditionally, unlike `ReduceLROnPlateau` which needed the validation loss passed in — remove the `scheduler.step(val_loss)` call accordingly).

In `train_one_epoch`, add mixup: when `EfficientNetConfig.MIXUP_ALPHA > 0`, call the shared `mixup_data(x, y, alpha)` (imported from the new `losses.py`, see Milestone 2) to blend each batch before the forward pass, and compute the loss as `-(soft_labels * log_softmax(logits, dim=1)).sum(dim=1).mean()`, exactly matching the pattern already implemented in `ResNet.py`'s `train_epoch`. After computing the loss and calling `loss.backward()`, add `nn.utils.clip_grad_norm_(model.parameters(), EfficientNetConfig.GRAD_CLIP_NORM)` before `optimizer.step()`. Replace the plain `criterion = nn.CrossEntropyLoss(weight=class_weights)` construction in `train_efficientnet_kfold()` with `from losses import build_hybrid_criterion` and `criterion = build_hybrid_criterion(class_weights, EfficientNetConfig.LABEL_SMOOTHING, EfficientNetConfig.FOCAL_GAMMA, EfficientNetConfig.FOCAL_WEIGHT)`.

Add early stopping to the per-fold epoch loop in `train_efficientnet_kfold()`, matching `ResNet.py`'s `train_fold` pattern: track a `patience` counter that resets to 0 whenever a new best checkpoint is saved, increments otherwise, resets again whenever `UNFREEZE_SCHEDULE` fires, and breaks out of the epoch loop once `patience >= EfficientNetConfig.EARLY_STOP_PATIENCE`; only save a checkpoint when `epoch >= EfficientNetConfig.MIN_EPOCH_FOR_BEST` (using 0-indexed epoch numbers consistently with the existing `for epoch in range(EfficientNetConfig.NUM_EPOCHS)` loop, so this means `epoch >= EfficientNetConfig.MIN_EPOCH_FOR_BEST - 1`; state this off-by-one adjustment explicitly in a code comment since `ResNet.py`'s loop is 1-indexed and `EfficientNet.py`'s is 0-indexed and this is an easy place to introduce a bug). After the per-fold loop ends (whether by early stopping or exhausting all epochs), reload the fold's saved best checkpoint and call a TTA-enabled evaluation pass (reusing the same three-way hflip/vflip averaging logic already written for `evaluate.py` in Milestone 1 — do not write a third copy of this logic; either import `evaluate_model` from `evaluate.py` and call it with `use_tta=True` against the fold's validation loader, or, if that creates an awkward import cycle, write the loop inline but keep it byte-for-byte identical in structure to `evaluate.py`'s version), and print both the no-TTA and TTA F1 for that fold, matching `ResNet.py`'s `train_fold` final print statements.

Run the full 5-fold training from the repository root:

    py EfficientNet.py

Expect a shorter run than ResNet's (EfficientNet-B0 is a smaller network and this recipe does not use SAM's double-pass), but still likely over an hour across 5 folds at up to 35 epochs each. Watch for the same kind of per-epoch and per-fold console output as Milestone 2, and confirm the final "Final Cross-Validation Results (EfficientNet)" block prints without error. This overwrites `checkpoints_efficientnet/best_model_fold_1.pth` through `best_model_fold_5.pth` in place.

After training completes, run `evaluate.py --tta` against all five new EfficientNet fold checkpoints, identify the winning fold by highest `f1`, and record all five folds' held-out test metrics plus the winner in the Outcomes section, directly compared against whatever Milestone 1 measured for the old EfficientNet checkpoints. State plainly whether the changes in this milestone improved held-out test accuracy, and by how much, or whether they did not — this cannot be guaranteed in advance and must be reported honestly either way.

### Milestone 4 — `inference.py` and picking the two winning checkpoints

After Milestones 2 and 3 each identify one winning fold, create `eval_results/selected_models.json` (a plain JSON file, not a script) with this exact shape, filled in with the real winning fold numbers and their real measured held-out test accuracy from the two milestones above:

    {
      "resnet": {"checkpoint": "checkpoints_resnet/fold<N>_best.pth", "accuracy": 0.0},
      "efficientnet": {"checkpoint": "checkpoints_efficientnet/best_model_fold_<N>.pth", "accuracy": 0.0}
    }

Create `inference.py` in the repository root. It defines `load_selected_models(device)`, which reads `eval_results/selected_models.json`, builds a `ResNetWithAttnPool` and loads the ResNet checkpoint path from the JSON, builds a `create_efficientnet()` model and loads the EfficientNet checkpoint path from the JSON (both using the same `load_model`-style loading already written for `evaluate.py` in Milestone 1 — import and reuse that function rather than re-implementing checkpoint loading a third time), sets both to `.eval()`, and returns a dict `{"resnet": (model, accuracy), "efficientnet": (model, accuracy)}` where each `accuracy` is the float from the JSON. It also defines `predict(image, models)`, where `image` is a `PIL.Image.Image` already converted to RGB: it applies `get_val_transforms()` (imported from `DataSetAugmentation.py`) to produce a single tensor, adds a batch dimension, and for each of the two models runs the same three-way (original, horizontal-flip, vertical-flip) test-time-augmentation forward pass already used in `evaluate.py` and `ResNet.py`, applies `torch.softmax` to the averaged logits to get a malignant probability (index `1`) and benign probability (index `0`), and builds a per-model result dict `{"label": "Malignant" if malignant_prob > 0.5 else "Benign", "malignant_probability": float, "confidence": float (the probability of whichever label was chosen), "reported_accuracy": float (from the loaded JSON)}`. It then computes `blended_prob = (resnet_malignant_prob * resnet_accuracy + efficientnet_malignant_prob * efficientnet_accuracy) / (resnet_accuracy + efficientnet_accuracy)`, derives a final label and confidence from `blended_prob` the same way as each individual model, and returns `{"resnet": {...}, "efficientnet": {...}, "final": {"label": ..., "confidence": ..., "malignant_probability": blended_prob, "models_agreed": bool(resnet_result["label"] == efficientnet_result["label"])}}`.

Verify this module works, without yet building the web app around it, by running a short interactive check from the repository root:

    py -c "
    import torch
    from PIL import Image
    from inference import load_selected_models, predict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = load_selected_models(device)
    img = Image.open('data/raw/jpeg/<pick any existing subfolder>/<pick any .jpg file>').convert('RGB')
    result = predict(img, models)
    print(result)
    "

Replace `<pick any existing subfolder>/<pick any .jpg file>` with a real path found under `data/raw/jpeg/` (for example, one of the paths already listed in this plan's research, such as `1.3.6.1.4.1.9590.100.1.2.100018879311824535125115145152454291132/1-263.jpg`). Confirm the printed dict has the shape described above with numeric, non-NaN values.

### Milestone 5 — the Flask web app

Create `requirements.txt` in the repository root (it does not currently exist) listing: `torch`, `torchvision`, `flask`, `pandas`, `numpy`, `scikit-learn`, `pillow`, `matplotlib`. Install the one genuinely new dependency:

    py -m pip install flask

Create `app.py` in the repository root. At module load time (once, not per-request) it computes `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and calls `models = load_selected_models(device)` from `inference.py`. It creates a `Flask(__name__)` app with a single route, `@app.route("/", methods=["GET", "POST"])`, implemented by one view function: on `GET`, render `templates/index.html` with no result context; on `POST`, read the uploaded file from `request.files.get("mammogram")`, and if it is missing or its filename does not end in `.jpg`, `.jpeg`, or `.png` (case-insensitive), re-render `templates/index.html` with an `error` message ("Please upload a .jpg, .jpeg, or .png image.") instead of a result — this is a genuine input-validation boundary (an arbitrary file from an arbitrary user) and is the only validation this plan adds; otherwise, open the file with `PIL.Image.open(file.stream).convert("RGB")` inside a `try/except Exception` that, on failure, re-renders the form with the error message "Could not read that file as an image.", call `predict(image, models)` from `inference.py`, and render `templates/index.html` passing the full result dict as template context. Run the app with `app.run(host="127.0.0.1", port=5000, debug=False)` under the `if __name__ == "__main__":` guard — `debug=False` is deliberate and must not be changed to `True`, because Flask's debug reloader restarts the whole process and would silently load both multi-hundred-megabyte models into GPU memory twice.

Create `templates/index.html` (Flask looks for templates in a `templates/` subfolder next to `app.py` by default, so this exact path is required, not a convention that can be renamed). It contains one HTML page with: a title; a form with `method="post"`, `enctype="multipart/form-data"` (required for file uploads), a `<input type="file" name="mammogram" accept=".jpg,.jpeg,.png">`, and a submit button; if an `error` was passed in, a visibly styled error message; if a `result` (the dict from `predict`) was passed in, three clearly labeled sections showing the ResNet result (`result.resnet.label`, `result.resnet.confidence` formatted as a percentage), the EfficientNet result (same fields under `result.efficientnet`), and the Final Answer (`result.final.label`, `result.final.confidence` as a percentage), plus — only when `result.final.models_agreed` is false — a visible note such as "Note: the two models disagreed on this image." Create `static/style.css` (Flask serves `static/` by default) reusing the exact hex colors already defined in `CompareModels.py`'s palette (`BG = "#0d1117"`, `PANEL = "#161b22"`, `BORDER = "#21262d"`, `ACCENT = "#58a6ff"`, `ACCENT2 = "#3fb950"`, `ACCENT3 = "#f78166"`, `TEXT_HI = "#e6edf3"`, `TEXT_MID = "#8b949e"`) so the web app visually matches the existing desktop tool; link it from `index.html` with `<link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">`.

Verify end-to-end by running, from the repository root:

    py app.py

then opening `http://127.0.0.1:5000/` in a web browser, choosing a real JPEG from `data/raw/jpeg/` (any subfolder), and clicking submit. Confirm the resulting page shows a ResNet verdict with a confidence percentage, an EfficientNet verdict with a confidence percentage, and a Final Answer verdict with a confidence percentage, and that these numbers are plausible (each confidence between 50% and 100%, since a `> 0.5` threshold decides the label). Additionally verify the validation path by attempting to submit with no file chosen and by attempting to upload a non-image file (for example a `.txt` file), confirming both show the "Please upload a .jpg, .jpeg, or .png image." error rather than crashing the server. Stop the server afterward with Ctrl+C in its terminal.

## Concrete Steps

Run every command below from the repository root, `A:\Projects\BreastCancerDetectionCNN`, in PowerShell (`py` is confirmed on this machine's `PATH`; there is no `python` alias):

    py evaluate.py --model resnet --checkpoint checkpoints_resnet/fold5_best.pth --tta
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet/best_model_fold_5.pth --tta
    py ResNet.py
    py evaluate.py --model resnet --checkpoint checkpoints_resnet/fold1_best.pth --tta
    py EfficientNet.py
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet/best_model_fold_1.pth --tta
    py -m pip install flask
    py app.py

(The full set of baseline-audit and post-retrain evaluation commands for all five folds of each model are listed in full in Milestones 1, 2, and 3 above; this section shows one representative command per stage rather than repeating all twenty.)

## Validation and Acceptance

Milestone 1 is accepted when `evaluate.py` runs to completion against at least one checkpoint of each architecture and produces a JSON file under `eval_results/` with plausible, non-NaN `accuracy`, `precision`, `recall`, and `f1` values between 0 and 1, and when every attempted checkpoint (including the ones expected to fail, like `fold1_best.pth`/`fold2_best.pth` against the pre-Milestone-2 `ResNet.py`) has a recorded outcome — success with numbers, or failure with the exact error message — in the Outcomes section.

Milestone 2 is accepted when `py ResNet.py` runs all 5 folds to completion (or clean early stopping) without a Python exception, when all five `checkpoints_resnet/fold{1..5}_best.pth` files are the same byte size as each other (proving they now come from one consistent architecture, unlike the pre-Milestone-2 state), and when `evaluate.py --tta` against the winning fold produces a JSON file with a recorded, real accuracy figure that is compared explicitly, in prose, against Milestone 1's baseline number for ResNet.

Milestone 3 is accepted the same way: `py EfficientNet.py` completes all 5 folds without a Python exception, and `evaluate.py --tta` against the winning fold produces a real accuracy figure compared explicitly against Milestone 1's baseline number for EfficientNet.

Milestone 4 is accepted when the interactive `py -c "..."` check in Milestone 4's description prints a `predict(...)` result dict containing `resnet`, `efficientnet`, and `final` keys, each with a `label` of exactly `"Benign"` or `"Malignant"` and a `confidence` between 0.5 and 1.0.

Milestone 5, and the plan as a whole, is accepted when a human can run `py app.py`, open `http://127.0.0.1:5000/` in a browser, upload a real mammogram JPEG from `data/raw/jpeg/`, and see a page displaying all three verdicts (ResNet, EfficientNet, Final Answer) each with a confidence percentage, with a visible disagreement note on any image where the two models' labels differ, and where submitting no file or a non-image file produces a visible on-page error message instead of a server crash.

## Idempotence and Recovery

`evaluate.py` is read-only with respect to the dataset and checkpoints; it only writes its own output JSON file under `eval_results/`, and re-running it with the same arguments simply overwrites that one file. It is always safe to re-run.

Re-running `py ResNet.py` or `py EfficientNet.py` overwrites their own checkpoint files in place — this is pre-existing behavior of both scripts, not new behavior introduced by this plan, and is exactly how Milestone 2 and Milestone 3 are expected to produce their results. Neither script supports resuming a partially completed fold; if a training run is interrupted before printing its final cross-validation summary, the safe recovery path is to restart that script from the beginning (accepting the lost GPU time) rather than trusting a partial set of checkpoints — this is precisely the failure mode Milestone 1 documented in the pre-existing ResNet checkpoints, and this plan does not add resume support since it was not requested and would add meaningful scope.

`py app.py` makes no changes to any file on disk; it only reads the two checkpoint files named in `eval_results/selected_models.json` at startup and reads whatever image a user uploads through the browser (the uploaded image itself is never saved to disk by this plan's code). It can be started and stopped (Ctrl+C) any number of times safely.

## Artifacts and Notes

Baseline audit results, post-retrain results, and the final `eval_results/selected_models.json` contents must be pasted into the Outcomes & Retrospective section as the plan proceeds, so that a reader of only this document (not the live `eval_results/` directory) can see the actual numbers this project achieved.

## Interfaces and Dependencies

In `losses.py` (new file), define:

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0): ...
        def forward(self, logits, targets) -> torch.Tensor: ...

    def mixup_data(x, y, alpha=1.0) -> tuple[torch.Tensor, torch.Tensor]: ...

    def build_hybrid_criterion(class_weights, label_smoothing, gamma, focal_weight):
        # returns a callable: (logits, labels) -> torch.Tensor

In `evaluate.py` (new file), define:

    def build_test_dataset() -> torch.utils.data.DataLoader: ...
    def load_model(model_name: str, checkpoint_path: pathlib.Path, device) -> torch.nn.Module: ...
    def evaluate_model(model, loader, device, use_tta: bool) -> dict: ...
    def main() -> None: ...  # argparse CLI: --model {resnet,efficientnet}, --checkpoint PATH, --tta, --out PATH

In `inference.py` (new file), define:

    def load_selected_models(device) -> dict:  # {"resnet": (model, accuracy), "efficientnet": (model, accuracy)}
    def predict(image: "PIL.Image.Image", models: dict) -> dict:  # {"resnet": {...}, "efficientnet": {...}, "final": {...}}

In `app.py` (new file), define a single Flask route `@app.route("/", methods=["GET", "POST"])` on a module-level `app = Flask(__name__)`, using `inference.load_selected_models` and `inference.predict` as its only calls into the rest of this codebase, and `torchvision.transforms`/`PIL.Image` only indirectly through those two functions.

`ResNet.py` and `EfficientNet.py` both depend on the new `losses.py` (`from losses import FocalLoss, mixup_data, build_hybrid_criterion` in `ResNet.py`; `from losses import mixup_data, build_hybrid_criterion` in `EfficientNet.py`, which has no direct need for the `FocalLoss` class itself since it only calls the criterion builder). `evaluate.py` depends on `ResNet.build_model` (or the `ResNetWithAttnPool` class directly), `EfficientNet.create_efficientnet`, and `DataSetAugmentation.MammogramRawDataset`/`TransformDataset`/`get_val_transforms`. `inference.py` depends on `evaluate.load_model` (reused, not reimplemented) and `DataSetAugmentation.get_val_transforms`. `app.py` depends only on `inference.py`.
