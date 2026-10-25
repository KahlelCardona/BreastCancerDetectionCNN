# Compare EfficientNet-B0 at 3-fold, 5-fold, and 10-fold cross-validation on the held-out test set

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not check in a copy of PLANS.md. The authoring and execution rules for this document live at `~/.agents/PLANS.md` on the machine of whoever is running this plan (a global, agent-agnostic file, not part of this git repository). This document must be maintained in accordance with that file.

This plan builds on three prior, completed-or-abandoned ExecPlans already checked into this repository: `specs/model-improvement-and-comparison-app.md` (built the shared training/evaluation/inference infrastructure), `specs/expanded-training-data-and-f1-improvements.md` (added per-model threshold tuning and checkpoint ensembling, later found not to net an improvement), and `specs/synthetic-augmentation-f1-improvement.md` (strengthened the training-image transform pipeline and reverted the dataset to its original, non-cropped composition; this plan's Milestone 2, a ResNet-50 retrain, was left permanently incomplete when the terminal window running it was closed on 2026-08-27, killing the process mid-fold-3-of-5 — that abandoned attempt is not resumed or touched by this plan). This plan does not repeat those documents' full contents, only the facts needed to follow this one.

## Approval

Status: Approved
Approved by: Repository owner (chat instruction: "Approved as written")
Date: 2026-09-07
Approved scope: All five milestones as written below.

## Purpose / Big Picture

Today, `EfficientNet.py`'s training function, `train_efficientnet_kfold()`, always splits its 2,864-image training set into exactly 5 cross-validation folds (a hardcoded `StratifiedKFold(n_splits=5, ...)` call) — there is no way to ask it to use a different number of folds without editing the source file by hand and there is no existing comparison, anywhere in this project's history, of how the number of folds affects either the cross-validation score or, more importantly, the honestly-measured accuracy on this project's untouched 704-image held-out test set.

After this plan, a person will be able to run `py EfficientNet.py --folds 3`, `py EfficientNet.py --folds 5`, or `py EfficientNet.py --folds 10` from the repository root, each producing a complete, independent 5-fold-style training run (just with 3, 5, or 10 folds respectively) written to its own checkpoint/plot/log directories that do not overwrite each other or the pre-existing, currently-live `checkpoints_efficientnet/` directory that the Flask web app (`app.py`, via `eval_results/selected_models.json` and `inference.py`) already depends on. Running `py EfficientNet.py` with no `--folds` flag at all will continue to behave exactly as it does today (5 folds, writing to `checkpoints_efficientnet/`), so nothing about the existing, live behavior changes.

Once all three runs finish, this document will report, honestly, for each fold count: the mean cross-validation F1 score the training script itself prints, and two numbers measured by `evaluate.py` against the untouched 704-image held-out test set — the single best-performing fold's own F1, and the full ensemble-of-all-folds' F1 — so a reader can see not just which fold count "trained better" internally but which one actually generalizes best to real unseen data, which is the only number this entire project has ever treated as trustworthy. The Outcomes section will state plainly which fold count is better and why, including the practical cost (total GPU time) of each, since more folds is not free.

## Progress

- [x] (2026-09-07) Milestone 1: added the backward-compatible `--folds` flag to `EfficientNet.py` (`train_efficientnet_kfold(n_folds=5, checkpoint_dir=None, plot_dir=None, log_name="efficientnet")`, argparse block in `__main__`). Verified `py -c "import EfficientNet"` succeeds and `py EfficientNet.py --help` prints the documented usage text without starting training. Confirmed via `git diff` that no other line changed beyond what the plan specified.
- [x] (2026-09-07) Milestone 2: `py -u EfficientNet.py --folds 3` completed all 3 folds cleanly in ~2h50m (13:01 wall-clock finish per checkpoint mtimes vs. start), zero content in `efficientnet_kfoldsweep_3fold.err.log`, no early stopping in any fold (all ran the full 35 epochs). `checkpoints_efficientnet_3fold/` has exactly 3 checkpoints (16,342,523 bytes each, matching the project's standard EfficientNet-B0 checkpoint size); `training_logs/efficientnet_3fold_training_log.xlsx` "Fold Summary" sheet has exactly 3 rows. Winning fold by TTA F1: fold 3 (0.6034). Both `evaluate.py` commands ran successfully; see Outcomes for the held-out numbers.
- [x] (2026-09-07) Milestone 3: `py -u EfficientNet.py --folds 5` completed cleanly, zero content in `efficientnet_kfoldsweep_5fold.err.log`. Fold 1 early-stopped at epoch 32 (best F1 0.5841); folds 2-5 ran the full 35 epochs. `checkpoints_efficientnet_5fold/` has exactly 5 checkpoints (16,342,523 bytes each); Excel log "Fold Summary" sheet has exactly 5 rows. Winning fold by TTA F1: fold 5 (0.6761). Both `evaluate.py` commands ran successfully; see Outcomes for the held-out numbers.
- [x] (2026-09-08) Milestone 4: `py -u EfficientNet.py --folds 10` completed cleanly, zero content in `efficientnet_kfoldsweep_10fold.err.log`. Folds 5, 9, 10 early-stopped; the other 7 ran the full 35 epochs. `checkpoints_efficientnet_10fold/` has exactly 10 checkpoints; Excel log "Fold Summary" sheet has exactly 10 rows. Winning fold by TTA F1: fold 5 (0.6860). Both `evaluate.py` commands ran successfully; see Outcomes for the held-out numbers.
- [x] (2026-09-08) Milestone 5: final three-way comparison and conclusion written below in Outcomes & Retrospective.

## Surprises & Discoveries

(None yet; this section will be updated as Milestones 1-5 proceed.)

## Decision Log

- Decision: scope this comparison to EfficientNet-B0 only, not ResNet-50, and use the current full production epoch budget (35 epochs per fold, no shortcuts) rather than a reduced-epoch quick comparison.
  Rationale: the repository owner chose both of these explicitly when asked. EfficientNet-B0 trains meaningfully faster than ResNet-50 (which uses a Sharpness-Aware-Minimization optimizer requiring two forward/backward passes per step), making three full runs practically feasible; a full production epoch budget was preferred over a faster but less representative reduced-epoch comparison despite the added time cost.
  Date/Author: 2026-09-07, drafted by agent per the repository owner's direct answers to two clarifying questions.
- Decision: give each fold-count run its own checkpoint directory (`checkpoints_efficientnet_{N}fold/`), plot directory (`plots_efficientnet_{N}fold/`), and Excel log workbook (`training_logs/efficientnet_{N}fold_training_log.xlsx`), and never write to the pre-existing `checkpoints_efficientnet/`, `plots_efficientnet/`, or `training_logs/efficientnet_training_log.xlsx` for any of the three experimental runs (5-fold included).
  Rationale: `checkpoints_efficientnet/` currently holds the checkpoints named in `eval_results/selected_models.json`, which `inference.py` loads and the live Flask app (`app.py`) serves; overwriting it mid-experiment (even with an eventually-comparable 5-fold run) would risk leaving the live app pointing at missing or inconsistent files if this plan were interrupted, and would destroy the ability to compare against the current production configuration afterward. Running the "5-fold" arm of this experiment as a fresh, separately-named run (rather than reusing the pre-existing `checkpoints_efficientnet/fold*.pth` files, which come from a different, crop-augmented dataset per the prior plan) also ensures all three points of this comparison are apples-to-apples: same code, same dataset, same augmentation, only the fold count differs.
  Date/Author: 2026-09-07, drafted by agent.
- Decision: make the new `--folds` flag opt-in and fully backward-compatible: omitting it must reproduce today's exact behavior (5 folds, writing to `checkpoints_efficientnet/`), not merely default to the same fold count while changing output paths.
  Rationale: this project's standing instruction (`~/.agents/PLANS.md` and this machine's global agent instructions) requires preserving existing behavior unless a plan explicitly calls for a change, and no change to the default, no-flag invocation was requested or is needed for this comparison.
  Date/Author: 2026-09-07, drafted by agent.
- Decision: measure each fold count's held-out performance two ways (single best-validation fold at the default 0.5 threshold, and the full ensemble of all N folds at the default 0.5 threshold), and deliberately not repeat the four-way threshold-tuning/ensembling comparison matrix from `specs/expanded-training-data-and-f1-improvements.md`.
  Rationale: threshold tuning and ensembling were already investigated as their own, separate question in a prior plan (with a mixed, inconclusive result); re-opening that question here would blur what this experiment is actually testing (the effect of fold count alone). The single-best-fold-at-0.5 number is this project's own established "isolate one variable" methodology (used identically in the prior plan's Milestone 1), and the full-ensemble-at-0.5 number is included because it costs nothing extra once all N checkpoints already exist and gives a second, complementary view of each fold count's held-out performance.
  Date/Author: 2026-09-07, drafted by agent.
- Decision: launch each of the three training runs as a background process managed by the coding agent's own tooling (not a manually-opened, separate terminal window), and run them one at a time, never simultaneously.
  Rationale: the most recent prior attempt at a multi-hour EfficientNet/ResNet retrain (`specs/synthetic-augmentation-f1-improvement.md`, Milestone 2) died because the terminal window it was running in was closed, producing a Fortran runtime error ("program aborting due to window-CLOSE event") and losing all progress after fold 2. Running one at a time is unchanged, pre-existing project practice, since the single available GPU (8GB) cannot comfortably run two training processes at once.
  Date/Author: 2026-09-07, drafted by agent.

## Outcomes & Retrospective

### Milestone 2 outcome (2026-09-07): 3-fold run

`py -u EfficientNet.py --folds 3` ran cleanly, zero errors, no early stopping (all 3 folds ran the full 35 epochs). Per-fold train-set size: 2,864 × 2/3 ≈ 1,909 train / ≈955 validation. Script's own final summary: Mean Val Accuracy 0.6854 ± 0.0186, Mean Val F1 0.5270 ± 0.0865 (a notably wide standard deviation across only 3 folds — fold TTA F1s were 0.5971, 0.5789, 0.6034). Held-out 704-image test set: single best fold (fold 3) accuracy 0.6179, F1 0.5009; full 3-fold ensemble accuracy 0.6548, F1 0.5389. Both numbers are well below this project's established EfficientNet baseline (F1 0.6375, single fold, from before the augmentation pipeline was strengthened) — an honest, if disappointing, result for 3-fold, recorded as measured.

### Milestone 3 outcome (2026-09-07): 5-fold run

`py -u EfficientNet.py --folds 5` ran cleanly, zero errors. Fold 1 early-stopped at epoch 32; folds 2-5 ran the full 35 epochs. Per-fold train-set size: 2,864 × 4/5 ≈ 2,291 train / ≈573 validation. Script's own final summary: Mean Val Accuracy 0.6903 ± 0.0194, Mean Val F1 0.5444 ± 0.0643 (fold TTA F1s: 0.5841, 0.5672, 0.6621, 0.5899, 0.6761). Held-out 704-image test set: single best fold (fold 5) accuracy 0.6349, F1 0.5862; full 5-fold ensemble accuracy 0.6577, F1 0.5735. Both figures are meaningfully better than the 3-fold run's (0.5009 single / 0.5389 ensemble) but still below the pre-augmentation EfficientNet baseline (F1 0.6375).

### Milestone 4 outcome (2026-09-08): 10-fold run

`py -u EfficientNet.py --folds 10` ran cleanly, zero errors. Folds 5, 9, and 10 early-stopped (epochs 30, 31, 33); the other 7 folds ran the full 35 epochs. Per-fold train-set size: 2,864 × 9/10 ≈ 2,578 train / ≈286 validation. Script's own final summary: Mean Val Accuracy 0.6899 ± 0.0309, Mean Val F1 0.5460 ± 0.0729 (fold TTA F1s ranged from 0.5561 to 0.6860 — the widest per-fold spread of the three configurations, consistent with each fold's validation set now being only ~286 images, a noisier estimate than 5-fold's ~573 or 3-fold's ~955). Held-out 704-image test set: single best fold (fold 5) accuracy 0.6463, F1 0.6103; full 10-fold ensemble accuracy 0.6634, F1 0.5835.

Actual wall-clock training time for all three runs (measured from each run's checkpoint-directory creation to its Excel log's final save, both logged automatically by the filesystem):

    3-fold:  2026-09-07 10:14:20 -> 14:05:00  =  3h 51m
    5-fold:  2026-09-07 14:07:52 -> 20:25:51  =  6h 18m
    10-fold: 2026-09-07 20:28:54 -> 2026-09-08 08:55:31  = 12h 27m

### Milestone 5 outcome (2026-09-08): final three-way comparison and conclusion

Full comparison table, all measured on the same untouched 704-image held-out test set (428 benign, 276 malignant), same code, same dataset (2,864 original full images, no cropped patches), same strengthened augmentation pipeline — only the fold count differs:

    Folds  Train/Val per fold   Mean CV F1 (script)   Held-out single-best F1 (acc)   Held-out ensemble F1 (acc)   Wall-clock time
    3      ~1,909 / ~955        0.5270 +/- 0.0865      0.5009  (0.6179)                0.5389  (0.6548)             3h 51m
    5      ~2,291 / ~573        0.5444 +/- 0.0643      0.5862  (0.6349)                0.5735  (0.6577)             6h 18m
    10     ~2,578 / ~286        0.5460 +/- 0.0729      0.6103  (0.6463)                0.5835  (0.6634)             12h 27m

**Which fold count is better, measured honestly:** by every held-out metric (single-best-fold F1 and accuracy, and full-ensemble F1 and accuracy), 10-fold scores highest, 5-fold second, 3-fold worst. This is a consistent, monotonic ordering across all four held-out numbers, not a mixed result — more folds genuinely helped here, both by giving each fold more training data (2,578 vs. 2,291 vs. 1,909 images) and, per this project's now-repeated observation that cross-validation-only numbers can be misleading (see `specs/expanded-training-data-and-f1-improvements.md`), the held-out numbers moving in the same direction as the mean CV F1 numbers this time is itself a mildly reassuring sign that this particular improvement is not an artifact of validation-set optimism.

**But the improvement has clearly diminishing returns relative to its cost.** Going from 3 to 5 folds bought a +0.0853 single-best F1 gain (0.5009 to 0.5862, a 17% relative improvement) for 1.6x the GPU time (3h51m to 6h18m). Going from 5 to 10 folds bought only a further +0.0241 gain (0.5862 to 0.6103, a 4% relative improvement) for another 2.0x the GPU time (6h18m to 12h27m) — roughly triple the total cost of the 3-fold run for a fold count whose marginal benefit over 5-fold is modest. The ensemble numbers show an even flatter pattern: 5-fold's ensemble F1 (0.5735) is barely below 10-fold's (0.5835), a difference of only 0.01 for twice the compute.

**A separate, honest caveat that applies to all three fold counts equally:** none of the three configurations recovers this project's own pre-augmentation EfficientNet baseline (F1 0.6375, single fold, accuracy 0.6705 — see `specs/synthetic-augmentation-f1-improvement.md`'s Context and Orientation section), which was measured under the older, milder augmentation pipeline (`RandomRotation(10)`, `ColorJitter(0.2, 0.2)`, no `RandomErasing`) on this exact same 2,864-image dataset composition. The 10-fold single-best result (0.6103) comes closest but still falls 0.0272 short. This is not a finding this plan was scoped to explain (its question was fold count, not augmentation strength), but it is worth surfacing plainly: the strengthened augmentation pipeline this project adopted, combined with any fold count tested here, has not yet been shown to beat the milder augmentation it replaced. That question — whether the strengthened augmentation itself helped, hurt, or was neutral — was left permanently unanswered for ResNet-50 (its retrain under this same augmentation died mid-run on 2026-08-27 and was never completed or re-attempted) and is only now partially informed for EfficientNet-B0 by this plan's own numbers, which suggest, tentatively, that it did not help. A future plan could test this directly by re-running the pre-augmentation transform pipeline at 10-fold and comparing.

**Conclusion:** 10-fold cross-validation produced this project's best-measured EfficientNet-B0 held-out result under the current dataset and augmentation, and the ordering (10 > 5 > 3) held consistently across every metric measured. Whether that modest, diminishing-returns improvement over 5-fold (+0.0241 F1 for roughly double the training time) is worth adopting depends on whether GPU time is the binding constraint: for a one-time model-selection decision, 10-fold's stronger held-out number is the better choice; for an iterative development workflow where many retrains are expected (as this project's own history of three prior plans demonstrates), 5-fold's much lower cost for only slightly worse held-out performance is a more practical default. This plan does not update `eval_results/selected_models.json` or change what the live web app serves, per its approved scope — adopting any of these three configurations for production would be a separate, explicitly-approved follow-up decision.

## Context and Orientation

The repository root is `A:\Projects\BreastCancerDetectionCNN` (equivalently `/a/Projects/BreastCancerDetectionCNN` from a Git Bash shell on this machine). Python is invoked as `py` from PowerShell or Git Bash on this machine (there is no plain `python` command). The installed PyTorch reports CUDA available, with an NVIDIA GeForce RTX 3070 Ti (8GB) as the GPU; both training scripts already fall back to CPU automatically if no GPU is present, but this plan assumes the GPU is used, since that is what every prior training run in this project's history has used.

The training dataset is built by `MammogramRawDataset(["mass_train", "calc_train"])`, defined in `DataSetAugmentation.py`. This reads two CSV files (`data/raw/csv/mass_case_description_train_set.csv` and `data/raw/csv/calc_case_description_train_set.csv`), drops rows with a missing `pathology` label, and locates each row's mammogram JPEG file under `data/raw/jpeg/`. With no `include_cropped_patches` argument passed (the current call in `EfficientNet.py`, unaffected by this plan), this produces exactly 2,864 labeled full-mammogram-image samples — the original, non-augmented-with-crops composition that the most recent plan (`specs/synthetic-augmentation-f1-improvement.md`) deliberately reverted to. A completely separate 704-image held-out test set (428 benign, 276 malignant) is built by `evaluate.py`'s `build_test_dataset()` function from `data/raw/csv/mass_case_description_test_set.csv` and `calc_case_description_test_set.csv`; this test set has never been used for training, validation-fold splitting, or threshold tuning by any script in this project's history, and this plan does not change that — it is the one number this whole project treats as trustworthy.

`DataSetAugmentation.py`'s `get_train_transforms()` function currently returns a pipeline of: resize to 224×224, a 50%-chance horizontal flip, a `RandomAffine` (up to 15 degrees of rotation, up to 10% translation, 90%-110% zoom, up to 5 degrees of shear), a `ColorJitter` (up to ±25% brightness and contrast), conversion to a normalized tensor (using the standard ImageNet mean/std values `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`, required because both this project's model backbones were pretrained on ImageNet with this normalization), and finally a `RandomErasing` step (30% chance of blanking out a random rectangle covering 2-15% of the image). This is the "moderate" strengthened augmentation from the most recent plan; this plan does not change it, and every fold count tested in this plan trains with this exact same pipeline, so any difference in results can only be attributed to the fold count, not the augmentation.

`EfficientNet.py` defines `EfficientNetConfig` (a class holding configuration constants, including `CHECKPOINT_DIR = Path("checkpoints_efficientnet")`, `PLOT_DIR = Path("plots_efficientnet")`, and `NUM_EPOCHS = 35`), `create_efficientnet()` (builds a pretrained `torchvision.models.efficientnet_b0` with a frozen backbone and a fresh 2-output classification head), and `train_efficientnet_kfold()`, the function that currently does all of the following in one hardcoded body: creates `checkpoints_efficientnet/` and `plots_efficientnet/` if they do not exist; builds the 2,864-sample training dataset; splits it with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` ("stratified" means each fold keeps the same benign/malignant ratio as the full dataset); calls `training_log.create_workbook("efficientnet")` (from the existing `training_log.py` module, which derives the actual file path as `training_logs/{name}_training_log.xlsx` from whatever name string is passed in, and returns an `openpyxl.Workbook` object that must be saved after every write) and separately computes `log_path = training_log.LOG_DIR / "efficientnet_training_log.xlsx"` (repeating the same naming pattern rather than asking `training_log` for it, which is why this plan must change both the `create_workbook(...)` call and this `log_path` line together, consistently); then, for each fold, trains a fresh EfficientNet-B0 model for up to `EfficientNetConfig.NUM_EPOCHS` (35) epochs with early stopping (`EfficientNetConfig.EARLY_STOP_PATIENCE = 10`), staged backbone unfreezing (`EfficientNetConfig.UNFREEZE_SCHEDULE = {4: [8, 7], 10: [6], 16: [5]}`, meaning at epoch 4 two backbone stages are unfrozen, at epoch 10 one more, and at epoch 16 one more), a hybrid focal-loss-plus-cross-entropy criterion (`losses.build_hybrid_criterion`, imported already), and mixup (`losses.mixup_data`, imported already); after each fold, it reloads that fold's saved best checkpoint, re-evaluates it with test-time augmentation (`evaluate.evaluate_model`, imported already) and searches for that fold's best decision threshold (`evaluate.find_best_threshold`, imported already), logs the fold's summary row, and saves a per-fold metrics plot image; finally, after all folds, it prints a `"========== Final Cross-Validation Results (EfficientNet) =========="` block with the mean and standard deviation of validation loss, accuracy, and F1 across all folds' last epoch, and a list of each fold's own best decision threshold. The function is called with no arguments from a bare `if __name__ == "__main__": train_efficientnet_kfold()` at the bottom of the file — this is the only place the function is invoked, and the only place this plan needs to add command-line argument parsing.

`evaluate.py` already fully supports everything this plan needs for measurement, with no code changes required to it: its command-line `main()` accepts `--model` (`resnet` or `efficientnet`), `--checkpoint` (one or more paths — `argparse`'s `nargs="+"`, so passing several checkpoint paths automatically evaluates them as an ensemble via the already-existing `evaluate_ensemble` function, which loads every named checkpoint, averages their predicted probabilities per image, and applies one shared decision threshold), `--tta` (a flag enabling three-way test-time augmentation: averaging the model's output for the original image, a horizontally flipped copy, and a vertically flipped copy), `--threshold` (defaulting to `0.5`), and `--out` (an explicit output JSON path; if omitted, `evaluate.py` picks a default name under `eval_results/` that this plan avoids relying on, per the Decision Log, in favor of always passing `--out` explicitly so no pre-existing file from an earlier plan is ever silently overwritten).

## Plan of Work

### Milestone 1 — a backward-compatible `--folds` flag for `EfficientNet.py`

Add `import argparse` to `EfficientNet.py`'s existing import block at the top of the file (it is not currently imported there). Change the function signature of `train_efficientnet_kfold()` to `def train_efficientnet_kfold(n_folds=5, checkpoint_dir=None, plot_dir=None, log_name="efficientnet"):`, so that calling it exactly as before (`train_efficientnet_kfold()`, with no arguments) reproduces today's behavior exactly — this default-argument design is what guarantees backward compatibility, not a runtime check inside the function.

At the very top of the function body, immediately after the existing `if torch.cuda.is_available(): ...` block, add two lines that resolve the two path parameters to their existing defaults when the caller did not override them: `checkpoint_dir = checkpoint_dir or EfficientNetConfig.CHECKPOINT_DIR` and `plot_dir = plot_dir or EfficientNetConfig.PLOT_DIR`. Then change the two existing lines `EfficientNetConfig.CHECKPOINT_DIR.mkdir(exist_ok=True)` and `EfficientNetConfig.PLOT_DIR.mkdir(exist_ok=True)` to `checkpoint_dir.mkdir(exist_ok=True)` and `plot_dir.mkdir(exist_ok=True)`.

Change every other reference to `EfficientNetConfig.CHECKPOINT_DIR` inside `train_efficientnet_kfold()` (there are two: the `torch.save(...)` call that writes each fold's best checkpoint, and the `model.load_state_dict(torch.load(...))` call that reloads it for TTA re-evaluation) to use the local `checkpoint_dir` variable instead. Change the one reference to `EfficientNetConfig.PLOT_DIR` (in the per-fold `plt.savefig(...)` call) to use the local `plot_dir` variable instead. Do not change any reference to `EfficientNetConfig.PLOT_DIR` in the three final cross-fold summary plots (`mean_val_loss.png`, `mean_val_acc.png`, `mean_val_f1.png`) — on reflection these must also change to `plot_dir`, since otherwise every fold-count run would overwrite the same three summary plot files in the original `plots_efficientnet/` directory regardless of `n_folds`, defeating the purpose of separating the runs; change all three of these `plt.savefig(...)` calls to use `plot_dir` as well.

Change `skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` to `skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)`. Change the per-fold print statement `print(f"\n========== Fold {fold+1}/5 ==========")` to `print(f"\n========== Fold {fold+1}/{n_folds} ==========")`, so console output correctly reflects whichever fold count is running.

Change `log_wb = training_log.create_workbook("efficientnet")` to `log_wb = training_log.create_workbook(log_name)`, and change `log_path = training_log.LOG_DIR / "efficientnet_training_log.xlsx"` to `log_path = training_log.LOG_DIR / f"{log_name}_training_log.xlsx"` — these two lines must always be changed together, since `training_log.create_workbook(name)` internally derives its own save path as `training_log.LOG_DIR / f"{name}_training_log.xlsx"` (see `training_log.py`, unchanged by this plan), and `log_path` here is a second, separate computation of that same path used later for `training_log.log_epoch(log_wb, log_path, ...)` and `training_log.log_fold_summary(log_wb, log_path, ...)` calls; if the two ever disagree, the workbook object in memory and the path it gets saved to would silently diverge.

Finally, replace the file's existing `if __name__ == "__main__": train_efficientnet_kfold()` with:

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="Train EfficientNet-B0 with k-fold cross-validation."
        )
        parser.add_argument(
            "--folds", type=int, default=None,
            help="Number of cross-validation folds. Omit to reproduce the original "
                 "behavior exactly (5 folds, writing to checkpoints_efficientnet/ and "
                 "plots_efficientnet/). When given explicitly, writes instead to "
                 "checkpoints_efficientnet_<N>fold/, plots_efficientnet_<N>fold/, and "
                 "training_logs/efficientnet_<N>fold_training_log.xlsx, so it never "
                 "overwrites the original files."
        )
        args = parser.parse_args()
        if args.folds is None:
            train_efficientnet_kfold()
        else:
            train_efficientnet_kfold(
                n_folds=args.folds,
                checkpoint_dir=Path(f"checkpoints_efficientnet_{args.folds}fold"),
                plot_dir=Path(f"plots_efficientnet_{args.folds}fold"),
                log_name=f"efficientnet_{args.folds}fold",
            )

`Path` is already imported in `EfficientNet.py` (`from pathlib import Path`), so no new import is needed for that name.

Verify this milestone without spending any real training time: run `py -c "import EfficientNet"` from the repository root and confirm it prints no error (this only checks the file parses and imports cleanly, since `train_efficientnet_kfold()` is never called at import time). Then run `py EfficientNet.py --help` from the repository root and confirm it prints usage text mentioning `--folds` with the help text above, without starting any training. Do not yet run a real training job in this milestone.

### Milestone 2 — the 3-fold run

From the repository root, start, as a background process:

    py -u EfficientNet.py --folds 3

Redirect its standard output and standard error to two new files, `efficientnet_kfoldsweep_3fold.log` and `efficientnet_kfoldsweep_3fold.err.log` (names chosen to be distinct from every existing log file in this repository, so nothing already on disk is overwritten), and launch it using the coding agent's own background-process tooling rather than a manually-opened terminal window, per the Decision Log entry above. Confirm at the start of the run that the console output includes the line `Loaded 2864 labelled samples from ['mass_train', 'calc_train'] (2864 full images + 0 cropped patches)`, confirming the dataset is exactly the original, non-augmented-with-crops composition this plan expects, and that the first `========== Fold 1/3 ==========` banner appears (proving the new `--folds` flag actually took effect, not silently defaulting to 5).

This run is expected, based on this project's own historical per-epoch timings (roughly 2-3 minutes per epoch for EfficientNet-B0 on this dataset size, though the exact figure has varied between 137 and 170 seconds per epoch across two historical runs), to take very roughly 3.5 to 5.25 hours for 3 folds at up to 35 epochs each (early stopping may end some folds sooner, as it has in prior runs). Wait for the run to finish — do not proceed to Milestone 3 until it prints its `"========== Final Cross-Validation Results (EfficientNet) =========="` block with zero content in `efficientnet_kfoldsweep_3fold.err.log`. Confirm `checkpoints_efficientnet_3fold/` now contains exactly `best_model_fold_1.pth`, `best_model_fold_2.pth`, and `best_model_fold_3.pth`, and that `training_logs/efficientnet_3fold_training_log.xlsx` exists with a "Fold Summary" sheet containing exactly 3 data rows (plus 1 header row).

Once training finishes, identify the single best fold by its own TTA-evaluated F1 (printed as `"Fold N best-val F1 (with TTA): ..."` for each fold, and also recorded in the "Fold Summary" sheet's `tta_f1` column) — call this fold's number `B3`. Run, from the repository root:

    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_3fold/best_model_fold_B3.pth --tta --out eval_results/kfoldsweep_efficientnet_3fold_singlebest_test_metrics.json
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_3fold/best_model_fold_1.pth checkpoints_efficientnet_3fold/best_model_fold_2.pth checkpoints_efficientnet_3fold/best_model_fold_3.pth --tta --out eval_results/kfoldsweep_efficientnet_3fold_ensemble_test_metrics.json

(replacing `best_model_fold_B3.pth` with the actual winning fold's filename). Record both resulting JSON files' `accuracy` and `f1` values, plus the training script's own final mean/std cross-validation F1, in the Outcomes section before proceeding to Milestone 3.

### Milestone 3 — the 5-fold run

Identical in structure to Milestone 2, but with `--folds 5`:

    py -u EfficientNet.py --folds 5

writing to `efficientnet_kfoldsweep_5fold.log`/`.err.log`, producing `checkpoints_efficientnet_5fold/best_model_fold_{1..5}.pth`, `plots_efficientnet_5fold/`, and `training_logs/efficientnet_5fold_training_log.xlsx`. This is a fresh run distinct from the pre-existing `checkpoints_efficientnet/` directory (which holds a 5-fold run from a different, crop-augmented dataset per a prior plan, and which `eval_results/selected_models.json` still points to for the live web app) — do not read from, evaluate, or otherwise treat the pre-existing `checkpoints_efficientnet/` files as part of this comparison; they are not a fair comparison point since they were trained on different data.

Expected duration: very roughly 5.8 to 8.75 hours for 5 folds. After it completes (same acceptance signals as Milestone 2, adjusted for 5 folds), identify the best fold (`B5`) and run:

    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_5fold/best_model_fold_B5.pth --tta --out eval_results/kfoldsweep_efficientnet_5fold_singlebest_test_metrics.json
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_5fold/best_model_fold_1.pth checkpoints_efficientnet_5fold/best_model_fold_2.pth checkpoints_efficientnet_5fold/best_model_fold_3.pth checkpoints_efficientnet_5fold/best_model_fold_4.pth checkpoints_efficientnet_5fold/best_model_fold_5.pth --tta --out eval_results/kfoldsweep_efficientnet_5fold_ensemble_test_metrics.json

Record results in Outcomes before proceeding.

### Milestone 4 — the 10-fold run

Identical in structure again, with `--folds 10`:

    py -u EfficientNet.py --folds 10

writing to `efficientnet_kfoldsweep_10fold.log`/`.err.log`, producing `checkpoints_efficientnet_10fold/best_model_fold_{1..10}.pth`, `plots_efficientnet_10fold/`, and `training_logs/efficientnet_10fold_training_log.xlsx`. Expected duration: very roughly 11.6 to 17.5 hours for 10 folds — the longest of the three runs, since each of the 10 folds independently trains up to 35 epochs, and 10 folds is twice as many independent training runs as the 5-fold case for only a slightly larger per-fold training set (2,864 × 9/10 ≈ 2,578 images per fold, versus 2,864 × 4/5 ≈ 2,291 for 5-fold, versus 2,864 × 2/3 ≈ 1,909 for 3-fold).

After completion, identify the best fold (`B10`) and run:

    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_10fold/best_model_fold_B10.pth --tta --out eval_results/kfoldsweep_efficientnet_10fold_singlebest_test_metrics.json
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_10fold/best_model_fold_1.pth checkpoints_efficientnet_10fold/best_model_fold_2.pth checkpoints_efficientnet_10fold/best_model_fold_3.pth checkpoints_efficientnet_10fold/best_model_fold_4.pth checkpoints_efficientnet_10fold/best_model_fold_5.pth checkpoints_efficientnet_10fold/best_model_fold_6.pth checkpoints_efficientnet_10fold/best_model_fold_7.pth checkpoints_efficientnet_10fold/best_model_fold_8.pth checkpoints_efficientnet_10fold/best_model_fold_9.pth checkpoints_efficientnet_10fold/best_model_fold_10.pth --tta --out eval_results/kfoldsweep_efficientnet_10fold_ensemble_test_metrics.json

Record results in Outcomes before proceeding to Milestone 5.

### Milestone 5 — final comparison and conclusion

Once all three fold counts have been trained and evaluated, write a comparison table (in prose, per the plan-wide formatting rule of favoring prose over tables, but a short indented table of numbers is acceptable here as captured evidence) into the Outcomes section covering, for each of 3, 5, and 10 folds: the number of training samples per fold (train/validation split size), the script's own mean ± standard deviation cross-validation F1, the held-out test F1 for the single best fold, the held-out test F1 for the full ensemble, and the total wall-clock training time actually observed (not estimated). State plainly which fold count produced the best held-out test F1 (the one number this project treats as trustworthy) — not merely the best cross-validation F1, which this project's own history (`specs/expanded-training-data-and-f1-improvements.md`) has already shown can diverge meaningfully from real held-out performance. Explicitly discuss the cost/benefit tradeoff: if a higher fold count wins by only a small margin at multiple times the GPU cost, say so plainly rather than declaring it an unqualified win. Do not update `eval_results/selected_models.json` or otherwise change what the live web app serves as part of this plan — this plan is a comparison experiment only; adopting a new configuration for the live app, if warranted, would be a separate, explicitly-approved follow-up decision.

## Concrete Steps

Run every command below from the repository root, `A:\Projects\BreastCancerDetectionCNN`:

    py -c "import EfficientNet"
    py EfficientNet.py --help
    py -u EfficientNet.py --folds 3
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_3fold/best_model_fold_<B3>.pth --tta --out eval_results/kfoldsweep_efficientnet_3fold_singlebest_test_metrics.json
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_3fold/best_model_fold_1.pth checkpoints_efficientnet_3fold/best_model_fold_2.pth checkpoints_efficientnet_3fold/best_model_fold_3.pth --tta --out eval_results/kfoldsweep_efficientnet_3fold_ensemble_test_metrics.json
    py -u EfficientNet.py --folds 5
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_5fold/best_model_fold_<B5>.pth --tta --out eval_results/kfoldsweep_efficientnet_5fold_singlebest_test_metrics.json
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_5fold/best_model_fold_1.pth checkpoints_efficientnet_5fold/best_model_fold_2.pth checkpoints_efficientnet_5fold/best_model_fold_3.pth checkpoints_efficientnet_5fold/best_model_fold_4.pth checkpoints_efficientnet_5fold/best_model_fold_5.pth --tta --out eval_results/kfoldsweep_efficientnet_5fold_ensemble_test_metrics.json
    py -u EfficientNet.py --folds 10
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_10fold/best_model_fold_<B10>.pth --tta --out eval_results/kfoldsweep_efficientnet_10fold_singlebest_test_metrics.json
    py evaluate.py --model efficientnet --checkpoint checkpoints_efficientnet_10fold/best_model_fold_1.pth checkpoints_efficientnet_10fold/best_model_fold_2.pth checkpoints_efficientnet_10fold/best_model_fold_3.pth checkpoints_efficientnet_10fold/best_model_fold_4.pth checkpoints_efficientnet_10fold/best_model_fold_5.pth checkpoints_efficientnet_10fold/best_model_fold_6.pth checkpoints_efficientnet_10fold/best_model_fold_7.pth checkpoints_efficientnet_10fold/best_model_fold_8.pth checkpoints_efficientnet_10fold/best_model_fold_9.pth checkpoints_efficientnet_10fold/best_model_fold_10.pth --tta --out eval_results/kfoldsweep_efficientnet_10fold_ensemble_test_metrics.json

(`<B3>`, `<B5>`, `<B10>` are placeholders for whichever fold number actually wins each run, determined from that run's own console output/Excel log — they cannot be known in advance.)

## Validation and Acceptance

Milestone 1 is accepted when `py -c "import EfficientNet"` succeeds with no output/error, and `py EfficientNet.py --help` prints usage text documenting `--folds` without starting training.

Milestones 2, 3, and 4 are each accepted when: the corresponding `py -u EfficientNet.py --folds N` run completes with zero content in its `.err.log`; the run's checkpoint directory contains exactly N `best_model_fold_*.pth` files; the run's Excel log's "Fold Summary" sheet contains exactly N data rows; and both of that fold count's `evaluate.py` commands (single-best-fold and full-ensemble) produce JSON files under `eval_results/` with plausible, non-NaN `accuracy`/`f1` values between 0 and 1.

Milestone 5 is accepted when the Outcomes section contains a complete, honest comparison of all three fold counts (cross-validation F1, held-out single-best F1, held-out ensemble F1, and actual wall-clock time for each), a plain statement of which fold count produced the best held-out F1, and an explicit discussion of whether that improvement (if any) justifies its GPU-time cost relative to the other two fold counts.

## Idempotence and Recovery

Each `py EfficientNet.py --folds N` run is independent of the others (separate checkpoint/plot/log directories) and independent of the pre-existing `checkpoints_efficientnet/` directory, so the three runs can be executed, re-run, or recovered from in any order without affecting each other or the live web app. Re-running `py EfficientNet.py --folds N` overwrites that fold count's own checkpoint, plot, and log files in place (the pre-existing per-fold-count behavior this plan's design carries forward from the original script) and cannot resume a partially completed fold; if a run is interrupted (for example, if the machine restarts or the coding agent's background process is killed), the safe recovery path is to restart that specific fold-count's run from the beginning, exactly as every prior training run in this project has required. Because each run is launched as a background process managed by the coding agent's own tooling rather than a separate, closable terminal window, the specific failure mode that ended the most recent prior attempt (a closed window sending a close signal to the training process) should not recur, but the repository owner should still avoid closing the coding agent's own session/terminal while a run is in progress, since doing so could still terminate the underlying process tree.

`evaluate.py` is read-only with respect to the dataset and checkpoints; re-running any of its commands in this plan simply overwrites that one JSON output file and is always safe to repeat.

## Artifacts and Notes

Every real number this plan produces — each fold count's mean cross-validation F1, held-out single-best F1, held-out ensemble F1, actual wall-clock training time, and the final comparison and conclusion — must be pasted into the Outcomes & Retrospective section as the plan proceeds, so a reader of only this document can see the actual results without re-running anything.

## Interfaces and Dependencies

In `EfficientNet.py`, `train_efficientnet_kfold()` changes from a zero-argument function to:

    def train_efficientnet_kfold(n_folds=5, checkpoint_dir=None, plot_dir=None, log_name="efficientnet"):
        ...

Calling it with no arguments (`train_efficientnet_kfold()`) must remain exactly equivalent to today's behavior. No other function in `EfficientNet.py`, and no function in any other file (`DataSetAugmentation.py`, `losses.py`, `training_log.py`, `evaluate.py`, `inference.py`, `app.py`, `ResNet.py`), changes signature or behavior as part of this plan. `evaluate.py`'s existing `main()`, `evaluate_model()`, and `evaluate_ensemble()` functions are used exactly as they exist today, with no modification.
