# Climb the image-resolution ladder for both models, then add lesion-patch pretraining to both

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not check in a copy of PLANS.md. The authoring and execution rules for this document live at `~/.agents/PLANS.md` on the machine of whoever is running this plan (a global, agent-agnostic file, not part of this git repository). This document must be maintained in accordance with that file.

This plan builds on the checked-in ExecPlan `specs/higher-resolution-1024x576-both-models.md`, which trained both models at 1024 by 576. Everything needed from it is restated below. (The file name of this plan predates its revision; see the revision note at the bottom.)

## Approval

Status: Approved

Approved by: Repository owner, chat instruction on 2026-10-05: "I want you to continuously run the resolution improvements first until it no longer shows improvement or the hardware cant handle it. Then you will create the pre-train for the lesions and run it for both models."

Date: 2026-10-05

Amendment approved 2026-10-07 (owner, in chat: "I approve of a-c"): (a) every run is scored in the precision it was trained in, so the ladder metric for a 16-bit run is its 16-bit validation TTA F1; (b) `evaluate.py` gains an opt-in `--amp` option (and `evaluate_model`, `evaluate_ensemble` and `find_best_threshold` an `amp=False` parameter, which both training scripts pass their `USE_AMP` setting to), and 16-bit-trained runs are tested on the held-out set with it; (c) `inference.py` and the web app stay unchanged in this plan, with the requirement recorded that 16-bit-trained models must be served in 16-bit.

Approved scope: Both models (ResNet-50 and EfficientNet-B0). (1) Add opt-in bfloat16 mixed precision, gradient accumulation and a loader-worker-count option to `ResNet.py` and `EfficientNet.py`. (2) For each model, train 5-fold runs at successively higher resolutions (1536 by 864, then 2048 by 1152, and so on) until a step fails to improve mean validation F1 or the hardware rule fails. (3) Add `make_patches.py` and `PatchPretrain.py`, pretrain each model's backbone on lesion and normal-tissue patches once per fold, add `--patch-init` to both training scripts, and train each model from its pretrained backbones at its best resolution. (4) Evaluate every completed run on the held-out test set. One line added to `.gitignore`. No change to the web app, `evaluate.py`, `inference.py` or `DataSetAugmentation.py`.

## Purpose / Big Picture

This project classifies mammograms from the public CBIS-DDSM dataset as benign or malignant with two networks, a ResNet-50 and an EfficientNet-B0, served by a Flask web app. Raising the training image size has been the most effective change so far: going from 512 by 288 to 1024 by 576 raised held-out F1 from 0.660 to 0.690 for ResNet-50 and from 0.647 to 0.667 for EfficientNet-B0. F1 is the main target; it balances precision (of the images called malignant, how many were) and recall (of the malignant images, how many were found).

The source mammograms are about 3,000 by 5,100 pixels, so 1024 by 576 still shrinks them about five times along each side, and microcalcifications (calcium specks 0.1 to 0.5 mm across, one of the two lesion types in this dataset) are only one or two pixels wide. The first part of this plan keeps raising the resolution, one step at a time and separately for each model, until a step no longer improves validation F1 or the 8 GB GPU cannot run it in reasonable time. Higher resolutions do not fit the GPU the way training works today, so the plan adds two standard memory savers. Mixed precision runs most of the network's arithmetic in a 16-bit number format (bfloat16) instead of 32-bit, roughly halving the memory that stored intermediate results ("activations") take. Gradient accumulation splits each batch of 8 images into smaller pieces processed one after another, adding up their gradients (the per-weight update directions) before taking one optimizer step, so a batch of 8 behaves almost the same as before while fewer images are in memory at once.

The second part teaches each network what lesions look like before it has to find them in a whole mammogram. A lesion covers well under 1 percent of a mammogram, so a network trained only on whole images with one label each gets a weak signal about where to look. The published approach for this dataset (Shen et al., 2019, "Deep Learning to Improve Breast Cancer Detection on Screening Mammography") first trains on small square patches cut around marked lesions and from normal tissue, then fine-tunes on whole images. CBIS-DDSM supplies a mask image for every lesion marking its outline, so patches can be cut automatically. This differs from an experiment this project ran in August that mixed the dataset's own lesion close-ups into whole-image training and gave no gain: those close-ups were at a different scale from the whole images, had no normal-tissue examples, and were used in the same training run. Here patches are cut at exactly the scale the whole-image network sees, include a normal-tissue class, and are used in a separate stage beforehand.

After this plan, the repository holds, for each model, its best resolution and a patch-pretrained run at that resolution, with a table of mean validation F1 and held-out accuracy, precision, recall and F1 for every run, so anyone can see what each change bought.

## Progress

- [x] (2026-10-05) Milestone 1: added `--amp`, `--accum-steps` and `--num-workers` to `ResNet.py` and `EfficientNet.py`. Equivalence: running the real `train_epoch`/`train_one_epoch` (SAM included) on CPU in float64, one update over 1 x 8 images and over 2 x 4 accumulated images give the same gradients at both SAM steps to within 9e-16 for both models (PASS). Smoke tests at the throwaway size 1024x577 printed `Config: ... accum_steps=1 effective_batch=8 amp=False workers=4` (ResNet) and `workers=8` (EfficientNet), loaded 2,864 samples and trained normally (ResNet two epochs, EfficientNet one) with empty stderr; their folders and Excel logs were deleted. Probes recorded below.
- [ ] Milestone 2: resolution ladder (repeat per rung, per model: probe, decide, train, evaluate, compare).
  - [x] (2026-10-06 12:31) ResNet-50 rung 1536x864: ran 2026-10-05 23:31 to 2026-10-06 12:25 (12 h 54 min), empty error log, folds 2 and 5 stopped early. Validation TTA F1 by fold 0.7702, 0.7405, 0.7753, 0.7650, 0.7848; mean 0.7671 ± 0.0149 (1024x576: 0.7382). Held-out ensemble acc 0.7656, prec 0.7079, rec 0.6848, F1 0.6961; best fold (5) acc 0.6932, F1 0.6550. Ladder rule: improvement by the logged (bfloat16) measure; provisional until the 32-bit per-fold TTA F1 is recomputed (see Surprises), which must beat 0.7432 for ResNet to climb to 2048x1152.
  - [x] (2026-10-06 20:19) EfficientNet-B0 rung 1536x864: ran 12:31 to 20:14 (7 h 43 min), empty error log, 5 checkpoints, fold 4 stopped early. Logged validation F1 by fold, no TTA (bfloat16) 0.7395, 0.7583, 0.7545, 0.7445, 0.7617 (mean 0.7517); with TTA (32-bit) 0.7110, 0.7153, 0.7531, 0.6603, 0.7206 (mean 0.7121; 1024x576: 0.7395). Held-out ensemble acc 0.7131, prec 0.6114, rec 0.7355, F1 0.6678 (1024x576: 0.7173, 0.6199, 0.7210, 0.6667); best fold (3) acc 0.6761, F1 0.6617. Ladder decision pending the 32-bit rescoring.
  - [x] (2026-10-06) 32-bit rescoring of every fold (TF32 disabled, F1 at 0.5). Means, no TTA / TTA both flips / TTA horizontal only: ResNet 1024x576 0.7335 / 0.7387 / 0.7344; ResNet 1536x864 0.7120 / 0.7169 / 0.7017; EfficientNet 1024x576 0.7283 / 0.7395 / 0.7363; EfficientNet 1536x864 0.7052 / 0.7122 / 0.7169. The 1024x576 values reproduce the original logs. (A first attempt hung for 1 h 50 min because the scratchpad script lacked a `__main__` guard, so Windows loader workers re-ran it; fixed and rerun.)
  - [x] (2026-10-07) Ladder decision for 1536x864, after an AUC diagnostic and the owner's approval of the precision amendment: each run scored in its training precision. ResNet 1024x576 (32-bit) 0.7387 vs 1536x864 (16-bit) 0.7671: improvement. EfficientNet 1024x576 (32-bit) 0.7395 vs 1536x864 (16-bit) 0.7614: improvement. Both climb to 2048x1152.
  - [x] (2026-10-07) `evaluate.py --amp` added. Regression: 32-bit ResNet 1024x576 ensemble reproduces acc 0.7443, F1 0.6897 exactly. 16-bit held-out evaluations of the 1536x864 runs (TTA, threshold 0.5): ResNet ensemble acc 0.7401, prec 0.6495, rec 0.7319, F1 0.6882; ResNet fold 5 acc 0.7145, F1 0.6564; EfficientNet ensemble acc 0.7230, prec 0.6302, rec 0.7101, F1 0.6678; EfficientNet fold 5 acc 0.7386, F1 0.6703. On the held-out set 1536x864 ties 1024x576 for both models (all differences within the roughly 0.02 to 0.03 noise of 704 images), so the validation gain is not confirmed there; per the plan, held-out results do not drive the ladder.
  - [x] (2026-10-08 10:51) 2048x1152 rung finished; both ladders stop, chosen size 1536x864 (16-bit, micro-batch 8, no accumulation) for both models. ResNet ran 09:04 to 00:19 (15 h 15 min), every fold stopped early; validation TTA F1 (16-bit) 0.7034, 0.7339, 0.7192, 0.7674, 0.7568, mean 0.7361 against 0.7671 at 1536. EfficientNet ran 00:26 to 10:47 (10 h 20 min); 0.6933, 0.6907, 0.7426, 0.7443, 0.7359, mean 0.7214 against 0.7614. Held-out (16-bit, TTA, 0.5): ResNet ensemble acc 0.6974, prec 0.5935, rec 0.7246, F1 0.6525, best fold (4) F1 0.6801; EfficientNet ensemble acc 0.6960, prec 0.6040, rec 0.6522, F1 0.6272, best fold (4) F1 0.6561. Both error logs empty, five checkpoints each.
  - [x] (2026-10-07 09:04) 2048x1152 rung launch details: Probe: 16-bit micro-batch 8 exceeds 4,500 MB (ResNet 6,054, EfficientNet 5,049), so 16-bit with micro-batch 4 and 2 accumulation steps (ResNet 4,007 MB, 3.89 GPU-min per epoch, 17.8 h upper bound; EfficientNet 3,233 MB, 2.29 min, 6.7 h; loading 1.92 min per epoch). ResNet first, then EfficientNet with 4 workers; chain evaluations use `--amp`. ResNet epoch 1 logged cleanly.
  - [ ] ResNet-50 rung 2048x1152 (probe when the GPU is free, after the EfficientNet run).
  - [ ] Further rungs as the ladder rule allows (add one line per rung here as they are decided).
- [x] (2026-10-08 11:04) Milestone 3: `make_patches.py` written, `data/patches_*/` added to `.gitignore`, patch set built at 1536x864 (one set serves both models) in about 10 minutes: 2,864 lesion rows on 2,458 mammograms, 66 skipped (2.3 percent); normal 7,096, benign 1,633, malignant 1,165 patches (161 MB). By-eye check of 20 random patches per class passed: lesion patches show masses or calcification clusters near the centre, normal patches show tissue; some patches include black edge padding, the scanner's white border strip or a film label.
- [x] (2026-10-08 12:59) Milestone 4: `PatchPretrain.py` written; pretraining ran 11:03 to 12:59 (ResNet 60 min, EfficientNet 55 min), empty error logs, five backbones each. Acceptance per fold (3-class accuracy minus largest-class share, must be at least +0.10; lesion F1 must exceed 0.55), at the best (lowest validation loss) epoch: ResNet +0.139, +0.120, +0.143, +0.108, +0.104 with lesion F1 0.713, 0.683, 0.779, 0.698, 0.706, all PASS. EfficientNet +0.117, +0.077, +0.127, +0.099, +0.082 with lesion F1 0.750, 0.701, 0.762, 0.697, 0.737: lesion F1 passes everywhere but the accuracy margin fails in folds 2, 4 and 5, so per the plan EfficientNet's whole-image run waits for the owner's decision. Best epochs were early (ResNet 5 to 8, EfficientNet 4 to 5 of 12).
- [x] (2026-10-20 04:30) Milestone 5: `--patch-init` added to both training scripts; checked on CPU that a saved backbone loads into each whole-image model with every tensor equal and the head untouched (ResNet 318 tensors, EfficientNet 358). ResNet patch-initialised run (1536x864, 16-bit, batch 8, same settings as the plain 1536 run) ran 2026-10-08 13:00 to 2026-10-09 01:35 (12 h 35 min), empty error log, five checkpoints of 107,226,339 bytes; folds 1, 2, 4 and 5 stopped early. Validation TTA F1 (16-bit) by fold 0.7709, 0.8090, 0.8208, 0.7016, 0.7847, mean 0.7774 ± 0.0418 (without pretraining: 0.7671 ± 0.0149). Held-out (16-bit, TTA, 0.5): ensemble acc 0.7741, prec 0.6931, rec 0.7609, F1 0.7254; best fold (3) acc 0.7812, prec 0.7103, rec 0.7464, F1 0.7279. EfficientNet patch-initialised run (1536x864, 16-bit, batch 8, 4 workers, same settings as the plain 1536 run), launched on the owner's instruction despite the Milestone 4 accuracy-margin miss: ran 2026-10-19 19:51 to 2026-10-20 04:13 (8 h 22 min), empty error log, five checkpoints of 16,342,523 bytes. Validation F1 by fold, no TTA 0.7511, 0.7689, 0.7983, 0.7727, 0.8009 (plain 1536: 0.7395, 0.7583, 0.7545, 0.7445, 0.7617; higher in every fold); with TTA (16-bit) 0.7424, 0.7833, 0.8026, 0.7702, 0.7879, mean 0.7773 ± 0.0227 (without pretraining: 0.7614). Held-out (16-bit, TTA, 0.5): ensemble acc 0.7514, prec 0.6613, rec 0.7500, F1 0.7029 (plain 1536: acc 0.7230, F1 0.6678); best fold (3) acc 0.7543, prec 0.6667, rec 0.7464, F1 0.7043.
- [ ] Milestone 6: write the comparison in Outcomes & Retrospective.

## Surprises & Discoveries

- Observation: on the GPU, the 32-bit equivalence check first appeared to fail for ResNet (gradients from 1 x 8 and 2 x 4 differed by 1.6 to 3.9 percent), while EfficientNet passed once TF32 was disabled. The cause is GPU arithmetic, not the accumulation code: by default this GPU runs 32-bit convolutions in TF32 (a reduced-precision mode), and ResNet's conv weight gradients are sensitive to it. The full-batch GPU gradient that every earlier run used is itself 4.4 percent away from a float64 reference, more than the 3.2 percent accumulation difference.
  Evidence: CPU float64, 1 x 8 vs 2 x 4: 3.96e-16. GPU default settings: identical full batch twice 0.0; 1 x 8 vs 2 x 4 3.21e-02; full batch vs GPU float64 4.44e-02. Forward outputs 1 x 8 vs 2 x 4: max abs difference 1.1e-08.

- Observation: this plan's probe reports higher peak memory than the previous plan's at the same setting (ResNet 1024x576 batch 8: 3,399 MB against 2,620 MB; EfficientNet: 5,355 MB against 1,967 MB), while GPU time matches (1.77 against 1.7, and 0.86 against 0.8 GPU-minutes per epoch). This probe enables `cudnn.benchmark`, as both training scripts do; its trial runs allocate temporary workspace that counts toward the peak. The hardware rule's 4,500 MB limit is applied to these higher, more conservative numbers. EfficientNet trained successfully at 1024x576 with a 5,355 MB peak by this measure, so the limit is safe.

- Observation: after the ResNet smoke test was stopped with `timeout`, one of its data-loader worker processes (PID 31640) stayed listed as a zombie: `taskkill` reports "There is no running instance of the task", it uses no CPU, holds about 560 MB of RAM, and has no GPU context. It cannot be killed and should disappear at the next reboot. Later smoke tests were stopped with `taskkill /T` on the whole process tree instead, which left no orphans.

- Observation: with `--amp`, ResNet's per-fold "best-val F1 (with TTA)" comes from `validate()`, which now runs under bfloat16, while the "best threshold" line comes from `evaluate.find_best_threshold`, which runs in 32-bit. The two disagree: in four of five folds the tuned-threshold F1 (which includes 0.5 among its candidates, so it is at least the 32-bit F1 at 0.5) is lower than the bfloat16 TTA F1. So bfloat16 inference shifts enough borderline predictions to move fold F1 by up to about 0.02, and the ladder metric is not measured in the same precision as the 1024x576 baseline (32-bit). The log only bounds the 32-bit numbers from above (per-fold 32-bit F1 at 0.5 is at most 0.7514, 0.7336, 0.7576, 0.7558, 0.7761, mean at most 0.7549), so the ResNet 1536 ladder decision is provisional. To settle it, the 32-bit per-fold TTA F1 will be recomputed for every bfloat16 run with `evaluate.evaluate_model` on the identical validation splits, and those numbers are what the ladder rule uses from now on. EfficientNet is not affected the same way, because its TTA F1 already uses the 32-bit `evaluate.evaluate_model`.
  Evidence: ResNet 1536 log, fold 1: `best-val F1 (with TTA): 0.7702` but `best threshold: 0.35 (val F1 0.7514 vs default-0.5 F1 0.7702)`.

- Observation: scoring the same 16-bit-trained ResNet 1536 checkpoints in 32-bit instead of 16-bit lowers mean validation TTA F1 at 0.5 from 0.7671 to 0.7169, a 0.05 drop, far more than ordinary rounding. Per fold (16-bit to 32-bit): 0.7702 to 0.7072, 0.7405 to 0.6934, 0.7753 to 0.7535, 0.7650 to 0.6667, 0.7848 to 0.7637. Leading hypothesis: models trained with 16-bit forward passes produce slightly shifted probabilities when run in 32-bit, and with many mammograms near the 0.5 cut-off, F1 at a fixed threshold swings a lot. Under this hypothesis ranking quality (ROC-AUC) should be almost the same in both precisions.
  Evidence: rescoring output above; held-out (32-bit) ensemble accuracy 0.7656 for ResNet 1536 against 0.7443 at 1024.
  Resolution (2026-10-07): the hypothesis was wrong. A threshold-free diagnostic shows ranking quality itself depends on precision, and every model scores best in the precision it was trained in. Mean validation TTA AUC, scored in 32-bit / 16-bit: ResNet 1024x576 (trained 32-bit) 0.8745 / 0.8594; ResNet 1536x864 (trained 16-bit) 0.8548 / 0.8913; EfficientNet 1024x576 (trained 32-bit) 0.8722 / 0.8357; EfficientNet 1536x864 (trained 16-bit) 0.8464 / 0.8869. This held in every fold for the 1536 runs. Consequences: the ladder must compare runs in their training precision (amendment (a)); the 32-bit held-out numbers first reported for the 1536 runs understate them; and any 16-bit-trained model must be tested and served in 16-bit.

- Observation: run times exceed the probe projections. ResNet 1536 took 12 h 54 min against a 9.6 h upper bound, and EfficientNet 1536 had finished only 2 of 5 folds after 4 h 40 min against a 4.6 h bound, with GPU utilisation around 43 percent. Image loading, not the GPU, limits speed, and the 40-image loading probe underestimates it.

- Observation: the desktop's GPU use was 644 MiB at the time of the probes (it was about 2,665 MiB during the previous plan).

## Decision Log

- Decision: both models, per the owner's instruction. Each model climbs its own resolution ladder, because the two networks differ in memory use and may peak at different sizes, and its patch pretraining uses its own best size.
  Rationale: owner's instruction of 2026-10-05. Patch features only transfer if patches are at the whole-image scale, so each model needs its own chosen size first.
  Date/Author: 2026-10-05, agent per owner.

- Decision: the ladder rungs are 1536 by 864, 2048 by 1152, 2560 by 1440 and 3072 by 1728 (height by width; every rung keeps the 16-to-9 shape used since 512 by 288, which is close to the mammograms' natural 1.7-to-1). Each model starts from its completed 1024 by 576 run as the current best.
  Rationale: steps of 1.33 to 1.5 times per side are large enough to show a measurable effect and few enough to finish in days. 3072 is near the source images' own short side (about 3,000 pixels), above which upsampling would add nothing.
  Date/Author: 2026-10-05, drafted by agent.

- Decision (ladder rule): after a rung's run finishes, compute its mean validation F1 with TTA (the mean of the five "best-val F1 (with TTA)" lines in its log). The rung counts as an improvement only if this beats the model's current best by at least 0.005. If it does, it becomes the current best and the model climbs to the next rung. If it does not, that model's ladder stops and its previous best stays. The ladder also stops for a model if the next rung fails the hardware rule. Held-out test scores are measured and reported for every run but never decide anything.
  Rationale: the five folds' spread is about 0.014 to 0.019, so differences under about 0.005 are indistinguishable from noise; requiring a margin stops the ladder from climbing on luck. Choosing by validation keeps the reported test numbers honest. Current bests: ResNet-50 0.7382 (per fold 0.7277, 0.7323, 0.7478, 0.7137, 0.7695); EfficientNet-B0 0.7395 (per fold 0.7329, 0.7243, 0.7221, 0.7582, 0.7598, fold 5 recomputed after that run's salvage).
  Date/Author: 2026-10-05, drafted by agent.

- Decision (hardware rule, per model, per rung): probe these configurations in order and take the first that passes: 32-bit with micro-batch 8; bfloat16 with micro-batch 8; bfloat16 with micro-batch 4 and 2 accumulation steps; bfloat16 with micro-batch 2 and 4 accumulation steps. A configuration passes if the probe completes without an out-of-memory error, peak allocated GPU memory is at most 4,500 MB, and the projected 5-fold run time is at most 30 hours. The projection is the larger of GPU minutes per epoch and image-loading minutes per epoch, times the model's epoch cap (55 ResNet, 35 EfficientNet), times 5 folds. If none passes, the hardware cannot handle the rung and that model's ladder stops.
  Rationale: the effective batch stays 8 (what both 1024 by 576 runs used) so the only intended change between rungs is resolution. A larger micro-batch is preferred because both networks contain batch normalisation layers, which normalise each micro-batch by its own statistics; with few images those are noisy, and accumulation does not fix that, so accumulation is used only as far as memory requires. 32-bit is tried first because it is what every earlier run used. Micro-batch 1 is excluded because ResNet's classifier head uses a batch normalisation layer that fails on a single image. About 3 GB of the card is used by the desktop, so 4,500 MB leaves headroom. The 30-hour cap reflects the owner's instruction to go as far as the hardware allows while keeping each run within a day and a bit.
  Date/Author: 2026-10-05, drafted by agent.

- Decision: mixed precision uses `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` with no gradient scaler.
  Rationale: the installed PyTorch 2.10.0+cu126 reports `torch.cuda.is_bf16_supported() == True` on this RTX 3070 Ti. bfloat16 has the same numeric range as 32-bit floats, so it does not need the "gradient scaler" that the other 16-bit format (float16) needs to keep small gradients from rounding to zero. That matters because both scripts train with SAM (explained in Context), which takes two gradient steps per batch and is awkward to combine with a scaler. Weights, optimizer state and gradients stay 32-bit.
  Date/Author: 2026-10-05, drafted by agent.

- Decision: when accumulation is on (`--accum-steps` above 1), the training loader drops its last incomplete batch (`drop_last=True`); otherwise the loader is unchanged.
  Rationale: a fold's training set has about 2,291 images, so with a micro-batch of 2 the last batch can be a single image, which crashes ResNet's classifier-head batch normalisation. Dropping at most a few images per epoch (in random order, so different ones each time) is negligible. Keeping the loader unchanged without accumulation preserves today's behaviour exactly.
  Date/Author: 2026-10-05, drafted by agent.

- Decision: add `--num-workers N` to both training scripts (defaults unchanged: 4 for ResNet, 8 for EfficientNet) and use 4 for EfficientNet in this plan's runs.
  Rationale: the 1024 by 576 EfficientNet run hung in fold 5 when a loader worker process failed to start with `OSError: [WinError 1455] The paging file is too small`. Each worker holds about 1.66 GB of Windows commit memory (RAM plus page file), and EfficientNet uses 8. ResNet's 4 workers have never failed. Image loading limits training speed, so 4 workers may slow EfficientNet somewhat; the probe measures this, and an unattended multi-day sequence cannot afford a silent hang.
  Date/Author: 2026-10-05, drafted by agent.

- Decision: run order is ResNet-50 rung 1536, EfficientNet-B0 rung 1536, then alternate between the models' next rungs while either ladder is still climbing. Each run starts only after the previous one finished cleanly. After both ladders stop, patch work begins.
  Rationale: one GPU, so one run at a time. Alternating gives both models results early.
  Date/Author: 2026-10-05, drafted by agent.

- Decision: patch pretraining is done separately for each of the 5 folds using only that fold's training patients.
  Rationale: whole-image training uses 5-fold cross-validation split by patient. Pretraining once on all 2,864 training images would mean every fold's validation lesions had been seen, with labels, before validation, inflating validation scores, which this project uses to choose models and thresholds without touching the test set. Patches are small, so five pretraining runs per model take hours, not days.
  Date/Author: 2026-10-05, drafted by agent.

- Decision: patches are cut from each mammogram after resizing it to the model's chosen training size: a 288 by 288 region centred on each lesion, plus up to 3 random normal-tissue regions per mammogram. The patch network trains on random 224 by 224 crops of them with three classes: normal tissue, benign lesion, malignant lesion.
  Rationale: features learned on patches only transfer if lesions look the same size as in whole images, so patches come from the same resized image. Storing 288 and cropping 224 at random moves the lesion around within the patch, so the network does not learn that lesions are always dead centre. The normal-tissue class teaches the network to tell lesions from the dense, bright tissue around them.
  Date/Author: 2026-10-05, drafted by agent.

- Decision: whole-image training from patch-pretrained weights keeps everything else identical to the same model's best ladder run (same size, micro-batch, accumulation, precision, unfreeze schedule).
  Rationale: one change at a time. Layers that the unfreeze schedule keeps fixed still benefit, because the fixed weights are now the patch-pretrained ones.
  Date/Author: 2026-10-05, drafted by agent.

- Decision (rung 1536 by 864, both models): bfloat16 with micro-batch 8 and no accumulation. 32-bit at micro-batch 8 fails the 4,500 MB limit (ResNet 6,039 MB, EfficientNet 6,329 MB); bfloat16 at micro-batch 8 passes (ResNet 3,615 MB, GPU 2.10 min per epoch; EfficientNet 2,879 MB, 1.24 min). Loading at 1536x864 is 1.58 min per epoch with 4 workers. Projected upper bounds: ResNet max(2.10, 1.58) x 55 x 5 = 9.6 h; EfficientNet max(1.24, 1.58) x 35 x 5 = 4.6 h.
  Rationale: the hardware rule's first passing configuration. Note that this rung changes number precision as well as resolution for both models; if it fails to improve, a 1024x576 bfloat16 control run would separate the two effects.
  Date/Author: 2026-10-05, agent.

- Decision (amendment, owner-approved 2026-10-07): score every run in the precision it was trained in. The ladder metric becomes the mean validation TTA F1 in training precision; held-out evaluation of 16-bit runs uses `evaluate.py --amp`; both training scripts' end-of-fold TTA and threshold scoring use their `USE_AMP` setting. 32-bit held-out results of 16-bit runs are kept on disk as a record but are not the reported numbers.
  Rationale: the AUC diagnostic (Surprises) shows models lose 0.02 to 0.04 AUC when run in a precision other than their training one, in both directions, so mixed-precision comparisons measure the mismatch rather than the model. The owner considered an all-32-bit alternative (retrain 1536 in 32-bit with accumulation; 2048 in 32-bit would take about 40 h per ResNet run with 2-image pieces) and chose this one.
  Date/Author: 2026-10-07, owner and agent.

- Decision (2026-10-08): both ladders stop; each model's chosen size is 1536x864, trained in 16-bit with micro-batch 8 and no accumulation (as in its 1536 run). Milestones 3 to 5 use 1536x864 for both, so one patch set serves both models, and the patch-pretrained whole-image runs use exactly the 1536 runs' settings.
  Rationale: the 2048x1152 rung fell short of 1536x864 by about 0.03 to 0.04 mean validation F1 for both models (ladder rule), and the held-out set agrees. The 2048 runs also changed micro-batch size from 8 to 4 (forced by memory), which may have contributed; separating that would need a 2048 run with micro-batch 8, which does not fit the hardware rule.
  Date/Author: 2026-10-08, agent per the ladder rule.

- Decision: when 16-bit-trained models are later served by the web app, `inference.py` must run them in 16-bit (not done in this plan).
  Rationale: amendment (c); the app runs everything in 32-bit today.
  Date/Author: 2026-10-07, owner and agent.

- Decision (2026-10-19): run EfficientNet's patch-initialised whole-image training even though its patch pretraining missed the +0.10 accuracy margin in folds 2, 4 and 5 (+0.077, +0.099, +0.082).
  Rationale: owner's instruction in chat ("run the EfficientNet patch-init training"). Lesion F1 passed in every fold (0.70 to 0.76, at or above ResNet's), and the margin is dominated by the 72 percent normal class, so it understates what the backbone learned about lesions; ResNet's patch-init run had already shown the transfer helps.
  Date/Author: 2026-10-19, owner and agent.

- Decision: this plan does not change what the web app serves.
  Rationale: the previous plan's Milestone 4 (switching the app to the 1024 by 576 models) is still waiting for the owner's decision. Serving is decided there or in a follow-up plan.
  Date/Author: 2026-10-05, drafted by agent.

## Outcomes & Retrospective

(Nothing yet.)

## Context and Orientation

All commands run from the repository root, `A:\Projects\BreastCancerDetectionCNN`, in PowerShell, using `py` (the Windows Python launcher; Python 3.13, PyTorch 2.10.0 with CUDA 12.6, torchvision 0.25.0). The machine has an NVIDIA RTX 3070 Ti with 8 GB of memory, of which the desktop uses about 3 GB. Model weights (`*.pth`) and JPEG images (`*.jpg`) are ignored by git. The A: drive has about 95 GB free.

The data lives in `data/raw/`. `data/raw/csv/mass_case_description_train_set.csv` and `calc_case_description_train_set.csv` list one row per lesion ("abnormality") with columns including `patient_id`, `pathology` (`MALIGNANT`, `BENIGN` or `BENIGN_WITHOUT_CALLBACK`; the last two count as benign), `image file path`, `cropped image file path` and `ROI mask file path`. The path columns look like `<case>/<uid1>/<uid2>/000000.dcm`; the JPEGs on disk are in `data/raw/jpeg/<uid2>/*.jpg`. A lesion's ROI mask (region of interest) is a black image the size of the full mammogram with the lesion's outline filled in white. The mask and the dataset's own cropped close-up share one folder and are told apart by counting distinct grey values: under 20 for a mask, over 100 for a crop (`DataSetAugmentation._count_distinct_grayscale_values`; this held for all 3,463 two-file folders). A patient can have several mammograms, and a mammogram several lesion rows. Training data: 2,864 lesion rows from 1,248 patients (1,683 benign, 1,181 malignant). The held-out test set is the 704 rows of `mass_test` and `calc_test` (428 benign, 276 malignant) and is never used for training or for choosing anything.

`DataSetAugmentation.py` defines `MammogramRawDataset(csv_types)` (full mammograms as `(path, label)` pairs in `self.samples`, label 0 benign and 1 malignant, with `self.patient_ids`), `TransformDataset`, and the transforms `get_light_train_transforms(size)` (resize to `(height, width)`, random horizontal flip, random rotation up to 10 degrees, normalised tensor) and `get_val_transforms(size)` (resize and normalise).

`ResNet.py` trains `ResNetWithAttnPool`, a ResNet-50 (50-layer image network starting from ImageNet-pretrained weights) whose parts are `layer0` (first convolution, batch normalisation, pooling), `layer1` to `layer4` (the four stages), `attn_pool` (learned weighting of the final feature map) and `fc` (classifier head with two batch-normalisation layers, producing benign and malignant scores). Only `attn_pool` and `fc` train at first; `UNFREEZE_SCHEDULE` makes `layer4` trainable from epoch 5 and `layer3` from epoch 10. Training uses SAM (Sharpness-Aware Minimisation): for each batch it computes the gradient, nudges the weights a small step uphill along it (`first_step`), computes the gradient again there, then undoes the nudge and takes the real AdamW step with that second gradient (`second_step`); this favours solutions that stay good under small weight changes. Up to 55 epochs per fold; it keeps the lowest-validation-loss checkpoint from epoch 8 on as `fold{k}_best.pth`, stops after 12 epochs without improvement, re-scores the checkpoint with TTA (test-time augmentation: averaging predictions for the image and its horizontal and vertical flips), and prints `Fold k best-val F1 (with TTA): x`, ending with `Val F1 TTA (mean ± std): ...`. Folds come from `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on patient ids, so no patient is in both a fold's training and validation data and the folds are identical in every run. Options: `--image-hw H W`, `--batch-size B`. Outputs: `checkpoints_resnet_sam_grouped_{H}x{W}_5fold/` and `training_logs/resnet_sam_grouped_{H}x{W}_5fold_training_log.xlsx`.

`EfficientNet.py` trains torchvision's EfficientNet-B0 (`create_efficientnet()`: ImageNet weights, all frozen, classifier replaced by dropout plus a 2-output linear layer). Its backbone is `model.features`, a sequence of blocks 0 to 8; `UNFREEZE_SCHEDULE = {4: [8, 7], 10: [6], 16: [5]}` makes blocks 8 and 7 trainable from the 5th epoch, 6 from the 11th and 5 from the 17th (the loop counts from 0). It uses the same SAM recipe (imported from `optimizers.py`) and loss (from `losses.py`), up to 35 epochs, early-stopping patience 10, saves `best_model_fold_{k}.pth`, and prints the same per-fold TTA lines (but no mean line; the mean is computed from the five fold lines). Options: `--folds N`, `--image-hw H W`, `--batch-size B`. Outputs: `checkpoints_efficientnet_sam_grouped_{H}x{W}_5fold/`, `plots_efficientnet_sam_grouped_{H}x{W}_5fold/` and a matching Excel log.

`evaluate.py` scores checkpoints on the held-out set: `py evaluate.py --model resnet|efficientnet --checkpoint <one or more .pth> --tta --image-hw H W --out <json>`, averaging several checkpoints as an ensemble, and prints a line ending `n=704 acc=... prec=... rec=... f1=...`. It builds the plain model and loads the checkpoint, so every checkpoint from this plan loads unchanged.

Reference numbers at 1024 by 576 (batch 8, 32-bit): ResNet-50 mean validation TTA F1 0.7382, held-out ensemble accuracy 0.7443, precision 0.6579, recall 0.7246, F1 0.6897. EfficientNet-B0 mean validation TTA F1 0.7395, held-out ensemble 0.7173, 0.6199, 0.7210, 0.6667. A 704-image test set gives an F1 sampling error of very roughly 0.02 to 0.03. Measured costs at 1024 by 576: ResNet-50 with `layer3` and `layer4` trainable at batch 8 peaked at 2,620 MB and used about 1.7 GPU-minutes per epoch; EfficientNet-B0 at batch 8 peaked at 1,967 MB and 0.8 GPU-minutes per epoch; loading one training image costs about 63 ms to decode plus about 59 ms to resize and augment on one CPU core. Long runs have failed once from Windows commit-memory exhaustion, so free commit memory is checked before every launch.

## Plan of Work

Milestone 1 edits `ResNet.py` and `EfficientNet.py` the same way. In each config class add `USE_AMP = False` and `ACCUM_STEPS = 1`. In the training-epoch function (`train_epoch` in ResNet, `train_one_epoch` in EfficientNet), buffer incoming batches until `ACCUM_STEPS` are held or the loader is exhausted, then do one update over the buffer. With SAM: run forward and backward on each buffered micro-batch with the loss divided by the number buffered, so gradients add up to the average over the effective batch; clip the gradient norm; `optimizer.first_step(zero_grad=True)`; repeat forward and backward over the same micro-batches; clip; `optimizer.second_step(zero_grad=True)`. The non-SAM branch does the same once with `optimizer.step()`. MixUp (off in practice) is applied per micro-batch when buffered, with its soft labels buffered alongside. Every forward pass is wrapped in `torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP)`. The reported training loss stays the average unscaled first-pass loss per image. In the validation function, wrap forward passes in the same autocast and convert logits with `.float()` before the loss. With `ACCUM_STEPS = 1` and `USE_AMP = False` the operations and their order are exactly today's. The training `DataLoader` gets `drop_last=CFG.ACCUM_STEPS > 1` (and the EfficientNet equivalent). In each `__main__` block add `--amp`, `--accum-steps N` (default 1) and `--num-workers N` (default the current value), set the config, and print one line at start such as `Config: image_hw=(1536, 864) micro_batch=4 accum_steps=2 effective_batch=8 amp=True workers=4`. `--batch-size` keeps its name and now means the micro-batch. Output folder names stay as today; they depend only on image size (and, from Milestone 5, on `--patch-init`).

Milestone 1 also uses throwaway scripts in the session scratchpad (not the repository). An equivalence check builds each model with its last unfreeze stage trainable, in evaluation mode (so batch normalisation uses fixed statistics and the comparison is exact), and compares the gradients of one pass over 8 random images with the summed gradients of two passes over 4 with the loss halved; they must agree within 1e-5 relative difference. A probe, given a model, size and configuration, runs a few real SAM updates on random tensors and prints peak allocated memory and GPU minutes per epoch (seconds per update times about 2,291 training images divided by the effective batch of 8), and times loading and transforming 40 real training JPEGs at that size.

Milestone 2 has no source edits. For each rung, it runs the probe through the hardware rule, launches the chosen configuration, evaluates the finished run, applies the ladder rule, and records all of it.

Milestone 3 adds `make_patches.py` at the repository root and the line `data/patches_*/` to `.gitignore` (the patch index is a CSV, which `*.jpg` alone would not ignore). `make_patches.py --image-hw H W` reads both training CSVs, finds each row's full mammogram and its ROI mask (the under-20-grey-values file in the `ROI mask file path` folder), and skips the row, counting it, if either is missing or the mask's size differs from the mammogram's. It groups rows by mammogram so each image is decoded once. For each mammogram it resizes the image (bilinear) and the union of its masks (nearest) to `(W, H)` exactly as the training transform does. For each lesion it writes the 288 by 288 region centred on the mask's bounding-box centre, padding with black past the edges, labelled 1 (benign) or 2 (malignant). Then it draws up to 3 normal-tissue patches per mammogram at random positions, accepted only if no masked pixel falls inside and at least half the pixels are brighter than 30 out of 255 (breast tissue rather than black background), giving up on that mammogram after 200 attempts; label 0. The random generator is seeded with 42. Patches are greyscale JPEGs (quality 95) in `data/patches_{H}x{W}/{normal,benign,malignant}/`, named `<patient_id>_<image folder uid>_<n>.jpg`, listed in `data/patches_{H}x{W}/index.csv` with columns `path,label,patient_id`. If that index already exists the script stops and says so. It prints totals per class and the number of skipped rows. One patch set is built per distinct chosen size (one if both models chose the same size).

Milestone 4 adds `PatchPretrain.py`. `py PatchPretrain.py --model resnet|efficientnet --image-hw H W` reads `data/patches_{H}x{W}/index.csv`, rebuilds the whole-image folds exactly as the training scripts do (same dataset call, same `StratifiedGroupKFold` with seed 42 on the same patient ids), and for each fold k uses patches of that fold's training patients to train and of its validation patients to measure. The network is the whole-image model with its final classifier swapped for 3 outputs and every parameter trainable: for ResNet, `ResNetWithAttnPool()` from `ResNet.py` with `fc` replaced by `nn.Linear(2048, 3)`; for EfficientNet, `create_efficientnet()` from `EfficientNet.py` with `classifier` replaced by `nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 3))`. So the saved backbone uses exactly the parameter names the whole-image model expects. Inputs: greyscale patch converted to 3-channel RGB, random 224 by 224 crop, random horizontal flip, random rotation up to 10 degrees, the ImageNet normalisation used in `DataSetAugmentation.py`; validation uses the centre 224 crop. AdamW, learning rate 1e-4, weight decay 1e-4, cosine decay to zero over 12 epochs, batch 32, class-weighted cross-entropy, bfloat16 autocast, 4 loader workers. At the start of each fold it prints per-class patch counts for training and validation and the largest class's share of the validation patches; after each epoch, validation loss, 3-class accuracy and, on lesion patches only, the benign-versus-malignant F1 (deciding between the two lesion scores). It keeps the epoch with the lowest validation loss and saves only the backbone (state-dict entries starting with `layer` for ResNet, `features.` for EfficientNet) to `checkpoints_patch_{model}_{H}x{W}/fold{k}_backbone.pth`, skipping folds whose file already exists so an interrupted run can resume.

Milestone 5 edits both training scripts once more: add `--patch-init DIR`. When given, right after the model is built for fold k, load `DIR/fold{k}_backbone.pth` with `load_state_dict(state, strict=False)`, then stop with an error if any loaded key is not in the model or any model backbone key (ResNet: starting with `layer`; EfficientNet: starting with `features.`) was not loaded. The run's folder and log names gain `_patchinit` (for example `checkpoints_resnet_sam_grouped_1536x864_patchinit_5fold/`). Then each model trains from its own backbones at its chosen configuration and is evaluated.

## Concrete Steps

All from `A:\Projects\BreastCancerDetectionCNN` in PowerShell. Before every long run: confirm with `nvidia-smi` that no training is running and that the target output folder does not exist, and check free commit memory with `(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB` (GB); at least 12 GB should be free, otherwise ask the owner to close applications. Every long run is a managed background process, never a closable terminal window. While a run is in progress, if its log gains no new epoch line for 60 minutes and the process uses no CPU, treat it as hung (the known Windows paging failure): stop it, record it in Surprises & Discoveries, and follow Idempotence and Recovery.

Milestone 1. Run the equivalence check for both models; expected:

    resnet: max relative grad difference (1 x 8 vs 2 x 4): ...e-07  PASS
    efficientnet: max relative grad difference (1 x 8 vs 2 x 4): ...e-07  PASS

Smoke-test the default path of each script with a throwaway size so no existing folder is touched, stopping each once the first epoch begins and deleting the folders and Excel logs it created:

    py -u ResNet.py --image-hw 1024 577 --batch-size 8
    py -u EfficientNet.py --image-hw 1024 577 --batch-size 8

Each must print its `Config:` line with `accum_steps=1 ... amp=False` and the usual `Loaded 2864 labelled samples` line, and begin training.

Milestone 2, per rung (H, W) and model. Run the probe over the hardware rule's configurations; expected lines resemble:

    resnet 1536x864 fp32 micro=8 accum=1 peak_MB=... gpu_min_per_epoch=...  (projected ... h)
    load+transform 1536x864: ... ms per image

Launch the chosen configuration, for example:

    py -u ResNet.py --image-hw 1536 864 --batch-size M --accum-steps A [--amp] > resnet_sam_grouped_1536x864_5fold.log 2> resnet_sam_grouped_1536x864_5fold.err.log
    py -u EfficientNet.py --image-hw 1536 864 --batch-size M --accum-steps A [--amp] --num-workers 4 > efficientnet_sam_grouped_1536x864_5fold.log 2> efficientnet_sam_grouped_1536x864_5fold.err.log

When it finishes, take fold K with the highest `best-val F1 (with TTA)` and evaluate the single fold and the ensemble, for example for ResNet:

    py evaluate.py --model resnet --checkpoint checkpoints_resnet_sam_grouped_1536x864_5fold/foldK_best.pth --tta --image-hw 1536 864 --out eval_results/resnetsamgrouped1536x864_5fold_singlebest_test_metrics.json
    py evaluate.py --model resnet --checkpoint (Get-ChildItem checkpoints_resnet_sam_grouped_1536x864_5fold\*.pth | Sort-Object Name).FullName --tta --image-hw 1536 864 --out eval_results/resnetsamgrouped1536x864_5fold_ensemble_test_metrics.json

(EfficientNet names follow the existing `samgrouped{H}x{W}_efficientnet_5fold_{singlebest,ensemble}_test_metrics.json` pattern.) Apply the ladder rule and record the outcome in Progress and the Decision Log.

Milestone 3, for each distinct chosen size:

    py make_patches.py --image-hw H W

Expected output resembles `Read 2864 lesion rows; skipped N` and `Wrote patches: normal ..., benign ..., malignant ...`. Then tile 20 random patches per class into one image (scratchpad script) and confirm by eye that lesion patches show a lesion near the centre and normal patches show breast tissue without one.

Milestone 4, per model:

    py -u PatchPretrain.py --model resnet --image-hw H W > patch_pretrain_resnet_HxW.log 2> patch_pretrain_resnet_HxW.err.log
    py -u PatchPretrain.py --model efficientnet --image-hw H W > patch_pretrain_efficientnet_HxW.log 2> patch_pretrain_efficientnet_HxW.err.log

Each fold ends with a line like `Fold 1: saved checkpoints_patch_resnet_HxW/fold1_backbone.pth (best epoch 9, val loss 0.61, 3-class acc 0.78, lesion F1 0.66)`.

Milestone 5, per model, with that model's chosen size and configuration:

    py -u ResNet.py --image-hw H W --batch-size M --accum-steps A [--amp] --patch-init checkpoints_patch_resnet_HxW > resnet_sam_grouped_HxW_patchinit_5fold.log 2> resnet_sam_grouped_HxW_patchinit_5fold.err.log
    py -u EfficientNet.py --image-hw H W --batch-size M --accum-steps A [--amp] --num-workers 4 --patch-init checkpoints_patch_efficientnet_HxW > efficientnet_sam_grouped_HxW_patchinit_5fold.log 2> efficientnet_sam_grouped_HxW_patchinit_5fold.err.log

Then evaluate single best fold and ensemble as in Milestone 2, with `patchinit` added to the output file names.

## Validation and Acceptance

Milestone 1 is accepted when both equivalence checks print PASS and both smoke tests print the expected `Config:` line and start training.

Each training run (Milestones 2 and 5) is accepted when its error log is empty, its checkpoint folder holds exactly five files of 107,226,339 bytes (ResNet) or 16,342,523 bytes (EfficientNet), its Excel "Fold Summary" sheet has five rows, its log ends with the final results block, and both `evaluate.py` runs print `n=704`.

Milestone 3 is accepted when each `index.csv` exists, every class has at least 1,000 patches, fewer than 5 percent of rows were skipped, and the by-eye check passes.

Milestone 4 is accepted when all five backbone files exist per model and, in every fold, 3-class validation accuracy beats always guessing the largest class by at least 0.10 and lesion benign-versus-malignant F1 is above 0.55. Failing either means the patch network learned little that is useful; the plan stops for that model and the owner is told before any whole-image run is spent on it.

The plan is accepted when Outcomes & Retrospective holds, for each model, a table of every run (1024 by 576 reference, each ladder rung, the patch-pretrained run) with mean validation TTA F1, held-out accuracy, precision, recall and F1 for single best fold and ensemble, configuration and wall-clock time, plus a plain statement of what each change did.

Non-regression: with no new flags both scripts behave as before (Milestone 1 smoke tests). No existing checkpoint folder, log or `eval_results` file is modified; every run writes to a new name. At the end `git diff --stat` shows changes only to `ResNet.py`, `EfficientNet.py`, `evaluate.py` (amendment (b)), `.gitignore` and this spec, plus the two new scripts. `evaluate.py` without `--amp` must reproduce earlier results exactly (checked on the ResNet 1024x576 ensemble, accuracy 0.7443).

## Idempotence and Recovery

Equivalence checks, probes and evaluations are safe to rerun. `make_patches.py` refuses to overwrite an existing patch set (delete `data/patches_{H}x{W}/` to rebuild). `PatchPretrain.py` skips folds already saved. A whole-image run cannot resume: if one fails or hangs, rename its checkpoint folder, plots folder, log files and Excel log to names ending `.aborted-<date>` and restart it from fold 1; never evaluate a partial run as this plan's result. If the same run fails twice, stop and tell the owner. The script edits are additive and opt-in; `git checkout -- ResNet.py EfficientNet.py` restores the originals.

## Artifacts and Notes

(Probe output, run timings and evaluation lines go here as they are produced.)

Milestone 1 probe (2026-10-05; peak memory includes cuDNN autotune workspace; GPU minutes cover training plus validation per epoch):

    resnet 1024x576 fp32 micro=8 accum=1 peak_MB=3399 gpu_min_per_epoch=1.77   (calibration)
    efficientnet 1024x576 fp32 micro=8 accum=1 peak_MB=5355 gpu_min_per_epoch=0.86   (calibration)
    resnet 1536x864 fp32 micro=8 accum=1 peak_MB=6039 gpu_min_per_epoch=3.55
    resnet 1536x864 bf16 micro=8 accum=1 peak_MB=3615 gpu_min_per_epoch=2.10
    resnet 1536x864 bf16 micro=4 accum=2 peak_MB=2499 gpu_min_per_epoch=2.21
    efficientnet 1536x864 fp32 micro=8 accum=1 peak_MB=6329 gpu_min_per_epoch=1.74
    efficientnet 1536x864 bf16 micro=8 accum=1 peak_MB=2879 gpu_min_per_epoch=1.24
    efficientnet 1536x864 bf16 micro=4 accum=2 peak_MB=1868 gpu_min_per_epoch=1.29
    load 1024x576: decode 56 ms + transform 57 ms per image; 1.34 min per epoch with 4 workers
    load 1536x864: decode 56 ms + transform 76 ms per image; 1.58 min per epoch with 4 workers

2048x1152 probe (2026-10-07):

    resnet 2048x1152 fp32 micro=4 accum=2 peak_MB=6096 gpu_min_per_epoch=6.72
    resnet 2048x1152 fp32 micro=2 accum=4 peak_MB=3961 gpu_min_per_epoch=6.88
    efficientnet 2048x1152 fp32 micro=4 accum=2 peak_MB=6983 gpu_min_per_epoch=3.22
    efficientnet 2048x1152 fp32 micro=2 accum=4 peak_MB=5810 gpu_min_per_epoch=3.30
    resnet 2048x1152 bf16 micro=8 accum=1 peak_MB=6054 gpu_min_per_epoch=3.84
    efficientnet 2048x1152 bf16 micro=8 accum=1 peak_MB=5049 gpu_min_per_epoch=2.25
    resnet 2048x1152 bf16 micro=4 accum=2 peak_MB=4007 gpu_min_per_epoch=3.89   (chosen)
    efficientnet 2048x1152 bf16 micro=4 accum=2 peak_MB=3233 gpu_min_per_epoch=2.29   (chosen)
    load 2048x1152: decode 63 ms + transform 98 ms per image; 1.92 min per epoch with 4 workers

## Interfaces and Dependencies

No new dependencies: PyTorch, torchvision, Pillow, NumPy, pandas and scikit-learn are already used by the project.

`ResNet.py`: `CFG` gains `USE_AMP = False` and `ACCUM_STEPS = 1`; `train_epoch` and `validate` keep their signatures; `train_fold` and `main` gain an optional final parameter `patch_init_dir=None`. `EfficientNet.py`: `EfficientNetConfig` gains the same two fields; `train_one_epoch` and `validate` keep their signatures; `train_efficientnet_kfold` gains `patch_init_dir=None`. Both gain the options `--amp`, `--accum-steps N`, `--num-workers N` and `--patch-init DIR`.

`make_patches.py`: option `--image-hw H W` (required); writes `data/patches_{H}x{W}/index.csv` with header `path,label,patient_id` (`path` relative to the repository root; `label` 0 normal, 1 benign, 2 malignant).

`PatchPretrain.py`: options `--model resnet|efficientnet` and `--image-hw H W` (both required); writes `checkpoints_patch_{model}_{H}x{W}/fold{k}_backbone.pth` for k = 1 to 5, each a state dict holding only the backbone entries.

Revision note (2026-10-05): the first draft covered ResNet-50 only, one 1536 by 864 run, and pretraining for ResNet only, pending approval. The owner then instructed in chat to keep raising resolution until it stops helping or the hardware cannot handle it, and then to do lesion pretraining for both models. This revision widens scope accordingly: both models, a resolution ladder with an explicit stopping rule and hardware rule, the same memory-saving flags in `EfficientNet.py`, a `--num-workers` option to avoid the known Windows paging hang, and pretraining and patch-initialised runs for both models. The owner's instruction is recorded as the approval.
