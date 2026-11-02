# SpectraSafe

Clean implementation of the SpectraSafe main experiments described in the
WACV 2027 submission **SpectraSafe: Spatio-Temporal Frequency-Aware Video
Anonymization**.

> Status: source-only paper-spec release. The code was assembled from the
> paper specification and audited historical sources, but has not been run or
> numerically verified in this release directory.

## Method

The anonymizer accepts RGB videos with shape `[B, 3, 16, 224, 224]` and uses:

1. a spatial `16 x 16` block-DCT branch with per-frame spatial attention;
2. a temporal real FFT branch with `16 / 2 + 1 = 9` frequency bins;
3. self-attention across the nine temporal-frequency tokens inside each
   spatial patch, followed by a coefficient-wise point-token head that
   predicts separate Y/Cb/Cr gate and noise values at every spatial position;
4. a detached guidance map obtained from equal-weight skin and motion
   responses, followed by spatial smoothing and `gamma = 0.6`;
5. learned gate/noise policies and Laplace replacement noise in frequency
   space.

The main protocol is:

- 30 epochs of adversarial anonymizer training;
- freeze the anonymizer;
- reinitialize VideoMAE V2-B and ResNet-50 evaluators;
- train the fresh evaluators for 70 epochs;
- AdamW learning rates: `7e-5` (anonymizer), `3e-5` (action), and `5e-5`
  (privacy);
- adversarial train/validation batch sizes: 8/32;
- evaluator train/validation batch sizes: 16/16;
- five privacy attributes with equal loss and metric weights.

## Repository layout

```text
SpectraSafe_release/
├── configs/
│   ├── vpucf.json
│   └── vphmdb.json
├── initialization/
│   └── initialize.py  # import pretrained weights; no identity training
├── training/
│   └── train.py       # 30-epoch adversarial + 70-epoch evaluator training
├── evaluation/
│   └── evaluate.py    # checkpoint-only evaluation
├── config.py          # flat validated experiment configuration
├── data.py            # VP-UCF and VP-HMDB datasets and frame sampling
├── fa_model.py        # self-contained SpectraSafe anonymizer
├── heads.py           # VideoMAE V2-B and ResNet-50 evaluators
├── runtime.py         # shared seeding and model-construction utilities
├── train.py           # shared training implementation
├── evaluate.py        # shared evaluation implementation
├── requirements.txt
└── PROVENANCE.md
```

No file imports code or parameters from a historical experiment directory.
The stage folders are inspired by the phase-oriented organization of
[Balancing Privacy and Action Performance: A Penalty-Driven Approach to Image
Anonymization](https://github.com/Rabusi/Balancing-Privacy-and-Action-Performance-A-Penalty-Driven-Approach-to-Image-Anonymization),
while keeping SpectraSafe's shared implementation in one place.

The implementation keeps the engineering changes found in the later
dependency-based experiment family: BF16 policy inference/training, fused
AdamW, AMP, anonymized-batch reuse, `zero_grad(set_to_none=True)`, and PyTorch
scaled-dot-product attention. On a supported CUDA build, PyTorch dispatches
the latter to its Flash Attention kernel automatically; no separate
`flash-attn` package is required. Unsupported devices use PyTorch's compatible
SDPA fallback.

## External assets

This repository does not include datasets or pretrained weights. The action
evaluator uses the official
[OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2) implementation.
The audited experiment used commit:

```text
29eab1e8a588d1b3ec0cdec7b03a86cca491b74b
```

and the official Kinetics-710 distilled ViT-B checkpoint named:

```text
vit_b_k710_dl_from_giant.pth
```

The privacy evaluator is torchvision ResNet-50 initialized with
`IMAGENET1K_V2` weights.

## Data configuration

The supplied JSON files use environment variables so that no machine-specific
absolute path is committed.

VP-UCF:

```bash
export VPUCF_ROOT=/path/to/UCF-101
export VPUCF_SPLITS=/path/to/ucfTrainTestlist
export VPUCF_PRIVACY_LABELS=/path/to/ucf101_privacy_attribute_label.pickle
```

VP-HMDB:

```bash
export VPHMDB_ROOT=/path/to/hmdb51/videos
export VPHMDB_SPLITS=/path/to/testTrainMulti_7030_splits
export VPHMDB_PRIVACY_LABELS=/path/to/hmdb51_privacy_attribute_label.pickle
export VPHMDB_CLASS_INDEX=/path/to/classInd.txt
```

Shared VideoMAE V2 assets:

```bash
export VIDEOMAEV2_ROOT=/path/to/VideoMAEv2
export VIDEOMAEV2_CHECKPOINT=/path/to/vit_b_k710_dl_from_giant.pth
```

The loaders use official split 1. Training samples one random frame from each
of 16 temporal segments. Evaluation uses 16 uniformly spaced frames; short
videos repeat rounded indices. Frames are resized directly to `224 x 224`.

## Three-stage workflow

All commands below are run from the repository root.

### 1. Initialization: weight import only

```bash
python -m initialization.initialize --config configs/vpucf.json
python -m initialization.initialize --config configs/vphmdb.json
```

This stage constructs SpectraSafe with its seeded parameter initialization,
imports the configured Kinetics-710 VideoMAE V2-B checkpoint and torchvision
ImageNet-1K V2 ResNet-50 weights, and writes:

```text
<output_dir>/checkpoints/initialization.pth
<output_dir>/checkpoints/initialization.json
```

It does **not** load VP-UCF/VP-HMDB, optimize an identity mapping, create an
optimizer, call `backward`, or run any training epoch.
Dataset environment variables are therefore not required for this stage; only
`VIDEOMAEV2_ROOT` and `VIDEOMAEV2_CHECKPOINT` must be configured.

### 2. Training

```bash
python -m training.train --config configs/vpucf.json
python -m training.train --config configs/vphmdb.json
```

Training requires the initialization checkpoint from stage 1, including when
resuming, because the same imported head weights are reused for the required
fresh evaluator initialization at the 30/70 phase boundary. An alternative
location can be supplied with `--initialization-checkpoint`.

Useful options:

```bash
python -m training.train --config configs/vpucf.json --no-resume
python -m training.train --config configs/vpucf.json --adversarial-only
python -m training.train --config configs/vpucf.json --output-dir /path/to/output
python -m training.train --config configs/vpucf.json \
  --initialization-checkpoint /path/to/initialization.pth
```

At the phase boundary the anonymizer is frozen and both evaluators are built
again from the untouched stage-1 weight-import checkpoint. Training therefore
never re-imports the external pretrained checkpoint directly. The final
paper-protocol point is benchmark epoch 70, or global epoch 100.

### 3. Evaluation

```bash
python -m evaluation.evaluate \
  --config configs/vpucf.json \
  --checkpoint outputs/vpucf/checkpoints/benchmark_epoch070.pth \
  --phase benchmark
```

The reported privacy metrics are the mean average precision over five valid
attributes (`cMAP`, lower is better for privacy) and macro F1 at threshold 0.5.

## Reproducibility note

This clean code combines the paper's rFFT-9/intra-patch-frequency specification
with the coefficient-wise policy and runtime improvements from the author's
later dependency-based implementations. It is a flattened reimplementation,
not a copy of the intermediate `clean_model`. Historical experiment folders
used to trace the paper tables contain a different full-FFT implementation and
are not imported here. See [PROVENANCE.md](PROVENANCE.md) for the audit
boundary. A fresh run is required before claiming that this source reproduces
any particular metric.
