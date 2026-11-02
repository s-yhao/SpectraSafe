# Source and provenance audit

## Scope

This directory is a clean-room consolidation of the paper specification and
small, targeted pieces of audited training infrastructure. It contains no
dynamic import from historical experiment directories and no copied dataset,
checkpoint, log, visualization, or generated result.

## Paper-spec implementation

`fa_model.py` implements the architecture stated in the manuscript:

- 16 input frames and 9 temporal rFFT bins;
- DCT spatial-frequency branch;
- attention across temporal-frequency bins within each spatial patch;
- coefficient-wise temporal gate/noise prediction from patch context plus
  each coefficient's real/imaginary YUV values;
- equal skin/motion guidance fusion;
- gamma 0.6;
- inverse real FFT reconstruction.

The training pipeline implements the stated 30+70 protocol, fresh downstream
evaluators, VideoMAE V2-B, ImageNet-1K V2 ResNet-50, AdamW, and the documented
learning rates and batch sizes.

The public entry points are separated into initialization, training, and
evaluation directories. Initialization is deliberately a weight-import-only
stage: unlike reconstruction-pretraining pipelines, it has no identity loss,
optimizer, dataset loader, or training loop. Its checkpoint records
`trained=false` and is validated before the training stage accepts it.

## How the clean source was formed

`clean_model` was an intermediate review snapshot, not the author's final
implementation. It was used only to identify self-contained transform and
interface fragments. The later dependency-based source family is the primary
engineering reference for:

- coefficient-wise point-token temporal policies (`context_head + point_head`);
- BF16 policy computation;
- PyTorch scaled-dot-product attention and automatic Flash-Attention dispatch;
- fused AdamW, global AMP, and reuse of the anonymized batch between adversary
  and classifier updates.

The long absolute-path inheritance chain has been manually flattened. No
historical helper is imported at runtime, and unused Mamba-era dependencies
have been removed.

## Historical evidence boundary

The historical logs that exactly match the manuscript tables came from a
full-complex FFT-16 implementation whose FFT attention operated across the
spatial patch grid independently for each frequency band. Those files also
formed a long chain of absolute-path dynamic imports. They are retained only
as provenance evidence outside this release and are deliberately not used by
this source tree.

Other audited historical candidates either used rFFT-9 with old guidance
defaults (`motion_weight=0.2`, `gamma=0.8`), lacked complete training logs, or
used intra-patch frequency attention with full FFT-16 and different metrics.
The clean release therefore combines independently evidenced pieces: rFFT-9
from the paper and rFFT point-token branch, intra-patch frequency attention
from the later implementation, and gamma 0.6/equal guidance from the final
main configuration. No audited historical run proves this exact combination.

Therefore this release must be described as a paper-spec implementation until
it has been trained and evaluated independently. Historical full-FFT
checkpoints are not compatible with `fa_model.py`.

## External implementation dependency

The action evaluator expects the official OpenGVLab VideoMAE V2 repository at
commit `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`. The repository path and
pretrained checkpoint path are supplied externally through configuration.
