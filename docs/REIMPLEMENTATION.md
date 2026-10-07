# FACS-Net v2 implementation

This release provides config-based training and evaluation while retaining the FACS-Net architecture and the BCE + soft-CTS + SEMEDA loss composition. The training recipe is specified in [cv12_ct_v2.json](../configs/cv12_ct_v2.json). The original source remains available in the repository history.

## Model and checkpoint compatibility

The hybrid ResNet50/MiT encoder, progressive decoder, CBAM, normalized FPCM with trainable cutoff, and fixed-affine LayerNorm are preserved. The historical CV12 checkpoint loads all 513 state entries with `strict=True`. Conversion changes the checkpoint container without training or changing its stored tensors. Historical weights are evaluated directly after conversion; they do not require BN recalibration.

`python -m facs` is the supported entry point. `code/src/main.py` forwards to this CLI, and `code/src/models/FACS.py` provides a model adapter returning `[B, 1, H, W]` logits. Other scripts under `code/src/` are historical references; their previous Lightning interfaces are not the v2 training interface.

## Implementation changes

- Model construction has no implicit training, checkpoint loading, or device allocation. ImageNet initialization is requested explicitly in the training config.
- Losses consume logits and binary targets. Sigmoid is applied within the relevant objective, and each active term is computed once. SEMEDA uses a frozen edge network pretrained on training masks only.
- Training and validation use independent data loaders and metric accumulators. Epoch checkpoints include optimizer, scheduler, and RNG states for deterministic resume; changed source, config, environment, or data are rejected on resume.
- Deterministic pooling and memory-efficient training attention preserve the model structure. Fixed-affine normalization and the uploaded FPCM behavior retain historical checkpoint compatibility.
- Dataset preparation preserves source PNG bytes, verifies image–mask pairing, reconstructs width subsets, and writes checksummed manifests. Evaluation exports binary masks, per-image scores, and explicit macro/micro summaries.

## Released training recipe

| Setting | Value |
| --- | --- |
| Segmentation initialization | Fresh segmentation weights; ResNet50 ImageNet V2 encoder |
| CT coefficients: BCE / soft-CTS / SEMEDA | 0.5 / 0.25 / 0.25 |
| SEMEDA feature weights | (0, 1, 0), mean absolute feature difference |
| Soft-CTS | Gaussian affinity, σ=2, kernel 13, sum normalization, 25 skeleton iterations |
| Edge pretraining | Five epochs on all 9,600 training masks, noise σ=0.1 |
| Optimizer | Adam, base LR 1e-4, segmentation head LR multiplier 10 |
| Batch size / precision | 12 / FP32 |
| Augmentation | Paired rotations and flips on training images |
| Seed | 0 |
| Scheduler | Raw validation macro IoU, plateau patience 5, factor 0.5 |
| Early stopping | Raw validation macro IoU, patience 15 |
| Reported checkpoint | Epoch-30 last, followed by full-train BN recalibration |
| BN recalibration | Training images only, 9,600 images, no augmentation, batch 12, seed 8106 |

The training config retains a 120-epoch ceiling; `--stop-after-epoch 30` specifies the reported budget. `best.pt` is selected using raw validation IoU, whereas the reported v2 result uses epoch-30 `last.pt` followed by BN recalibration. Recalibration changes only the 171 BN buffers; it does not update learned parameters. The resulting checkpoint uses ordinary `model.eval()` inference.

The reported model was trained independently of the historical segmentation weights. Validation informed recipe selection. The new result is a single-seed validation measurement; additional seeds, final test evaluation, external datasets, and a full comparison of models were not rerun for this release. It should not be substituted for the paper's test tables.

## Data and evaluation

CV12 uses original IDs 1–9,600 for training, 9,601–10,800 for validation, and 10,801–12,000 for testing, at the supplied 256×256 resolution. Mean crack width is foreground area divided by Zhang skeleton pixel count. The six positive test width bins contain 63, 64, 141, 252, 330, and 132 images, with 218 empty masks. The thinnest positive-test decile contains 99 images and differs from the width≤2px subset. The original split is retained, including five detected exact image–mask duplicates between training and validation.

The preserved metric profile uses a 0.5 threshold, image-macro IoU, CL-IoU with `thin` skeletons and disk tolerances δ=0,2,4,8, and `legacy_exact` CTS matching with square buffer radius 10 and overlap threshold 0.5. Empty–empty IoU and CL-IoU are 1; empty CTS is 0. Positive-only and width-bin results are reported separately. The default metric definitions are unchanged.

[Historical validation summary](results/historical_validation.json), [historical per-image scores](results/historical_per_image.csv), [v2 CT validation summary](results/v2_ct_validation.json), [v2 CT per-image scores](results/v2_ct_per_image.csv). The summaries record the source hashes of the evaluated release code. Compatibility checks and source changes are recorded in [release verification](results/release_verification.json).
