# FACS-Net

Official implementation of **Frequency-aware crack segmentation network (FACS-net) and crack topology loss (CT-loss) for thin cracks**, Siheon Joo, Seokhwan Kim, and Hongjo Kim, *Automation in Construction* 182 (2026), 106719. [Paper](https://doi.org/10.1016/j.autcon.2025.106719).

## Reimplementation v2

This release provides config-based training and evaluation with the original FACS-Net architecture and **BCE + soft-CTS + SEMEDA** loss composition. All three loss coefficients remain positive. Model, data, metric, and checkpoint handling are explicit. See [implementation changes and protocol](docs/REIMPLEMENTATION.md).

For benchmarking the published model, use the [original CV12 weights](https://doi.org/10.6084/m9.figshare.29849432) and the evaluation commands below. Newly trained v2 segmentation weights are available upon request at [sh.joo@yonsei.ac.kr](mailto:sh.joo@yonsei.ac.kr).

## Installation

Tested on Linux with Python 3.12, PyTorch 2.8.0, CUDA 12.8, and an RTX 5090.

```bash
git clone https://github.com/SH-Joo/FACS.git
cd FACS
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-train.lock
.venv/bin/python -m facs --help
```

For CPU inference, install `requirements-data.txt`, then CPU builds of PyTorch 2.8.0 and torchvision 0.23.0, and pass `--device cpu`.

## CrackVision12K preparation

Place `CrackVision12K.zip` in the repository root, or supply its path:

```bash
.venv/bin/python scripts/prepare_cv12.py --archive CrackVision12K.zip
```

Preparation retains the original PNG files and **9,600 train / 1,200 validation / 1,200 test** split. Width subsets are reconstructed from foreground area divided by skeleton pixel count. Images and masks retain their supplied 256×256 resolution. The archive is expected to contain `split_dataset_final/{train,val,test}/{IMG,GT}`. Obtain datasets from their authors.

## Evaluation with published weights

Download `CV12.ckpt` from [Figshare](https://doi.org/10.6084/m9.figshare.29849432) and place it under `code/ckpts/`. Convert it once, then evaluate:

```bash
.venv/bin/python -m facs migrate-legacy \
  --checkpoint code/ckpts/CV12.ckpt \
  --config configs/cv12_legacy_inference.json \
  --output runs/cv12_published/checkpoint.pt
.venv/bin/python -m facs evaluate \
  --checkpoint runs/cv12_published/checkpoint.pt \
  --output-dir runs/cv12_published_test --split test
```

Conversion preserves all 513 checkpoint state entries and requires no training. Historical weights should be used without BN recalibration. To reproduce the validation table below, use `--split val` and a new output directory. Evaluation writes predictions, per-image CSV, and overall, positive-only, negative, and width-bin summaries. The original metric profile is preserved: threshold 0.5, image-macro IoU, CL-IoU with δ=0,2,4,8, and `legacy_exact` CTS. See [metric definitions](docs/REIMPLEMENTATION.md#data-and-evaluation).

## Training v2

Pretrain the SEMEDA edge network on CV12 training masks, then train the segmentation model:

```bash
.venv/bin/python -m facs pretrain-edge \
  --config configs/cv12_ct_v2.json --run-dir runs/cv12_edge_v1 --epochs 5
.venv/bin/python -m facs train \
  --config configs/cv12_ct_v2.json --run-dir runs/cv12_ct_v2 --stop-after-epoch 30
.venv/bin/python scripts/recalibrate_batchnorm.py \
  --checkpoint runs/cv12_ct_v2/last.pt --output-dir runs/cv12_ct_v2_bn \
  --sample-count 9600 --seed 8106 --batch-size 12 --workers 4 --device cuda \
  --cpu-threads 4
.venv/bin/python -m facs evaluate \
  --checkpoint runs/cv12_ct_v2_bn/checkpoint.pt \
  --output-dir runs/cv12_ct_v2_val --split val
```

The recipe uses **0.5 BCE + 0.25 soft-CTS + 0.25 SEMEDA**, SEMEDA feature weights `(0,1,0)`, batch size 12, Adam base LR 1e-4, and segmentation-head LR multiplier 10. Segmentation is initialized independently of the published checkpoint, with an ImageNet V2 ResNet50 encoder. The reported export uses epoch-30 `last.pt` with BN statistics recalculated using **training images only**; learned parameters and the evaluator are unchanged.

Run directories must be new. `last.pt` stores optimizer, scheduler, and RNG states for epoch-boundary resume:

```bash
.venv/bin/python -m facs train \
  --config configs/cv12_ct_v2.json --run-dir runs/cv12_ct_v2 \
  --resume runs/cv12_ct_v2/last.pt --stop-after-epoch 30
```

`best.pt` uses raw validation IoU and is separate from the reported BN-recalibrated epoch-30 export. The v2 settings are documented separately from historical training conditions. Training does not automatically evaluate the test split.

## Published results

The following tables retain the historical results reported for CrackVision12K. They are not new v2 measurements.

### Overall performance on CrackVision12K

| Model            | IoU   | CTS   |
| ---------------- | ----- | ----- |
| **FACS-Net**     | 0.663 | 0.651 |
| Hybrid-Segmentor | 0.625 | 0.619 |
| DECS-Net         | 0.564 | 0.626 |
| FCN              | 0.610 | 0.614 |

### Extremely thin cracks (≤2px)

| Model            | IoU       | CTS       |
| ---------------- | --------- | --------- |
| **FACS-Net**     | **0.466** | **0.945** |
| Hybrid-Segmentor | 0.160     | 0.585     |
| DECS-Net         | 0.275     | 0.896     |
| FCN              | 0.136     | 0.717     |

## Reimplementation results

These measurements use the same original **validation split of 1,200 images** and unchanged evaluator. They are separate from the paper's test tables above. The new model is a **single-seed, 30-epoch CT-Loss training run** followed by full-train BN recalibration. Independent-seed and final-test results are not included in this release.

| Model | Overall IoU | Overall CTS | ≤2px IoU | ≤2px CTS |
| --- | ---: | ---: | ---: | ---: |
| Published CV12 checkpoint | 0.6696 | 0.6393 | 0.5197 | 0.9550 |
| FACS-Net v2, newly trained | 0.6608 | 0.6890 | 0.4308 | 0.9495 |

The new run has higher overall CTS, indicating greater agreement under the connectivity metric. For cracks ≤2px, binary-mask IoU is sensitive to pixel quantization and small boundary shifts; IoU and CTS should be considered together.

[Historical scores](docs/results/historical_validation.json) · [V2 CT scores](docs/results/v2_ct_validation.json) · [Protocol and verification](docs/REIMPLEMENTATION.md).

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Checks cover losses, metrics against historical evaluators, paired transforms, dataset validation, and exact epoch-boundary resume. [Release verification](docs/results/release_verification.json) records compatibility checks with the historical weights.

## Citation

```bibtex
@article{Joo2026FACSNet,
  title = {Frequency-aware crack segmentation network (FACS-net) and crack topology loss (CT-loss) for thin cracks},
  author = {Joo, Siheon and Kim, Seokhwan and Kim, Hongjo},
  journal = {Automation in Construction},
  volume = {182},
  pages = {106719},
  year = {2026},
  doi = {10.1016/j.autcon.2025.106719}
}
```

## Terms and contact

This code is released for research and academic use only, consistent with the repository's existing terms. Use datasets under their creators' licenses. Please cite the paper when using this implementation. Questions and v2 weight requests: [sh.joo@yonsei.ac.kr](mailto:sh.joo@yonsei.ac.kr).
