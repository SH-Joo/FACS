# FACS-Net: Frequency-Aware Crack Segmentation Network for Thin Cracks via Topology Preservation

## Introduction

**FACS-Net** is a deep learning framework targeting the **segmentation of thin structural cracks** in images, a crucial task for infrastructure safety monitoring. Traditional crack segmentation models often suffer from **spectral bias**, meaning they favor low-frequency (coarse) features and struggle with high-frequency details like very thin cracks. This leads to fragmented or missed detections of fine cracks, compromising the analysis of crack continuity and topology. FACS-Net directly addresses this issue with a two-fold strategy: a **frequency-aware architecture** and a **topology-preserving loss function**. The result is a model that more reliably detects **thin cracks (width \u2264 2px)** and maintains their connectivity in segmentation outputs.

### Highlights

* **Frequency-Aware Design:** A novel segmentation network that counteracts spectral bias by explicitly learning high-frequency crack features.
* **Topology Preservation:** A custom **Crack Topology Loss (CT-Loss)** that enforces crack connectivity and continuous thin structures in the predicted masks.
* **State-of-the-Art Performance:** On the **CrackVision12K** benchmark, FACS-Net significantly outperforms prior models on thin cracks (IoU improved by 0.306 and CTS by 0.360) and sets new overall best scores (IoU 0.663, CTS 0.651).
* **Exceptional Thin Crack Detection:** On the thinnest cracks (≤ 2px), FACS-Net outperforms previous state-of-the-art methods by a large margin, achieving +0.306 IoU and +0.360 CTS gains over the best existing model. This highlights the effectiveness of FACS-Net's frequency-aware design in the most challenging cases.

## Paper Link

**Preprint available on SSRN**: [https://ssrn.com/abstract=xxxxxxx](https://ssrn.com/abstract=xxxxxxx)
*This paper is currently under review.

## Model Description

FACS-Net consists of a **hybrid encoder** and a **frequency-aware decoder**:

* **Encoder:** Combines CNN (ResNet50) and Transformer (MixVision Transformer from SegFormer) for local and global feature extraction.
* **Decoder:** Uses CBAM (attention) and FPCM (Fourier-based frequency modulation) to recover high-frequency details lost due to downsampling.

**CT-Loss** supervises:

* Pixel-wise accuracy via **BCE loss**
* Edge alignment via **SEMEDA edge-aware loss**
* Crack continuity via differentiable **soft-CTS loss**

## Installation

```bash
git clone https://github.com/yourusername/FACS-Net.git
```

## Datasets & Training

* **CrackVision12K**: Main training dataset (9600 train / 1200 test split).
* **OmniCrack30K**: Used for cross-domain evaluation.
* Images resized to 256x256; training with Adam optimizer, LR=5e-5, batch=12, early stopping within 100 epochs.

## Evaluation Metrics

* **IoU**: Pixel-level accuracy
* **CL-IoU**: Alignment of predicted vs. true crack centerlines
* **CTS**: Measures crack continuity & segment-level matching

## Results Summary

### Overall Performance on CrackVision12K

| Model            | IoU   | CTS   |
| ---------------- | ----- | ----- |
| **FACS-Net**     | 0.663 | 0.651 |
| Hybrid-Segmentor | 0.625 | 0.619 |
| DECS-Net         | 0.564 | 0.626 |
| FCN              | 0.610 | 0.614 |

### Performance on Extremely Thin Cracks (≤ 2px in CrackVision12K)

| Model            | IoU       | CTS       |
| ---------------- | --------- | --------- |
| **FACS-Net**     | **0.466** | **0.945** |
| Hybrid-Segmentor | 0.160     | 0.585     |
| DECS-Net         | 0.275     | 0.896     |
| FCN              | 0.136     | 0.717     |

FACS-Net shows **exceptional performance** on the most challenging thin-crack range (τ ≤ 2 px):

* **+0.306 IoU** and **+0.360 CTS** over Hybrid-Segmentor
* Maintains topological continuity better than all prior models (CTS = 0.945)

These results confirm FACS-Net's superiority in segmenting very fine cracks, which are critical for structural safety analysis and where previous SOTA methods perform poorly.

## Visualization

* Thin crack examples show FACS-Net capturing continuous paths vs. fragmented outputs from prior models.
* Figures from the paper (6 & 7) demonstrate robustness under variable crack widths.

## Pretrained Models & Results

* 🔗 Model Weights & Outputs: [figshare](https://doi.org/10.6084/m9.figshare.29849432)

## Citation

```bibtex
@unpublished{Joo2025FACSNet,
  title     = {Frequency-Aware Crack Segmentation Network (FACS-Net) for Thin-Cracks via Topology Preservation},
  author    = {Siheon Joo and Seokhwan Kim and Hongjo Kim},
  note      = {Manuscript under review. Preprint available at SSRN: \url{https://ssrn.com/abstract/XXXXXXX}},
  year      = {2025}
}
```

## License & Disclaimer

This code is released for **research and academic use only**.
The paper is under review at *Automation in Construction*, and the version shared here is a **preprint** in compliance with Elsevier's sharing policy.
Do not redistribute the publisher version. Cite appropriately.

---

\u26a0\ufe0f CrackVision12K and OmniCrack30K datasets are owned by their creators. Use them under their respective licenses.

For questions, please contact the corresponding author.
