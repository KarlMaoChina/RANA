# RANA: Reference-Anchored Neural Anatomy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](LICENSE)
[![Framework](https://img.shields.io/badge/built%20on-nnU--Net%20v2-orange.svg?style=flat-square)](https://github.com/MIC-DKFZ/nnUNet)
> Official repository for **RANA: Extrinsic Canonical Coordinate Conditioning for Pathological Gallbladder Segmentation on Computed Tomography** (Reference-Anchored Neural Anatomy).

---

## Overview

In advanced gallbladder cancer and acute cholecystitis, local radiographic contrast frequently collapses: aggressive neoplastic infiltration and severe inflammatory adhesions dissolve the pericholecystic fat plane, leaving purely intensity-driven 3D neural networks "lost" at tissue interfaces.

**RANA** resolves this fundamental ill-posedness by conditioning convolutional representations on an **extrinsic canonical coordinate frame** anchored to macroscopic anatomical landmarks:
1. **The visceral hepatic surface** (fitted local tangent plane $\Pi = (\mathbf{p}_0, \mathbf{n}, \mathbf{t}_1, \mathbf{t}_2)$).
2. **The main portal venous axis** (fitted line segment $\mathcal{V} = (\mathbf{q}_0, \mathbf{t}_d)$).

Together, they establish a continuous, four-channel spatial field:
$$\gamma(\mathbf{x}) = \left[ \frac{\text{SDF}_\Pi(\mathbf{x})}{L_0}, \frac{t_1(\mathbf{x})}{L_0}, \frac{t_2(\mathbf{x})}{L_0}, \min\left(\frac{d(\mathbf{x}, \mathcal{V})}{L_0}, d_{\max}\right) \right] \in \mathbb{R}^4$$

### Key Highlights

- **Exact Affine Pullbacks:** Spatial data augmentations (3D rotation, anisotropic scaling, reflection) transform the coordinate channels in **exact closed form**. Coordinates are regenerated directly on the augmented grid without 3D volume resampling, eliminating interpolation blur and preserving zero-divergence geometry.
- **Minimal Capacity Overhead:** Extends nnU-Net's first convolutional layer from 1 to 5 input channels—adding merely **3,456 weights** to a 101.9M-parameter residual encoder backbone (<0.003% increase).
- **Strong Clinical & Zero-Shot Transfer:** Evaluated across 721 multicenter and public CT series (+0.0331 internal cross-validation Dice over nnU-Net ResEnc) with zero-shot domain transfer to public organ benchmarks (AMOS22 median Dice 0.9216, WORD median Dice 0.8381).
- **Reference-Free Quality Assurance:** Spatial disagreement across independent training seeds serves as an unsupervised detector for difficult segmentations (AUC 0.79–0.84).

---

## Repository Structure

```text
RANA/
├── rana/
│   ├── nnUNetTrainerRANA.py       # Core trainer with exact affine pullbacks
│   ├── nnUNetTrainerRANA_abl.py   # Coordinate ablation trainers (zero-channel baselines)
│   ├── nnUNetPredictorRANA.py     # Inference predictor with on-the-fly coordinate injection
│   ├── geometry.py                # Closed-form frame fitting and coordinate evaluation
│   ├── feat_match.py              # Geometric metadata matching utilities
│   └── constants.py               # Normalization scales and anatomical organ indices
├── scripts/
│   ├── install_trainers.py        # Registers RANA trainers into local nnUNet installation
│   ├── prepare_features.py        # Fits anatomical frames from TotalSegmentator masks
│   ├── prepare_sidecar.py         # Builds preprocessed-space training sidecars
│   └── predict.py                 # Sliding-window inference pipeline
├── docs/
│   └── DATA_POLICY.md             # Data and ethics guidelines
├── CITATION.cff                   # Citation metadata
└── pyproject.toml                 # Package installation specification
```

---

## Getting Started

### 1. Requirements & Installation

- Linux / Windows
- Python \(\ge 3.10\)
- CUDA-compatible PyTorch
- [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) installed in your environment

Clone and install RANA:

```bash
git clone https://github.com/KarlMaoChina/RANA.git
cd RANA
pip install -e .
python scripts/install_trainers.py
```

`install_trainers.py` copies `nnUNetTrainerRANA.py` and `nnUNetTrainerRANA_abl.py` into your active `nnunetv2` package directory so they can be invoked natively via `nnUNetv2_train`.

---

### 2. Workflow: Preparing Your Data

RANA extracts geometric anchors from standard automated segmentations (e.g., [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)):

#### Step A: Fit the Anatomical Frame
Extract the hepatic plane and portal axis from organ masks. This writes a lightweight geometric parameter file (`.npz` containing vectors $\mathbf{p}_0, \mathbf{n}, \mathbf{t}_1, \mathbf{t}_2, \mathbf{q}_0, \mathbf{t}_d$):

```bash
python scripts/prepare_features.py \
  --image /path/to/ct.nii.gz \
  --ts-dir /path/to/totalsegmentator_masks \
  --out /path/to/features/CASE_001.npz
```

#### Step B: Build Training Sidecars
After running standard nnU-Net preprocessing (`nnUNetv2_plan_and_preprocess`), map the world-coordinate anchors into preprocessed voxel space:

```bash
export RANA_FEAT_ROOT=/path/to/features
export RANA_SIDECAR_ROOT=/path/to/sidecars

python scripts/prepare_sidecar.py \
  --preprocessed /path/to/nnUNet_preprocessed/DatasetXXX_Name/nnUNetResEncUNetMPlans_3d_fullres \
  --raw-image /path/to/ct.nii.gz \
  --case CASE_001 \
  --out-dir $RANA_SIDECAR_ROOT/DatasetXXX_Name
```

---

### 3. Training

Set the sidecar root and launch training with nnU-Net:

```bash
export RANA_SIDECAR_ROOT=/path/to/sidecars
export RANA_FEAT_ROOT=/path/to/features

nnUNetv2_train DatasetXXX_Name 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerRANA
```

#### Coordinate Ablations
To evaluate the contribution of individual coordinate channels, run the corresponding ablation trainers (matching Table 3 in the paper):

| Trainer Name | Active Channels | Description |
|---|---|---|
| `nnUNetTrainerRANA` | All ($\gamma_1, \gamma_2, \gamma_3, \gamma_4$) | Full 4-channel reference frame |
| `nnUNetTrainerRANA_SDFOnly` | $\gamma_1$ only | Liver-surface signed distance field |
| `nnUNetTrainerRANA_NoTangential` | $\gamma_1, \gamma_4$ | Surface distance + portal axis distance |
| `nnUNetTrainerRANA_NoPortal` | $\gamma_1, \gamma_2, \gamma_3$ | Surface distance + in-plane planar basis |

---

### 4. Inference

Run full-volume 3D sliding-window prediction (default 50% patch overlap with Gaussian blending and test-time mirroring):

```bash
python scripts/predict.py \
  -i /path/to/input_images \
  -o /path/to/output_predictions \
  -m /path/to/nnUNet_results/DatasetXXX_Name/nnUNetTrainerRANA__nnUNetResEncUNetMPlans__3d_fullres \
  -f 0 1 2 3 4
```

---

## Data & Model Weights

- **Public Benchmarks:** Experiments on public cohorts reported in the paper (AMOS22, WORD) utilize publicly available CT volumes directly from their respective challenge hosts.
- **Model Checkpoints:** Pretrained neural network weights will be published on [GitHub Releases](https://github.com/KarlMaoChina/RANA/releases) upon formal acceptance of the manuscript.
- **Clinical Cohorts:** In accordance with Institutional Review Board (IRB) ethics protocols, in-house clinical patient CT volumes and manual segmentations cannot be publicly redistributed.

---

## Citation

If you find RANA useful in your research, please cite this repository:

```bibtex
@software{rana2026code,
  author = {{RANA Authors}},
  title = {{RANA: Reference-Anchored Neural Anatomy}},
  year = {2026},
  url = {https://github.com/KarlMaoChina/RANA},
  version = {1.0.0}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE). Built on top of [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet).
