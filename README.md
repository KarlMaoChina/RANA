# RANA

**Reference-Anchored Neural Anatomy** for pathological gallbladder segmentation on CT.

This repository releases the training and inference code that accompanies the paper. It does **not** contain hospital CT volumes, DICOM files, reference-standard contours, or any other patient-identifiable imaging.

Code: https://github.com/karlmaochina/RANA

## What is released

- Closed-form anatomical frame: hepatic-plane coordinates \(\gamma_{1:3}\) and portal-axis distance \(\gamma_4\)
- nnU-Net residual-encoder trainer that injects the four channels after affine spatial augmentation
- Ablation trainers that zero selected channels while keeping the five-channel architecture unchanged
- Inference predictor that regenerates \(\gamma\) on the preprocessed grid
- Feature and sidecar builders that store only geometric parameters (`p0`, `n`, `t1`, `t2`, `q0`, `td`, `A_pre`)

## What is not released

- In-house multicenter CT and masks (ethics protocols prohibit public redistribution)
- Precomputed sidecars or dense \(\gamma\) volumes from the private cohort
- Qualitative figure source images

Public organ benchmarks used in the paper (AMOS22, WORD) remain available from their original distributors. Pretrained weights will be attached to GitHub Releases when the manuscript is accepted; they are neural-network parameters and do not include patient images.

## Requirements

- Python 3.10+
- [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) with residual-encoder plans
- TotalSegmentator (or any model that yields liver, gallbladder, and portal/splenic-vein masks)
- PyTorch with CUDA for training and sliding-window inference

## Install trainers

```bash
pip install -e .
python scripts/install_trainers.py
```

Set the sidecar directory before training:

```bash
export RANA_SIDECAR_ROOT=/path/to/sidecars
export RANA_FEAT_ROOT=/path/to/feature_npz
nnUNetv2_train DATASETID 3d_fullres FOLD -p nnUNetResEncUNetMPlans -tr nnUNetTrainerRANA
```

Ablations matching the paper:

| Trainer | Zeroed channels |
|---|---|
| `nnUNetTrainerRANA_SDFOnly` | \(\gamma_2,\gamma_3,\gamma_4\) |
| `nnUNetTrainerRANA_NoTangential` | \(\gamma_2,\gamma_3\) |
| `nnUNetTrainerRANA_NoPortal` | \(\gamma_4\) |

## Prepare coordinates on your own data

1. Segment liver, gallbladder, and portal/splenic vein (TotalSegmentator labels).
2. Fit the frame (writes a compact npz, no CT voxels):

```bash
python scripts/prepare_features.py \
  --image /path/ct.nii.gz \
  --ts-dir /path/totalsegmentator_masks \
  --out /path/features/CASEID.npz
```

3. After nnU-Net preprocessing, build the training sidecar:

```bash
python scripts/prepare_sidecar.py \
  --preprocessed /path/nnUNet_preprocessed/DatasetXXX_Name/nnUNetResEncUNetMPlans_3d_fullres \
  --raw-image /path/ct.nii.gz \
  --case CASEID \
  --out-dir $RANA_SIDECAR_ROOT/DatasetXXX
```

## Inference

```bash
python scripts/predict.py -i /path/imagesTs -o /path/preds -m /path/nnUNet_results/.../nnUNetTrainerRANA__nnUNetResEncUNetMPlans__3d_fullres
```

## Protocol used in the paper

- Residual-encoder U-Net, patch \(96\times160\times160\), 1000 epochs, SGD with Nesterov momentum 0.9
- Soft Dice + cross-entropy, deep supervision over the finer decoder stages
- Intensity augmentations restricted to the CT channel; spatial augmentations regenerate \(\gamma\) analytically
- Inference: 50% sliding-window overlap, Gaussian weighting, test-time mirroring, five-fold probability average

## License

Apache License 2.0. RANA builds on nnU-Net (Apache-2.0).
