#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Register RANA trainers with an installed nnU-Net v2 environment."""
from __future__ import annotations

import shutil
from pathlib import Path


def main():
    import nnunetv2

    dest = Path(nnunetv2.__file__).resolve().parent / "training" / "nnUNetTrainer"
    src = Path(__file__).resolve().parents[1] / "rana"
    files = [
        "nnUNetTrainerRANA.py",
        "nnUNetTrainerRANA_abl.py",
    ]
    for name in files:
        target = dest / name
        shutil.copy2(src / name, target)
        print("installed", target)
    print("Train with: nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNetTrainerRANA")


if __name__ == "__main__":
    main()
