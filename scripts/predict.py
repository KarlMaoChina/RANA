#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run RANA sliding-window inference on a folder of CT NIfTIs."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rana.nnUNetPredictorRANA import nnUNetPredictorRANA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-m", "--model", required=True, help="nnU-Net results folder for RANA")
    ap.add_argument("-f", "--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    args = ap.parse_args()
    pred = nnUNetPredictorRANA(tile_step_size=0.5, use_gaussian=True,
                               use_mirroring=True, perform_everything_on_device=True,
                               device="cuda", verbose=False, gamma_verbose=True)
    pred.initialize_from_trained_model_folder(args.model, args.folds)
    pred.predict_from_files(args.input, args.output, save_probabilities=False,
                            overwrite=True, num_processes_preprocessing=2,
                            num_processes_segmentation_export=2)


if __name__ == "__main__":
    main()
