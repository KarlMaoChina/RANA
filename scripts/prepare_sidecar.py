#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build nnU-Net preprocessed-space sidecars from RANA feature npz files.

The sidecar stores only the affine map A_pre and the six geometric vectors.
It does not copy CT intensities or reference-standard masks.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rana.feat_match import find_feat  # noqa: E402


_FLIP = np.diag([-1.0, -1.0, 1.0])


def affine_nn_ras_from_sitk(sitk_stuff):
    sp = np.asarray(sitk_stuff["spacing"], dtype=np.float64)
    O = np.asarray(sitk_stuff["origin"], dtype=np.float64)
    D = np.asarray(sitk_stuff["direction"], dtype=np.float64).reshape(3, 3)
    A = np.eye(4)
    for ax in range(3):
        A[:3, ax] = D[:, 2 - ax] * sp[2 - ax]
    A[:3, 3] = O
    A[:3, :] = _FLIP @ A[:3, :]
    return A


def build_a_pre(A_nn, bbox, new_shape):
    bmin = np.array([b[0] for b in bbox], dtype=np.float64)
    crop_shape = np.array([b[1] - b[0] for b in bbox], dtype=np.float64)
    f = np.asarray(new_shape, dtype=np.float64) / crop_shape
    A = np.eye(4)
    A[:3, :3] = A_nn[:3, :3] / f
    A[:3, 3] = A_nn[:3, :3] @ (bmin + 0.5 / f - 0.5) + A_nn[:3, 3]
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed", required=True, help="nnU-Net 3d_fullres folder")
    ap.add_argument("--raw-image", required=True, help="One raw CT NIfTI used to bind affine")
    ap.add_argument("--case", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import nibabel as nib
    im = nib.load(args.raw_image)
    feat, status, detail = find_feat(args.case, im.shape, im.affine)
    if feat is None:
        raise RuntimeError(f"feature npz not found: {status} {detail}")
    z = np.load(feat, allow_pickle=False)
    pkl = Path(args.preprocessed) / f"{args.case}.pkl"
    props = pickle.load(open(pkl, "rb"))
    A_nn = affine_nn_ras_from_sitk(props["sitk_stuff"])
    # prefer preprocessed array shape if a seg exists; else use pkl
    new_shape = tuple(int(x) for x in props.get("shape_after_cropping_and_before_resampling",
                                                [1, 1, 1]))
    if "shape_after_resampling" in props:
        new_shape = tuple(int(x) for x in props["shape_after_resampling"])
    A_pre = build_a_pre(A_nn, props["bbox_used_for_cropping"], new_shape)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{args.case}.npz",
        A_pre=A_pre.astype(np.float64),
        pre_shape=np.array(new_shape, dtype=np.int32),
        p0=z["p0"], n=z["n"], t1=z["t1"], t2=z["t2"], q0=z["q0"], td=z["td"],
    )
    print(out / f"{args.case}.npz")


if __name__ == "__main__":
    main()
