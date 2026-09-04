#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fit RANA anatomical-frame parameters from TotalSegmentator masks.

Writes one compact npz per case containing p0, n, t1, t2, q0, td and the CT
affine. Dense four-channel volumes are optional and off by default so that
patient images are never written to disk by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import nibabel.processing  # noqa: F401  resample_from_to
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rana.constants import ANCHOR_ORGANS, CHANNEL_NAMES  # noqa: E402
from rana.geometry import fit_frame, gamma_at_world  # noqa: E402


def load_mask(path: Path, shape, affine):
    m = nib.load(str(path))
    if m.shape != tuple(shape) or not np.allclose(m.affine, affine, atol=1e-2):
        m = nib.processing.resample_from_to(m, (tuple(shape), affine), order=0)
    return np.asanyarray(m.dataobj) > 0.5


def process(image: Path, mask_dir: Path, out_path: Path, write_dense: bool) -> dict:
    img = nib.load(str(image))
    shape = img.shape
    affine = img.affine.astype(np.float64)
    masks = {}
    for organ in ANCHOR_ORGANS:
        p = mask_dir / f"{organ}.nii.gz"
        if not p.is_file():
            raise FileNotFoundError(p)
        masks[organ] = load_mask(p, shape, affine)
    frame = fit_frame(masks["liver"], masks["portal_vein_and_splenic_vein"],
                      masks["gallbladder"], affine)
    payload = dict(
        channel_names=np.array(CHANNEL_NAMES),
        channel_norm_mm=np.float64(frame["norm_mm"]),
        spar_clamp_mm=np.float64(frame["spar_mm"]),
        dt_clamp=np.float64(frame["dt_clamp"]),
        shape=np.array(shape, dtype=np.int32),
        spacing_mm=np.linalg.norm(affine[:3, :3], axis=0),
        affine=affine,
        p0=frame["p0"], n=frame["n"], t1=frame["t1"], t2=frame["t2"],
        q0=frame["q0"], td=frame["td"],
        plane_fit_points=np.int64(frame["plane_fit_points"]),
        plane_fit_radius_mm=np.float64(frame["plane_fit_radius_mm"]),
        plane_eigvals=frame["plane_eigvals"],
    )
    if write_dense:
        zz, yy, xx = np.meshgrid(
            np.arange(shape[0], dtype=np.float64),
            np.arange(shape[1], dtype=np.float64),
            np.arange(shape[2], dtype=np.float64),
            indexing="ij",
        )
        world = (affine[:3, :3] @ np.stack([zz, yy, xx], 0).reshape(3, -1)).T + affine[:3, 3]
        coords = gamma_at_world(world, frame).reshape(4, *shape).astype(np.float16)
        payload["coords"] = coords
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return dict(out=str(out_path), plane_points=int(frame["plane_fit_points"]))


def main():
    ap = argparse.ArgumentParser(description="Prepare RANA feature npz files")
    ap.add_argument("--image", required=True, help="CT NIfTI path")
    ap.add_argument("--ts-dir", required=True, help="Directory of TotalSegmentator organ NIfTIs")
    ap.add_argument("--out", required=True, help="Output npz path")
    ap.add_argument("--write-dense", action="store_true",
                    help="Also store dense gamma volumes (off by default)")
    args = ap.parse_args()
    rec = process(Path(args.image), Path(args.ts_dir), Path(args.out), args.write_dense)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
