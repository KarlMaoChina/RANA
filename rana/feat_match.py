"""Locate a feature npz for a case identifier.

The public matcher looks for an exact `{case_id}.npz` under RANA_FEAT_ROOT,
then for `{case_id}.npz` in one-level subdirectories. No hospital-specific
naming rules are applied.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def feat_root(explicit: str | None = None) -> Path:
    root = explicit or os.environ.get("RANA_FEAT_ROOT", "")
    if not root:
        raise EnvironmentError("Set RANA_FEAT_ROOT to the directory of feature npz files")
    return Path(root)


def find_feat(case_id: str, img_shape, img_affine, feat_dir: str | None = None):
    root = feat_root(feat_dir)
    candidates = [root / f"{case_id}.npz"]
    candidates.extend(sorted(root.glob(f"*/{case_id}.npz")))
    img_shape = tuple(int(x) for x in img_shape)
    for path in candidates:
        if not path.is_file():
            continue
        z = np.load(path, allow_pickle=False)
        if "shape" in z.files and "affine" in z.files:
            fshape = tuple(int(x) for x in z["shape"])
            faffine = np.asarray(z["affine"], dtype=np.float64)
            if fshape != img_shape or not np.allclose(faffine, img_affine, atol=1e-2):
                continue
        return str(path), "ok_exact", path.stem
    return None, "no_npz", f"no geometry-matched npz for {case_id}"
