"""Closed-form anatomical frame used by RANA.

All quantities are in RAS world millimetres. The functions here match the
equations in the paper: hepatic-plane channels gamma_1:3 and the portal-axis
distance gamma_4.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from rana.constants import (
    DT_CLAMP,
    MIN_PLANE_PTS,
    MIN_PV_VOXELS,
    NORM_MM,
    PLANE_RADII_MM,
    SPAR_MM,
)


def tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, n))) > 0.90:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)
    return t1.astype(np.float64), t2.astype(np.float64)


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = points.mean(0)
    d = points - p0
    cov = (d.T @ d) / max(len(points) - 1, 1)
    w, V = np.linalg.eigh(cov)
    return p0, V[:, 0], w


def centroid_world(mask: np.ndarray, affine: np.ndarray):
    idx = np.nonzero(mask)
    if idx[0].shape[0] == 0:
        return None, 0
    c = np.array([idx[0].mean(), idx[1].mean(), idx[2].mean()], dtype=np.float64)
    return affine[:3, :3] @ c + affine[:3, 3], int(idx[0].shape[0])


def fit_frame(liver_mask: np.ndarray, pv_mask: np.ndarray, gb_mask: np.ndarray,
              affine: np.ndarray) -> dict:
    """Fit the four-channel frame from binary masks on the CT grid."""
    affine = np.asarray(affine, dtype=np.float64)
    gb_cent, gb_n = centroid_world(gb_mask, affine)
    liv_cent, liv_n = centroid_world(liver_mask, affine)
    if gb_n == 0 or liv_n == 0:
        raise ValueError("gallbladder or liver mask is empty")

    boundary = liver_mask & ~ndimage.binary_erosion(liver_mask)
    bidx = np.nonzero(boundary)
    bpts = (affine[:3, :3] @ np.stack(bidx, 0)).T + affine[:3, 3]
    sel = bpts
    used_radius = "all"
    for r in PLANE_RADII_MM:
        if r is None:
            sel = bpts
            used_radius = "all"
            break
        d = np.linalg.norm(bpts - gb_cent, axis=1)
        if int((d <= r).sum()) >= MIN_PLANE_PTS:
            sel = bpts[d <= r]
            used_radius = r
            break
    if len(sel) < MIN_PLANE_PTS:
        sel = bpts
        used_radius = "all"
    p0, n, eigvals = fit_plane(sel)
    if float(np.dot(gb_cent - liv_cent, n)) < 0:
        n = -n
    t1, t2 = tangent_basis(n)

    pidx = np.nonzero(pv_mask)
    if pidx[0].shape[0] < MIN_PV_VOXELS:
        raise ValueError("portal/splenic-vein mask is too small")
    pv_pts = (affine[:3, :3] @ np.stack(pidx, 0)).T + affine[:3, 3]
    q0 = pv_pts.mean(0)
    dp = pv_pts - q0
    covp = (dp.T @ dp) / max(len(pv_pts) - 1, 1)
    _w, Vp = np.linalg.eigh(covp)
    td = Vp[:, -1]
    td = td / np.linalg.norm(td)
    return dict(
        p0=p0.astype(np.float64),
        n=n.astype(np.float64),
        t1=t1,
        t2=t2,
        q0=q0.astype(np.float64),
        td=td.astype(np.float64),
        plane_fit_points=int(len(sel)),
        plane_fit_radius_mm=-1.0 if used_radius == "all" else float(used_radius),
        plane_eigvals=np.asarray(eigvals, dtype=np.float64),
        norm_mm=NORM_MM,
        spar_mm=SPAR_MM,
        dt_clamp=DT_CLAMP,
    )


def gamma_at_world(xyz: np.ndarray, frame: dict) -> np.ndarray:
    """Evaluate gamma at world points xyz of shape (..., 3)."""
    p0, n, t1, t2 = frame["p0"], frame["n"], frame["t1"], frame["t2"]
    q0, td = frame["q0"], frame["td"]
    d = xyz - p0
    g1 = (d * n).sum(-1) / NORM_MM
    g2 = (d * t1).sum(-1) / NORM_MM
    g3 = (d * t2).sum(-1) / NORM_MM
    q = xyz - q0
    s = np.clip((q * td).sum(-1), -SPAR_MM, SPAR_MM)
    dist = np.linalg.norm(q - s[..., None] * td, axis=-1) / NORM_MM
    g4 = np.minimum(dist, DT_CLAMP)
    return np.stack([g1, g2, g3, g4], axis=0)
