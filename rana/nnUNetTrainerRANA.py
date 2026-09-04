# RANA: Reference-Anchored Neural Anatomy coordinate channels.
#
# gamma(x) = [liver-plane sdf, t1, t2, portal-vein tube distance] / 64 mm,
# computed ANALYTICALLY from per-case sidecar parameters in nnUNet preprocessed
# geometry (sidecars: $RANA_SIDECAR_ROOT/<DatasetID>/<case>.npz,
# built by scripts/prepare_sidecar.py).
#
# Design decisions:
# - Training pipeline (v2, 2026-08-27): gamma channels are appended AFTER the
#   spatial/mirror augmentation, not before. gamma is analytic, and the
#   bgv2 SpatialTransform applies a pure affine warp (p_elastic_deform=0 in the
#   nnUNet default chain), so the warped coordinate channels are recomputed
#   exactly by composing the recorded affine map with the sidecar geometry
#   (_RANAGammaCompose). This removes the two dominant CPU costs measured by
#   dataloader profiling (initial patch 191x257x219):
#     * analytic gamma on the 10.7M-voxel initial patch: 0.34 s/sample
#     * 5-channel grid_sample in SpatialTransform:        0.32 s/sample (mean)
#   and replaces them with analytic gamma on the 2.46M-voxel final patch
#   (~0.08 s/sample). Side effects, all improvements:
#     * sdf/t1/t2 are affine fields: trilinear interpolation reproduces them
#       exactly, and the analytic composition reproduces them exactly, so these
#       channels are numerically unchanged for those voxels; dtube changes by
#       interpolation error on the distance channel.
#     * padded regions (out-of-volume crop/rotation margins) now carry exact
#       analytic gamma values instead of zeros pulled in by grid_sample padding.
# - Validation/inference path is unchanged: nnUNetDatasetRANA.load_case returns
#   the lazily joined 5-channel array (RANAJoinedArray), used by dl_val and by
#   perform_actual_validation.
# - Intensity augmentations remain restricted to image channel 0
#   (RestrictToChannelsTransform, zero-copy variant).
# - build_network_architecture forces +4 input channels both at training and
#   inference time (nnUNet passes plans-derived channel count = 1 in both
#   paths, so the trained 5-channel network is reconstructed consistently).
import os
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import isfile, join
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

NORM_MM = 64.0
SPAR_MM = 40.0
DT_CLAMP = 1.5
GAMMA_CHANNEL_NAMES = ("sdf", "t1", "t2", "dtube")

# ctor parameter names of bgv2 SpatialTransform (all stored as same-name attrs)
_SPATIAL_PARAM_NAMES = (
    "patch_size", "patch_center_dist_from_border", "random_crop",
    "p_elastic_deform", "elastic_deform_scale", "elastic_deform_magnitude",
    "p_synchronize_def_scale_across_axes", "p_rotation", "rotation",
    "p_rot_per_axis", "p_scaling", "scaling",
    "p_synchronize_scaling_across_axes", "bg_style_seg_sampling", "mode_seg",
    "border_mode_seg", "center_deformation", "mode_image", "padding_mode_image",
    "padding_value_seg", "padding_value_image", "align_corners")


def gamma_slabs(out, positions, cis, G, p0, n, t1, t2, q0, td, slab=16):
    """Analytic gamma channels into out[positions] for channel ids cis.

    out: (C, D, H, W) float32. G: (3, 4) mapping array-local voxel index
    (i, j, k) to world RAS mm: world = G[:, :3] @ (i, j, k) + G[:, 3].
    Single numpy kernel shared by the joined-array (validation/inference),
    the training-time compose transform, and the inference predictor.
    """
    A = G.astype(np.float32)
    p0 = p0.astype(np.float32)
    q0 = q0.astype(np.float32)
    vecs = (n.astype(np.float32), t1.astype(np.float32), t2.astype(np.float32))
    td = td.astype(np.float32)
    D = out.shape[1]
    for a0 in range(0, D, slab):
        a1 = min(a0 + slab, D)
        I = np.arange(a0, a1, dtype=np.float32)
        J = np.arange(out.shape[2], dtype=np.float32)
        K = np.arange(out.shape[3], dtype=np.float32)
        gx = A[0, 0] * I[:, None, None] + A[0, 1] * J[None, :, None] + A[0, 2] * K[None, None, :] + A[0, 3]
        gy = A[1, 0] * I[:, None, None] + A[1, 1] * J[None, :, None] + A[1, 2] * K[None, None, :] + A[1, 3]
        gz = A[2, 0] * I[:, None, None] + A[2, 1] * J[None, :, None] + A[2, 2] * K[None, None, :] + A[2, 3]
        o0, o1 = a0, a1
        for pos, ci in zip(positions, cis):
            if ci < 3:
                v = vecs[ci]
                out[pos, o0:o1] = ((gx - p0[0]) * v[0] + (gy - p0[1]) * v[1]
                                   + (gz - p0[2]) * v[2]) / NORM_MM
            else:
                qx, qy, qz = gx - q0[0], gy - q0[1], gz - q0[2]
                spar = np.clip(qx * td[0] + qy * td[1] + qz * td[2], -SPAR_MM, SPAR_MM)
                dx = qx - spar * td[0]
                dy = qy - spar * td[1]
                dz = qz - spar * td[2]
                np.minimum(np.sqrt(dx * dx + dy * dy + dz * dz) / NORM_MM, DT_CLAMP,
                           out=out[pos, o0:o1])


def g_from_ranges(A_pre, ranges):
    """G (3,4) for an array whose local index (0,0,0) sits at volume index
    (ranges[0][0], ranges[1][0], ranges[2][0])."""
    A_pre = np.asarray(A_pre, dtype=np.float64)
    G = np.zeros((3, 4), dtype=np.float64)
    G[:, :3] = A_pre[:3, :3]
    G[:, 3] = A_pre[:3, :3] @ np.array([r[0] for r in ranges], dtype=np.float64) + A_pre[:3, 3]
    return G


class RANAJoinedArray:
    """Lazy (1 + 4, D, H, W) float32 view: channel 0 = preprocessed image
    (blosc2 NDArray, lazy), channels 1..4 = gamma computed analytically for the
    requested patch subgrid. Supports the slicing pattern used by
    acvl_utils.cropping_and_padding.bounding_boxes.crop_and_pad_nd
    (tuple of 4 slices, channel slice first).

    Used on the validation/inference path (no spatial augmentation there).
    """

    def __init__(self, image, sidecar_npz):
        self.image = image
        self.A_pre = np.asarray(sidecar_npz["A_pre"], dtype=np.float64)
        self.p0 = np.asarray(sidecar_npz["p0"], dtype=np.float64)
        self.n = np.asarray(sidecar_npz["n"], dtype=np.float64)
        self.t1 = np.asarray(sidecar_npz["t1"], dtype=np.float64)
        self.t2 = np.asarray(sidecar_npz["t2"], dtype=np.float64)
        self.q0 = np.asarray(sidecar_npz["q0"], dtype=np.float64)
        self.td = np.asarray(sidecar_npz["td"], dtype=np.float64)
        pre_shape = tuple(int(x) for x in sidecar_npz["pre_shape"])
        img_shape = tuple(int(x) for x in image.shape[1:])
        if img_shape != pre_shape:
            raise ValueError(f"RANA sidecar shape {pre_shape} != image shape {img_shape}")
        self.shape = (5,) + img_shape
        self.dtype = np.float32
        self.ndim = 4

    def _gamma_into(self, out, positions, cis, ranges):
        G = g_from_ranges(self.A_pre, ranges)
        gamma_slabs(out, positions, cis, G, self.p0, self.n, self.t1, self.t2,
                    self.q0, self.td)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = idx + (slice(None),) * (4 - len(idx))
        cidx, s0, s1, s2 = idx[0], idx[1], idx[2], idx[3]
        chans = np.atleast_1d(np.arange(self.shape[0])[cidx])
        ranges = []
        for s, dim in zip((s0, s1, s2), self.shape[1:]):
            if not isinstance(s, slice) or s.step not in (None, 1):
                raise NotImplementedError(f"RANAJoinedArray only supports unit-step slices, got {s}")
            start = 0 if s.start is None else s.start
            stop = dim if s.stop is None else s.stop
            ranges.append((start, stop))
        out = np.empty((len(chans),) + tuple(b - a for a, b in ranges), dtype=np.float32)
        gpos, gcis = [], []
        for pos, c in enumerate(chans):
            if c == 0:
                out[pos] = self.image[(0, s0, s1, s2)]
            else:
                gpos.append(pos)
                gcis.append(int(c) - 1)
        if gcis:
            self._gamma_into(out, gpos, gcis, ranges)
        return out


class nnUNetDatasetRANA(nnUNetDatasetBlosc2):
    def __init__(self, folder: str, identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None,
                 sidecar_dir: str = None):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        self.sidecar_dir = sidecar_dir
        if sidecar_dir is not None:
            self.identifiers = [i for i in self.identifiers
                                if isfile(join(sidecar_dir, i + ".npz"))]

    def load_case(self, identifier):
        """Validation/inference path: lazily joined 5-channel array."""
        data, seg, seg_prev, properties = super().load_case(identifier)
        sc = np.load(join(self.sidecar_dir, identifier + ".npz"))
        return RANAJoinedArray(data, sc), seg, seg_prev, properties

    def load_case_train(self, identifier):
        """Training path: plain 1-channel image; gamma geometry travels in
        properties['rana'] and is composed after spatial augmentation by
        _RANAGammaCompose."""
        data, seg, seg_prev, properties = super().load_case(identifier)
        sc = np.load(join(self.sidecar_dir, identifier + ".npz"))
        properties = dict(properties)
        properties["rana"] = dict(
            A_pre=np.asarray(sc["A_pre"], dtype=np.float64),
            p0=np.asarray(sc["p0"], dtype=np.float64),
            n=np.asarray(sc["n"], dtype=np.float64),
            t1=np.asarray(sc["t1"], dtype=np.float64),
            t2=np.asarray(sc["t2"], dtype=np.float64),
            q0=np.asarray(sc["q0"], dtype=np.float64),
            td=np.asarray(sc["td"], dtype=np.float64),
        )
        return data, seg, seg_prev, properties


class RestrictToChannelsTransform(BasicTransform):
    """Apply an inner image-only transform to a subset of image channels.
    Segmentation and other entries pass through unchanged. Contiguous channel
    subsets are handled with zero-copy views (slice in, write back in place)."""

    def __init__(self, inner: BasicTransform, channels: Tuple[int, ...]):
        super().__init__()
        self.inner = inner
        self.channels = list(channels)

    def __call__(self, **data_dict) -> dict:
        img = data_dict.get('image')
        if img is None or img.shape[0] <= max(self.channels):
            return self.inner(**data_dict)
        contiguous = self.channels == list(range(self.channels[0], self.channels[0] + len(self.channels)))
        if contiguous:
            sl = slice(self.channels[0], self.channels[0] + len(self.channels))
            sub = dict(data_dict)
            sub['image'] = img[sl]  # basic slice: view, no copy
            out = self.inner(**sub)
            res = out['image']
            if isinstance(img, torch.Tensor):
                if not isinstance(res, torch.Tensor):
                    res = torch.from_numpy(np.ascontiguousarray(res))
                if res is not img[sl]:
                    img[sl].copy_(res.to(dtype=img.dtype))
            else:
                img[sl][...] = np.asarray(res)
            out['image'] = img
            return out
        # non-contiguous channel subsets: rare path, full-tensor copy semantics
        sub = dict(data_dict)
        sub['image'] = img[self.channels]
        out = self.inner(**sub)
        img = img.clone() if isinstance(img, torch.Tensor) else img.copy()
        img[self.channels] = out['image']
        out['image'] = img
        return out

    def __repr__(self):
        return f"RestrictToChannelsTransform(channels={self.channels}, inner={self.inner!r})"


class _RANASpatialRecorder(SpatialTransform):
    """SpatialTransform that records its sampled parameters so that
    _RANAGammaCompose can reproduce the identical coordinate map analytically.
    Requires p_elastic_deform=0 (nnUNet default): gamma composition is affine."""

    def __init__(self, *args, recorder: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder

    def get_parameters(self, **data_dict) -> dict:
        params = super().get_parameters(**data_dict)
        if params.get('elastic_offsets') is not None:
            raise RuntimeError("RANA gamma composition requires p_elastic_deform=0")
        if self.recorder is not None:
            self.recorder['spatial_affine'] = params['affine']
            self.recorder['shape_in'] = tuple(int(x) for x in data_dict['image'].shape[1:])
        return params


class _RANAMirrorRecorder(MirrorTransform):
    """MirrorTransform that records the sampled axes."""

    def __init__(self, *args, recorder: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder

    def get_parameters(self, **data_dict) -> dict:
        params = super().get_parameters(**data_dict)
        if self.recorder is not None:
            self.recorder['mirror_axes'] = list(params['axes'])
        return params


class _RANAGammaCompose(BasicTransform):
    """Compose the affine map of the final patch and store it as
    data_dict['rana_G'] (3x4 float32, patch-local index -> world RAS mm).
    The gamma channels themselves are computed on GPU in
    nnUNetTrainerRANA.train_step (the analytic evaluation is memory-bound
    single-pass work, essentially free on device), which keeps DA workers free
    of the 4-channel gamma evaluation (~0.09 s/sample measured on the
    96x160x160 final patch).

    The coordinate map of the final patch is

        v(y) = bbox_lb + M (F y + f) + b

    with M, b from the recorded spatial affine (or the integer center-crop
    offset when no rotation/scale was sampled), F, f from the recorded mirror
    axes, bbox_lb the lower corner of the initial crop in volume coordinates
    (may be negative: gamma is analytic and defined there as well). World
    position is A_pre @ (v, 1); G = A_pre o [M F | M f + b + bbox_lb].
    """

    def __init__(self, recorder: dict):
        super().__init__()
        self.recorder = recorder

    def __call__(self, **data_dict) -> dict:
        img = data_dict.get('image')
        geom = data_dict.get('rana_geom')
        if img is None or geom is None:
            raise RuntimeError("_RANAGammaCompose requires 'image' and 'rana_geom' entries")
        if img.ndim != 4:
            raise RuntimeError(f"_RANAGammaCompose expects 3D patches, got shape {img.shape}")
        P = tuple(int(x) for x in img.shape[1:])
        S_crop = self.recorder['shape_in']
        affine = self.recorder.get('spatial_affine')
        if affine is None:
            # grid-None path: crop_tensor with integer start (see bgv2
            # spatial.py _apply_to_image and cropping.py crop_tensor)
            M = np.eye(3)
            b = np.array([int(np.floor(S_crop[d] / 2)) - P[d] // 2 for d in range(3)],
                         dtype=np.float64)
        else:
            A = np.asarray(affine, dtype=np.float64)
            c_patch = np.array([(P[d] - 1) / 2 for d in range(3)])
            c_crop = np.array([(S_crop[d] - 1) / 2 for d in range(3)])
            # grid_sample source position: u(w) = A @ (w - c_patch) + c_crop
            M = A
            b = c_crop - A @ c_patch
        F = np.eye(3)
        f = np.zeros(3)
        for ax in self.recorder.get('mirror_axes', []):
            F[ax, ax] = -1.0
            f[ax] = P[ax] - 1
        M2 = M @ F
        b2 = M @ f + b + np.asarray(geom['bbox_lb'], dtype=np.float64)
        A_pre = np.asarray(geom['A_pre'], dtype=np.float64)
        G = np.zeros((3, 4), dtype=np.float64)
        G[:, :3] = A_pre[:3, :3] @ M2
        G[:, 3] = A_pre[:3, :3] @ b2 + A_pre[:3, 3]
        data_dict['rana_G'] = G.astype(np.float32)
        return data_dict

    def __repr__(self):
        return "_RANAGammaCompose()"


class nnUNetDataLoaderRANA(nnUNetDataLoader):
    """nnUNetDataLoader with the RANA training pipeline: 1-channel crop,
    augmentation, then analytic gamma composition on the final patch.
    generate_train_batch mirrors the parent implementation, adding the
    per-sample 'rana_geom' entry consumed by _RANAGammaCompose."""

    def generate_train_batch(self):
        from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

        selected_keys = self.get_indices()
        data_all = None
        seg_all = None
        G_all = None
        V_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)

                    data, seg, seg_prev, properties = self._data.load_case_train(i)

                    shape = data.shape[1:]
                    bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg,
                                                       properties['class_locations'])
                    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(
                        crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    rana_geom = dict(properties['rana'])
                    rana_geom['bbox_lb'] = [int(b[0]) for b in bbox]

                    if self.transforms is not None:
                        transformed = self.transforms(**{'image': data_cropped,
                                                         'segmentation': seg_cropped,
                                                         'rana_geom': rana_geom})
                        data_sample = transformed['image']
                        seg_sample = transformed['segmentation']
                        if 'rana_G' in transformed:
                            if G_all is None:
                                G_all = np.empty((self.batch_size, 3, 4), dtype=np.float32)
                                V_all = np.empty((self.batch_size, 6, 3), dtype=np.float32)
                            G_all[j] = transformed['rana_G']
                            V_all[j] = np.stack([rana_geom[k] for k in
                                                 ('p0', 'n', 't1', 't2', 'q0', 'td')])
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    if data_all is None:
                        data_all = torch.empty((self.batch_size, *data_sample.shape),
                                               dtype=torch.float32)
                    data_all[j] = data_sample

                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype)
                                       for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape),
                                                  dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample
        batch = {'data': data_all, 'target': seg_all, 'keys': selected_keys}
        if G_all is not None:
            batch['rana_G'] = G_all
            batch['rana_vecs'] = V_all
        return batch


class nnUNetTrainerRANA(nnUNetTrainer):
    NUM_GAMMA_CHANNELS = 4
    DEFAULT_SIDECAR_ROOT = os.environ.get("RANA_SIDECAR_ROOT", "")
    INTENSITY_TRANSFORMS = {
        "GaussianNoiseTransform", "GaussianBlurTransform",
        "MultiplicativeBrightnessTransform", "ContrastTransform",
        "GammaTransform", "SimulateLowResolutionTransform",
    }

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # +4 gamma channels at training AND inference (nnUNet passes the
        # plans-derived channel count in both paths).
        return get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels + nnUNetTrainerRANA.NUM_GAMMA_CHANNELS,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)

    @staticmethod
    def gamma_channels_torch(G: torch.Tensor, vecs: torch.Tensor,
                             patch_size: Tuple[int, int, int]) -> torch.Tensor:
        """Analytic gamma channels on device. G: (B,3,4) float32 mapping
        patch-local voxel index to world RAS mm; vecs: (B,6,3) float32 with
        rows [p0, n, t1, t2, q0, td]. Returns (B,4,*patch_size) float32 with
        channels [sdf, t1, t2, dtube] (same formulas as gamma_slabs)."""
        B = G.shape[0]
        P = patch_size
        dev = G.device
        I = torch.arange(P[0], device=dev, dtype=torch.float32).view(1, P[0], 1, 1)
        J = torch.arange(P[1], device=dev, dtype=torch.float32).view(1, 1, P[1], 1)
        K = torch.arange(P[2], device=dev, dtype=torch.float32).view(1, 1, 1, P[2])
        wx = G[:, 0, 0].view(B, 1, 1, 1) * I + G[:, 0, 1].view(B, 1, 1, 1) * J \
            + G[:, 0, 2].view(B, 1, 1, 1) * K + G[:, 0, 3].view(B, 1, 1, 1)
        wy = G[:, 1, 0].view(B, 1, 1, 1) * I + G[:, 1, 1].view(B, 1, 1, 1) * J \
            + G[:, 1, 2].view(B, 1, 1, 1) * K + G[:, 1, 3].view(B, 1, 1, 1)
        wz = G[:, 2, 0].view(B, 1, 1, 1) * I + G[:, 2, 1].view(B, 1, 1, 1) * J \
            + G[:, 2, 2].view(B, 1, 1, 1) * K + G[:, 2, 3].view(B, 1, 1, 1)

        def affine_channel(vrow):
            # ((w - p0) . v) / NORM_MM, vrow = vecs[:, k]
            return ((wx - vecs[:, 0, 0].view(B, 1, 1, 1)) * vrow[:, 0].view(B, 1, 1, 1)
                    + (wy - vecs[:, 0, 1].view(B, 1, 1, 1)) * vrow[:, 1].view(B, 1, 1, 1)
                    + (wz - vecs[:, 0, 2].view(B, 1, 1, 1)) * vrow[:, 2].view(B, 1, 1, 1)) / NORM_MM

        sdf = affine_channel(vecs[:, 1])
        t1c = affine_channel(vecs[:, 2])
        t2c = affine_channel(vecs[:, 3])
        q0 = vecs[:, 4]
        td = vecs[:, 5]
        qx = wx - q0[:, 0].view(B, 1, 1, 1)
        qy = wy - q0[:, 1].view(B, 1, 1, 1)
        qz = wz - q0[:, 2].view(B, 1, 1, 1)
        spar = (qx * td[:, 0].view(B, 1, 1, 1) + qy * td[:, 1].view(B, 1, 1, 1)
                + qz * td[:, 2].view(B, 1, 1, 1)).clamp_(-SPAR_MM, SPAR_MM)
        dx = qx - spar * td[:, 0].view(B, 1, 1, 1)
        dy = qy - spar * td[:, 1].view(B, 1, 1, 1)
        dz = qz - spar * td[:, 2].view(B, 1, 1, 1)
        dtube = ((dx * dx + dy * dy + dz * dz).sqrt_() / NORM_MM).clamp_(max=DT_CLAMP)
        return torch.stack([sdf, t1c, t2c, dtube], dim=1)

    def train_step(self, batch: dict) -> dict:
        G = batch.pop('rana_G', None)
        V = batch.pop('rana_vecs', None)
        if G is None or V is None:
            raise RuntimeError("nnUNetTrainerRANA.train_step requires 'rana_G'/'rana_vecs' "
                               "in the batch (train dataloader must be nnUNetDataLoaderRANA)")
        data = batch['data'].to(self.device, non_blocking=True)
        G = torch.from_numpy(np.ascontiguousarray(G)).to(self.device, non_blocking=True)
        V = torch.from_numpy(np.ascontiguousarray(V)).to(self.device, non_blocking=True)
        gamma = self.gamma_channels_torch(G, V, tuple(int(x) for x in data.shape[2:]))
        batch['data'] = torch.cat([data, gamma], dim=1)
        return super().train_step(batch)

    def initialize(self):
        super().initialize()
        # bind the RANA dataset (with sidecar join + missing-sidecar filtering) as
        # self.dataset_class so that every downstream path uses it, including
        # perform_actual_validation which instantiates dataset_class directly.
        sd = self._sidecar_dir()

        class _BoundRANADataset(nnUNetDatasetRANA):
            def __init__(self, folder, identifiers=None,
                         folder_with_segs_from_previous_stage=None):
                super().__init__(folder, identifiers,
                                 folder_with_segs_from_previous_stage, sidecar_dir=sd)

        self.dataset_class = _BoundRANADataset

    def _sidecar_dir(self) -> str:
        root = os.environ.get("RANA_SIDECAR_ROOT", self.DEFAULT_SIDECAR_ROOT)
        dsid = self.plans_manager.dataset_name.split("_")[0]
        return join(root, dsid)

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        sd = self._sidecar_dir()
        tr_keep = [k for k in tr_keys if isfile(join(sd, k + ".npz"))]
        val_keep = [k for k in val_keys if isfile(join(sd, k + ".npz"))]
        self.print_to_log_file(
            f"RANA sidecar filter ({sd}): train {len(tr_keys)} -> {len(tr_keep)}, "
            f"val {len(val_keys)} -> {len(val_keep)}; "
            f"dropped train={sorted(set(tr_keys) - set(tr_keep))} "
            f"val={sorted(set(val_keys) - set(val_keep))}")
        dataset_tr = nnUNetDatasetRANA(self.preprocessed_dataset_folder, tr_keep,
                                       folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                       sidecar_dir=sd)
        dataset_val = nnUNetDatasetRANA(self.preprocessed_dataset_folder, val_keep,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        sidecar_dir=sd)
        return dataset_tr, dataset_val

    def get_training_transforms(self, patch_size, rotation_for_DA,
                                deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, **kwargs) -> BasicTransform:
        if do_dummy_2d_data_aug:
            raise RuntimeError("RANA gamma-compose training pipeline does not "
                               "support dummy 2D augmentation")
        tr = super().get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, **kwargs)
        recorder: dict = {}
        wrapped = []
        for t in tr.transforms:
            if isinstance(t, SpatialTransform):
                sp_kwargs = {name: getattr(t, name) for name in _SPATIAL_PARAM_NAMES}
                t = _RANASpatialRecorder(recorder=recorder, **sp_kwargs)
            elif isinstance(t, MirrorTransform):
                t = _RANAMirrorRecorder(t.allowed_axes, recorder=recorder)
            # intensity augmentations stay restricted to the image channel 0
            # (after the pipeline change the image is 1-channel here anyway;
            # the wrap is kept as a guard against chain reordering)
            if isinstance(t, RandomTransform) and type(t.transform).__name__ in self.INTENSITY_TRANSFORMS:
                t.transform = RestrictToChannelsTransform(t.transform, (0,))
            elif type(t).__name__ in self.INTENSITY_TRANSFORMS:
                t = RestrictToChannelsTransform(t, (0,))
            wrapped.append(t)
            if isinstance(t, _RANAMirrorRecorder):
                wrapped.append(_RANAGammaCompose(recorder))
        return ComposeTransforms(wrapped)

    def get_dataloaders(self):
        # identical to nnUNetTrainer.get_dataloaders except that the training
        # loader is nnUNetDataLoaderRANA (1-channel crop + gamma composition).
        # The validation loader is the stock class on the joined 5-channel
        # dataset. Referenced names are imported lazily to match the parent
        # module's import list.
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

        if self.dataset_class is None:
            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size,
         mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnUNetDataLoaderRANA(dataset_tr, self.batch_size,
                                     initial_patch_size,
                                     self.configuration_manager.patch_size,
                                     self.label_manager,
                                     oversample_foreground_percent=self.oversample_foreground_percent,
                                     sampling_probabilities=None, pad_sides=None,
                                     transforms=tr_transforms,
                                     probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None,
                                  transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class nnUNetTrainerRANA_5epochs(nnUNetTrainerRANA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5


class nnUNetTrainerRANA_4epochs(nnUNetTrainerRANA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 4
