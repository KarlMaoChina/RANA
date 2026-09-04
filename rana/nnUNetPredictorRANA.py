#!/usr/bin/env python
# RANA inference predictor: injects the 4 analytic gamma channels into the
# preprocessed 1-channel image inside the prediction pipeline.
#
# Injection point: nnUNetPredictor._internal_get_data_iterator_from_lists_of_filenames
# yields per-case dicts {'data': (1, X, Y, Z) float32 tensor in preprocessed
# geometry, 'data_properties', 'ofile'}. The fromfiles iterator preserves input
# order (round-robin over shard workers with queue maxsize 1), so the k-th item
# corresponds to the k-th input file list. For each case we:
#   1. derive the case id from the input filename (strip _0000 + file_ending)
#   2. locate the feat npz via feat_match.find_feat (needs native shape/affine
#      from the input nifti)
#   3. build A_pre from data_properties (sitk_stuff -> RAS-flipped native
#      affine, bbox_used_for_cropping, resampled shape) with the same geometry
#      functions as scripts/prepare_sidecar.py
#   4. compute gamma on the full preprocessed grid with the shared gamma_slabs
#      kernel (identical to training-side values up to float32 rounding)
#   5. concatenate to a 5-channel tensor and yield the modified item
import os
import sys
from typing import List, Union

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rana.feat_match import find_feat  # noqa: E402

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402
from rana.nnUNetTrainerRANA import gamma_slabs  # noqa: E402

_FLIP = np.diag([-1.0, -1.0, 1.0])  # RAS <-> LPS (involutive)


def affine_nn_ras_from_sitk(sitk_stuff):
    """native-space affine for nnUNet (z,y,x) array axes -> RAS world.

    SimpleITK reports LPS for the same nifti whose nibabel affine (used for the
    feat vectors) is RAS, so the sitk-derived affine is flipped back to RAS.
    The sitk-derived affine is flipped back to RAS.
    """
    sp = np.asarray(sitk_stuff["spacing"], dtype=np.float64)      # sitk (x,y,z)
    O = np.asarray(sitk_stuff["origin"], dtype=np.float64)
    D = np.asarray(sitk_stuff["direction"], dtype=np.float64).reshape(3, 3)
    A = np.eye(4)
    for ax in range(3):  # nnUNet axis ax corresponds to sitk axis 2-ax
        A[:3, ax] = D[:, 2 - ax] * sp[2 - ax]
    A[:3, 3] = O
    A[:3, :] = _FLIP @ A[:3, :]
    return A


def build_a_pre(A_nn, bbox, new_shape):
    """Map native nnU-Net affine through crop and resample to A_pre."""
    bmin = np.array([b[0] for b in bbox], dtype=np.float64)
    crop_shape = np.array([b[1] - b[0] for b in bbox], dtype=np.float64)
    f = np.asarray(new_shape, dtype=np.float64) / crop_shape
    A = np.eye(4)
    A[:3, :3] = A_nn[:3, :3] / f
    A[:3, 3] = A_nn[:3, :3] @ (bmin + 0.5 / f - 0.5) + A_nn[:3, 3]
    return A


class nnUNetPredictorRANA(nnUNetPredictor):
    def __init__(self, *args, gamma_verbose: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_verbose = gamma_verbose
        # debug hook: if set, the injected gamma channels of each case are
        # saved as <case>_gamma.npy in this directory (used by the self-check)
        self.gamma_debug_dir = os.environ.get("RANA_PRED_DEBUG_DIR") or None

    def _internal_get_data_iterator_from_lists_of_filenames(
            self, input_list_of_lists: List[List[str]],
            seg_from_prev_stage_files: Union[List[str], None],
            output_filenames_truncated: Union[List[str], None],
            num_processes: int):
        base = super()._internal_get_data_iterator_from_lists_of_filenames(
            input_list_of_lists, seg_from_prev_stage_files,
            output_filenames_truncated, num_processes)
        return self._wrap_with_gamma(base, input_list_of_lists)

    def _wrap_with_gamma(self, iterator, input_list_of_lists):
        import nibabel as nib
        file_ending = self.dataset_json["file_ending"]
        for item, files in zip(iterator, input_list_of_lists):
            data = item["data"]
            assert isinstance(data, torch.Tensor) and data.shape[0] == 1, \
                f"RANA predictor expects 1-channel preprocessed data, got {type(data)} {getattr(data, 'shape', None)}"
            fname = os.path.basename(files[0])
            case = fname[:-(len(file_ending) + 5)]  # strip _0000.nii.gz
            im = nib.load(files[0])
            feat_npz, mstatus, mdetail = find_feat(case, im.shape, im.affine)
            if feat_npz is None:
                raise RuntimeError(f"RANA predictor: no feat npz for {case}: {mstatus} {mdetail}")
            z = np.load(feat_npz, allow_pickle=False)

            props = item["data_properties"]
            A_nn_ras = affine_nn_ras_from_sitk(props["sitk_stuff"])
            new_shape = tuple(int(x) for x in data.shape[1:])
            A_pre = build_a_pre(A_nn_ras, props["bbox_used_for_cropping"], new_shape)

            gamma = np.empty((4,) + new_shape, dtype=np.float32)
            gamma_slabs(gamma, [0, 1, 2, 3], [0, 1, 2, 3], A_pre[:3, :],
                        z["p0"], z["n"], z["t1"], z["t2"], z["q0"], z["td"])
            if self.gamma_debug_dir:
                os.makedirs(self.gamma_debug_dir, exist_ok=True)
                np.save(os.path.join(self.gamma_debug_dir, case + "_gamma.npy"), gamma)
            item["data"] = torch.cat([data, torch.from_numpy(gamma)], dim=0)
            if self.gamma_verbose:
                print(f"RANA gamma injected for {case}: shape {tuple(item['data'].shape)}")
            yield item
