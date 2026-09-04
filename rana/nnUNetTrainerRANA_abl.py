# RANA ablation trainers that zero selected coordinate channels.
#
# Ablation method: all-zero gamma
# placeholder. The input channel count and the network architecture are kept
# exactly as in the full RANA model (5 input channels); only the values of the
# ablated gamma channels are forced to zero, so the ablation reading reflects
# the information contribution of the channel and not a capacity change.
#
# Gamma channel order (GAMMA_CHANNEL_NAMES in nnUNetTrainerRANA):
#   index 0 = sdf    (liver-surface signed distance)
#   index 1 = t1     (in-plane tangential coordinate 1)
#   index 2 = t2     (in-plane tangential coordinate 2)
#   index 3 = dtube  (portal-vein tube distance)
#
# The zeroing is applied at every gamma injection point:
#   1. train_step: after the on-GPU analytic composition in
#      gamma_channels_torch (subclass wrapper below);
#   2. validation dataloader: nnUNetDatasetRANAZero.load_case returns
#      RANAJoinedArrayZero, which zeroes the selected channels after the
#      analytic evaluation in _gamma_into;
#   3. perform_actual_validation: self.dataset_class is re-bound to the
#      zeroing dataset in initialize().
# The inference-side counterparts live in nnUNetPredictorRANA_abl.py
# (this repository / ), same GAMMA_ZERO_CHANNELS convention.
#
# Epoch policy: ablation variants use 1000 epochs to match the paper protocol.

from typing import List, Tuple

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import isfile, join

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerRANA import (
    RANAJoinedArray,
    nnUNetDatasetRANA,
    nnUNetTrainerRANA,
)


class RANAJoinedArrayZero(RANAJoinedArray):
    """RANAJoinedArray with selected gamma channels forced to zero after the
    analytic evaluation. zero_channels holds indices into [sdf, t1, t2, dtube].
    Channel 0 (image) is never touched."""

    zero_channels: Tuple[int, ...] = ()

    def _gamma_into(self, out, positions, cis, ranges):
        super()._gamma_into(out, positions, cis, ranges)
        if self.zero_channels:
            zc = frozenset(self.zero_channels)
            for pos, ci in zip(positions, cis):
                if ci in zc:
                    out[pos] = 0.0


class nnUNetDatasetRANAZero(nnUNetDatasetRANA):
    """nnUNetDatasetRANA whose load_case returns the zero-placeholder joined
    array (validation/inference path). The training path is unchanged: gamma
    is composed on GPU in train_step and zeroed there."""

    zero_channels: Tuple[int, ...] = ()

    def load_case(self, identifier):
        data, seg, seg_prev, properties = nnUNetDatasetBlosc2.load_case(self, identifier)
        sc = np.load(join(self.sidecar_dir, identifier + ".npz"))
        arr = RANAJoinedArrayZero(data, sc)
        arr.zero_channels = self.zero_channels
        return arr, seg, seg_prev, properties


class nnUNetTrainerRANA_ZeroGammaBase(nnUNetTrainerRANA):
    """Base class for all-zero gamma placeholder ablations.

    GAMMA_ZERO_CHANNELS: indices into [sdf, t1, t2, dtube] that are forced to
    zero on every gamma injection path. Architecture (5 input channels) is
    inherited unchanged from nnUNetTrainerRANA."""

    GAMMA_ZERO_CHANNELS: Tuple[int, ...] = ()
    NUM_EPOCHS_ABLATION = 1000

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.NUM_EPOCHS_ABLATION

    def gamma_channels_torch(self, G: torch.Tensor, vecs: torch.Tensor,
                             patch_size: Tuple[int, int, int]) -> torch.Tensor:
        """Analytic gamma channels with the ablated channels zeroed. Overrides
        the base staticmethod with an instance method; the only call site is
        nnUNetTrainerRANA.train_step (self.gamma_channels_torch(...))."""
        gamma = nnUNetTrainerRANA.gamma_channels_torch(G, vecs, patch_size)
        if self.GAMMA_ZERO_CHANNELS:
            gamma[:, list(self.GAMMA_ZERO_CHANNELS)] = 0
        return gamma

    def _zero_dataset_class(self):
        """nnUNetDatasetRANAZero bound to this trainer's sidecar dir and zero
        channel set (same binding pattern as nnUNetTrainerRANA.initialize)."""
        sd = self._sidecar_dir()
        zc = tuple(self.GAMMA_ZERO_CHANNELS)

        class _BoundRANAZeroDataset(nnUNetDatasetRANAZero):
            zero_channels = zc

            def __init__(self, folder, identifiers=None,
                         folder_with_segs_from_previous_stage=None):
                super().__init__(folder, identifiers,
                                 folder_with_segs_from_previous_stage,
                                 sidecar_dir=sd)

        return _BoundRANAZeroDataset

    def initialize(self):
        super().initialize()
        # re-bind dataset_class so that perform_actual_validation (which
        # instantiates dataset_class directly) also sees the zeroed channels
        self.dataset_class = self._zero_dataset_class()

    def get_tr_and_val_datasets(self):
        # same body as nnUNetTrainerRANA.get_tr_and_val_datasets, with the
        # zeroing dataset class in place of nnUNetDatasetRANA
        tr_keys, val_keys = self.do_split()
        sd = self._sidecar_dir()
        tr_keep = [k for k in tr_keys if isfile(join(sd, k + ".npz"))]
        val_keep = [k for k in val_keys if isfile(join(sd, k + ".npz"))]
        self.print_to_log_file(
            f"RANA ablation zero-channels {tuple(self.GAMMA_ZERO_CHANNELS)} of "
            f"(sdf, t1, t2, dtube); sidecar filter ({sd}): "
            f"train {len(tr_keys)} -> {len(tr_keep)}, "
            f"val {len(val_keys)} -> {len(val_keep)}; "
            f"dropped train={sorted(set(tr_keys) - set(tr_keep))} "
            f"val={sorted(set(val_keys) - set(val_keep))}")
        ds_class = self._zero_dataset_class()
        dataset_tr = ds_class(self.preprocessed_dataset_folder, tr_keep,
                              folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = ds_class(self.preprocessed_dataset_folder, val_keep,
                               folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        return dataset_tr, dataset_val

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            f"RANA ablation trainer {type(self).__name__}: "
            f"GAMMA_ZERO_CHANNELS={tuple(self.GAMMA_ZERO_CHANNELS)} "
            f"(indices into [sdf, t1, t2, dtube]), num_epochs={self.num_epochs}")


class nnUNetTrainerRANA_SDFOnly(nnUNetTrainerRANA_ZeroGammaBase):
    """Keep only the liver-surface SDF channel; t1/t2/dtube zeroed."""

    GAMMA_ZERO_CHANNELS = (1, 2, 3)


class nnUNetTrainerRANA_NoTangential(nnUNetTrainerRANA_ZeroGammaBase):
    """Tangential coordinates zeroed; SDF and portal-axis distance kept."""

    GAMMA_ZERO_CHANNELS = (1, 2)


class nnUNetTrainerRANA_NoPortal(nnUNetTrainerRANA_ZeroGammaBase):
    """Portal-axis distance zeroed; SDF and tangential coordinates kept."""

    GAMMA_ZERO_CHANNELS = (3,)

