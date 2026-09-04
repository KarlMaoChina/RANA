#!/usr/bin/env python
# RANA ablation predictors: all-zero gamma placeholder on the inference side.
#
# Counterpart of nnUNetTrainerRANA_abl.py (training side). Same convention:
# GAMMA_ZERO_CHANNELS holds indices into [sdf, t1, t2, dtube]; the named
# variants below match the training-side ablation trainers. The full model
# leaves gamma unchanged and uses nnUNetPredictorRANA.
#
# The zeroing wraps nnUNetPredictorRANA._wrap_with_gamma: the base class
# concatenates [image, sdf, t1, t2, dtube] into item["data"], so gamma channel
# ci sits at channel index 1 + ci. Zeroing after the base injection keeps the
# geometry code path identical to the full model.
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rana.nnUNetPredictorRANA import nnUNetPredictorRANA  # noqa: E402


class nnUNetPredictorRANAZero(nnUNetPredictorRANA):
    """nnUNetPredictorRANA with selected gamma channels forced to zero after
    injection (all-zero placeholder ablation)."""

    GAMMA_ZERO_CHANNELS: Tuple[int, ...] = ()

    def _wrap_with_gamma(self, iterator, input_list_of_lists):
        zc = tuple(self.GAMMA_ZERO_CHANNELS)
        for item in super()._wrap_with_gamma(iterator, input_list_of_lists):
            if zc:
                data = item["data"]
                for ci in zc:
                    data[1 + ci] = 0
                if self.gamma_verbose:
                    print(f"RANA ablation: zeroed gamma channels {zc} "
                          f"(indices into [sdf, t1, t2, dtube])")
            yield item


class nnUNetPredictorRANA_SDFOnly(nnUNetPredictorRANAZero):
    """Inference counterpart: only SDF kept."""

    GAMMA_ZERO_CHANNELS = (1, 2, 3)


class nnUNetPredictorRANA_NoTangential(nnUNetPredictorRANAZero):
    """Inference counterpart: t1/t2 zeroed."""

    GAMMA_ZERO_CHANNELS = (1, 2)


class nnUNetPredictorRANA_NoPortal(nnUNetPredictorRANAZero):
    """Inference counterpart: portal-axis distance zeroed."""

    GAMMA_ZERO_CHANNELS = (3,)
