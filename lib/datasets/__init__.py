# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from .aflw import AFLW
from .cofw import COFW
from .face300w import Face300W
from .wflw import WFLW
from .ds_300w_lp import DS_300W_LP
from .mix_300w_lp_wflw import Mix300WLP_WFLW

__all__ = ['AFLW', 'COFW', 'Face300W', 'WFLW', 'DS_300W_LP', 'Mix300WLP_WFLW', 'get_dataset']


def get_dataset(config):

    if config.DATASET.DATASET == 'AFLW':
        return AFLW
    elif config.DATASET.DATASET == 'COFW':
        return COFW
    elif config.DATASET.DATASET == '300W':
        return Face300W
    elif config.DATASET.DATASET == 'WFLW':
        return WFLW
    elif config.DATASET.DATASET == '300W_LP':
        return DS_300W_LP
    elif config.DATASET.DATASET == '300W_LP_WFLW':
        return Mix300WLP_WFLW
    else:
        raise NotImplemented()

