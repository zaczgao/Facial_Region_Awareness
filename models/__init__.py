# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from .lewel import LEWELB, LEWELB_EMAN
from .fra import FRAB, FRAB_EMAN



def get_model(model):
    """
    Args:
        model (str or callable):

    Returns:
        Model
    """
    if isinstance(model, str):
        model = {
            "LEWELB": LEWELB,
            "LEWELB_EMAN": LEWELB_EMAN,
            "FRAB": FRAB,
            "FRAB_EMAN": FRAB_EMAN,
        }[model]
    return model