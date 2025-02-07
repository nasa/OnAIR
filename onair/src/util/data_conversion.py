# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
data_conversion.py
Utility file to perform conversions for supervised learning, and beyond
"""
import numpy as np

CLASSES = {"RED": 0, "YELLOW": 1, "GREEN": 2, "---": 3}


def status_to_one_hot(status):
    """
    Convert status string to one hot positional array
    """
    if isinstance(status, np.ndarray):
        return status
    one_hot = [0.0, 0.0, 0.0, 0.0]
    one_hot[CLASSES[status]] = 1.0
    return list(one_hot)
