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

classes = {'RED' : 0,
        'YELLOW' : 1,
         'GREEN' : 2,
           '---' : 3}

def floatify_input(_input, remove_str=False):
    floatified = []
    for i in _input:
        try:
            x = float(i)
            floatified.append(x)
        except ValueError:
            try:
                x = i.replace('-', '').replace(':', '').replace('.', '')
                floatified.append(float(x))
            except:
                if remove_str == False:
                    floatified.append(0.0)
                else:
                    continue
                continue
    return floatified

def status_to_oneHot(status):
    if isinstance(status, np.ndarray):
        return status
    one_hot = [0.0, 0.0, 0.0, 0.0]
    one_hot[classes[status]] = 1.0
    return list(one_hot)
