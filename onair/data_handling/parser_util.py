# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import os
from .tlm_json_parser import parseTlmConfJson, str2lst
from pandas import to_datetime
import datetime

## Method to extract configuration data and return 3 dictionaries
def extract_meta_data(meta_data_file):
    """
    Extracts metadata from a telemetry configuration file.

    Args:
    --------
        meta_data_file (str): The path to the telemetry configuration file.

    Returns:
    --------
        dict: A dictionary containing extracted metadata including subsystem assignments and test assignments.
    """
    assert meta_data_file != ''

    configs = parseTlmConfJson(meta_data_file)

    configs_len = len(configs['subsystem_assignments'])

    for i in range(configs_len):
        if configs['subsystem_assignments'][i] != 'NONE':
            configs['subsystem_assignments'][i] = [configs['subsystem_assignments'][i]]
        else:
            configs['subsystem_assignments'][i] = []

        test_assign = configs['test_assignments'][i]

        for j in range(len(test_assign)):
            if len(test_assign[j]) > 1:
                test = [test_assign[j][0]]
                limits = str2lst(test_assign[j][1])
                test_assign[j] = test + limits

    return configs

def floatify_input(_input, remove_str=False):
     """
    Convert a list of inputs to float values.

    Args:
    --------
        _input (list): A list of input values.
        remove_str (bool): If True, remove non-convertible string values; otherwise, set them to 0.0.

    Returns:
    --------
        list: A list of float values.
    """
    floatified = []
    for i in _input:
        try:
            x = float(i)
            floatified.append(x)
        except ValueError:
            try:
                x = convert_str_to_timestamp(i)
                floatified.append(x)
            except:
                if remove_str == False:
                    floatified.append(0.0)
                else:
                    continue
                continue
    return floatified

def convert_str_to_timestamp(time_str):
    """
    Convert a timestamp string to a Unix timestamp.

    Args:
    --------
        time_str (str): The timestamp string in a valid format.

    Returns:
    --------
        float: The Unix timestamp.
    """
    try:
        t = to_datetime(time_str)
        return t.timestamp()
    except:
        t = datetime.datetime.strptime(time_str[:24], '%Y-%j-%H:%M:%S.%f')
        return t.timestamp()
