# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

from .tlm_json_parser import parseTlmConfJson, str2lst
import datetime


def extract_meta_data_handle_ss_breakdown(meta_data_file, ss_breakdown):
    parsed_meta_data = extract_meta_data(meta_data_file)
    if ss_breakdown == False:
        num_elements = len(parsed_meta_data["subsystem_assignments"])
        parsed_meta_data["subsystem_assignments"] = [
            ["MISSION"] for elem in range(num_elements)
        ]
    return parsed_meta_data


## Method to extract configuration data and return 3 dictionaries
def extract_meta_data(meta_data_file):
    assert meta_data_file != ""

    configs = parseTlmConfJson(meta_data_file)

    configs_len = len(configs["subsystem_assignments"])

    for i in range(configs_len):
        if configs["subsystem_assignments"][i] != "NONE":
            configs["subsystem_assignments"][i] = [configs["subsystem_assignments"][i]]
        else:
            configs["subsystem_assignments"][i] = []

        test_assign = configs["test_assignments"][i]

        for j in range(len(test_assign)):
            if len(test_assign[j]) > 1:
                test = [test_assign[j][0]]
                limits = str2lst(test_assign[j][1])
                test_assign[j] = test + limits

    return configs


def floatify_input(_input, remove_str=False):
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
    try:
        t = datetime.datetime.strptime(time_str, "%Y-%j-%H:%M:%S.%f")
        return t.timestamp()
    except:
        min_sec = time_str.split(":")
        # Use 1 am on Jan 1st, 2000 as the date if only minutes and seconds are specified
        t = datetime.datetime(2000, 1, 1, 1, int(min_sec[0]), int(min_sec[1]), 0)
        return t.timestamp()
