# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import os
from data_handling.parsers.tlm_json_parser import parseTlmConfJson, str2lst
from pandas import to_datetime
import datetime

## Method to extract configuration data and return 3 dictionaries
def extract_configs(configFilePath, configFile, csv = False):
    assert configFile != ''

    configs = parseTlmConfJson(configFilePath + configFile)

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
        
def process_filepath(path, return_config=False, csv = False):
    if csv:
        filename =  path.split(os.sep)[-1].replace('_CONFIG', '')
        filename = filename.replace('.txt', '.csv')
        if return_config == True:
            filename = filename.replace('.csv', '_CONFIG.txt')
        return filename
    else:
        filename =  path.split(os.sep)[-1].replace('_CONFIG', '')
        if return_config == True:
            filename = filename.replace('.txt', '_CONFIG.txt')
        return filename

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
        t = to_datetime(time_str)
        return t.timestamp()
    except:
        t = datetime.datetime.strptime(time_str[:24], '%Y-%j-%H:%M:%S.%f')
        return t.timestamp()
