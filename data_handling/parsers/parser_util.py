import ast
import os
from data_handling.parsers.tlm_json_parser import parseTlmConfJson, str2lst

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
        if len(test_assign) > 1:
            test = [test_assign[0]]
            limits = str2lst(test_assign[1])
            test_assign = test + limits
        elif test_assign[0] != 'NOOP':
            test_assign = str2lst(test_assign[0])

        configs['test_assignments'][i] = [test_assign]

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
