import ast
import os
from data_handling.parsers.tlm_json_parser import parseTlmConfJson

## Method to extract configuration data and return 3 dictionaries
def extract_configs(configFilePath, configFiles, csv = False):
    if configFiles == []:
        return {'subsystem_assignments' : {},
                'test_assignments' : {},
                'description_assignments' : {}}

    configs = parseTlmConfJson(configFilePath + configFiles[0])

    configs_len = len(configs['subsystem_assignments'])

    for i in range(configs_len):
        configs['subsystem_assignments'][i] = [configs['subsystem_assignments'][i]]

        test_assign = configs['test_assignments'][i]
        if len(test_assign) > 1:
            test = [test_assign[0]]
            limits = str2lst(test_assign[1])
            test_assign = test + limits
        elif test_assign[0] != 'NOOP':
            test_assign = str2lst(test_assign[0])

        configs['test_assignments'][i] = [test_assign]

    return configs

def str2lst(string):
    try:
        return ast.literal_eval(string)
    except:
        print("Unable to process string representation of list")
        # return string
        
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
