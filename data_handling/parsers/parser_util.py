import ast
import os
from data_handling.parsers.tlm_json_parser import * 

## Method to extract configuration data and return 3 dictionaries
def extract_configs(configFilePath, configFiles, csv = False):
    ss_assigns = {}
    test_assigns = {}
    desc_assigns = {}

    for cFile in configFiles:
        subsystem_assignments, tests, descs = extract_config(configFilePath, cFile, csv=csv)
        ss_assigns = subsystem_assignments
        test_assigns = tests
        desc_assigns = descs

    configs = {'subsystem_assignments' : ss_assigns,
               'test_assignments' : test_assigns,
               'description_assignments' : desc_assigns}
    return configs

## Helper method for extract_configs
def extract_config(configFilePath, configFile, csv = False):
    subsystem_assignments = []
    mnemonic_tests = []
    descriptions = []

    descriptor_file = open(configFilePath + configFile, "r+")
    data_str = descriptor_file.read()
    descriptor_file.close()

    dataPts = data_str.split('\n')
    dataPts = [i for i in dataPts if i]
    # if not csv:
    #     dataPts = dataPts[:len(dataPts)-1]

    for field_info in dataPts:
        data = field_info.split(' : ')
        if len(data) == 1:
            description = 'No description'
        else:
            description = data[1]

        descriptors = data[0].split(' ')

        subsystem_assignments.append(ast.literal_eval(descriptors[1]))
        descriptions.append(description)

        test_list = []
        for test in descriptors[2:]:
            test_list.append(ast.literal_eval(test))

        mnemonic_tests.append(test_list)

    return subsystem_assignments, mnemonic_tests, descriptions

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
