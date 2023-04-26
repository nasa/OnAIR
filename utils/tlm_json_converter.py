import orjson
import os
import ast
import argparse

# Functions for converting old configs to new format

# convert tlm txt config file to json file
def convertTlmToJson(tlm, json):
    tlm_path = getConfigPath(tlm)
    json_path = getConfigPath(json)
    data = parseTlmConfTxt(tlm_path)
    json_data = convertTlmDictToJsonDict(data)
    writeToJson(json_path, json_data)
    
# convert tlm data format to readable json
def convertTlmDictToJsonDict(data):
    labels, subsys_assigns, mnemonics, descriptions = data
    num_elems = len(labels)
    assert num_elems == len(subsys_assigns) and num_elems == len(mnemonics) and num_elems == len(descriptions)
    
    json_data = {}
    json_data['NONE'] = {}
    for s_list in subsys_assigns:
        for s in s_list:
            json_data[s] = {}
    
    for i in range(num_elems):
        data = getJsonData(labels[i], mnemonics[i], descriptions[i])
        
        if subsys_assigns[i] == []:
            mergeDicts(json_data['NONE'], data)
        for s in subsys_assigns[i]:
            mergeDicts(json_data[s], data)
    
    json_data = {'subsystems' : json_data}
    json_data['order'] = labels
    return json_data

# helper function to organize data parsed from tlm file into desired json format
def getJsonData(label, mnemonics, description):
    if str.upper(label) == 'TIME':
        return {label : {'conversion' : '', str(mnemonics[0][0]) : '[]', str(mnemonics[0][1]) : '[]', 'description' : str(description)}}
    
    test_attr = mnemonics[0][0]
    limits_attr = mnemonics[0][1:]
    json_data = {label : {'conversion' : '', str(test_attr) : str(limits_attr), 'description' : str(description)}}
    
    return json_data

# parse tlm config files in original txt format
def parseTlmConfTxt(file_path):   
    f = open(file_path, 'r')
    data_str = f.read()
    f.close()

    labels = []
    subsys_assignments = []
    mnemonic_tests = []
    descriptions = []
    
    data_pts = [i for i in data_str.split('\n') if i]
    
    for field_info in data_pts:
        data = field_info.split(' : ')
        if len(data) == 1:
            desc = 'No description'
        else:
            desc = data[1]
    
        descriptors = data[0].split(' ')
        
        labels.append(descriptors[0])
        subsys_assignments.append(str2lst(descriptors[1]))
        descriptions.append(desc)
    
        test_list = []
        for test in descriptors[2:]:
            test_list.append(str2lst(test))
            
        mnemonic_tests.append(test_list)
    
    return [labels, subsys_assignments, mnemonic_tests, descriptions]

# helper function to get path to config file
def getConfigPath(file_name):
    parent_dir = __file__
    for i in range(2):
        parent_dir = os.path.dirname(parent_dir)
    data_dir = os.path.join(parent_dir, 'data')
    configs_dir = os.path.join(data_dir, 'telemetry_configs')
    return os.path.join(configs_dir, file_name)

# helper function to merge two dicts together
def mergeDicts(dict1, dict2):
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return -1

    for key in dict2:
        if key in dict1:
            mergeDicts(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]

def writeToJson(path, data):
    file = open(path, 'wb')

    file.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    file.close()

def str2lst(string):
    try:
        return ast.literal_eval(string)
    except:
        print("Unable to process string representation of list")

def main():
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('text_config', nargs='?', help='Config file to be converted')
    arg_parser.add_argument('json_config', nargs='?', help='Config file to be written to')
    args = arg_parser.parse_args()
    
    try:
        convertTlmToJson(args.text_config, args.json_config)
    except:
        print ('failed to convert file to json')

def init():
    if __name__ == '__main__':
        main()

init()