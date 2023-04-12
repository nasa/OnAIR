import ast
import orjson
import os

# parse tlm config json file
def parseTlmConfJson(file_path):
    data = parseJson(file_path)
    reorg_data = reorganizeTlmDict(data)

    labels = []
    subsys_assignments = []
    mnemonic_tests = []
    descriptions = []
    
    for label in reorg_data:
        subsys = reorg_data[label]['subsystem']
        temp = str2lst(reorg_data[label]['limits'])
        if temp == []:
            mnemonics = [reorg_data[label]['tests']]
        else:
            mnemonics = [reorg_data[label]['tests'], reorg_data[label]['limits']]
        desc = reorg_data[label]['description']
        
        labels.append(label)
        subsys_assignments.append(subsys)
        mnemonic_tests.append(mnemonics)
        descriptions.append(desc)

    # if given an order, reorder data to match
    if 'order' in data and data['order'] != []:
        original_order = {}
        for i in range(len(data['order'])):
            original_order[data['order'][i]] = i

        ordering_list = []
        for label in labels:
            ordering_list.append(original_order[label])

        labels = [y for x, y in sorted(zip(ordering_list, labels))]
        subsys_assignments = [y for x, y in sorted(zip(ordering_list, subsys_assignments))]
        mnemonic_tests = [y for x, y in sorted(zip(ordering_list, mnemonic_tests))]
        descriptions = [y for x, y in sorted(zip(ordering_list, descriptions))]

    configs = {}
    configs['subsystem_assignments'] = subsys_assignments
    configs['test_assignments'] = mnemonic_tests
    configs['description_assignments'] = descriptions
    
    return configs

# process tlm dict into dict of labels and their attributes
def reorganizeTlmDict(data):
    processed_data = {}
    
    for s in data['subsystems']:
        for app in data['subsystems'][s]:
            app_data = reorganizeTlmDictRecursiveStep(app, s, data['subsystems'][s][app])
            processed_data.update(app_data)
    
    return processed_data

# recursive helper function for reorganizeTlmDict
def reorganizeTlmDictRecursiveStep(label, subsys, data):    
    if 'description' in data:
        ret_data = {label: {}}
        ret_data[label]['subsystem'] = subsys
        ret_data[label]['tests'] = data['test']
        ret_data[label]['limits'] = data['limits']
        ret_data[label]['description'] = data['description']
        return ret_data
    
    processed_data = {}
    for elem in data:
        ret = reorganizeTlmDictRecursiveStep(f'{label}.{elem}', subsys, data[elem])
        processed_data.update(ret)

    return processed_data

def str2lst(string):
    try:
        return ast.literal_eval(string)
    except:
        print("Unable to process string representation of list")
        # return string

def parseJson(path):
    file = open(path, 'rb')
    file_str = file.read()

    data = orjson.loads(file_str)
    file.close()
    return data

def writeToJson(path, data):
    file = open(path, 'wb')

    file.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    file.close()

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
    if label == 'TIME':
        return {label : {'conversion' : '', 'test' : str(mnemonics[0]), 'limits' : '[]', 'description' : str(description)}}
    
    attributes = label.split('.')
    num_attr = len(attributes)
    test_attr = mnemonics[0][0]
    limits_attr = mnemonics[0][1:]
    json_data = {attributes[num_attr-1] : {'conversion' : '', 'test' : str(test_attr), 'limits' : str(limits_attr), 'description' : str(description)}}
    
    for attr in reversed(attributes[:num_attr-1]):
        json_data = {attr : json_data}
    
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
            desc = 'No Description'
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
    for i in range(3):
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