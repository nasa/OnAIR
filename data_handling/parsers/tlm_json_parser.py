# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import ast
import orjson

# parse tlm config json file
def parseTlmConfJson(file_path):
    data = parseJson(file_path)
    reorg_data = reorganizeTlmDict(data)

    labels = []
    subsys_assignments = []
    mnemonic_tests = []
    descriptions = []
    
    for label in reorg_data:
        curr_datapt = reorg_data[label]
        subsys = curr_datapt['subsystem']

        tests = curr_datapt['tests'] if 'tests' in curr_datapt else {}
        if tests == {}:
            mnemonics = [['NOOP']]
        else:
            mnemonics = []
            for key in tests:
                mnemonics.append([key, curr_datapt['tests'][key]])
        desc = curr_datapt['description'] if 'description' in curr_datapt else ['No description']
        
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
        for label in data['subsystems'][s]:
            processed_data[label] = data['subsystems'][s][label]
            processed_data[label]['subsystem'] = s
    
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
