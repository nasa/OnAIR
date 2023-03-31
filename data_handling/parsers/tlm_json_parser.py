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
