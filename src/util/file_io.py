"""
file_io.py
Utility file for parsing file data
"""

import json

def parse_associations_from_json(filepath):
    with open(filepath) as f:
      data = json.load(f)

    associations_list = []
    raw_associations = data['children']
    for association in raw_associations:
        antecedant = association['name']
        for connection in association['connections']:
            consequent = connection['target']
            weight = connection['weight']
            relationship = (antecedant, consequent)
            weighted_relationship = (relationship, weight)
            associations_list.append(weighted_relationship)
    
    associations_list.sort(key = lambda x: x[1], reverse=True)
    
    for elem in associations_list:
        ant = elem[0][0]
        cons = elem[0][1]
        wei = elem[1]
        print(str(ant) + ' --> ' + str(cons) + ', ' + str(wei))

def aggregate_results():
    return