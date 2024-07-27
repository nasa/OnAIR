# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
file_io.py
Utility file for parsing file data
"""

import json


def parse_associations_from_json(filepath):
    with open(filepath) as f:
        data = json.load(f)

    associations_list = []
    raw_associations = data["children"]
    for association in raw_associations:
        antecedant = association["name"]
        for connection in association["connections"]:
            consequent = connection["target"]
            weight = connection["weight"]
            relationship = (antecedant, consequent)
            weighted_relationship = (relationship, weight)
            associations_list.append(weighted_relationship)

    associations_list.sort(key=lambda x: x[1], reverse=True)

    for elem in associations_list:
        ant = elem[0][0]
        cons = elem[0][1]
        wei = elem[1]
        print(str(ant) + " --> " + str(cons) + ", " + str(wei))


def aggregate_results():
    return
