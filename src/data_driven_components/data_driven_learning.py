# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
Data driven learning class for managing all data driven AI components
"""
import importlib

from src.util.data_conversion import *

class DataDrivenLearning:
    def __init__(self, headers, _ai_plugins:list=[]):
        assert(len(headers)>0)
        self.headers = headers
        self.ai_constructs = [
            importlib.import_module('src.data_driven_components.' + plugin_name + '.' + f'{plugin_name}_plugin').Plugin(plugin_name, headers) for plugin_name in _ai_plugins
        ]

    def update(self, curr_data, status):
        input_data = floatify_input(curr_data)
        output_data = status_to_oneHot(status)
        for plugin in self.ai_constructs:
            plugin.update(input_data)

    def apriori_training(self, batch_data):
        for plugin in self.ai_constructs:
            plugin.apriori_training(batch_data)

    def render_diagnosis(self):
        diagnoses = {}
        for plugin in self.ai_constructs:
            diagnoses[plugin.component_name] = plugin.render_diagnosis()
        return diagnoses



