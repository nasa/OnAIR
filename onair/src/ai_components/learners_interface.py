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
import importlib.util

from ..util.data_conversion import *

class LearnersInterface:
    def __init__(self, headers, _ai_plugins={}):
        assert(len(headers)>0), 'Headers are required'
        self.headers = headers
        self.ai_constructs = []
        for module_name in list(_ai_plugins.keys()):
            spec = importlib.util.spec_from_file_location(module_name, _ai_plugins[module_name])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.ai_constructs.append(module.Plugin(module_name,headers))
    
    def apriori_training(self, batch_data):
        for plugin in self.ai_constructs:
            plugin.apriori_training(batch_data)

    def update(self, low_level_data, high_level_data):
        for plugin in self.ai_constructs:
            plugin.update(low_level_data)

    def check_for_salient_event(self):
        pass

    def render_reasoning(self):
        diagnoses = {}
        for plugin in self.ai_constructs:
            diagnoses[plugin.component_name] = plugin.render_reasoning()
        return diagnoses
