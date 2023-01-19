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
            importlib.import_module('src.data_driven_components.' + plugin + '.core').AIPlugIn(plugin, headers) for plugin in _ai_plugins
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



