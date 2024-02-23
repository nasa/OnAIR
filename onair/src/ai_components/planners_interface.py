# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
Planners interface class for managing all planning-based AI components
"""
from ..util.plugin_import import import_plugins
from ..util.data_conversion import *

class PlannersInterface:
    def __init__(self, headers, _planner_plugins={}):
        assert(len(headers)>0), 'Headers are required'
        self.headers = headers
        self.planner_constructs = import_plugins(self.headers,_planner_plugins)

    def update(self, high_level_data):
        # Raw TLM should be transformed into high-leve state representation here
        # Can store something as stale unless a planning thread is launched
        for plugin in self.planner_constructs:
            plugin.update(high_level_data=high_level_data)

    def check_for_salient_event(self):
        pass

    def render_reasoning(self):
        diagnoses = {}
        for plugin in self.planner_constructs:
            diagnoses[plugin.component_name] = plugin.render_reasoning()
        return diagnoses
