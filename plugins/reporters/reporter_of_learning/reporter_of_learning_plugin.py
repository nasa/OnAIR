# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin

class Plugin(AIPlugin):
    def update(self,low_level_data=[], high_level_data={}):
        """
        Given streamed data point, system should update internally
        """
        print(f"  LEARNING: UPDATE")
        print(f"          :   low_level_data {low_level_data.__class__} = '{low_level_data}'")
        print(f"          :   high_level_data {high_level_data.__class__} = '{high_level_data}'")

    def render_reasoning(self):
        """
        System should return its diagnosis
        """
        print("  LEARNING: REASONING")
        print("          :   No args")