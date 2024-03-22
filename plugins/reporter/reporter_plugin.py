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
    verbose_mode = False

    def update(self, low_level_data=[], high_level_data={}):
        """
        Reporter outputs that it is updating and outputs known headers and
        given low and high level data.
        """
        self.low_level_data = low_level_data
        self.high_level_data = high_level_data
        print(f"{self.component_name}: UPDATE")
        if self.verbose_mode:
            print(f" : headers {self.headers}")
            print(f" : low_level_data {low_level_data.__class__} = '{low_level_data}'")
            print(f" : high_level_data {high_level_data.__class__} = '{high_level_data}'")

    def render_reasoning(self):
        """
        Reporter outputs that it is reasoning and gives its known low and
        high level data.
        """
        print(f"{self.component_name}: RENDER_REASONING")
        if self.verbose_mode:
            print(f" : My low_level_data is {self.low_level_data}")
            print(f" : My high_level_data is {self.high_level_data}")
