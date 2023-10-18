# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import numpy as np
from onair.src.ai_components.ai_plugin_abstract.core import AIPlugIn

class Plugin(AIPlugIn):
    def apriori_training(self,batch_data=[]):
        """
        Given data, system should learn any priors necessary for realtime diagnosis.
        """
        pass
            
    def update(self,low_level_data=[], high_level_data={}):
        """
        Given streamed data point, system should update internally
        """
        pass

    def render_reasoning(self):
        """
        System should return its diagnosis
        """
        pass