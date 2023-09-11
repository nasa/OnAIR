# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import numpy as np
from onair.src.data_driven_components.generic_component.core import AIPlugIn

class Plugin(AIPlugIn):
    def __init__(self, _name, _headers):
        """
        Superclass for data driven components: VAE, PPO, etc. Allows for easier modularity.
        """
        assert(len(_headers)>0)
        self.component_name = _name
        self.headers = _headers

    def apriori_training(self, batch_data=[]):
        """
        Given data, system should learn any priors necessary for realtime diagnosis.
        """
        pass
            
    def update(self, frame=[]):
        """
        Given streamed data point, system should update internally
        """
        pass

    def render_reasoning(self):
        """
        System should return its diagnosis
        """
        pass