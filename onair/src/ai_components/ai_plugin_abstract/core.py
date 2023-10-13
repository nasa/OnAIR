# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

from abc import ABC, abstractmethod
"""This object serves as a proxy for all plug-ins.
   Therefore, the AIPlugIn object is meant to induce 
   standards and structures of compliance for user-created 
   and/or imported plug-ins/libraries
"""
class AIPlugIn(ABC):
    def __init__(self, _name, _headers):
        """
        Superclass for data driven components: VAE, PPO, etc. Allows for easier modularity.
        """
        assert(len(_headers)>0)
        self.component_name = _name
        self.headers = _headers

    @abstractmethod
    def apriori_training(self, batch_data=[]):
        """
        Given data, system should learn any priors necessary for realtime diagnosis.
        """
        # I dont know yet whether we should allow empty frames from updates 
        # The batch data format could change 
        # depending on how the tutorial fleshes out (too early to tell) 
        # There will be no return from this function (user can pull training)
        # data from the construct itself) 
        # I dont know yet whether we should allow empty batch data
        raise NotImplementedError
        
    @abstractmethod
    def update(self, low_level_data=[], high_level_data={}):
        """
        Given streamed data point, system should update internally
        """
        raise NotImplementedError

    @abstractmethod
    def render_reasoning(self):
        """
        System should return its diagnosis
        """
        raise NotImplementedError

