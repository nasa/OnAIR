
from src.data_driven_components.generic_intelligence_construct import GenericIntelligenceConstruct

"""This object serves as a proxy for all plug-ins.
   Therefore, the AIPlugIn object is meant to induce 
   standards and structures of compliance for user-created 
   and/or imported plug-ins/libraries
"""
class AIPlugIn(GenericIntelligenceConstruct):

    def __init__(self, _name, _headers):
        assert(len(_headers)>0)
        self.component_name = _name
        self.headers = _headers

    def apriori_training(self, batch_data): 
        # The batch data format could change 
        # depending on how the tutorial fleshes out (too early to tell) 
        # There will be no return from this function (user can pull training)
        # data from the construct itself) 
        # I dont know yet whether we should allow empty batch data
        pass

    def update(self, frame=[]):
        # I dont know yet whether we should allow empty frames from updates 
        assert(len(frame) == len(self.headers))

    def render_diagnosis(self):
        return []
