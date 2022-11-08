
from src.data_driven_components.generic_intelligence_construct import GenericIntelligenceConstruct

class AIPlugIn(GenericIntelligenceConstruct):

    def __init__(self, _headers):
        assert(len(_headers)>0)
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
