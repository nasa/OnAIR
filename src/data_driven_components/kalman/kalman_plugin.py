from src.data_driven_components.generic_component.core import AIPlugIn

class Plugin(AIPlugIn):
    def __init__(self, _name, _headers):
        super().__init__(_name, _headers)

    def apriori_training(self, batch_data=[]):
        pass

    def update(self, frame=[]):
        pass

    def render_diagnosis(self):
        return dict()