"""
Data driven learning class for managing all data driven AI components
"""

from src.util.data_conversion import *

class DataDrivenLearning:
    def __init__(self, headers=[], AI_constructs=[]):
        assert(len(headers)>0)
        self.headers = headers
        #  INIT the construct here: self.intellgence = []

    def update(self, curr_data, status):
        input_data = floatify_input(curr_data)
        output_data = status_to_oneHot(status)
        return input_data, output_data 

    def apriori_training(self, batch_data):
        return 


