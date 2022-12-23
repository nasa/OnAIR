"""
Data driven learning class for managing all data driven AI components
"""
import numpy as np

class DataDrivenLearning:
    def __init__(self, headers=[], sample_input=[]):
        self.classes = {'RED' : 0,
                     'YELLOW' : 1,
                      'GREEN' : 2,
                        '---' : 3}
        self.inverted_classes = {0 : 'RED',
                                 1 : 'YELLOW',
                                 2 : 'GREEN',
                                 3 : '---'}
        try:
            self.init_learning_systems(headers, sample_input)
        except:
            self.headers = []

    def init_learning_systems(self, headers, sample=[]):
        assert(len(headers)>0)
        self.headers = headers
        if sample == []:
            sample_input = [0.0]*len(headers)
        else:
            sample_input = self.floatify_input(sample)
        sample_output = self.status_to_oneHot('---')
        return sample_input, sample_output

    def update(self, curr_data, status):
        input_data = self.floatify_input(curr_data)
        output_data = self.status_to_oneHot(status)
        return input_data, output_data 


    def apriori_training(self, batch_data):
        return 

    ###### HELPER FUNCTIONS
    def floatify_input(self, _input, remove_str=False):
        floatified = []
        for i in _input:
            if type(i) is str:
                try:
                    x = float(i)
                    floatified.append(x)
                except:
                    try:
                        x = i.replace('-', '').replace(':', '').replace('.', '')
                        floatified.append(float(x))
                    except:
                        if remove_str == False:
                            floatified.append(0.0)
            else:
                floatified.append(float(i))
        return floatified

    def status_to_oneHot(self, status):
        if isinstance(status, np.ndarray):
            return status
        one_hot = [0.0, 0.0, 0.0, 0.0]
        one_hot[self.classes[status]] = 1.0
        return list(one_hot)

