"""
Data driven learning class for managing all data driven AI components
"""

import copy
import os
import numpy as np

from src.util.print_io import *

from src.data_driven_components.associativity.associativity import Associativity

class DataDrivenLearning:
    def __init__(self, headers=[]):
        self.classes = {'RED' : 0,
                     'YELLOW' : 1,
                      'GREEN' : 2,
                        '---' : 3}
        self.inverted_classes = {0 : 'RED',
                                 1 : 'YELLOW',
                                 2 : 'GREEN',
                                 3 : '---'}
        # try:
        self.init_learning_systems(headers)
        # except:
        #     self.headers = []

    """ Initialize all learning systems with necessary dimensional info """
    def init_learning_systems(self, headers):
        assert(len(headers)>0)
        sample_input = [0.0]*len(headers) 
        sample_output = self.status_to_oneHot('---')
        self.headers = headers

        self.associations = Associativity(headers, 20, True)
        # self.vae = VAE(headers, 20)
        
        return sample_input, sample_output

    def update(self, curr_data, status):
        input_data = self.floatify_input(curr_data)
        output_data = self.status_to_oneHot(status)

        self.associations.update(input_data)
        # self.vae.update(input_data)
        
        return input_data, output_data 

    def apriori_training(self, batch_data):
        self.associations.apriori_training(batch_data)
        # self.vae.apriori_training(batch_data)

    def diagnose(self):
        return self.associations.render_diagnosis()
    # def set_historical_data(self, input_samples, output_samples):
    #     assert len(input_samples) == len(output_samples)
    #     for system in self.ssNames:
    #         input_hist = [sample[system]['data'] for sample in input_samples]
    #         output_hist = [sample[system] for sample in output_samples]
    #         processed_input = [self.floatify_input(elem) for elem in input_hist]
    #         processed_output = [self.status_to_oneHot(elem) for elem in output_hist]
    #         self.input_history[system] = processed_input
    #         self.output_history[system] = processed_output

    # def set_benchmark_data(self, filepath, files, indices):
    #     self.associations.set_benchmark_data(filepath, files, indices)

    # def train_all(self):
    #     x = copy.deepcopy(self.input_history['MISSION'])
    #     y = copy.deepcopy(self.output_history['MISSION'])
    #     self.LSTM.bulkTrainModel(x, y)

    #     # save files
    #     if os.environ.get('RAISR_GRAPHS_SAVE_PATH'):
    #         self.LSTM.saveGraphs(os.environ.get('RAISR_GRAPHS_SAVE_PATH'))
    #     if os.environ.get('RAISR_MODELS_SAVE_PATH'):
    #         self.LSTM.saveModel(os.environ.get('RAISR_MODELS_SAVE_PATH'))
            
    #     self.NNs['MISSION'].train(x, y)


    # ###### HELPER FUNCTIONS
    # def get_importance_sampling(self):
    #     importance_sampling = copy.deepcopy(self.LSTM.getImportanceSampling())
    #     return importance_sampling

    # def get_associativity_graph(self):
    #     return copy.deepcopy(self.associations.get_association_graph())

    # def get_benchmark_graph(self):
    #     return copy.deepcopy(self.associations.get_benchmark_graph())

    # def get_associativity_metrics(self):
    #     return copy.deepcopy(self.associations.compare_to_benchmark())        

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
                            continue
                        continue
            else:
                floatified.append(float(i))
        return floatified

    def status_to_oneHot(self, status):
        if isinstance(status, np.ndarray):
            return status
        one_hot = [0.0, 0.0, 0.0, 0.0]
        one_hot[self.classes[status]] = 1.0
        return list(one_hot)

