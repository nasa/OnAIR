"""
Data driven learning class for managing all data driven AI components
"""

import copy
import os
import numpy as np

from src.util.print_io import *
from src.util.data_reformatting import *

from src.data_driven_components.associativity.associativity import Associativity
from src.data_driven_components.vae.vae import VAE

class DataDrivenLearning:
    def __init__(self, headers=[], window_size=10):
        self.classes = {'RED' : 0,
                     'YELLOW' : 1,
                      'GREEN' : 2,
                        '---' : 3}
        self.inverted_classes = {0 : 'RED',
                                 1 : 'YELLOW',
                                 2 : 'GREEN',
                                 3 : '---'}

        assert(len(headers)>0)

        self.headers = headers
        self.window_size = window_size
        self.associations = Associativity(headers, self.window_size, True)
        self.vae = VAE(headers, self.window_size)

    def apriori_training(self, data):
        batch_data = prep_apriori_training_data(data, self.window_size)
        self.associations.apriori_training(batch_data)
        self.vae.apriori_training(batch_data)

    def update(self, curr_data, status):
        input_data = floatify_input(curr_data, self.window_size)
        output_data = self.status_to_oneHot(status)

        self.associations.update(input_data)
        self.vae.update(input_data)
        
        return input_data, output_data 

    def diagnose(self):
        diagnosis = {}
        diagnosis['associativity_diagnosis'] = self.associations.render_diagnosis()
        diagnosis['vae_diagnosis'] = self.vae.render_diagnosis()
        return diagnosis     

    ###### HELPER FUNCTIONS
    def status_to_oneHot(self, status):
        if isinstance(status, np.ndarray):
            return status
        one_hot = [0.0, 0.0, 0.0, 0.0]
        one_hot[self.classes[status]] = 1.0
        return list(one_hot)

