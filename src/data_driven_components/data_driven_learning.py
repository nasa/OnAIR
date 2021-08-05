
"""
Data driven learning class for managing all data driven AI components
"""

import copy
import os
import numpy as np

from src.util.print_io import *
from src.util.data_reformatting import *
from src.util.config import get_config


from src.data_driven_components.causality.causal_graph import CausalGraph
from src.data_driven_components.vae.vae_model import VAEModel
from src.data_driven_components.pomdp.ppo_model import PPOModel
from src.data_driven_components.kalman.kalman_model import KalmanModel

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

        self.config = get_config()

        self.headers = headers
        self.window_size = window_size

        self.causal_graph = CausalGraph(headers=headers, window_size=self.window_size)
        self.vae = VAEModel(headers=headers, window_size=self.window_size)
        self.ppo = PPOModel(headers=headers, window_size=self.window_size)
        self.kalman = KalmanModel(headers=headers, window_size=self.window_size)


    def apriori_training(self, data):
        if not data == []:
            batch_data = prep_apriori_training_data(data, self.window_size)
        else:
            batch_data = []
        self.vae.apriori_training(batch_data)
        self.ppo.apriori_training(batch_data)

    def load_models(self):
        # self.associations.apriori_training(batch_data) ## TODO: load model
        self.vae.load_model(self.config['VAE']['Path'])
        self.ppo.load_model()

    def update(self, curr_data, status):
        """
        :param curr_data: (numpy array) 3d tensor (batch, window_size, input_dim)
        :status: ('RED' | 'YELLOW' | 'GREEN' | '---')
        """
        input_data = floatify_input(curr_data, self.window_size)
        output_data = self.status_to_oneHot(status)

        #self.associations.update(input_data)
        self.causal_graph.update(input_data)
        self.ppo.update(input_data)
        self.vae.update(input_data, self.classes[status])
        self.kalman.update(input_data)
        return input_data, output_data 

    def diagnose(self, faulting_mnemonics):
        diagnosis = {}
        diagnosis['vae_diagnosis'] =  self.vae.render_diagnosis()
        diagnosis['pomdp_diagnosis'] = self.ppo.render_diagnosis()
        diagnosis['kalman_diagnosis'] = self.kalman.render_diagnosis()
        diagnosis['causality_diagnosis'] = self.causal_graph.render_diagnosis(faulting_mnemonics)
        return diagnosis
        
    ####################################################################################
    
    ###### HELPER FUNCTIONS
    def status_to_oneHot(self, status):
        if isinstance(status, np.ndarray):
            return status
        one_hot = [0.0, 0.0, 0.0, 0.0]
        one_hot[self.classes[status]] = 1.0
        return list(one_hot)
