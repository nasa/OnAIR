'''
Computes a causal graph using real-time mnemonic data and returns 
the significant causal links from a specified node (i.e. the faulty mnemonic)
'''

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction

from src.data_driven_components.causality.causality_utils import CausalityUtils
from src.data_driven_components.causality.causality_viz import CausalityViz

class CausalGraph:
    def __init__(self, headers=[], window_size=10):
        self.data = []
        self.headers = headers
        self.causality_viz = CausalityViz()
        self.causality_utils = CausalityUtils()
        self.viz = False
        self.verbose = 0
        self.filtered_columns = None
        self.window_size = window_size
        self.datum_index = 0
        self.diagnosis = None
        self.binary_threshold = 0.9 ##tune this parameter to increase acc if using binary filtering
        self.binarize = False
        self.faulty_mnemonic = None
        self.formatted_output_matrix = None
        self.aggregate_formatted_output_matrix = []

    def update(self, frame):
        self.data.append(frame)
        self.datum_index = self.datum_index + 1
        if self.datum_index % self.window_size == 0: ##increase window size for speed increase
            self.run_causal_model()

    def render_diagnosis(self, faulting_mnemonic):
        if faulting_mnemonic == None:
            return None
        else:
            if 'Time' in self.formatted_output_matrix:
                self.formatted_output_matrix = self.formatted_output_matrix.drop(['Time'])
            if 'TIME' in self.formatted_output_matrix:
                self.formatted_output_matrix = self.formatted_output_matrix.drop(['TIME'])
            for mnemonic in faulting_mnemonic:
                if mnemonic in self.formatted_output_matrix: ##if this is false then the self.faulty_mnemonic column is constant therefore not in the dataframe (see convert to df function below)
                    self.aggregate_formatted_output_matrix.append(self.formatted_output_matrix.sort_values(mnemonic)[mnemonic])
            return self.aggregate_formatted_output_matrix

    def set_faulty_nuemonic(self, faulty):
        self.faulty_mnemonic = faulty

    def compute_pcmci_algorithm(self, linear, data):
        if linear == True:
            parcorr = ParCorr()
            return PCMCI(dataframe=data, cond_ind_test=parcorr, verbosity=self.verbose)
        else: ##if there are nonlinear relationships present in the data parcorr above with provide misleading results. Run GPDC below to mitigate this if data is highly nonlinear
            gpdc = GPDC(significance='analytic', gp_params=None)
            return PCMCI(dataframe=dataframe, cond_ind_test=gpdc, verbosity=self.verbose)
            
    def compute_causal_graph(self, pcmci):        
        results = pcmci.run_pcmci(tau_max=0, pc_alpha=0.2)
        return results

    def compute_significant_links(self, print_links, pcmci_parcorr, results):
        link_matrix = pcmci_parcorr.return_significant_links(pq_matrix=results['p_matrix'], val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']

        if print_links == True:
                pcmci_parcorr.print_significant_links(
                    p_matrix = results['p_matrix'], 
                    val_matrix = results['val_matrix'],
                    alpha_level = 0.01)    
    
        return link_matrix

    def run_causal_model(self):
        data = self.causality_utils.convert_to_df(self.data, self.headers)           
        pcmci = self.compute_pcmci_algorithm(True, data)
        results = self.compute_causal_graph(pcmci)        
        significant_links = self.compute_significant_links(False, pcmci, results)
        self.filtered_columns = self.causality_utils.get_filtered_columns()
        output_binary_matrix = self.causality_utils.convert_to_matrix(significant_links, self.filtered_columns)
        output_val_matrix = self.causality_utils.convert_to_matrix(results['val_matrix'], self.filtered_columns) ##this uses parcorr... can also use p-vals or other techniques if accuracy needs to be improved
        if self.binarize == True:
            output_val_matrix = self.causality_utils.binarize_output(output_val_matrix, output_val_matrix)
            self.formatted_output_matrix = output_val_matrix
        else:
            self.formatted_output_matrix = output_val_matrix

        if self.verbose == True:
            self.render_diagnosis()

