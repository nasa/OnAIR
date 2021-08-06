'''
Class for causality visualization
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

class CausalityViz:
    def __init__(self, causal_graph = None):
        self.causal_graph = causal_graph

    def plot_graph(self, data):
        dataframe = self.convert_to_df(self.data)
        tp.plot_timeseries(dataframe)
        plt.show()

    def set_causal_graph(self, causal_graph):
        self.causal_graph = causal_graph

    def plot_time_series_graph(self, results, link_matrix, var_names):
        # Plot time series graph
        tp.plot_time_series_graph(
        val_matrix=results['val_matrix'],
        link_matrix=link_matrix,
        var_names=var_names,
        link_colorbar_label='MCI',
        )
        plt.show()
      

 
