'''
Class for causality utils
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction

class CausalityUtils:
    def __init__(self):
        self.filtered_columns = None

    def convert_to_df(self, data, headers):
        dataframe = pd.DataFrame (data, columns=headers)
        dataframe = dataframe.loc[:, dataframe.apply(pd.Series.nunique) != 1] #drops constant columns (requirement for the calculations)
        self.filtered_columns = dataframe.columns
        dataframe = pp.DataFrame(dataframe.values, var_names=dataframe.columns)
        return dataframe

    def convert_to_matrix(self, data, filtered_columns):
        new_arr = []
        for datum in data:
            new_arr.append([float(x) for [x] in datum])
        df = pd.DataFrame(new_arr, filtered_columns).transpose()
        df.index = filtered_columns
        return df

    def get_filtered_columns(self):
        return self.filtered_columns

    def binarize_output(self, output_val_matrix, binary_threshold):
        for col in output_val_matrix:
            for idx in output_val_matrix.index:
                if (output_val_matrix[col][idx] > binary_threshold).any().any():
                    output_val_matrix[col][idx] = 0
                else:
                    output_val_matrix[col][idx] = 1
        return output_val_matrix





