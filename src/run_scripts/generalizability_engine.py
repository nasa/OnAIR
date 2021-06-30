"""
Generalizability Engine, which tests how generalizable components of the system are 
A priori
"""

import importlib
import os
import sys 
import shutil
import pandas as pd
import csv 

from src.data_driven_components.associativity.associativity import Associativity
from src.data_driven_components.curve_characterizer.curve_characterizer import CurveCharacterizer
from src.data_driven_components.vae.vae import VAE
from src.data_driven_components.pomdp.pomdp import POMDP

# -----------------------------------------------------
# data has a folder for each 
#  -- My generated Data 
#  -- Nicks data (CSV's of set size)
#  -- the actual sounding rocket TLM 
#  -- KSP 
#   
# Two modes: either generate data, or do not. 
# Maybe try running on each of the static sets, 
# then do a few iterations of each of the data generated 
# -----------------------------------------------------

# TODO WRITE A TEST FOR THIS!! 
class GeneralizabilityEngine:
    def __init__(self, run_path='', construct_name='Associativity', construct_inits=[], 
                       sample_paths=['2020_handmade_data/',
                                     'data_physics_generation/Errors/',
                                     'data_physics_generation/No_Errors/']): 
        self.run_path = run_path
        self.construct_files = {'Associativity' : 'associativity',
                                'VAE' : 'vae',
                                'POMDP' : 'pomdp',
                                'CurveCharacterizer' : 'curve_characterizer'}
        self.sample_paths = sample_paths
        self.data_samples = self.init_samples(run_path + '/data/raw_telemetry_data/' )
        self.construct = self.init_construct(construct_name)

    def init_construct(self, construct_name, construct_inits=[]):
        _construct = importlib.import_module('src.data_driven_components.' + self.construct_files[construct_name] + '.' + self.construct_files[construct_name])
        construct_class = getattr(_construct, construct_name)
        construct_inits = self.extract_dimensional_info(construct_name) if construct_inits == [] else construct_inits
        return construct_class(*construct_inits)

    def init_samples(self, data_path):
        data_sets = []
        for sample in self.sample_paths:
            files = [f for f in os.listdir(data_path + sample) if os.path.isfile(os.path.join(data_path + sample, f))]
            for file in files:
                path = data_path + sample + file
                data_sets.append(DataWrapper(path, *parse_data(path)))
        return data_sets

    def extract_dimensional_info(self, construct_name):
        return [self.data_samples[0].get_headers(), 10]

        # sample = self.data_samples[0]

        # input_dim = len(sample.get_headers())       # VAE
        # seq_len = sample.get_num_frames()           # VAE
        # window_size = 10 # Window size

        # headers = sample.get_headers()              # ASSOC
        # sample_input = sample.get_sample()          # ASSOC

        # name = sample.get_name()                    # POMDP
        # path = sample.get_path()                    # POMDP
        # telemetry_headers = sample.get_headers()    # POMDP

        # args = {'VAE' : [headers, window_size], 
        #         'Associativity' : [headers, window_size+10],
        #         'POMDP' : [name, path, telemetry_headers],
        #         'CurveCharacterizer' : [self.run_path  + 'data/']}

        # return args[construct_name]

    def run_integration_test(self):
        data = self.data_samples[0].get_data()

        frames = data[:10]

        self.construct.apriori_training(frames)
        self.construct.update(frames[0])
        
        # for frame in frame:
        #     self.construct.update(frame)

        # cc = CurveCharacterizer(self.run_path + 'data/')
        # vae = VAE()

    # def run_generalizability_tests(self):
    #     for sample in self.data_samples:

# -----------------------------------------------------------
# ---------------- PULL THIS STUFF OUT ----------------------
"""
DataWrapper Class
"""
class DataWrapper:
    def __init__(self, _path,  _headers=[], _input_data_frames=[], _output_data_frames=[]):
        if len(_output_data_frames)>0:
            assert(len(_input_data_frames) == len(_output_data_frames))
        self.path = _path
        self.name = _path.split('/')[-1]
        self.headers = _headers
        self.input_data = [floatify_input(elem) for elem in _input_data_frames]
        self.output_data = [floatify_input(elem) for elem in _output_data_frames]

    def get_name(self):
        return self.name

    def get_path(self):
        return self.path

    def get_headers(self):
        return self.headers

    def get_num_frames(self):
        return len(self.input_data)

    def get_sample(self):
        return self.input_data[0]

    def get_data(self, labels=False):
        if labels == True:
            return self.input_data, self.output_data
        return self.input_data

# -----------------------------------------------------

"""Can abstract this out of this file"""
def parse_data(dataFile):
    with open(dataFile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        all_data = []
        for row in reader:
            all_data.append(row)
    return all_data[0], all_data[1:]

# This needs to be able to handle sci notation e+..
def floatify_input(_input, remove_str=False):
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
# -----------------------------------------------------------


