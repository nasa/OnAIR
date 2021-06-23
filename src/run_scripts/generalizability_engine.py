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

from src.data_driven_components.associativity import Associativity
from src.data_driven_components.vae import VAE
from src.data_driven_components.pomdp import POMDP

# -----------------------------------------------------
# data has a folder for each 
#  -- My generated Data 
#  -- Nicks data (CSV's of set size)
#  -- the actual sounding rocket TLM 
#  -- KSP 

# Two modes: either generate data, or do not. 
# Maybe try running on each of the static sets, 
# then do a few iterations of each of the data generated 
# -----------------------------------------------------

# TODO WRITE A TEST FOR THIS!! 
class GeneralizabilityEngine:
    def __init__(self, run_path='', construct_name=None, construct_inits=[], 
                       sample_paths=['2020_handmade_data/',
                                     'data_physics_generation/Errors/',
                                     'data_physics_generation/No_Errors/']):
        try: 
            self.construct_files = {'Associativity' : 'associativity',
                                    'VAE' : 'vae',
                                    'POMDP' : 'pomdp'}
            self.sample_paths = sample_paths

            self.data_samples = self.init_samples(run_path + '/data/raw_telemetry_data/' )
            self.construct = self.init_construct(construct_name)
        except:
            self.construct_files = {'Associativity' : 'associativity',
                                    'VAE' : 'vae',
                                    'POMDP' : 'pomdp'}
            self.sample_paths = []

            self.data_samples = []
            self.construct = None

    def init_construct(self, construct_name, construct_inits=[]):
        _construct = importlib.import_module('src.data_driven_components.' + self.construct_files[construct_name])
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

    def extract_dimensional_info(self, construct_name, sample=None):
        sample = self.data_samples[0] if sample == None else sample
        input_dim = len(sample.get_headers())         # VAE
        seq_len = sample.get_num_frames()           # VAE

        headers = sample.get_headers()              # ASSOC
        sample_input = sample.get_sample()          # ASSOC

        name = sample.get_name()                    # POMDP
        path = sample.get_path()                    # POMDP
        telemetry_headers = sample.get_headers()    # POMDP

        args = {'VAE' : [input_dim, seq_len],
                'Associativity' : [headers, sample_input],
                'POMDP' : [name, path, telemetry_headers]}
        return args[construct_name]


    # def run_generalizability_tests(self):
    #     for sample in self.data_samples:
            
# VAE: input_dim=30, seq_len=15, z_units=5, hidden_units=100
# ASSOC: headers=[], sample_input=[]
# POMDP: name, path, telemetry_headers, 
#        print_on=False, save_me=True, reportable_states=['no_error', 'error'], 
#        alpha=0.01, discount=0.8, epsilon=0.2, run_limit=100, reward_correct=100, 
#        reward_incorrect=-100, reward_action=-1


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
        self.input_data = _input_data_frames
        self.output_data = _output_data_frames

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

# -----------------------------------------------------

"""Can abstract this out of this file"""
def parse_data(dataFile):
    with open(dataFile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        all_data = []
        for row in reader:
            all_data.append(row)
    return all_data[0], all_data[1:]
# -----------------------------------------------------------


