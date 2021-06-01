"""
Execution Engine, which sets configs and sets up the simulation
"""

import os
import sys
import configparser
import importlib
import ast
import shutil
from distutils.dir_util import copy_tree
from time import gmtime, strftime   

# from src.run_scripts.sim import Simulator
from src.data_handling.binner import Binner
# from src.util.file_io import * 

class ExecutionEngine:
    def __init__(self, config_file='', run_name='', save_flag=False):
        
        # Init Housekeeping 
        self.run_name = run_name

        # Init Flags 
        self.IO_Flag = False
        self.Dev_Flag = False
        self.SBN_Flag = False
        self.Viz_Flag = False
        
        # Init Paths 
        self.dataFilePath = ''
        self.metadataFilePath = ''
        self.benchmarkFilePath = ''
        self.metaFiles = ''
        self.telemetryFiles = ''
        self.benchmarkFiles = ''
        self.benchmarkIndices = ''

        # Init parsing/sim info
        self.parser_file_name = ''
        self.parser_name = ''
        self.sim_name = ''
        self.data = None
        self.binnedData = None
        self.sim = None

        self.save_flag = save_flag
        self.save_name = run_name

        if config_file != '':
            self.init_save_paths()
            self.parse_configs(config_file)
            self.parse_data(self.parser_name, self.parser_file_name, self.dataFilePath, self.metadataFilePath)
            self.setup_sim(self.data)

    def parse_configs(self, config_file_path):
        # print("Using config file: {}".format(config_file_path))
        config = configparser.ConfigParser()
        config.read(config_file_path)
        ## Sort Data: Telementry Data & Configuration
        self.dataFilePath = config['DEFAULT']['TelemetryDataFilePath']
        self.metadataFilePath = config['DEFAULT']['TelemetryMetadataFilePath']
        self.metaFiles = config['DEFAULT']['MetaFiles'] # Config for spacecraft telemetry
        self.telemetryFiles = config['DEFAULT']['TelemetryFiles'] # Spacecraft telemetry data
        try:
            self.benchmarkFilePath = config['DEFAULT']['BenchmarkFilePath']
            self.benchmarkFiles = config['DEFAULT']['BenchmarkFiles'] # Spacecraft telemetry data
            self.benchmarkIndices = config['DEFAULT']['BenchmarkIndices']
        except:
            pass
        ## Sort Data: Names
        self.parser_file_name = config['DEFAULT']['ParserFileName']
        self.parser_name = config['DEFAULT']['ParserName']
        self.sim_name = config['DEFAULT']['SimName']

        ## Sort Data: Flags
        self.IO_Flag = config['RUN_FLAGS'].getboolean('IO_Flag')
        self.Dev_Flag = config['RUN_FLAGS'].getboolean('Dev_Flag')
        self.SBN_Flag = config['RUN_FLAGS'].getboolean('SBN_Flag')
        self.Viz_Flag = config['RUN_FLAGS'].getboolean('Viz_Flag')

    def parse_data(self, parser_name, parser_file_name, dataFilePath, metadataFilePath, subsystems_breakdown=False):
        parser = importlib.import_module('src.data_handling.parsers.' + parser_file_name)
        parser_class = getattr(parser, parser_name) # This could be simplified if the parsers all extend a parser class... but this works for now
        tm_data_path = os.environ['RUN_PATH'] + dataFilePath
        tm_metadata_path = os.environ['RUN_PATH'] +  metadataFilePath
        self.data = parser_class(tm_data_path, tm_metadata_path, self.telemetryFiles, self.metaFiles, subsystems_breakdown)

    def setup_sim(self, data):
        self.binnedData = Binner(*data.get_sim_data())
        self.sim = Simulator(self.sim_name, self.binnedData, self.SBN_Flag)
        try:
            fls = ast.literal_eval(self.benchmarkFiles)
            fp = os.path.dirname(os.path.realpath(__file__)) + '/../..' + self.benchmarkFilePath
            bi = ast.literal_eval(self.benchmarkIndices)
            self.sim.set_benchmark_data(fp, fls, bi)
        except:
            pass
        # except:
        #     print("Unable to set benchmark matrices")

    def run_sim(self):
        self.sim.run_sim(self.IO_Flag, self.Dev_Flag, self.Viz_Flag)
        if self.save_flag:
            self.save_results(self.save_name)

    def init_save_paths(self):
        save_path = os.environ['RESULTS_PATH']
        temp_save_path = os.path.join(save_path, 'tmp')

        self.delete_save_paths()
        os.mkdir(temp_save_path)
        os.mkdir(os.path.join(temp_save_path, 'tensorboard'))
        os.mkdir(os.path.join(temp_save_path, 'associativity'))
        os.mkdir(os.path.join(temp_save_path, 'viz'))
        os.mkdir(os.path.join(temp_save_path, 'models'))
        os.mkdir(os.path.join(temp_save_path, 'graphs'))
        os.mkdir(os.path.join(temp_save_path, 'diagnosis'))
    
        os.environ['RAISR_SAVE_PATH'] = save_path
        os.environ['RAISR_TMP_SAVE_PATH'] = temp_save_path
        os.environ['RAISR_TENSORBOARD_SAVE_PATH'] = os.path.join(temp_save_path, 'tensorboard')
        os.environ['RAISR_ASSOCIATIVITY_SAVE_PATH'] = os.path.join(temp_save_path, 'associativity')
        os.environ['RAISR_VIZ_SAVE_PATH'] = os.path.join(temp_save_path, 'viz')
        os.environ['RAISR_MODELS_SAVE_PATH'] = os.path.join(temp_save_path, 'models')
        os.environ['RAISR_GRAPHS_SAVE_PATH'] = os.path.join(temp_save_path, 'graphs')
        os.environ['RAISR_DIAGNOSIS_SAVE_PATH'] = os.path.join(temp_save_path, 'diagnosis')

    def delete_save_paths(self):
        save_path = os.environ['RESULTS_PATH']
        sub_dirs = os.listdir(save_path)
        if 'tmp' in sub_dirs: 
            try:
                shutil.rmtree(save_path + '/tmp')
            except OSError as e:
                print("Error: %s : %s" % (save_path, e.strerror))

    def save_results(self, save_name):
        complete_time = strftime("%H-%M-%S", gmtime())
        save_path = os.environ['RAISR_SAVE_PATH'] + '/saved/' + save_name + '_' + complete_time
        os.mkdir(save_path)
        copy_tree(os.environ['RAISR_TMP_SAVE_PATH'], save_path)
        # move everything from temp into here

    #### SETTERS AND GETTERS 
    def set_run_param(self, name, val):
        setattr(self, name, val)





