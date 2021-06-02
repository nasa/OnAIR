"""
Sim class
Helper class to create and run a simulation
"""

import csv
import time
import os
import sys
import importlib
import copy
import webbrowser
import json
import random

from src.reasoning.brain import Brain
from src.subsystems.spacecraft import Spacecraft
from src.util.file_io import *
from src.util.print_io import *
from src.util.sim_io import *
from src.data_handling.data_source import DataSource

class Simulator:
    def __init__(self, simType, parsedData, SBN_Flag):

        self.simulator = simType
        spaceCraft = Spacecraft(parsedData.get_sc_configs())

        if SBN_Flag:
            # TODO: This is ugly, but sbn_client is only available when built for cFS...
            # ...from sbn_adapter import AdapterDataSource
            sbn_adapter = importlib.import_module('src.run_scripts.sbn_adapter')
            AdapterDataSource = getattr(sbn_adapter, 'AdapterDataSource')

            self.simData = AdapterDataSource(parsedData.get_binned_data())
            self.simData.connect() # this also subscribes to the msgIDs
            
        else:
            self.simData = DataSource(parsedData.get_binned_data())
        self.brain = Brain(spaceCraft)

    def run_sim(self, IO_Flag=False, dev_flag=False, viz_flag = True):
        print_sim_header() if (IO_Flag == True) else ''
        print_msg('Please wait...\n') if (IO_Flag == 'strict') else ''
        diagnosis_list = []
        time_step = 0
        last_diagnosis = time_step
        last_fault = time_step
        if dev_flag == True:
            self.generate_supervised_learning_data()
            self.train_supervised_learning()

        while self.simData.has_more() and time_step < 2050:
            next = self.simData.get_next()
            self.brain.reason(next)
            self.IO_check(time_step, IO_Flag)
            
            ### Stop when a fault is reached  
            if self.brain.interpreted_status == 'RED':
                if last_fault == time_step - 1: #if they are consecutive
                    if (time_step - last_diagnosis) % 100 == 0:
                        diagnosis_list.append(self.brain.diagnose(time_step))
                        last_diagnosis = time_step
                else:
                    diagnosis_list.append(self.brain.diagnose(time_step))
                    last_diagnosis = time_step
                last_fault = time_step
            time_step += 1
            
        # Final diagnosis processing
        if len(diagnosis_list) == 0:
            diagnosis_list.append(self.brain.diagnose(time_step))
        final_diagnosis = diagnosis_list[-1]

        # Renderings
        render_viz(self.brain.get_hierarchical_status_report(), 
                   self.brain.supervised_learning.associations.render_associations(),
                   self.simulator,
                   final_diagnosis)
        render_diagnosis(diagnosis_list)
        print_diagnosis(final_diagnosis) if (IO_Flag == True) else ''
        if viz_flag:
            print('To start viz server, use \033[36mpython3 src/viz/server.py', os.path.join(os.environ.get('RAISR_SAVE_PATH'), 'tmp', 'viz'), '\033[0m')
            print_msg("Press (v) followed by \'enter\' to see visualization, otherwise hit \'enter\' to quit")
            show_viz = input('\033[95m >>> \033[0m')
            if show_viz == 'v' or show_viz == 'V':
                webbrowser.open('http://localhost:5000')

        return final_diagnosis

    def generate_supervised_learning_data(self):
        historicalData = copy.deepcopy(self.simData)
        self.brain.set_past_history(historicalData)

    def train_supervised_learning(self):
        self.brain.supervised_learning.train_all()

    def set_benchmark_data(self, filepath, files, indices):
        self.brain.supervised_learning.set_benchmark_data(filepath, files, indices)

    def IO_check(self, time_step, IO_Flag):
        if IO_Flag == True:
            print_sim_step(time_step + 1)
            print(self.brain.spacecraft_rep.get_status_object())
            print_interpreted_status(self.brain)
        else:
            # print_dots(time_step)
            pass


