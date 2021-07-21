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
from src.systems.spacecraft import Spacecraft
from src.util.print_io import *
from src.util.sim_io import *
from src.data_handling.data_source import DataSource

from src.util.config import get_config

class Simulator:
    def __init__(self, simType, parsedData, SBN_Flag):

        self.simulator = simType
        spaceCraft = Spacecraft(*parsedData.get_spacecraft_metadata())

        if SBN_Flag:
            # TODO: This is ugly, but sbn_client is only available when built for cFS...
            # ...from sbn_adapter import AdapterDataSource
            sbn_adapter = importlib.import_module('src.run_scripts.sbn_adapter')
            AdapterDataSource = getattr(sbn_adapter, 'AdapterDataSource')
            self.simData = AdapterDataSource(parsedData.get_sim_data())
            self.simData.connect() # this also subscribes to the msgIDs
            
        else:
            self.simData = DataSource(parsedData.get_sim_data())
        self.brain = Brain(spaceCraft)

        self.Run_Model_Flag = get_config().getboolean('TESTING', 'RunModels', fallback=True)


    #####################################################
    def run_sim(self, IO_Flag=False, dev_flag=False, viz_flag = True):
        """
        :param Run_Model_Flag: (bool) whether to run models, false for testing purposes
        """
        if self.Run_Model_Flag:
            self.apriori_training()

        print_sim_header() if (IO_Flag == True) else ''
        print_msg('Please wait...\n') if (IO_Flag == 'strict') else ''
        diagnosis_list = []
        time_step = 0
        last_diagnosis = time_step
        last_fault = time_step

        while self.simData.has_more() and time_step < 2050:
            _next = self.simData.get_next()
            self.brain.reason(_next)
            self.IO_check(time_step, IO_Flag)
            
            ### Stop when a fault is reached  
            if self.brain.mission_status == 'RED':
                diagnosis_list.append(self.brain.diagnose(time_step))
                break 
            time_step += 1
            
        # Final diagnosis processing
        if len(diagnosis_list) == 0:
            diagnosis_list.append(self.brain.diagnose(time_step))
        
        final_diagnosis = diagnosis_list[-1]
        print("**** FINAL DIAGNOSIS ****") if (IO_Flag == True) else ''
        print(final_diagnosis) if (IO_Flag == True) else ''
        
        return final_diagnosis

    def apriori_training(self):
        self.brain.learning_systems.apriori_training(self.simData.data)

    def IO_check(self, time_step, IO_Flag):
        if IO_Flag == True:
            print_sim_step(time_step + 1)
            curr_data = self.brain.spacecraft_rep.curr_data
            print_mission_status(self.brain, curr_data)
        else:
            # print_dots(time_step)
            pass


