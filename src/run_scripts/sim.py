"""
Sim class
Helper class to create and run a simulation
"""

import importlib

from src.reasoning.agent import Agent
from src.systems.vehicle_rep import VehicleRepresentation
from src.util.file_io import *
from src.util.print_io import *
from src.util.sim_io import *
from data_handling.data_source import DataSource

MAX_STEPS = 2050
PLACEHOLDER_NAME = 100

class Simulator:
    def __init__(self, simType, parsedData, SBN_Flag):
        self.simulator = simType
        vehicle = VehicleRepresentation(*parsedData.get_vehicle_metadata())

        if SBN_Flag:
            # TODO: This is ugly, but sbn_client is only available when built for cFS...
            # ...from sbn_adapter import AdapterDataSource
            sbn_adapter = importlib.import_module('src.run_scripts.sbn_adapter')
            AdapterDataSource = getattr(sbn_adapter, 'AdapterDataSource')
            self.simData = AdapterDataSource(parsedData.get_sim_data())
            self.simData.connect() # this also subscribes to the msgIDs
            
        else:
            self.simData = DataSource(parsedData.get_sim_data())
        self.agent = Agent(vehicle)

    def run_sim(self, IO_Flag=False, dev_flag=False, viz_flag = True):
        if IO_Flag == True: print_sim_header()
        if IO_Flag == 'strict': print_msg('Please wait...\n')
        diagnosis_list = []
        time_step = 0
        last_diagnosis = time_step
        last_fault = time_step

        while self.simData.has_more() and time_step < MAX_STEPS:

            next = self.simData.get_next()
            self.agent.reason(next)
            self.IO_check(time_step, IO_Flag)
            
            ### Stop when a fault is reached  
            if self.agent.mission_status == 'RED':
                if last_fault == time_step - 1: #if they are consecutive
                    if (time_step - last_diagnosis) % PLACEHOLDER_NAME == 0:
                        diagnosis_list.append(self.agent.diagnose(time_step))
                        last_diagnosis = time_step
                else:
                    diagnosis_list.append(self.agent.diagnose(time_step))
                    last_diagnosis = time_step
                last_fault = time_step
            time_step += 1
            
        # Final diagnosis processing
        if len(diagnosis_list) == 0:
            diagnosis_list.append(self.agent.diagnose(time_step))
        final_diagnosis = diagnosis_list[-1]
        return final_diagnosis


    def set_benchmark_data(self, filepath, files, indices):
        self.agent.supervised_learning.set_benchmark_data(filepath, files, indices)

    def IO_check(self, time_step, IO_Flag):
        if IO_Flag == True:
            print_sim_step(time_step + 1)
            curr_data = self.agent.vehicle_rep.curr_data
            print_mission_status(self.agent, curr_data)
        else:
            # print_dots(time_step)
            pass


