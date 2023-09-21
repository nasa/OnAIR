# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
Sim class
Helper class to create and run a simulation
"""

import importlib

from ..reasoning.agent import Agent
from ..systems.vehicle_rep import VehicleRepresentation
from ..util.file_io import *
from ..util.print_io import *
from ..util.sim_io import *
from ...data_handling.data_source import DataSource

MAX_STEPS = 2050
DIAGNOSIS_INTERVAL = 100

class Simulator:
    def __init__(self, simType, dataParser, plugin_list, SBN_Flag):
        self.simulator = simType

        if SBN_Flag:
            # TODO: This is ugly, but sbn_client is only available when built for cFS...
            # ...from sbn_adapter import AdapterDataSource
            sbn_adapter = importlib.import_module('onair.src.run_scripts.sbn_adapter')
            AdapterDataSource = getattr(sbn_adapter, 'AdapterDataSource')
            self.simData = AdapterDataSource(dataParser.get_just_data())
            self.simData.connect() # this also subscribes to the msgIDs
            
        else:
            #self.simData = DataSource(dataParser.get_just_data())
            self.simData = dataParser

        headers, tests = dataParser.get_vehicle_metadata()
        vehicle = VehicleRepresentation(headers, tests)
        self.agent = Agent(vehicle, plugin_list)

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
                    if (time_step - last_diagnosis) % DIAGNOSIS_INTERVAL == 0:
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
            print_system_status(self.agent, curr_data)
        else:
            # print_dots(time_step)
            pass
