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

MAX_STEPS = 2050
DIAGNOSIS_INTERVAL = 100

class Simulator:
    """
    Simulator class for creating and running simulations.

    Args:
        dataParser: The data parser for simulation data.
        plugin_list (list): List of plugins for the simulation agent.
    """
    def __init__(self, dataParser, plugin_list):
        self.simData = dataParser

        headers, tests = dataParser.get_vehicle_metadata()
        vehicle = VehicleRepresentation(headers, tests)
        self.agent = Agent(vehicle, plugin_list)

    def run_sim(self, IO_Flag=False, dev_flag=False, viz_flag = True):
        """
        Run the simulation.

        Args:
            IO_Flag (bool or str): Flag for controlling simulation output.
            dev_flag (bool): Development flag.
            viz_flag (bool): Visualization flag.

        Returns:
            dict: Final diagnosis information.
        """
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
        """
        Set benchmark data for supervised learning.

        Args:
            filepath (str): Path to the benchmark data file.
            files (list): List of file names.
            indices (list): List of indices.
        """
        self.agent.supervised_learning.set_benchmark_data(filepath, files, indices)

    def IO_check(self, time_step, IO_Flag):
        """
        Perform IO checks and printing for the simulation.

        Args:
            time_step (int): Current time step.
            IO_Flag (bool or str): Flag for controlling simulation output.
        """
        if IO_Flag == True:
            print_sim_step(time_step + 1)
            curr_data = self.agent.vehicle_rep.curr_data
            print_system_status(self.agent, curr_data)
        else:
            # print_dots(time_step)
            pass
