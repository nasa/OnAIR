# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
sim_io.py
Utility file for sim io
"""

import os
import json 

def render_reasoning(diagnosis_list):
    with open(os.path.join(os.environ.get('ONAIR_DIAGNOSIS_SAVE_PATH'), 'diagnosis.txt'), mode='a') as out:
        out.write('==========================================================\n')
        out.write('                        DIAGNOSIS                         \n')
        out.write('==========================================================\n')
        for diagnosis in diagnosis_list:
            out.write('\n----------------------------------------------------------\n')
            out.write('***                DIAGNOSIS AT FRAME {}               ***\n'.format(diagnosis.get_time_step()))
            out.write(diagnosis.__str__())
            out.write('----------------------------------------------------------\n')
    with open(os.path.join(os.environ.get('ONAIR_DIAGNOSIS_SAVE_PATH'), 'diagnosis.csv'), mode='a') as out:
        out.write('time_step, cohens_kappa, faults, subgraph\n')
        for diagnosis in diagnosis_list:
            out.write(diagnosis.results_csv())

def render_viz(status_data, sensor_data, sim_name, diagnosis=None):
    # Status Staburst
    status_report = {} 
    status_report['filename'] = sim_name
    status_report['data'] = status_data
    with open(os.path.join(os.environ.get('ONAIR_VIZ_SAVE_PATH'), 'system.json'), 'w') as outfile:
        json.dump(status_report, outfile)

    # Associativity
    sensor_status_report = {}
    sensor_status_report['name'] = 'MISSION'
    sensor_status_report['children'] = sensor_data

    with open(os.path.join(os.environ.get('ONAIR_VIZ_SAVE_PATH'), 'faults.json'), 'w') as outfile:
        json.dump(sensor_status_report, outfile)

    # Diagnosis info
    if diagnosis is not None:
        results = diagnosis.get_diagnosis_viz_json()
        with open(os.path.join(os.environ.get('ONAIR_VIZ_SAVE_PATH'), 'results.json'), 'w') as outfile:
            json.dump(results, outfile)

def print_dots(ts):
    incrFlag = ts % 20
    if incrFlag < 10:
        dots = ts % 10
    else:
        dots = 10 - (ts % 10)
    print('\033[95m' + (dots+1)*'.' + '\033[0m')

####################################################################################
# print('\033[95m [TIMESTEP] \033[0m' + str(time_step))
# print('\033[95m [FAULTING MNEMONICS] \033[0m' + str(self.agent.vehicle_rep.get_faulting_mnemonics()))
# # print('\033[95m [IMPORTANCE] \033[0m' + str(self.agent.supervised_learning.get_importance_sampling()))
# record_input = input('\033[95m record diagnosis? (y/n/exit) >>> \033[0m')
# if record_input == 'exit': 
#     diagnosis_list.append(self.agent.diagnose(time_step))
#     # print(self.agent.supervised_learning.get_benchmark_graph())
#     break
# if record_input == 'y': diagnosis_list.append(self.agent.diagnose(time_step))
####################################################################################

