"""
sim_io.py
Utility file for sim io
"""

import os
import json 

def render_diagnosis(diagnosis_list):
    with open(os.path.join(os.environ.get('RAISR_DIAGNOSIS_SAVE_PATH'), 'diagnosis.txt'), mode='a') as out:
        out.write('==========================================================\n')
        out.write('                        DIAGNOSIS                         \n')
        out.write('==========================================================\n')
        for diagnosis in diagnosis_list:
            out.write('\n----------------------------------------------------------\n')
            out.write('***                DIAGNOSIS AT FRAME {}               ***\n'.format(diagnosis.get_time_step()))
            out.write(diagnosis.__str__())
            out.write('----------------------------------------------------------\n')
    with open(os.path.join(os.environ.get('RAISR_DIAGNOSIS_SAVE_PATH'), 'diagnosis.csv'), mode='a') as out:
        out.write('time_step, cohens_kappa, faults, subgraph\n')
        for diagnosis in diagnosis_list:
            out.write(diagnosis.results_csv())


def print_dots(ts):
    incrFlag = ts % 20
    if incrFlag < 10:
        dots = ts % 10
    else:
        dots = 10 - (ts % 10)
    print('\033[95m' + (dots+1)*'.' + '\033[0m')


