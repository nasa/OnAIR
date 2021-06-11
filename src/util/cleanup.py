"""
cleanup.py
Utility file to clean and remove unwanted files pre-Run and post-Run
"""

import os

def clean(PreRun, path=''):
    if PreRun: # PreRun: Perform cleaning that must occur at the START of driver.py
        os.system('find . -name \'.DS_Store\' -type f -delete') # Removes those pesky .DS_Store files that Macs make
        if not os.path.isdir(path + 'results'):
            os.system('mkdir ' + path + 'results')
    else: # PostRun: Perform cleaning that must occur at the END of driver.py
        pass
