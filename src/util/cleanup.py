"""
cleanup.py
Utility file to clean and remove unwanted files post-run, or setup folders pre-run
"""

import os

def clean_all(run_path):
    run_path = os.path.dirname(os.path.realpath(__file__)) + "/../../" if run_path == '' else run_path
    os.chdir(run_path)
    # os.system('find . -name \'.DS_Store\' -type f -delete') # Removes those pesky .DS_Store files that Macs make
    os.system('find . | grep -E "(__pycache__|.pyc|.pyo$)" | xargs rm -rf') # Removes those pesky .DS_Store files that Macs make
    os.system('find . | grep -E ".DS_Store" | xargs rm -rf') 

def setup_folders(results_path):
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
