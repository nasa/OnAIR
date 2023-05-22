"""
cleanup.py
Utility file to clean and remove unwanted files post-run, or setup folders pre-run
"""

import os

# TODO: These two lines used in one place do not warrant their own function
def setup_folders(results_path):
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
