# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
cleanup.py
Utility file to clean and remove unwanted files post-run, or setup folders pre-run
"""

import os


def setup_folders(results_path):
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
