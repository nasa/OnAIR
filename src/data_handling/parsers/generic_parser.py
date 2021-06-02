"""
Generic Parser
"""

import csv
import time
import os
import re
import time

from src.util.file_io import *
from src.util.print_io import *
from src.data_handling.parsers.parser_util import * 

class Generic:
    def __init__(self, _headers, _data, _configs):
        self.headers = _headers
        self.data = _data
        self.configs = _configs

    ##### GETTERS ##################################
    def get_sim_data(self):
        return self.headers, self.data, self.configs

