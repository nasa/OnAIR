"""
Driver
Source of the main function for the RAISR repo
"""

import os
import sys
import io
import importlib
import argparse
import pathlib
from datetime import datetime
from test_all import *
from src.util.cleanup import *

import src.util.config

from src.run_scripts.execution_engine import ExecutionEngine

def main():
    """
    This is the standard naming format, for now.
    filename.txt and filename_CONFIG.txt
    Additional (for now), the files need to live in the following locations:
     filename.txt: RAISR/src/data/raw_telemetry_data/
     filename_CONFIG.txt: RAISR/src/data/telemetry_configs/
    Check the .ini file for the filenames used
    """

    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('configfile', nargs='?', default='./src/config/default_config.ini', help='Config file to be used')
    arg_parser.add_argument('--save', '-s', action='store_true', help='Should log files be saved?')
    arg_parser.add_argument('--save_name', '--name', '-n', help='Name of saved log files')
    arg_parser.add_argument('--mute', '-m', action='store_true', help='Mute all non-error output')
    arg_parser.add_argument('--test', '-t', action='store_true', help='Run tests')
    args = arg_parser.parse_args()

    if args.mute:
        blockPrint()

    init_global_paths(args.configfile, args.test)
    setup_folders(os.environ['RESULTS_PATH'])

    importlib.reload(src.util.config) # Reload config after changing env variables

    if args.test:
        run_unit_tests()
    else:
        save_name = args.save_name if args.save_name else datetime.now().strftime("%m%d%Y_%H%M%S")
        RAISR = ExecutionEngine(False, save_name, args.save)
        RAISR.run_sim()

    clean_all(os.environ['SRC_ROOT_PATH']) 

""" Runs all unit tests """
def run_unit_tests():
    suite = create_suite()
    run_tests(suite)

""" Initializes global paths, used throughout execution """
def init_global_paths(config_path, test=False):
    """
    Initializes environment variables
    :param config_path: (string) path to configuration file from src directory (include src/)
    :param test: (optional bool) if true, environment variables point to 'src/test' instead of 'src'
    """
    run_path = 'src/test' if test == True else 'src/'
    results_path = 'src/test/results' if test == True else 'results/'

    os.environ['RUN_PATH'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), run_path)
    os.environ['RESULTS_PATH'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), results_path)
    os.environ['SRC_ROOT_PATH'] = os.path.dirname(os.path.realpath(__file__))
    os.environ['CONFIG_PATH'] = os.path.join(os.environ['SRC_ROOT_PATH'], 'src/test/config/default_config.ini') if test == True else config_path

""" Disable terminal output """
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

""" Restore terminal output """
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
