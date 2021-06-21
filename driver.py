"""
Driver
Source of the main function for the RAISR repo
"""

import os
import sys
import io
import configparser
import argparse
import pathlib
from datetime import datetime
from test_all import *
import src.util.cleanup as cleanup

from src.run_scripts.execution_engine import ExecutionEngine

def main():
    cleanup.clean(True) # Perform PreRun Cleanup

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

    init_global_paths(args.test)

    if args.test:
        cleanup.clean(True, path='src/test/') # Perform PreRun Cleanup
        suite = create_suite()
        run_tests(suite)
        cleanup.clean(False) # Perform PostRun Cleanup
        return

    save_name = args.save_name if args.save_name else datetime.now().strftime("%m%d%Y_%H%M%S")

    if args.mute:
        blockPrint()

    RAISR = ExecutionEngine(args.configfile, save_name, args.save)
    RAISR.run_sim()

    cleanup.clean(False) # Perform PostRun Cleanup

def init_global_paths(test=False):
    run_path = 'src/test' if test == True else 'src/'
    results_path = 'src/test/results' if test == True else 'results/'
    os.environ['RUN_PATH'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), run_path)
    os.environ['RESULTS_PATH'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), results_path)
    os.environ['SRC_ROOT_PATH'] = os.path.dirname(os.path.realpath(__file__))

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
