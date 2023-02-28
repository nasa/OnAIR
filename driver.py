"""
Driver
Source of the main function for the OnAIR repo
"""
import pytest
import coverage
# coverage started early to see all lines in all files (def and imports were being missed with programmatic runs)
cov = coverage.Coverage(source=['src'], branch=True)
cov.start()

import os
import sys
import io
import configparser
import argparse
import pathlib
from datetime import datetime
# from test_all import *
from src.util.cleanup import *

from src.run_scripts.execution_engine import ExecutionEngine

cov.stop()

def main():
    """
    This is the standard naming format, for now.
    filename.txt and filename_CONFIG.txt
    Additional (for now), the files need to live in the following locations:
     filename.txt: OnAIR/src/data/raw_telemetry_data/
     filename_CONFIG.txt: OnAIR/src/data/telemetry_configs/
    Check the .ini file for the filenames used
    """

    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('configfile', nargs='?', default='./config/default_config.ini', help='Config file to be used')
    arg_parser.add_argument('--save', '-s', action='store_true', help='Should log files be saved?')
    arg_parser.add_argument('--save_name', '--name', '-n', help='Name of saved log files')
    arg_parser.add_argument('--mute', '-m', action='store_true', help='Mute all non-error output')
    arg_parser.add_argument('--test', '-t', action='store_true', help='Run tests')
    args = arg_parser.parse_args()

    if args.mute:
        blockPrint()

    init_global_paths(args.test)

    if args.test:
        run_unit_tests(cov)
    else:
        setup_folders(os.environ['RESULTS_PATH'])
        save_name = args.save_name if args.save_name else datetime.now().strftime("%m%d%Y_%H%M%S")
        OnAIR = ExecutionEngine(args.configfile, save_name, args.save)
        OnAIR.run_sim()

    clean_all(os.environ['BASE_PATH']) 


""" Runs all unit tests """
def run_unit_tests(Coverage: cov):
    cov.start()
    retval=pytest.main(['test'])
    cov.stop()
    cov.save()
    cov.html_report()

""" Initializes global paths, used throughout execution """
def init_global_paths(test=False):
    run_path = 'src/test' if test == True else './'
    results_path = 'src/test/results' if test == True else 'results/'

    os.environ['BASE_PATH'] = os.path.dirname(os.path.realpath(__file__))
    os.environ['RUN_PATH'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), run_path)
    os.environ['RESULTS_PATH'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), results_path)
    os.environ['SRC_ROOT_PATH'] = os.path.dirname(os.path.realpath(__file__))

""" Disable terminal output """
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

""" Restore terminal output """
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
