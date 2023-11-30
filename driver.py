# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
Driver
Source of the main function for the OnAIR repo
"""
import os
import sys
import argparse
from datetime import datetime


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
    arg_parser.add_argument('configfile', nargs='?',
                            default='./onair/config/default_config.ini',
                            help='Config file to be used')
    arg_parser.add_argument('--save', '-s', action='store_true',
                            help='Should log files be saved?')
    arg_parser.add_argument('--save_name', '--name', '-n',
                            help='Name of saved log files')
    arg_parser.add_argument('--mute', '-m', action='store_true',
                            help='Mute all non-error output')

    """
    Testing specific arguments
    """
    arg_parser.add_argument('--test', '-t', action='store_true',
                            help='Run tests')
    arg_parser.add_argument('--verbose', '-v', action='count', default=0,
                            help="Increase verbosity in tests")
    arg_parser.add_argument('-k', action='store', dest='keyword', default="",
                            metavar='EXPRESSION',
                            help="Pass thru for pytest's -k option. Runs only"
                                 " tests with names that match EXPRESSION.")
    arg_parser.add_argument('--conftest-seed', action='store',
                            type=int, default=None,
                            help="Set the random seed for test values")
    arg_parser.add_argument('--randomly-seed', action='store',
                            type=int, default=None,
                            help="Set the random seed for test run order")
    args = arg_parser.parse_args()

    """
    In test mode, covergage must start before imports from onair,
    otherwise lines are missed.
    """
    if args.test:
        import coverage
        cov = coverage.Coverage(source=['onair', 'plugins'], branch=True)
        cov.start()

    """
    Imports from onair that load with or without test mode enabled.
    """
    from onair.src.util.cleanup import setup_folders
    from onair.src.run_scripts.execution_engine import ExecutionEngine

    if args.mute:
        blockPrint()

    init_global_paths(args.test)

    """ Runs all unit tests """
    if args.test:
        import pytest
        test_directory_name = "test"
        pytest_args = [test_directory_name]

        pytest_args.extend(['-v'] * args.verbose)
        if args.conftest_seed:
            pytest_args.extend([f"--conftest-seed={args.conftest_seed}"])
        if args.randomly_seed:
            pytest_args.extend([f"--randomly-seed={args.randomly_seed}"])
        pytest_args.extend([f"-k {args.keyword}"])

        pytest.main(pytest_args)
        cov.stop()
        cov.save()
        cov.html_report()
    else:
        setup_folders(os.environ['RESULTS_PATH'])
        if args.save_name:
            save_name = args.save_name
        else:
            save_name = datetime.now().strftime("%m%d%Y_%H%M%S")
        OnAIR = ExecutionEngine(args.configfile, save_name, args.save)
        OnAIR.run_sim()


def init_global_paths(test=False):
    """
    Initializes global paths, used throughout execution
    """
    run_path = 'onair/src/test' if test else './'
    results_path = 'onair/src/test/results' if test else 'results/'

    os.environ['BASE_PATH'] = os.path.dirname(os.path.realpath(__file__))
    os.environ['RUN_PATH'] = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), run_path)
    os.environ['RESULTS_PATH'] = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), results_path)
    os.environ['SRC_ROOT_PATH'] = os.path.dirname(os.path.realpath(__file__))


def blockPrint():
    """ Disable terminal output """
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    """ Restore terminal output """
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
