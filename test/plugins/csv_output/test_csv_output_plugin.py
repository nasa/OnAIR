# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test CSV Output Plugin Functionality """
import pytest
from unittest.mock import MagicMock
import os


from plugins.csv_output import csv_output_plugin
from plugins.csv_output.csv_output_plugin import Plugin as CSV_Output


def test_init_initalizes_expected_default_variables():

    arg_name = MagicMock()
    arg_headers = [MagicMock(), MagicMock()]

    csv_out = CSV_Output(arg_name, arg_headers)

    # could we not just csv_out = CSV_Output(arg_name, arg_headers)

    # assert
    assert csv_out.component_name == arg_name
    assert csv_out.headers == arg_headers
    assert csv_out.first_frame == True
    assert csv_out.lines_per_file == 10
    assert csv_out.lines_current == 0
    assert csv_out.current_buffer == [] 
    assert csv_out.filename_preamble == "csv_out_"
    assert csv_out.filename == ""

def test_update_leaves_buffer_empty_when_given_no_data():
    arg_name = MagicMock()
    arg_headers = [MagicMock(), MagicMock()]
    csv_out = CSV_Output(arg_name, arg_headers)

    csv_out.update(low_level_data=[],high_level_data={})
    assert csv_out.current_buffer == []

def test_update_fills_buffer_with_low_level_data():
    arg_name = MagicMock()
    arg_headers = [MagicMock(), MagicMock()]
    csv_out = CSV_Output(arg_name, arg_headers)

    low_level_data = [MagicMock() for x in range(10)]

    csv_out.update(low_level_data)

    assert csv_out.current_buffer == [str(item) for item in low_level_data]

def test_update_fills_buffer_with_high_level_data():
    arg_name = MagicMock()
    arg_headers = [MagicMock(), MagicMock()]
    csv_out = CSV_Output(arg_name, arg_headers)

    example = {
        'vehicle_rep':{
            'plugin_1': ['1', '2', '3']
        },
        'learning_system':{
            'plugin_2': ['a', 'b', 'c'],
            'plugin_3': ['x', 'y', 'z']
        },
        'planning_system':{}
    }
    expected_buffer = ['1', '2', '3', 'a', 'b', 'c', 'x' ,'y', 'z']

    csv_out.update(low_level_data= [], high_level_data = example)

    assert csv_out.current_buffer == expected_buffer

def test_render_reasoning_creates_expected_file():
    arg_name = MagicMock()
    arg_headers = ['header1', 'header2']
    csv_out = CSV_Output(arg_name, arg_headers)
    csv_out.filename_preamble = 'test_'

    csv_out.render_reasoning()

    assert csv_out.file_name.startswith(csv_out.filename_preamble)
    assert os.path.exists(csv_out.file_name)

    # cleanup
    if os.path.exists(csv_out.file_name):
        os.remove(csv_out.file_name)