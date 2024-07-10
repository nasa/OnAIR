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
from copy import copy

from plugins.csv_output import csv_output_plugin
from plugins.csv_output.csv_output_plugin import Plugin as CSV_Output


def test_init_initalizes_expected_default_variables():

    arg_name = MagicMock()
    arg_headers = [MagicMock(), MagicMock()]

    csv_out = CSV_Output(arg_name, arg_headers)

    # assert
    assert csv_out.component_name == arg_name
    assert csv_out.headers == arg_headers
    assert csv_out.first_frame == True
    assert csv_out.lines_per_file == 10
    assert csv_out.lines_current == 0
    assert csv_out.current_buffer == [] 
    assert csv_out.filename_preamble == "csv_out_"
    assert csv_out.filename == ""

def test_update_adds_plugins_to_headers_on_first_frame():
    arg_name = MagicMock()
    arg_name = MagicMock()
    arg_headers = ['header1', 'header2']
    csv_out = CSV_Output(arg_name, arg_headers)

    # test specific conditions
    csv_out.first_frame = True
    high_level_data = {
        'layer1':{
            'plugin1':[]
        },
        'layer2':{
            'plugin2':[],
            'plugin3':[]
        }
    }
    
    # initial state
    intial_headers = copy(csv_out.headers)

    # expected state
    expected_headers = intial_headers + ['plugin1', 'plugin2', 'plugin3']

    # check
    csv_out.update([], high_level_data)
    assert csv_out.headers == expected_headers

def test_update_does_not_add_headers_on_first_frame_when_missing_plugins():
    arg_name = MagicMock()
    arg_name = MagicMock()
    arg_headers = ['header1', 'header2']
    csv_out = CSV_Output(arg_name, arg_headers)

    # test specific conditions
    csv_out.first_frame = True
    high_level_data = {
        'layer1':{},
        'layer2':{}
    }
    
    # initial state
    intial_headers = copy(csv_out.headers)

    # expected state
    expected_headers = intial_headers

    # check
    csv_out.update([], high_level_data)
    assert csv_out.headers == expected_headers

def test_update_does_not_add_headers_on_first_frame_when_missing_layers():
    arg_name = MagicMock()
    arg_name = MagicMock()
    arg_headers = ['header1', 'header2']
    csv_out = CSV_Output(arg_name, arg_headers)

    # test specific conditions
    csv_out.first_frame = True
    high_level_data = {}
    
    # initial state
    intial_headers = copy(csv_out.headers)

    # expected state
    expected_headers = intial_headers

    # check
    csv_out.update([], high_level_data)
    assert csv_out.headers == expected_headers

def test_update_skips_headers_after_first_frame():
    arg_name = MagicMock()
    arg_name = MagicMock()
    arg_headers = ['header1', 'header2']
    csv_out = CSV_Output(arg_name, arg_headers)

    # test specific conditions
    csv_out.first_frame = False
    
    # initial state
    intial_headers = copy(csv_out.headers)

    # expected state
    expected_headers = intial_headers

    # check
    csv_out.update(low_level_data=[], high_level_data={})
    assert csv_out.headers == expected_headers

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
    high_level_data = {}

    csv_out.update(low_level_data, high_level_data)

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

def test_render_reasoning_creates_expected_file_on_first_frame():
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

def test_render_reasoning_does_not_create_new_file_when_not_first_frame():
    arg_name = MagicMock()
    arg_headers = ['header1', 'header2']
    csv_out = CSV_Output(arg_name, arg_headers)
    csv_out.filename_preamble = 'test_'

    # test specific conditions
    target_file_name = csv_out.filename_preamble + "render_reasoning_not_first_frame" + ".csv"
    csv_out.file_name = target_file_name
    csv_out.first_frame = False

    # create file first so csv_out pluging has something to write to
    with open(target_file_name, 'a') as file: 
        delimiter = ','
        file.write(delimiter.join(csv_out.headers) + '\n')

    # act
    csv_out.render_reasoning()    
    
    try:
        # if the current file name attribute doesn't match the original target file name
        # then a new file was created
        assert(csv_out.file_name == target_file_name)

    except:
        os.remove(csv_out.file_name)
        raise AssertionError("Unexpected new file created after first frame")
    
    finally:
        os.remove(target_file_name)

def test_render_reasoning_changes_file_when_lines_per_file_reached():
    arg_name = MagicMock()
    arg_headers = ['header1', 'header2']
    csv_out = CSV_Output(arg_name, arg_headers)
    csv_out.filename_preamble = 'test_'

    # test specific conditions
    old_file_name = csv_out.filename_preamble + "render_reasoning_end_of_file" + ".csv"
    csv_out.file_name = old_file_name
    csv_out.first_frame = False
    csv_out.lines_per_file = 1
    csv_out.lines_current = 0

    # act
    csv_out.render_reasoning()

    # check if a new file was set, but NOT yet created
    try:
        assert csv_out.file_name != old_file_name
        assert not os.path.exists(csv_out.file_name)
    except:
        raise AssertionError("file name was not updated or new file was created prematurely")
    finally:
        os.remove(old_file_name)
