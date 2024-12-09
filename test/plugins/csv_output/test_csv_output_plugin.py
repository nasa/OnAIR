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

from plugins.csv_output.csv_output_plugin import Plugin as CSV_Output


def test_csv_output_plugin_init_initalizes_expected_default_variables(mocker):

    # Arrange
    arg_name = MagicMock()
    arg_headers = [MagicMock(), MagicMock()]

    cut = CSV_Output.__new__(CSV_Output)

    mocker.patch('onair.src.ai_components.ai_plugin_abstract.ai_plugin.ServiceManager')

    # Act

    cut.__init__(arg_name, arg_headers)

    # Assert
    assert cut.component_name == arg_name
    assert cut.headers == arg_headers
    assert cut.first_frame == True
    assert cut.lines_per_file == 10
    assert cut.lines_current == 0
    assert cut.current_buffer == []
    assert cut.filename_preamble == "csv_out_"
    assert cut.filename == ""


def test_csv_output_plugin_update_adds_plugins_to_headers_on_first_frame(mocker):
    # Arrange
    fake_headers = ["header1", "header2"]
    cut = CSV_Output.__new__(CSV_Output)
    cut.headers = fake_headers
    cut.first_frame = True

    high_level_data = {
        "layer1": {"plugin1": []},
        "layer2": {"plugin2": [], "plugin3": []},
    }

    intial_headers = copy(cut.headers)
    expected_headers = intial_headers + ["plugin1", "plugin2", "plugin3"]

    mocker.patch('onair.src.ai_components.ai_plugin_abstract.ai_plugin.ServiceManager')

    # Act
    cut.update([], high_level_data)

    # Assert
    assert cut.headers == expected_headers


def test_csv_output_plugin_update_does_not_add_headers_on_first_frame_when_missing_plugins(mocker):
    # Arrange
    fake_headers = ["header1", "header2"]
    cut = CSV_Output.__new__(CSV_Output)
    cut.headers = fake_headers

    cut.first_frame = True
    high_level_data = {"layer1": {}, "layer2": {}}

    intial_headers = copy(cut.headers)
    expected_headers = intial_headers

    # Act
    cut.update([], high_level_data)

    # Assert
    assert cut.headers == expected_headers


def test_csv_output_plugin_update_does_not_add_headers_on_first_frame_when_missing_layers(mocker):
    # Arrange
    fake_headers = ["header1", "header2"]
    cut = CSV_Output.__new__(CSV_Output)
    cut.headers = fake_headers
    cut.first_frame = True

    high_level_data = {}

    intial_headers = copy(cut.headers)
    expected_headers = intial_headers

    # Act
    cut.update([], high_level_data)
    
    # Assert
    assert cut.headers == expected_headers


def test_csv_output_plugin_update_skips_headers_after_first_frame(mocker):
    # Arrange
    fake_headers = ["header1", "header2"]
    cut = CSV_Output.__new__(CSV_Output)
    cut.headers = fake_headers
    cut.first_frame = False

    intial_headers = copy(cut.headers)
    expected_headers = intial_headers

    # Act
    cut.update(low_level_data=[], high_level_data={})
 
    # Assert
    assert cut.headers == expected_headers


def test_csv_output_plugin_update_leaves_buffer_empty_when_given_no_data(mocker):
    # Arrange
    cut = CSV_Output.__new__(CSV_Output)
    cut.first_frame = False

    # Act
    cut.update(low_level_data=[], high_level_data={})
    
    # Assert
    assert cut.current_buffer == []


def test_csv_output_plugin_update_fills_buffer_with_low_level_data(mocker):
    # Arrange
    cut = CSV_Output.__new__(CSV_Output)
    cut.first_frame = False

    low_level_data = [MagicMock() for x in range(10)]
    high_level_data = {}

    # Act
    cut.update(low_level_data, high_level_data)

    # Assert
    assert cut.current_buffer == [str(item) for item in low_level_data]


def test_csv_output_plugin_update_fills_buffer_with_high_level_data(mocker):
    # Arrange
    cut = CSV_Output.__new__(CSV_Output)
    cut.first_frame = False

    example = {
        "vehicle_rep": {"plugin_1": ["1", "2", "3"]},
        "learning_system": {"plugin_2": ["a", "b", "c"], "plugin_3": ["x", "y", "z"]},
        "planning_system": {},
    }
    expected_buffer = ["1", "2", "3", "a", "b", "c", "x", "y", "z"]

    # Act
    cut.update(low_level_data=[], high_level_data=example)

    # Assert
    assert cut.current_buffer == expected_buffer


def test_csv_output_plugin_render_reasoning_creates_expected_file_on_first_frame(mocker):
    # Arrange
    cut = CSV_Output.__new__(CSV_Output)
    cut.filename_preamble = "test_"
    cut.first_frame = True
    cut.headers = ["header1", "header2"]
    cut.current_buffer = []
    cut.lines_per_file = 10
    cut.lines_current = 0

    # Act
    cut.render_reasoning()

    # Assert
    assert cut.file_name.startswith(cut.filename_preamble)
    assert os.path.exists(cut.file_name)

    # cleanup
    if os.path.exists(cut.file_name):
        os.remove(cut.file_name)


def test_csv_output_plugin_render_reasoning_does_not_create_new_file_when_not_first_frame(mocker):
    # Arrange 
    cut = CSV_Output.__new__(CSV_Output)
    cut.filename_preamble = "test_"
    cut.first_frame = True
    cut.headers = ["header1", "header2"]
    cut.current_buffer = []
    cut.lines_per_file = 10
    cut.lines_current = 0

    # test specific conditions
    target_file_name = (
        cut.filename_preamble + "render_reasoning_not_first_frame" + ".csv"
    )
    cut.file_name = target_file_name
    cut.first_frame = False

    # create file first so csv_out pluging has something to write to
    with open(target_file_name, "a") as file:
        delimiter = ","
        file.write(delimiter.join(cut.headers) + "\n")

    # act
    cut.render_reasoning()

    try:
        # if the current file name attribute doesn't match the original target file name
        # then a new file was created
        assert cut.file_name == target_file_name

    except:
        os.remove(cut.file_name)
        raise AssertionError("Unexpected new file created after first frame")

    finally:
        os.remove(target_file_name)


def test_csv_output_plugin_render_reasoning_changes_file_when_lines_per_file_reached(mocker):
    # Arrange
    cut = CSV_Output.__new__(CSV_Output)
    cut.filename_preamble = "test_"
    cut.headers = ["header1", "header2"]
    cut.current_buffer = []
    cut.first_frame = False
    cut.lines_per_file = 1
    cut.lines_current = 0

    old_file_name = cut.filename_preamble + "render_reasoning_end_of_file" + ".csv"
    cut.file_name = old_file_name

    # Act
    cut.render_reasoning()

    # Assert
    try:
        assert cut.file_name != old_file_name
        assert not os.path.exists(cut.file_name)
    except:
        raise AssertionError(
            "file name was not updated or new file was created prematurely"
        )
    finally:
        os.remove(old_file_name)
