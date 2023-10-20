# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test PlannersInterface Functionality """
import pytest
from mock import MagicMock

import onair.src.ai_components.planners_interface as planners_interface
from onair.src.ai_components.planners_interface import PlannersInterface

import importlib
from typing import Dict

# __init__ tests
def test_PlannersInterface__init__raises_AssertionError_when_given_headers_len_is_0():
    # Arrange
    arg_headers = MagicMock()

    arg_headers.__len__.return_value = 0

    cut = PlannersInterface.__new__(PlannersInterface)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg_headers)

    # Assert
    assert e_info.match('Headers are required')

def test_PlannersInterface__init__sets_self_headers_to_given_headers_and_sets_self_ai_constructs_to_return_value_of_import_plugins(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg__planner_plugins = MagicMock()

    arg_headers.__len__.return_value = 1

    forced_return_ai_constructs = MagicMock()

    mocker.patch(planners_interface.__name__ + '.import_plugins', return_value=forced_return_ai_constructs)
    
    cut = PlannersInterface.__new__(PlannersInterface)

    # Act
    cut.__init__(arg_headers, arg__planner_plugins)

    # Assert
    assert cut.headers == arg_headers
    assert planners_interface.import_plugins.call_count == 1
    assert planners_interface.import_plugins.call_args_list[0].args == (arg_headers, arg__planner_plugins)
    assert cut.ai_constructs == forced_return_ai_constructs

def test_update_does_nothing():
    # Arrange
    arg_high_level_data = MagicMock()

    cut = PlannersInterface.__new__(PlannersInterface)

    # Act
    result = cut.update(arg_high_level_data)

    # Assert
    assert result == None

def test_check_for_salient_event_does_nothing():
    # Arrange
    cut = PlannersInterface.__new__(PlannersInterface)

    # Act
    result = cut.check_for_salient_event()

    # Assert
    assert result == None

def test_render_reasoning_does_nothing():
    # Arrange
    cut = PlannersInterface.__new__(PlannersInterface)
    
    # Act
    result = cut.render_reasoning()

    # Assert
    assert result == None