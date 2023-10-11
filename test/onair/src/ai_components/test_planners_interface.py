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
def test_PlannersInterface__init__sets_instance_headers_to_given_headers_and_does_nothing_else_when_given__ai_plugins_is_empty(mocker):
    # Arrange
    arg_headers = []
    arg__ai_plugins = {}

    num_fake_headers = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 headers (0 has own test)
    for i in range(num_fake_headers):
        arg_headers.append(MagicMock())

    cut = PlannersInterface.__new__(PlannersInterface)

    # Act
    cut.__init__(arg_headers, arg__ai_plugins)

    # Assert
    assert cut.headers == arg_headers

def test_PlannersInterface__init__sets_instance_ai_constructs_to_a_list_of_the_calls_AIPlugIn_with_plugin_and_given_headers_for_each_item_in_given__ai_plugins_when_given__ai_plugins_is_occupied(mocker):
    # Arrange
    arg_headers = []
    arg__ai_plugins = {}
    fake_spec_list = []
    fake_module_list = []

    num_fake_headers = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 headers (0 has own test)
    for i in range(num_fake_headers):
        arg_headers.append(MagicMock())
    num_fake_ai_plugins = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_ai_plugins):
        arg__ai_plugins[str(i)] = str(MagicMock())
        fake_spec_list.append(MagicMock())
        fake_module_list.append(MagicMock())
        
    expected_ai_constructs = []
    for i in range(num_fake_ai_plugins):
        expected_ai_constructs.append(MagicMock())

    # mocker.patch('importlib.import_module', return_value=fake_imported_module)
    mocker.patch('importlib.util.spec_from_file_location',side_effect=fake_spec_list)
    mocker.patch('importlib.util.module_from_spec',side_effect=fake_module_list)
    for spec in fake_spec_list:
        mocker.patch.object(spec,'loader.exec_module')
    for i, module in enumerate(fake_module_list):
        mocker.patch.object(module,'Plugin',return_value=expected_ai_constructs[i])

    cut = PlannersInterface.__new__(PlannersInterface)

    # Act
    cut.__init__(arg_headers, arg__ai_plugins)

    # Assert
    assert importlib.util.spec_from_file_location.call_count == len(arg__ai_plugins)
    assert importlib.util.module_from_spec.call_count == len(fake_spec_list)

    for i in range(num_fake_ai_plugins):
        fake_name = list(arg__ai_plugins.keys())[i]
        fake_path = arg__ai_plugins[fake_name]
        assert importlib.util.spec_from_file_location.call_args_list[i].args == (fake_name,fake_path)
        assert importlib.util.module_from_spec.call_args_list[i].args == (fake_spec_list[i],)
        assert fake_spec_list[i].loader.exec_module.call_count == 1
        assert fake_spec_list[i].loader.exec_module.call_args_list[0].args == (fake_module_list[i],)
        assert fake_module_list[i].Plugin.call_count == 1
        assert fake_module_list[i].Plugin.call_args_list[0].args == (fake_name,arg_headers)

    assert cut.ai_constructs == expected_ai_constructs

def test_update_does_nothing():
    # Arrange
    arg_curr_raw_tlm = MagicMock()
    arg_status = MagicMock()

    cut = PlannersInterface.__new__(PlannersInterface)

    # Act
    result = cut.update(arg_curr_raw_tlm, arg_status)

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
