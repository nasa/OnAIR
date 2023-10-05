# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test LearnersInterface Functionality """
import pytest
from mock import MagicMock

import onair.src.ai_components.learners_interface as learners_interface
from onair.src.ai_components.learners_interface import LearnersInterface

import importlib
from typing import Dict

# __init__ tests
def test_LearnersInterface__init__sets_instance_headers_to_given_headers_and_does_nothing_else_when_given__ai_plugins_is_empty(mocker):
    # Arrange
    arg_headers = []
    arg__ai_plugins = {}

    num_fake_headers = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 headers (0 has own test)
    for i in range(num_fake_headers):
        arg_headers.append(MagicMock())

    cut = LearnersInterface.__new__(LearnersInterface)

    # Act
    cut.__init__(arg_headers, arg__ai_plugins)

    # Assert
    assert cut.headers == arg_headers

def test_LearnersInterface__init__throws_AttributeError_when_given_module_file_has_no_attribute_Plugin(mocker):
    # Arrange
    fake_module_name = MagicMock()
    arg_headers = []

    arg__ai_plugins = {MagicMock()}

    

    # Act

    # Assert

def test_LearnersInterface__init__sets_instance_ai_constructs_to_a_list_of_the_calls_AIPlugIn_with_plugin_and_given_headers_for_each_item_in_given__ai_plugins_when_given__ai_plugins_is_occupied(mocker):
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

    cut = LearnersInterface.__new__(LearnersInterface)

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

# update tests
def test_LearnersInterface_update_only_calls_flotify_input_with_given_curr_data_and_status_to_oneHot_with_given_status_when_instance_ai_constructs_is_empty(mocker):
    # Arrange
    arg_curr_data = MagicMock()
    arg_status = MagicMock()

    mocker.patch(learners_interface.__name__ + '.status_to_oneHot')

    cut = LearnersInterface.__new__(LearnersInterface)
    cut.ai_constructs = []

    # Act
    result = cut.update(arg_curr_data, arg_status)

    # Assert
    assert learners_interface.status_to_oneHot.call_count == 1
    assert learners_interface.status_to_oneHot.call_args_list[0].args == (arg_status,)
    assert result == None

def test_LearnersInterface_update_calls_flotify_input_with_given_curr_data_and_status_to_oneHot_with_given_status_and_calls_update_on_each_ai_construct_with_input_data_when_instance_ai_constructs_is_occupied(mocker):
    # Arrange
    arg_curr_data = MagicMock()
    arg_status = MagicMock()

    mocker.patch(learners_interface.__name__ + '.status_to_oneHot')

    cut = LearnersInterface.__new__(LearnersInterface)
    cut.ai_constructs = []

    num_fake_ai_constructs = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_ai_constructs):
        fake_ai_construct = MagicMock()
        cut.ai_constructs.append(fake_ai_construct)
        mocker.patch.object(fake_ai_construct, 'update')

    # Act
    result = cut.update(arg_curr_data, arg_status)

    # Assert
    assert learners_interface.status_to_oneHot.call_count == 1
    assert learners_interface.status_to_oneHot.call_args_list[0].args == (arg_status,)
    for i in range(num_fake_ai_constructs):
        assert cut.ai_constructs[i].update.call_count == 1
        assert cut.ai_constructs[i].update.call_args_list[0].args == (arg_curr_data,)
    assert result == None

# apriori_training tests
def test_LearnersInterface_apriori_training_does_nothing_when_instance_ai_constructs_is_empty():
    # Arrange
    arg_batch_data = MagicMock()

    cut = LearnersInterface.__new__(LearnersInterface)
    cut.ai_constructs = []

    # Act
    result = cut.apriori_training(arg_batch_data)

    # Assert
    assert result == None

def test_LearnersInterface_apriori_training_calls_apriori_training_on_each_ai_constructs_item(mocker):
    # Arrange
    arg_batch_data = MagicMock()

    cut = LearnersInterface.__new__(LearnersInterface)
    cut.ai_constructs = []

    num_fake_ai_constructs = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_ai_constructs):
        cut.ai_constructs.append(MagicMock())

    # Act
    result = cut.apriori_training(arg_batch_data)

    # Assert
    for i in range(num_fake_ai_constructs):
        assert cut.ai_constructs[i].apriori_training.call_count == 1
        assert cut.ai_constructs[i].apriori_training.call_args_list[0].args == (arg_batch_data, )
    assert result == None

# render_reasoning tests
def test_LearnersInterface_render_reasoning_returns_empty_dict_when_instance_ai_constructs_is_empty(mocker):
    # Arrange
    cut = LearnersInterface.__new__(LearnersInterface)
    cut.ai_constructs = []

    # Act
    result = cut.render_reasoning()

    # Assert
    assert result == {}

def test_LearnersInterface_render_reasoning_returns_dict_of_each_ai_construct_as_key_to_the_result_of_its_render_reasoning_when_instance_ai_constructs_is_occupied(mocker):
    # Arrange
    cut = LearnersInterface.__new__(LearnersInterface)
    cut.ai_constructs = []

    expected_result = {}

    num_fake_ai_constructs = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_ai_constructs):
        fake_ai_construct = MagicMock()
        forced_return_ai_construct_render_reasoning = MagicMock()
        cut.ai_constructs.append(fake_ai_construct)
        mocker.patch.object(fake_ai_construct, 'render_reasoning', return_value=forced_return_ai_construct_render_reasoning)
        fake_ai_construct.component_name = MagicMock()
        expected_result[fake_ai_construct.component_name] = forced_return_ai_construct_render_reasoning

    # Act
    result = cut.render_reasoning()

    # Assert
    for i in range(num_fake_ai_constructs):
        assert cut.ai_constructs[i].render_reasoning.call_count == 1
        assert cut.ai_constructs[i].render_reasoning.call_args_list[0].args == ()
    assert result == expected_result