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
from unittest.mock import MagicMock

import onair.src.ai_components.learners_interface as learners_interface
from onair.src.ai_components.learners_interface import LearnersInterface

# __init__ tests
def test_LearnersInterface__init__raises_AssertionError_when_given_headers_len_is_0():
    # Arrange
    arg_headers = MagicMock()

    arg_headers.__len__.return_value = 0

    cut = LearnersInterface.__new__(LearnersInterface)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg_headers)

    # Assert
    assert e_info.match('Headers are required')

def test_LearnersInterface__init__sets_self_headers_to_given_headers_and_sets_self_learner_constructs_to_return_value_of_import_plugins(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg__learner_plugins = MagicMock()

    arg_headers.__len__.return_value = 1

    forced_return_learner_constructs = MagicMock()

    mocker.patch(learners_interface.__name__ + '.import_plugins', return_value=forced_return_learner_constructs)


    cut = LearnersInterface.__new__(LearnersInterface)

    # Act
    cut.__init__(arg_headers, arg__learner_plugins)

    # Assert
    assert cut.headers == arg_headers
    assert learners_interface.import_plugins.call_count == 1
    assert learners_interface.import_plugins.call_args_list[0].args == (arg_headers, arg__learner_plugins)
    assert cut.learner_constructs == forced_return_learner_constructs

# update tests
def test_LearnersInterface_update_does_nothing_when_instance_learner_constructs_is_empty():
    # Arrange
    arg_low_level_data = MagicMock()
    arg_high_level_data = MagicMock()

    cut = LearnersInterface.__new__(LearnersInterface)
    cut.learner_constructs = []

    # Act
    result = cut.update(arg_low_level_data, arg_high_level_data)

    # Assert
    assert result == None

def test_LearnersInterface_update_calls_update_with_given_low_level_and_high_level_data_on_each_learner_constructs_item(mocker):
    # Arrange
    arg_low_level_data = MagicMock()
    arg_high_level_data = MagicMock()

    cut = LearnersInterface.__new__(LearnersInterface)
    cut.learner_constructs = []

    num_fake_learner_constructs = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_learner_constructs):
        cut.learner_constructs.append(MagicMock())

    # Act
    result = cut.update(arg_low_level_data, arg_high_level_data)

    # Assert
    for i in range(num_fake_learner_constructs):
        assert cut.learner_constructs[i].update.call_count == 1
        assert cut.learner_constructs[i].update.call_args_list[0].args == (arg_low_level_data, arg_high_level_data)

# check_for_salient_event
def test_LearnersInterface_salient_event_does_nothing():
    # Arrange
    cut = LearnersInterface.__new__(LearnersInterface)

    # Act
    result = cut.check_for_salient_event()

    # Assert
    assert result == None

# render_reasoning tests
def test_LearnersInterface_render_reasoning_returns_empty_dict_when_instance_learner_constructs_is_empty(mocker):
    # Arrange
    cut = LearnersInterface.__new__(LearnersInterface)
    cut.learner_constructs = []

    # Act
    result = cut.render_reasoning()

    # Assert
    assert result == {}

def test_LearnersInterface_render_reasoning_returns_dict_of_each_ai_construct_as_key_to_the_result_of_its_render_reasoning_when_instance_learner_constructs_is_occupied(mocker):
    # Arrange
    cut = LearnersInterface.__new__(LearnersInterface)
    cut.learner_constructs = []

    expected_result = {}

    num_fake_learner_constructs = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_learner_constructs):
        fake_ai_construct = MagicMock()
        forced_return_ai_construct_render_reasoning = MagicMock()
        cut.learner_constructs.append(fake_ai_construct)
        mocker.patch.object(fake_ai_construct, 'render_reasoning', return_value=forced_return_ai_construct_render_reasoning)
        fake_ai_construct.component_name = MagicMock()
        expected_result[fake_ai_construct.component_name] = forced_return_ai_construct_render_reasoning

    # Act
    result = cut.render_reasoning()

    # Assert
    for i in range(num_fake_learner_constructs):
        assert cut.learner_constructs[i].render_reasoning.call_count == 1
        assert cut.learner_constructs[i].render_reasoning.call_args_list[0].args == ()
    assert result == expected_result