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
from unittest.mock import MagicMock

import onair.src.reasoning.complex_reasoning_interface as complex_reasoning_interface
from onair.src.reasoning.complex_reasoning_interface import ComplexReasoningInterface

# __init__ tests
def test_ComplexReasoningInterface__init__raises_AssertionError_when_given_headers_len_is_0():
    # Arrange
    arg_headers = MagicMock()

    arg_headers.__len__.return_value = 0

    cut = ComplexReasoningInterface.__new__(ComplexReasoningInterface)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg_headers)

    # Assert
    assert e_info.match('Headers are required')

def test_ComplexReasoningInterface__init__sets_self_headers_to_given_headers_and_sets_self_reasoning_constructs_to_return_value_of_import_plugins(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg__reasoning_plugins = MagicMock()

    arg_headers.__len__.return_value = 1

    forced_return_reasoning_constructs = MagicMock()

    mocker.patch(complex_reasoning_interface.__name__ + '.import_plugins', return_value=forced_return_reasoning_constructs)


    cut = ComplexReasoningInterface.__new__(ComplexReasoningInterface)

    # Act
    cut.__init__(arg_headers, arg__reasoning_plugins)

    # Assert
    assert cut.headers == arg_headers
    assert complex_reasoning_interface.import_plugins.call_count == 1
    assert complex_reasoning_interface.import_plugins.call_args_list[0].args == (arg_headers, arg__reasoning_plugins)
    assert cut.reasoning_constructs == forced_return_reasoning_constructs

# update_and_render_reasoning
def test_ComplexReasoningInterface_update_and_render_reasoning_returns_given_high_level_data_with_complex_systems_as_empty_dict_when_no_reasoning_constructs(mocker):
    # Arrange
    fake_high_level_key = MagicMock(name='fake_high_level_key')
    fake_high_level_value = MagicMock(name='fake_high_level_value')
    arg_high_level_data = {fake_high_level_key:fake_high_level_value}
    expected_result = arg_high_level_data
    expected_result.update({'complex_systems':{}})

    cut = ComplexReasoningInterface.__new__(ComplexReasoningInterface)
    cut.reasoning_constructs = []

    # Act
    result = cut.update_and_render_reasoning(arg_high_level_data)

    # Assert
    assert result == expected_result

def test_ComplexReasoningInterface_update_and_render_reasoning_invokes_on_all_reasoning_constructs_then_returns_their_results_added_to_the_hgih_level_data(mocker):
    # Arrange
    fake_high_level_key = MagicMock(name='fake_high_level_key')
    fake_high_level_value = MagicMock(name='fake_high_level_value')
    arg_high_level_data = {fake_high_level_key:fake_high_level_value}
    expected_result = arg_high_level_data
    expected_result.update({'complex_systems':{}})

    cut = ComplexReasoningInterface.__new__(ComplexReasoningInterface)
    cut.reasoning_constructs = []
    for i in range(0, pytest.gen.randint(1, 10)):
        cut.reasoning_constructs.append(MagicMock(name=f"fake_plugin_{i}"))
        cut.reasoning_constructs[-1].component_name = f"fake_plugin_{i}"
        mocker.patch.object(cut.reasoning_constructs[-1], 'update')
        rv = f"{i}"
        mocker.patch.object(cut.reasoning_constructs[-1], 'render_reasoning', return_value=rv)
        expected_result['complex_systems'].update({cut.reasoning_constructs[-1].component_name : rv})

    # Act
    result = cut.update_and_render_reasoning(arg_high_level_data)

    # Assert
    assert result == expected_result
    assert cut.reasoning_constructs[0].update.call_count == 1

# check_for_salient_event tests
def test_ComplexReasoningInterface_salient_event_does_nothing():
    # Arrange
    cut = ComplexReasoningInterface.__new__(ComplexReasoningInterface)

    # Act
    result = cut.check_for_salient_event()

    # Assert
    assert result == None
