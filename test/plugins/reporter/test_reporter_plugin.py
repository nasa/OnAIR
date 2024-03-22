# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test Reporter Plugin Functionality """
import pytest
from unittest.mock import MagicMock
import onair

from plugins.reporter import reporter_plugin
from plugins.reporter.reporter_plugin import Plugin as Reporter

# test update
def test_Reporter_update_saves_given_args_and_only_outputs_update_when_not_verbose_mode(mocker):
    # Arrange
    arg_low_level_data = MagicMock(name='arg_low_level_data')
    arg_high_level_data =  MagicMock(name='arg_high_level_data')

    cut = Reporter.__new__(Reporter)
    cut.component_name = MagicMock(name='fake.cut.component_name')

    mocker.patch(reporter_plugin.__name__ + '.print')

    # Act
    cut.update(arg_low_level_data, arg_high_level_data)

    # Assert
    assert reporter_plugin.print.call_count == 1
    assert reporter_plugin.print.call_args_list[0].args == (f"{cut.component_name}: UPDATE", )
    assert cut.low_level_data == arg_low_level_data
    assert cut.high_level_data == arg_high_level_data

def test_Reporter_update_saves_given_args_and_outputs_all_info_when_verbose_mode(mocker):
    # Arrange
    arg_low_level_data = MagicMock(name='arg_low_level_data')
    arg_high_level_data =  MagicMock(name='arg_high_level_data')

    cut = Reporter.__new__(Reporter)
    cut.component_name = MagicMock(name='fake.cut.component_name')
    cut.headers = MagicMock(name='fake.cut.headers')
    cut.verbose_mode = True

    mocker.patch(reporter_plugin.__name__ + '.print')

    # Act
    cut.update(arg_low_level_data, arg_high_level_data)

    # Assert
    assert reporter_plugin.print.call_count == 4
    assert reporter_plugin.print.call_args_list[0].args == (f"{cut.component_name}: UPDATE", )
    assert reporter_plugin.print.call_args_list[1].args == (f" : headers {cut.headers}", )
    assert reporter_plugin.print.call_args_list[2].args == (f" : low_level_data {arg_low_level_data.__class__} = '{arg_low_level_data}'", )
    assert reporter_plugin.print.call_args_list[3].args == (f" : high_level_data {arg_high_level_data.__class__} = '{arg_high_level_data}'", )
    assert cut.low_level_data == arg_low_level_data
    assert cut.high_level_data == arg_high_level_data

# test render_reasoning
def test_Reporter_render_reasoning_only_outputs_render_reasoning_when_not_verbose_mode(mocker):
    # Arrange
    cut = Reporter.__new__(Reporter)
    cut.component_name = MagicMock(name='fake.cut.component_name')
    cut.verbose_mode = False

    mocker.patch(reporter_plugin.__name__ + '.print')

    # Act
    cut.render_reasoning()

    # Assert
    assert reporter_plugin.print.call_count == 1
    assert reporter_plugin.print.call_args_list[0].args == (f"{cut.component_name}: RENDER_REASONING", )

def test_Reporter_render_reasoning_outputs_all_info_when_verbose_mode(mocker):
    # Arrange
    cut = Reporter.__new__(Reporter)
    cut.component_name = MagicMock(name='fake.cut.component_name')
    fake_low_level_data = MagicMock(name='fake_low_level_data')
    cut.low_level_data = fake_low_level_data
    fake_high_level_data = MagicMock(name='fake_high_level_data')
    cut.high_level_data = fake_high_level_data
    cut.verbose_mode = True

    mocker.patch(reporter_plugin.__name__ + '.print')

    # Act
    cut.render_reasoning()

    # Assert
    assert reporter_plugin.print.call_count == 3
    assert reporter_plugin.print.call_args_list[0].args == (f"{cut.component_name}: RENDER_REASONING", )
    assert reporter_plugin.print.call_args_list[1].args == (f" : My low_level_data is {fake_low_level_data}", )
    assert reporter_plugin.print.call_args_list[2].args == (f" : My high_level_data is {fake_high_level_data}", )
