# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test Generic Component Core (abstract class) Functionality """
import pytest
from mock import MagicMock

import onair.src.ai_components.ai_plugin_abstract.ai_plugin as ai_plugin
from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin

class FakeAIPlugin(AIPlugin):
    def __init__(self, _name, _headers):
        return super().__init__(_name, _headers)

    def update(self):
        return None

    def render_reasoning(self):
        return dict()

class IncompleteFakeAIPlugin(AIPlugin):
    def __init__(self, _name, _headers):
        return super().__init__(_name, _headers)

class BadFakeAIPlugin(AIPlugin):
    def __init__(self, _name, _headers):
        return super().__init__(_name, _headers)

    def update(self):
        return super().update()

    def render_reasoning(self):
        return super().render_reasoning()

# abstract methods tests
def test_AIPlugin_raises_error_because_of_unimplemented_abstract_methods():
    # Arrange - None
    # Act
    with pytest.raises(TypeError) as e_info:
        cut = AIPlugin.__new__(AIPlugin)

    # Assert
    assert "Can't instantiate abstract class AIPlugin with" in e_info.__str__()
    assert "update" in e_info.__str__()
    assert "render_reasoning" in e_info.__str__()

# Incomplete plugin call tests
def test_AIPlugin_raises_error_when_an_inherited_class_is_instantiated_because_abstract_methods_are_not_implemented_by_that_class():
    # Arrange - None
    # Act
    with pytest.raises(TypeError) as e_info:
        cut = IncompleteFakeAIPlugin.__new__(IncompleteFakeAIPlugin)

    # Assert
    assert "Can't instantiate abstract class IncompleteFakeAIPlugin with" in e_info.__str__()
    assert "update" in e_info.__str__()
    assert "render_reasoning" in e_info.__str__()

def test_AIPlugin_raises_error_when_an_inherited_class_calls_abstract_methods_in_parent():
    # Act
    cut = BadFakeAIPlugin.__new__(BadFakeAIPlugin)

    # populate list with the functions that should raise exceptions when called.
    not_implemented_functions = [cut.update, cut.render_reasoning]
    for fnc in not_implemented_functions:
        with pytest.raises(NotImplementedError) as e_info:
            fnc()
        assert "NotImplementedError" in e_info.__str__()

# Complete plugin call tests
def test_AIPlugin_does_not_raise_error_when_an_inherited_class_is_instantiated_because_abstract_methods_are_implemented_by_that_class():
    # Arrange
    exception_raised = False
    try:
        fake_ic = FakeAIPlugin.__new__(FakeAIPlugin)
    except:
        exception_raised = True

    # Assert
    assert exception_raised == False

# Complete plugin call tests

# __init__ tests
def test_AIPlugin__init__raises_assertion_error_when_given__headers_len_is_not_greater_than_0():
    # Arrange
    arg__name = MagicMock()
    arg__headers = []

    cut = FakeAIPlugin.__new__(FakeAIPlugin)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg__name, arg__headers)

    # Assert
    assert e_info.match('')

def test_AIPlugin__init__sets_instance_values_to_given_args_when_given__headers_len_is_greater_than_0(mocker):
    # Arrange
    arg__name = MagicMock()
    arg__headers = MagicMock()

    cut = FakeAIPlugin.__new__(FakeAIPlugin)

    mocker.patch(ai_plugin.__name__ + '.len', return_value=pytest.gen.randint(1, 200)) # arbitrary, from 1 to 200 (but > 0)

    # Act
    cut.__init__(arg__name, arg__headers)

    # Assert
    assert ai_plugin.len.call_count == 1
    assert ai_plugin.len.call_args_list[0].args == (arg__headers,)
    assert cut.component_name == arg__name
    assert cut.headers == arg__headers
