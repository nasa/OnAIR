""" Test Generic Component Core (abstract class) Functionality """
import pytest
from mock import MagicMock


import src.data_driven_components.generic_component.core as core
from src.data_driven_components.generic_component.core import AIPlugIn

class FakeAIPlugIn(AIPlugIn):
    def __init__(self, _name, _headers):
        return super().__init__(_name, _headers)

    def apriori_training(self):
        return None

    def update(self):
        return None

    def render_diagnosis(self):
        return dict()

class IncompleteFakeAIPlugIn(AIPlugIn):
    def __init__(self, _name, _headers):
        return super().__init__(_name, _headers)

# abstract methods tests
def test_AIPlugIn_raises_error_because_of_unimplemented_abstract_methods():
    # Arrange - None
    # Act
    with pytest.raises(TypeError) as e_info:
        cut = AIPlugIn.__new__(AIPlugIn)
    
    # Assert
    assert "Can't instantiate abstract class AIPlugIn with" in e_info.__str__()
    assert "apriori_training" in e_info.__str__()
    assert "update" in e_info.__str__()
    assert "render_diagnosis" in e_info.__str__()

def test_Incomplete_AIPlugIn_raises_error_because_of_unimplemented_abstract_methods():
    # Arrange - None
    # Act
    with pytest.raises(TypeError) as e_info:
        cut = IncompleteFakeAIPlugIn.__new__(IncompleteFakeAIPlugIn)
    
    # Assert
    assert "Can't instantiate abstract class IncompleteFakeAIPlugIn with" in e_info.__str__()
    assert "apriori_training" in e_info.__str__()
    assert "update" in e_info.__str__()
    assert "render_diagnosis" in e_info.__str__()

def test_Complete_AIPlugIn_does_not_raise_error_because_abstract_methods_are_implemented():
    # Arrange
    exception_raised = False
    try:
        fake_ic = FakeAIPlugIn.__new__(FakeAIPlugIn)
    except:
        exception_raised = True
    
    # Assert
    assert exception_raised == False

# Complete plugin call tests

# __init__ tests
def test_AIPlugIn__init__raises_assertion_error_when_given__headers_len_is_not_greater_than_0():
    # Arrange
    arg__name = MagicMock()
    arg__headers = []

    cut = FakeAIPlugIn.__new__(FakeAIPlugIn)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg__name, arg__headers)

    # Assert
    assert e_info.match('')

def test_AIPlugIn__init__sets_instance_values_to_given_args_when_given__headers_len_is_greater_than_0(mocker):
    # Arrange
    arg__name = MagicMock()
    arg__headers = MagicMock()

    cut = FakeAIPlugIn.__new__(FakeAIPlugIn)

    mocker.patch('src.data_driven_components.generic_component.core.len', return_value=pytest.gen.randint(1, 200)) # arbitrary, from 1 to 200 (but > 0)

    # Act
    cut.__init__(arg__name, arg__headers)

    # Assert
    assert core.len.call_count == 1
    assert core.len.call_args_list[0].args == (arg__headers,)
    assert cut.component_name == arg__name
    assert cut.headers == arg__headers
