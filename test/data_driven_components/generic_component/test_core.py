""" Test Generic Component Core (abstract class) Functionality """
import pytest
from mock import MagicMock


import src.data_driven_components.generic_component.core as core
from src.data_driven_components.generic_component.core import AIPlugIn

class FakeAIPlugIn(AIPlugIn):
    def __init__(self, _name, _headers):
        return super().__init__(_name, _headers)

    def apriori_training(self):
        return super().apriori_training()

    def update(self):
        return super().update()

    def render_diagnosis(self):
        return super().render_diagnosis()

# abstract methods tests
def test_AIPlugIn_has_expected_abstract_methods():
    # Arrange - None
    # Act
    with pytest.raises(TypeError) as e_info:
        cut = AIPlugIn.__new__(AIPlugIn)
    
    # Assert
    assert "Can't instantiate abstract class AIPlugIn with" in e_info.__str__()
    assert "apriori_training" in e_info.__str__()
    assert "update" in e_info.__str__()
    assert "render_diagnosis" in e_info.__str__()

# super call tests
def test_AIPlugIn_apriori_training_returns_None():
    # Arrange
    fake_ic = FakeAIPlugIn.__new__(FakeAIPlugIn)

    # Act
    result = fake_ic.apriori_training()
    
    # Assert
    assert result == None

def test_AIPlugIn_update_returns_None():
    # Arrange
    fake_ic = FakeAIPlugIn.__new__(FakeAIPlugIn)

    # Act
    result = fake_ic.update()
    
    # Assert
    assert result == None

def test_AIPlugIn_render_diagnosis_returns_None(mocker):
    # Arrange
    fake_ic = FakeAIPlugIn.__new__(FakeAIPlugIn)

    # Act
    result = fake_ic.render_diagnosis()
    
    # Assert
    assert result == None

# __init__ tests
def test_AIPlugIn__init__asserts_when_given__headers_len_is_less_than_0():
    # Arrange
    arg__name = MagicMock()
    arg__headers = []

    cut = FakeAIPlugIn.__new__(FakeAIPlugIn)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg__name, arg__headers)

    # Assert
    assert e_info.match('')

def test_AIPlugIn__init__returns_None_when_headers_len_is_greater_than_0():
    # Arrange
    fake_ic = FakeAIPlugIn.__new__(FakeAIPlugIn)

    # Act
    fake_headers = ["fake_item"]
    fake_name = MagicMock()
    result = fake_ic.__init__(fake_name, fake_headers)
    
    # Assert
    assert result == None

def test_AIPlugIn__init__sets_instance_values_to_given_args_when_when_given__headers_len_is_greater_than_0(mocker):
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
