""" Test GenericIntelligenceConstruct [Abstract Class] Functionality """
import pytest

import src.data_driven_components.generic_intelligence_construct as generic_intelligence_construct
from src.data_driven_components.generic_intelligence_construct import GenericIntelligenceConstruct

class FakeIntelligenceConstruct(GenericIntelligenceConstruct):

    def __init__(self):
        return super().__init__()

    def apriori_training(self):
        return super().apriori_training()

    def update(self):
        return super().update()

    def render_diagnosis(self):
        return super().render_diagnosis()

# abstract methods tests
def test_GenericIntelligenceConstruct_has_expected_abstract_methods():
    # Arrange - None
    # Act
    with pytest.raises(TypeError) as e_info:
        cut = GenericIntelligenceConstruct.__new__(GenericIntelligenceConstruct)
    
    # Assert
    assert "Can't instantiate abstract class GenericIntelligenceConstruct with" in e_info.__str__()
    assert "__init__" in e_info.__str__()
    assert "apriori_training" in e_info.__str__()
    assert "update" in e_info.__str__()
    assert "render_diagnosis" in e_info.__str__()

# super call tests
def test_GenericIntelligenceConstruct__init__returns_None():
    # Arrange
    fake_ic = FakeIntelligenceConstruct.__new__(FakeIntelligenceConstruct)

    # Act
    result = fake_ic.__init__()
    
    # Assert
    assert result == None

def test_GenericIntelligenceConstruct_apriori_training_returns_None():
    # Arrange
    fake_ic = FakeIntelligenceConstruct.__new__(FakeIntelligenceConstruct)

    # Act
    result = fake_ic.apriori_training()
    
    # Assert
    assert result == None

def test_GenericIntelligenceConstruct_update_returns_None():
    # Arrange
    fake_ic = FakeIntelligenceConstruct.__new__(FakeIntelligenceConstruct)

    # Act
    result = fake_ic.update()
    
    # Assert
    assert result == None

def test_GenericIntelligenceConstruct_render_diagnosis_returns_None(mocker):
    # Arrange
    fake_ic = FakeIntelligenceConstruct.__new__(FakeIntelligenceConstruct)

    # Act
    result = fake_ic.render_diagnosis()
    
    # Assert
    assert result == None

