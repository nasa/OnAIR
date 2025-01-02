# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import pytest
from unittest.mock import MagicMock

@pytest.fixture
def Singleton():
    from onair.src.util.singleton import Singleton
    yield Singleton

def test_Singleton__new__returns_new_instance_when_class_does_not_have_instance_attribute(
    mocker,
    Singleton
):
    # Arrange
    # TestSingleton inherits Singleton
    class TestSingleton(Singleton):
        pass

    # special assert to show TestSingleton does not have instance
    assert not hasattr(TestSingleton, "instance")

    # NotSingleton is not a Singleton
    class NotSingleton:
        pass

    NotSingleton.instance = MagicMock()
    # special assert to show NotSingleton has instance
    assert hasattr(NotSingleton, "instance")

    # fake_test_singleton is an instance of NotSingleton
    fake_test_singleton = NotSingleton()
    # true_class is added to track what it really is
    fake_test_singleton.true_class = NotSingleton
    # __class__ is set to TestSingleton to fool system to think it is one
    fake_test_singleton.__class__ = TestSingleton

    # Mock Singleton's usage of super to return a NotSingleton class object
    mock_super = mocker.patch(
        "onair.src.util.singleton.super", return_value=NotSingleton
    )
    # Mock NotSingleton's __new__ call to return our fake_test_singleton
    mock_new = mocker.patch.object(
        NotSingleton, "__new__", return_value=fake_test_singleton
    )

    # Act
    test_singleton = TestSingleton()

    # Assert
    assert mock_super.call_count == 1
    assert mock_super.call_args_list[0].args == (Singleton, TestSingleton)
    assert isinstance(test_singleton, TestSingleton)
    assert hasattr(test_singleton, "true_class")
    assert test_singleton.true_class == NotSingleton
    assert mock_new.call_count == 1
    assert mock_new.call_args_list[0].args == (TestSingleton,)
    assert TestSingleton.instance is test_singleton


def test_Singleton__new__returns_singleton_instance_when_class_has_instance_attribute(
    mocker,
    Singleton
):
    # Arrange
    # TestSingleton inherits Singleton
    class TestSingleton(Singleton):
        pass

    fake_instance = object()
    TestSingleton.instance = fake_instance
    # special assert to show TestSingleton has instance
    assert hasattr(TestSingleton, "instance")

    # NotSingleton is not a Singleton
    class NotSingleton:
        pass

    # fake_test_singleton is an instance of NotSingleton
    fake_test_singleton = NotSingleton()
    # true_class is added to track what it really is
    fake_test_singleton.true_class = NotSingleton
    # __class__ is set to TestSingleton to fool system to think it is one
    fake_test_singleton.__class__ = TestSingleton

    # Mock Singleton's usage of super to return a NotSingleton class object
    mock_super = mocker.patch(
        "onair.src.util.singleton.super", return_value=NotSingleton
    )
    # Mock NotSingleton's __new__ call to return our fake_test_singleton
    mock_new = mocker.patch.object(
        NotSingleton, "__new__", return_value=fake_test_singleton
    )

    # Act
    test_singleton = TestSingleton()

    # Assert
    assert mock_super.call_count == 0
    assert mock_new.call_count == 0
    assert not isinstance(test_singleton, TestSingleton)
    assert not hasattr(test_singleton, "true_class")
    assert test_singleton is fake_instance
