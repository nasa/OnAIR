# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"
#
# NOTE: For testing singleton-like classes, a teardown procedure must be implemented
# to delete the instance after every test. Otherwise, proceeding tests will have
# access to the last test's instance. This happens due to the nature of singletons,
# which have a single instance per global scope (which the tests are running in).
#
import pytest
from onair.src.util.singleton import Singleton


def test_Singleton_creates_only_one_instance():
    # Arrange/Act
    instance1 = Singleton()
    instance2 = Singleton()

    # Assert
    assert instance1 is instance2

    # Teardown
    del Singleton.instance


def test_Singleton_with_inheritance():
    # Arrange
    class DerivedSingleton(Singleton):
        pass

    # Act
    instance1 = DerivedSingleton()
    instance2 = DerivedSingleton()

    # Assert
    assert instance1 is instance2

    # Teardown
    del DerivedSingleton.instance


def test_Singleton_maintains_state():
    # Arrange
    instance1 = Singleton()
    instance1.data = "test data"

    # Act
    instance2 = Singleton()

    # Assert
    assert instance2.data == "test data"

    # Teardown
    del Singleton.instance
