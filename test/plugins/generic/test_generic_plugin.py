# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""Test Generic Plugin Functionality"""
import pytest
from unittest.mock import MagicMock

from plugins.generic.generic_plugin import Plugin


def test_update_does_nothing():
    # Arrange
    cut = Plugin.__new__(Plugin)

    # Act
    result = cut.update()

    # Assert
    assert result == None


def test_render_reasoning_does_nothing():
    # Arrange
    cut = Plugin.__new__(Plugin)

    # Act
    result = cut.render_reasoning()

    # Assert
    assert result == None
