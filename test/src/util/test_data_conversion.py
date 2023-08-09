# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test Data Conversion Functionality """
import pytest
from mock import MagicMock

import onair.src.util.data_conversion

from numpy import ndarray
    
# status_to_oneHot tests
def test_data_conversion_status_to_oneHot_returns_given_status_when_status_isinstance_of_np_ndarray(mocker):
    # Arrange
    arg_status = MagicMock()

    mocker.patch('onair.src.util.data_conversion.isinstance', return_value=True)

    # Act
    result = onair.src.util.data_conversion.status_to_oneHot(arg_status)

    # Assert
    assert onair.src.util.data_conversion.isinstance.call_count == 1
    assert onair.src.util.data_conversion.isinstance.call_args_list[0].args == (arg_status, ndarray)
    assert result == arg_status

def test_data_conversion_status_to_oneHot_returns_one_hot_set_to_list_of_four_zeros_and_the_value_of_the_classes_status_to_1_point_0(mocker):
    # Arrange
    arg_status = MagicMock()

    fake_status = pytest.gen.randint(0,3) # size of array choice, from 0 to 3

    expected_result = [0.0, 0.0, 0.0, 0.0]
    expected_result[fake_status] = 1.0

    onair.src.util.data_conversion.classes = {arg_status: fake_status}

    mocker.patch('onair.src.util.data_conversion.isinstance', return_value=False)

    # Act
    result = onair.src.util.data_conversion.status_to_oneHot(arg_status)

    # Assert
    assert result == expected_result

