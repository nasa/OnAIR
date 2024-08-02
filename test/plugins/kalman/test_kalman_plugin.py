# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test Kalman Plugin Functionality """
import pytest
from unittest.mock import MagicMock
import onair
import numpy as np

from plugins.kalman import kalman_plugin
from plugins.kalman.kalman_plugin import Plugin as Kalman

# test init
def test_Kalman__init__initializes_variables_to_expected_values_when_given_all_args_except_window_size_and_residual_threshold(mocker):
    # Arrange
    arg_name = MagicMock()
    arg_headers = [MagicMock(), MagicMock()]

    fake_var = MagicMock()
    class Fake_KalmanFilter():
        def __init__(self, state_transition, process_noise, observation_model, observation_noise):
            self.test_var = fake_var
            self.state_transition = state_transition
            self.process_noise = process_noise
            self.observation_model = observation_model
            self.observation_noise = observation_noise

    forced_diag_return_value = MagicMock()
    forced_array_return_value = MagicMock()
    mocker.patch(kalman_plugin.__name__ + '.simdkalman.KalmanFilter', Fake_KalmanFilter)
    mocker.patch(kalman_plugin.__name__ + '.np.diag', return_value=forced_diag_return_value)
    mocker.patch(kalman_plugin.__name__ + '.np.array', return_value=forced_array_return_value)

    cut = Kalman.__new__(Kalman)

    # Act
    cut.__init__(arg_name, arg_headers)

    # Assert
    assert cut.frames == []
    assert cut.component_name == arg_name
    assert cut.headers == arg_headers
    assert cut.window_size == 15
    assert cut.residual_threshold == 1.5
    assert isinstance(cut.kf, Fake_KalmanFilter)
    assert cut.kf.test_var == fake_var
    assert cut.kf.state_transition == [[1,1],[0,1]]
    assert cut.kf.process_noise == forced_diag_return_value
    assert cut.kf.observation_model == forced_array_return_value
    assert cut.kf.observation_noise == 1.0

def test_Kalman__init__initializes_variables_to_expected_values_when_given_all_args(mocker):
    # Arrange
    arg_name = MagicMock()
    arg_headers = [MagicMock(), MagicMock()]
    arg_window_size = MagicMock()
    arg_residual_threshold = MagicMock()

    fake_var = MagicMock()
    class Fake_KalmanFilter():
        def __init__(self, state_transition, process_noise, observation_model, observation_noise):
            self.test_var = fake_var
            self.state_transition = state_transition
            self.process_noise = process_noise
            self.observation_model = observation_model
            self.observation_noise = observation_noise

    forced_diag_return_value = MagicMock()
    forced_array_return_value = MagicMock()
    mocker.patch(kalman_plugin.__name__ + '.simdkalman.KalmanFilter', Fake_KalmanFilter)
    mocker.patch(kalman_plugin.__name__ + '.np.diag', return_value=forced_diag_return_value)
    mocker.patch(kalman_plugin.__name__ + '.np.array', return_value=forced_array_return_value)

    cut = Kalman.__new__(Kalman)

    # Act
    cut.__init__(arg_name, arg_headers, arg_window_size, arg_residual_threshold)

    # Assert
    assert cut.frames == []
    assert cut.component_name == arg_name
    assert cut.headers == arg_headers
    assert cut.window_size == arg_window_size
    assert cut.residual_threshold == arg_residual_threshold
    assert isinstance(cut.kf, Fake_KalmanFilter)
    assert cut.kf.test_var == fake_var
    assert cut.kf.state_transition == [[1,1],[0,1]]
    assert cut.kf.process_noise == forced_diag_return_value
    assert cut.kf.observation_model == forced_array_return_value
    assert cut.kf.observation_noise == 1.0

# test update
def test_Kalman_update_does_not_mutate_frames_attribute_when_arg_frame_is_empty():
    # Arrange
    fake_frames = MagicMock()
    arg_frame = []

    cut = Kalman.__new__(Kalman)
    cut.frames = fake_frames

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == fake_frames

def test_Kalman_update_mutates_frames_attribute_as_expected_when_frames_is_empty_and_arg_frame_is_not_empty():
    # Arrange
    fake_frames = []
    len_arg_frame = pytest.gen.randint(1, 10) # arbitrary, random integer from 1 to 10
    arg_frame = [pytest.gen.randint(-10, 10)] * len_arg_frame

    cut = Kalman.__new__(Kalman)
    cut.frames = fake_frames

    expected_result = []
    for data_pt in arg_frame:
        expected_result.append([float(data_pt)])

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == expected_result

def test_Kalman_update_mutates_frames_attribute_as_expected_when_both_frames_and_arg_frame_are_not_empty_and_len_arg_frame_less_than_len_frames():
    # Arrange
    len_fake_frames = pytest.gen.randint(6, 10) # arbitrary int greater than max len of arg_frame, from 6 to 10
    fake_frames = [[MagicMock()]] * len_fake_frames
    fake_window_size = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10

    len_arg_frame = pytest.gen.randint(1, 5) # arbitrary, random int from 1 to 5
    arg_frame = [MagicMock()] * len_arg_frame

    cut = Kalman.__new__(Kalman)
    cut.frames = fake_frames
    cut.window_size = fake_window_size

    expected_result = fake_frames.copy()

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == expected_result

# test render diagnosis
def test_Kalman_render_reasoning_returns_values_above_threshold(mocker):
    # Arrange

    cut = Kalman.__new__(Kalman)
    cut.frames = ["fake_frames"] * 10
    cut.headers = ["fake_headers"] * 10
    cut.residual_threshold = pytest.gen.randint(5, 10)
    forced__generate_residuals_return = np.array([pytest.gen.randint(cut.residual_threshold + 1, cut.residual_threshold + 10)] * 10)

    mocker.patch.object(cut, '_generate_residuals', return_value=forced__generate_residuals_return)

    # Act
    result = cut.render_reasoning()

    # Assert
    assert len(result) == len(forced__generate_residuals_return)

def test_Kalman_render_reasoning_returns_values_below_threshold(mocker):
    # Arrange

    cut = Kalman.__new__(Kalman)
    cut.frames = ["fake_frames"] * 10
    cut.headers = ["fake_headers"] * 10
    cut.residual_threshold = pytest.gen.randint(5, 10)
    forced__generate_residuals_return = np.array([pytest.gen.randint(0, cut.residual_threshold - 1)] * 10)

    mocker.patch.object(cut, '_generate_residuals', return_value=forced__generate_residuals_return)

    # Act
    result = cut.render_reasoning()

    # Assert
    assert len(result) == 0

# test predict
def test_Kalman__predict_smoothes_data_and_predicts_result_using_KalmanFilter_functions_as_expected_when_initial_val_is_not_set(mocker):
    # Arrange
    arg_subframe = []
    arg_forward_steps = MagicMock()

    fake_kf = MagicMock()

    forced_predict_return_value = MagicMock()
    mocker.patch.object(fake_kf, 'smooth')
    mocker.patch.object(fake_kf, 'predict', return_value=forced_predict_return_value)

    cut = Kalman.__new__(Kalman)
    cut.kf = fake_kf

    # Act
    result = cut._predict(arg_subframe, arg_forward_steps)

    # Assert
    assert result == forced_predict_return_value
    assert fake_kf.smooth.call_count == 1
    assert fake_kf.smooth.call_args_list[0].args == (arg_subframe, )
    assert fake_kf.predict.call_count == 1
    assert fake_kf.predict.call_args_list[0].args == (arg_subframe, arg_forward_steps)

def test_Kalman__predict_smoothes_data_and_predicts_result_using_KalmanFilter_functions_as_expected_when_initial_val_is_set(mocker):
    # Arrange
    arg_subframe = []
    arg_forward_steps = MagicMock()
    arg_initial_val = pytest.gen.uniform(-10.0, 10.0) # arbitrary, random float from -10.0 to 10.0

    fake_kf = MagicMock()

    forced_predict_return_value = MagicMock()
    mocker.patch.object(fake_kf, 'smooth')
    mocker.patch.object(fake_kf, 'predict', return_value=forced_predict_return_value)

    cut = Kalman.__new__(Kalman)
    cut.kf = fake_kf

    # Act
    result = cut._predict(arg_subframe, arg_forward_steps, arg_initial_val)

    # Assert
    assert result == forced_predict_return_value
    assert fake_kf.smooth.call_count == 1
    assert fake_kf.smooth.call_args_list[0].args == (arg_subframe, )
    assert fake_kf.smooth.call_args_list[0].kwargs == {'initial_value' : arg_initial_val}
    assert fake_kf.predict.call_count == 1
    assert fake_kf.predict.call_args_list[0].args == (arg_subframe, arg_forward_steps)
