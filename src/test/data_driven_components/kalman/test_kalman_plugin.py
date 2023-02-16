""" Test Kalman Plugin Functionality """

import pytest
from mock import MagicMock
import src.data_driven_components.kalman.kalman_plugin as kalman
from src.data_driven_components.kalman.kalman_plugin import Kalman

import importlib

# test init
def test_Kalman__init__initializes_variables_to_expected_values_when_given_all_args_except_window_size(mocker):
    # Arrange
    arg_name = MagicMock()
    arg_headers = MagicMock()

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
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.simdkalman.KalmanFilter', Fake_KalmanFilter)
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.np.diag', return_value=forced_diag_return_value)
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.np.array', return_value=forced_array_return_value)

    cut = Kalman.__new__(Kalman)

    # Act
    cut.__init__(arg_name, arg_headers)

    # Assert
    assert cut.frames == []
    assert cut.component_name == arg_name
    assert cut.headers == arg_headers
    assert cut.window_size == 3
    assert isinstance(cut.kf, Fake_KalmanFilter)
    assert cut.kf.test_var == fake_var
    assert cut.kf.state_transition == [[1,1],[0,1]]
    assert cut.kf.process_noise == forced_diag_return_value
    assert cut.kf.observation_model == forced_array_return_value
    assert cut.kf.observation_noise == 1.0
  
def test_Kalman__init__initializes_variables_to_expected_values_when_given_all_args(mocker):
    # Arrange
    arg_name = MagicMock()
    arg_headers = MagicMock()
    arg_window_size = MagicMock()

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
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.simdkalman.KalmanFilter', Fake_KalmanFilter)
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.np.diag', return_value=forced_diag_return_value)
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.np.array', return_value=forced_array_return_value)

    cut = Kalman.__new__(Kalman)

    # Act
    cut.__init__(arg_name, arg_headers, arg_window_size)

    # Assert
    assert cut.frames == []
    assert cut.component_name == arg_name
    assert cut.headers == arg_headers
    assert cut.window_size == arg_window_size
    assert isinstance(cut.kf, Fake_KalmanFilter)
    assert cut.kf.test_var == fake_var
    assert cut.kf.state_transition == [[1,1],[0,1]]
    assert cut.kf.process_noise == forced_diag_return_value
    assert cut.kf.observation_model == forced_array_return_value
    assert cut.kf.observation_noise == 1.0
    
# test apiori training
def test_Kalman_apiori_training_returns_none():
    # Arrange
    cut = Kalman.__new__(Kalman)

    # Act
    result = cut.apriori_training()  

    # Assert
    assert result == None

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
    arg_frame = [MagicMock()] * len_arg_frame

    cut = Kalman.__new__(Kalman)
    cut.frames = fake_frames

    expected_result = []
    for data_pt in arg_frame:
        expected_result.append([data_pt])

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == expected_result

def test_Kalman_update_mutates_frames_attribute_as_expected_when_both_frames_and_arg_frame_are_not_empty_and_len_arg_frame_greater_than_len_frames():
    # Arrange
    len_fake_frames = pytest.gen.randint(1, 5) # arbitrary, random int from 1 to 5
    fake_frames = [[MagicMock()]] * len_fake_frames
    fake_window_size = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    
    len_arg_frame = pytest.gen.randint(6, 10) # arbitrary int greater than max len of fake_frames, from 6 to 10
    arg_frame = [MagicMock()] * len_arg_frame

    cut = Kalman.__new__(Kalman)
    cut.frames = fake_frames
    cut.window_size = fake_window_size

    len_dif = len_arg_frame - len_fake_frames
    expected_result = fake_frames.copy()

    for i in range(len_dif):
        expected_result.append([arg_frame[i]])

    for i in range(len_dif, len_arg_frame):
        expected_result[i].append(arg_frame[i])
        if len(expected_result[i]) > fake_window_size:
            expected_result[i].pop(0)

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
    for i in range(len_arg_frame):
        expected_result[i].append(arg_frame[i])
        if len(expected_result[i]) > fake_window_size:
            expected_result[i].pop(0)

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == expected_result


def test_Kalman_update_pops_first_index_of_frames_data_points_when_window_size_is_exceeded():
    # Arrange
    len_fake_frames = pytest.gen.randint(6, 10) # arbitrary int greater than max len of arg_frame, from 6 to 10
                                                # choosing to keep len of fake_frames greater than arg_frame in order to guarantee 'popping'
    fake_frames = [[MagicMock()]] * len_fake_frames
    fake_window_size = 1 # arbitrary, chosen to guarantee 'popping'

    len_arg_frame = pytest.gen.randint(1, 5) # arbitrary, random int from 1 to 5
    arg_frame = [MagicMock()] * len_arg_frame

    cut = Kalman.__new__(Kalman)
    cut.frames = fake_frames
    cut.window_size = fake_window_size

    expected_result = fake_frames.copy()

    for i in range(len_arg_frame):
        expected_result[i].append(arg_frame[i])
        expected_result[i].pop(0)

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == expected_result

# test render diagnosis
def test_Kalman_render_diagnosis_returns_value_returned_by_frame_diagnosis_function(mocker):
    # Arrange
    fake_frames = MagicMock()
    fake_headers = MagicMock()
    forced_frame_diagnose_return = MagicMock()
    
    cut = Kalman.__new__(Kalman)
    cut.frames = fake_frames
    cut.headers = fake_headers

    mocker.patch.object(cut, 'frame_diagnosis', return_value=forced_frame_diagnose_return)
    
    # Act
    result = cut.render_diagnosis()

    # Assert
    assert result == forced_frame_diagnose_return

# test mean
def test_Kalman_():
    # Arrange

    # Act

    # Assert
    assert True

# test residual
def test_Kalman_():
    # Arrange

    # Act

    # Assert
    assert True

# test std_dev
def test_Kalman_():
    # Arrange

    # Act

    # Assert
    assert True
    
# test predict
def test_Kalman_():
    # Arrange

    # Act

    # Assert
    assert True
    
# test predictions_for_given_data
def test_Kalman_():
    # Arrange

    # Act

    # Assert
    assert True
    
# test generate_residuals_for_given_data
def test_Kalman_():
    # Arrange

    # Act

    # Assert
    assert True
    
# test current_attribute_chunk_get_error
def test_Kalman_():
    # Arrange

    # Act

    # Assert
    assert True