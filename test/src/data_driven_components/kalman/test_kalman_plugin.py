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
from mock import MagicMock
import src.data_driven_components.kalman.kalman_plugin as kalman
from src.data_driven_components.kalman.kalman_plugin import Plugin as Kalman

# test init
def test_Kalman__init__initializes_variables_to_expected_values_when_given_all_args_except_window_size(mocker):
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
    arg_headers = [MagicMock(), MagicMock()]
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
                                                # choosing to keep len of fake_frames greater than arg_frame in order to guarantee 'else' statement is reached
    # fake_frames = [[MagicMock()]] * len_fake_frames
    expected_result = []
    fake_frames = []
    for i in range(len_fake_frames):
        fake_frame = [MagicMock()]
        fake_frames.append([fake_frame])
        expected_result.append([fake_frame])
    fake_window_size = 1 # arbitrary, chosen to guarantee 'popping'

    len_arg_frame = pytest.gen.randint(1, 5) # arbitrary, random int from 1 to 5
    arg_frame = []
    for i in range(len_arg_frame):
        arg_frame.append(MagicMock())

    cut = Kalman.__new__(Kalman)
    cut.frames = fake_frames
    cut.window_size = fake_window_size

    for i in range(len_arg_frame):
        expected_result[i].append(arg_frame[i])
        expected_result[i].pop(0)

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == expected_result

def test_Kalman_update_will_not_pop_first_index_of_frames_data_points_when_window_size_is_never_exceeded():
    # Arrange
    len_fake_frames = pytest.gen.randint(6, 10) # arbitrary int greater than max len of arg_frame, from 6 to 10
                                                # choosing to keep len of fake_frames greater than arg_frame in order to guarantee 'else' statement is reached
    # fake_frames = [[MagicMock()]] * len_fake_frames
    expected_result = []
    fake_frames = []
    for i in range(len_fake_frames):
        fake_frame = [MagicMock()]
        fake_frames.append([fake_frame])
        expected_result.append([fake_frame])
    fake_window_size = 99 # arbitrary, chosen to guarantee no 'popping' will occur

    len_arg_frame = pytest.gen.randint(1, 5) # arbitrary, random int from 1 to 5
    arg_frame = []
    for i in range(len_arg_frame):
        arg_frame.append(MagicMock())

    cut = Kalman.__new__(Kalman)
    cut.frames = fake_frames
    cut.window_size = fake_window_size

    for i in range(len_arg_frame):
        expected_result[i].append(arg_frame[i])

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
def test_Kalman_mean_calculates_return_value_by_dividing_sum_by_len(mocker):
    # Arrange
    arg_values = MagicMock()

    forced_sum_return_value = pytest.gen.uniform(1.0, 10.0) # arbitrary, random float from 1.0 to 10.0
    forced_len_return_value = pytest.gen.uniform(1.0, 10.0) # arbitrary, random float from 1.0 to 10.0
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.sum', return_value=forced_sum_return_value)
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.len', return_value=forced_len_return_value)

    cut = Kalman.__new__(Kalman)

    # Act
    result = cut.mean(arg_values)

    # Assert
    assert kalman.sum.call_count == 1
    assert kalman.sum.call_args_list[0].args == (arg_values, )
    assert kalman.len.call_count == 1
    assert kalman.len.call_args_list[0].args == (arg_values, )
    assert result == forced_sum_return_value/forced_len_return_value

# test residual
def test_Kalman_residual_calculates_return_value_by_finding_the_abs_difference_of_given_args(mocker):
    # Arrange
    arg_predicted = pytest.gen.uniform(-10.0, 10.0) # arbitrary, random float from -10.0 to 10.0
    arg_actual = pytest.gen.uniform(-10.0, 10.0) # arbitrary, random float from -10.0 to 10.0

    forced_abs_return_value = MagicMock()
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.abs', return_value=forced_abs_return_value)

    cut = Kalman.__new__(Kalman)

    # Act
    result = cut.residual(arg_predicted, arg_actual)

    # Assert
    assert result == forced_abs_return_value
    assert kalman.abs.call_count == 1
    assert kalman.abs.call_args_list[0].args == (arg_actual - arg_predicted, )

# test std_dev
def test_Kalman_std_dev_calculates_return_value_by_using_np_std_function_on_arg_data(mocker):
    # Arrange
    arg_data = MagicMock()

    forced_std_return_value = MagicMock()
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.np.std', return_value=forced_std_return_value)

    cut = Kalman.__new__(Kalman)

    # Act
    result = cut.std_dev(arg_data)

    # Assert
    assert result == forced_std_return_value
    assert kalman.np.std.call_count == 1
    assert kalman.np.std.call_args_list[0].args == (arg_data, )

# test predict
def test_Kalman_predict_smoothes_data_and_predicts_result_using_KalmanFilter_functions_as_expected_when_data_is_empty_and_initial_val_equals_None(mocker):
    # Arrange
    arg_data = []
    arg_forward_steps = MagicMock()
    arg_initial_val = None

    fake_kf = MagicMock()

    forced_predict_return_value = MagicMock()
    mocker.patch.object(fake_kf, 'smooth')
    mocker.patch.object(fake_kf, 'predict', return_value=forced_predict_return_value)

    cut = Kalman.__new__(Kalman)
    cut.kf = fake_kf

    # Act
    result = cut.predict(arg_data, arg_forward_steps, arg_initial_val)

    # Assert
    assert result == forced_predict_return_value
    assert fake_kf.smooth.call_count == 1
    assert fake_kf.smooth.call_args_list[0].args == (arg_data, )
    assert fake_kf.predict.call_count == 1
    assert fake_kf.predict.call_args_list[0].args == (arg_data, arg_forward_steps)
   
def test_Kalman_predict_smoothes_data_and_predicts_result_using_KalmanFilter_functions_as_expected_when_data_is_empty_and_initial_val_is_not_None(mocker):
    # Arrange
    arg_data = []
    arg_forward_steps = MagicMock()
    arg_initial_val = pytest.gen.uniform(-10.0, 10.0) # arbitrary, random float from -10.0 to 10.0

    fake_kf = MagicMock()

    forced_predict_return_value = MagicMock()
    mocker.patch.object(fake_kf, 'smooth')
    mocker.patch.object(fake_kf, 'predict', return_value=forced_predict_return_value)

    cut = Kalman.__new__(Kalman)
    cut.kf = fake_kf

    # Act
    result = cut.predict(arg_data, arg_forward_steps, arg_initial_val)

    # Assert
    assert result == forced_predict_return_value
    assert fake_kf.smooth.call_count == 1
    assert fake_kf.smooth.call_args_list[0].args == (arg_data, )
    assert fake_kf.smooth.call_args_list[0].kwargs == {'initial_value' : [arg_initial_val,0]}
    assert fake_kf.predict.call_count == 1
    assert fake_kf.predict.call_args_list[0].args == (arg_data, arg_forward_steps)

def test_Kalman_predict_smoothes_data_and_predicts_result_using_KalmanFilter_functions_as_expected_when_initial_val_equals_None(mocker):
    # Arrange
    len_arg_data = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    arg_data = []
    for i in range(len_arg_data):
        rand_float = pytest.gen.uniform(-10.0, 10.0) # arbitrary, random float from -10.0 to 10.0
        arg_data.append(rand_float)
    arg_forward_steps = MagicMock()
    arg_initial_val = None

    fake_kf = MagicMock()

    forced_predict_return_value = MagicMock()
    mocker.patch.object(fake_kf, 'smooth')
    mocker.patch.object(fake_kf, 'predict', return_value=forced_predict_return_value)

    cut = Kalman.__new__(Kalman)
    cut.kf = fake_kf

    # Act
    result = cut.predict(arg_data, arg_forward_steps, arg_initial_val)

    # Assert
    assert result == forced_predict_return_value
    assert fake_kf.smooth.call_count == 1
    assert fake_kf.smooth.call_args_list[0].args == (arg_data, )
    assert fake_kf.predict.call_count == 1
    assert fake_kf.predict.call_args_list[0].args == (arg_data, arg_forward_steps)
   
def test_Kalman_predict_when_not_given_initial_val_arg_sets_initial_val_arg_equal_to_None(mocker):
    # Arrange
    len_arg_data = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    arg_data = []
    for i in range(len_arg_data):
        rand_float = pytest.gen.uniform(-10.0, 10.0) # arbitrary, random float from -10.0 to 10.0
        arg_data.append(rand_float)
    arg_forward_steps = MagicMock()

    fake_kf = MagicMock()

    forced_predict_return_value = MagicMock()
    mocker.patch.object(fake_kf, 'smooth')
    mocker.patch.object(fake_kf, 'predict', return_value=forced_predict_return_value)

    cut = Kalman.__new__(Kalman)
    cut.kf = fake_kf

    # Act
    result = cut.predict(arg_data, arg_forward_steps)

    # Assert
    assert result == forced_predict_return_value
    assert fake_kf.smooth.call_count == 1
    assert fake_kf.smooth.call_args_list[0].args == (arg_data, )
    assert fake_kf.predict.call_count == 1
    assert fake_kf.predict.call_args_list[0].args == (arg_data, arg_forward_steps)
   
def test_Kalman_predict_smoothes_data_and_predicts_result_using_KalmanFilter_functions_as_expected_when_initial_val_is_not_None(mocker):
    # Arrange
    len_arg_data = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    arg_data = []
    for i in range(len_arg_data):
        rand_float = pytest.gen.uniform(-10.0, 10.0) # arbitrary, random float from -10.0 to 10.0
        arg_data.append(rand_float)
    arg_forward_steps = MagicMock()
    arg_initial_val = pytest.gen.uniform(-10.0, 10.0) # arbitrary, random float from -10.0 to 10.0

    fake_kf = MagicMock()

    forced_predict_return_value = MagicMock()
    mocker.patch.object(fake_kf, 'smooth')
    mocker.patch.object(fake_kf, 'predict', return_value=forced_predict_return_value)

    cut = Kalman.__new__(Kalman)
    cut.kf = fake_kf

    # Act
    result = cut.predict(arg_data, arg_forward_steps, arg_initial_val)

    # Assert
    assert result == forced_predict_return_value
    assert fake_kf.smooth.call_count == 1
    assert fake_kf.smooth.call_args_list[0].args == (arg_data, )
    assert fake_kf.smooth.call_args_list[0].kwargs == {'initial_value' : [arg_initial_val,0]}
    assert fake_kf.predict.call_count == 1
    assert fake_kf.predict.call_args_list[0].args == (arg_data, arg_forward_steps)

def test_Kalman_predict_floatifies_args_and_smoothes_data_and_predicts_result_using_KalmanFilter_functions_as_expected_when_args_are_not_float_values(mocker):
    # Arrange
    len_arg_data = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    arg_data = []
    arg_data_float = []
    for i in range(len_arg_data):
        rand_float = pytest.gen.uniform(-10.0, 10.0) # arbitrary, random float from -10.0 to 10.0
        arg_data_float.append(rand_float)
        arg_data.append(str(rand_float))
    arg_forward_steps = MagicMock()
    arg_initial_val_float = pytest.gen.uniform(-10.0, 10.0) # arbitrary, random float from -10.0 to 10.0
    arg_initial_val = str(arg_initial_val_float)

    fake_kf = MagicMock()

    forced_predict_return_value = MagicMock()
    mocker.patch.object(fake_kf, 'smooth')
    mocker.patch.object(fake_kf, 'predict', return_value=forced_predict_return_value)

    cut = Kalman.__new__(Kalman)
    cut.kf = fake_kf

    # Act
    result = cut.predict(arg_data, arg_forward_steps, arg_initial_val)

    # Assert
    assert result == forced_predict_return_value
    assert arg_data == arg_data_float
    assert arg_initial_val != arg_initial_val_float
    assert fake_kf.smooth.call_count == 1
    assert fake_kf.smooth.call_args_list[0].args == (arg_data_float,)
    assert fake_kf.smooth.call_args_list[0].kwargs == {'initial_value' : [arg_initial_val_float,0]}
    assert fake_kf.predict.call_count == 1
    assert fake_kf.predict.call_args_list[0].args == (arg_data_float, arg_forward_steps)

# test predictions_for_given_data
def test_Kalman_predictions_for_given_data_raises_error_when_data_arg_is_empty(mocker):
    # Arrange
    arg_data = []

    cut = Kalman.__new__(Kalman)
    mocker.patch.object(cut, 'predict')

    # Act
    with pytest.raises(IndexError) as e_info:
        cut.predictions_for_given_data(arg_data)

    # Assert
    assert e_info.match('list index out of range')
    assert cut.predict.call_count == 0

def test_Kalman_predictions_for_given_data_returns_expected_result_when_data_arg_has_only_one_element(mocker):
    # Arrange
    arg_data = [MagicMock()]

    cut = Kalman.__new__(Kalman)

    mocker.patch.object(cut, 'predict')

    expected_result = [0]

    # Act
    result = cut.predictions_for_given_data(arg_data)

    # Assert
    assert result == expected_result
    assert cut.predict.call_count == 0

def test_Kalman_predictions_for_given_data_returns_expected_result_when_data_arg_has_more_than_one_element(mocker):
    # Arrange
    len_data = pytest.gen.randint(2, 10) # arbitrary, random int from 1 to 10
    arg_data = [MagicMock()] * len_data

    cut = Kalman.__new__(Kalman)

    forced_predict_return_value = MagicMock()
    forced_pred_mean = MagicMock()
    mocker.patch.object(forced_predict_return_value, 'observations', forced_predict_return_value)
    mocker.patch.object(forced_predict_return_value, 'mean', forced_pred_mean)
    mocker.patch.object(cut, 'predict', return_value=forced_predict_return_value)

    expected_result = []
    for i in range(len_data - 1):
        expected_result.append(forced_pred_mean)

    # Act
    result = cut.predictions_for_given_data(arg_data)

    # Assert
    assert result == expected_result
    assert cut.predict.call_count == len_data - 1
    for i in range(len_data - 1):
        cut.predict.call_args_list[i].args == (arg_data[0:i+1], 1, arg_data[0])
    
# test generate_residuals_for_given_data
def test_Kalman_generate_residuals_for_given_data_raises_error_when_data_arg_is_empty(mocker):
    # Arrange
    arg_data = []

    cut = Kalman.__new__(Kalman)
    mocker.patch.object(cut, 'predict')

    # Act
    with pytest.raises(IndexError) as e_info:
        cut.generate_residuals_for_given_data(arg_data)

    # Assert
    assert e_info.match('list index out of range')
    assert cut.predict.call_count == 0

def test_Kalman_generate_residuals_for_given_data_returns_expected_result_when_data_arg_has_only_one_element(mocker):
    # Arrange
    arg_data = [MagicMock()]

    cut = Kalman.__new__(Kalman)

    mocker.patch.object(cut, 'predict')

    expected_result = [0]

    # Act
    result = cut.generate_residuals_for_given_data(arg_data)

    # Assert
    assert result == expected_result
    assert cut.predict.call_count == 0

def test_Kalman_generate_residuals_for_given_data_returns_expected_result_when_data_arg_has_more_than_one_element(mocker):
    # Arrange
    len_data = pytest.gen.randint(2, 10) # arbitrary, random int from 1 to 10
    arg_data = [MagicMock()] * len_data

    cut = Kalman.__new__(Kalman)

    forced_predict_return_value = MagicMock()
    forced_pred_mean = MagicMock()
    forced_residual_side_effect = []
    for i in range(len_data - 1):
        rand_float = pytest.gen.uniform(-10.0, 10.0) # arbitrary, random float from -10.0 to 10.0
        forced_residual_side_effect.append(rand_float)
    mocker.patch.object(forced_predict_return_value, 'observations', forced_predict_return_value)
    mocker.patch.object(forced_predict_return_value, 'mean', forced_pred_mean)
    mocker.patch.object(cut, 'residual', side_effect=forced_residual_side_effect)
    mocker.patch.object(cut, 'predict', return_value=forced_predict_return_value)

    expected_result = []
    for i in range(len_data - 1):
        expected_result.append(forced_residual_side_effect[i])

    # Act
    result = cut.generate_residuals_for_given_data(arg_data)

    # Assert
    assert result == expected_result
    assert cut.predict.call_count == len_data - 1
    for i in range(len_data - 1):
        cut.predict.call_args_list[i].args == (arg_data[0:i+1], 1, arg_data[0])
    assert cut.residual.call_count == len_data - 1
    for i in range(len_data - 1):
        cut.residual.call_args_list[i].args == (forced_pred_mean, arg_data[i + 1])

# test current_attribute_chunk_get_error
def test_Kalman_current_attribute_chunk_get_error_returns_true_when_abs_of_mean_residuals_equal_to_or_greater_than_one_point_five(mocker):
    # Arrange
    arg_data = MagicMock()

    cut = Kalman.__new__(Kalman)
    forced_generate_residuals_return_value = MagicMock()
    forced_mean_return_value = MagicMock()
    forced_abs_return_value = pytest.gen.uniform(1.5, 10.0) # random float, greater than cutoff value 1.5

    mocker.patch.object(cut, 'generate_residuals_for_given_data', return_value=forced_generate_residuals_return_value)
    mocker.patch.object(cut, 'mean', return_value=forced_mean_return_value)
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.abs', return_value=forced_abs_return_value)

    # Act
    result = cut.current_attribute_chunk_get_error(arg_data)

    # Assert
    assert result == True
    assert cut.generate_residuals_for_given_data.call_count == 1
    assert cut.generate_residuals_for_given_data.call_args_list[0].args == (arg_data, )
    assert cut.mean.call_count == 1
    assert cut.mean.call_args_list[0].args == (forced_generate_residuals_return_value, )
    assert kalman.abs.call_count == 2
    assert kalman.abs.call_args_list[0].args == (forced_mean_return_value, )
    assert kalman.abs.call_args_list[1].args == (forced_abs_return_value, )
    
def test_Kalman_current_attribute_chunk_get_error_returns_false_when_abs_of_mean_residuals_less_than_one_point_five(mocker):
    # Arrange
    arg_data = MagicMock()

    cut = Kalman.__new__(Kalman)
    forced_generate_residuals_return_value = MagicMock()
    forced_mean_return_value = MagicMock()
    forced_abs_return_value = pytest.gen.uniform(0.0, 1.49) # random float, less than cutoff value 1.5

    mocker.patch.object(cut, 'generate_residuals_for_given_data', return_value=forced_generate_residuals_return_value)
    mocker.patch.object(cut, 'mean', return_value=forced_mean_return_value)
    mocker.patch('src.data_driven_components.kalman.kalman_plugin.abs', return_value=forced_abs_return_value)

    # Act
    result = cut.current_attribute_chunk_get_error(arg_data)

    # Assert
    assert result == False
    assert cut.generate_residuals_for_given_data.call_count == 1
    assert cut.generate_residuals_for_given_data.call_args_list[0].args == (arg_data, )
    assert cut.mean.call_count == 1
    assert cut.mean.call_args_list[0].args == (forced_generate_residuals_return_value, )
    assert kalman.abs.call_count == 2
    assert kalman.abs.call_args_list[0].args == (forced_mean_return_value, )
    assert kalman.abs.call_args_list[1].args == (forced_abs_return_value, )

# test frame_diagnosis
def test_Kalman_frame_diagnosis_returns_empty_list_when_args_frame_and_headers_are_empty():
    # Arrange
    arg_frame = []
    arg_headers = []

    cut = Kalman.__new__(Kalman)

    # Act
    result = cut.frame_diagnosis(arg_frame, arg_headers)

    # Assert
    assert result == []

def test_Kalman_frame_diagnosis_returns_empty_list_when_current_attribute_chunk_get_error_always_returns_false_and_args_not_empty(mocker):
    # Arrange
    len_args = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    arg_frame = []
    for i in range(len_args):
        arg_frame.append(MagicMock())
    arg_headers = MagicMock()

    cut = Kalman.__new__(Kalman)
    forced_get_error_return_value = False
    mocker.patch.object(cut, 'current_attribute_chunk_get_error', return_value=forced_get_error_return_value)

    # Act
    result = cut.frame_diagnosis(arg_frame, arg_headers)

    # Assert
    assert result == []
    assert cut.current_attribute_chunk_get_error.call_count == len_args
    for i in range(len_args):
        assert cut.current_attribute_chunk_get_error.call_args_list[i].args == (arg_frame[i], )

def test_Kalman_frame_diagnosis_returns_empty_list_when_all_elements_in_headers_arg_match_time_str(mocker):
    # Arrange
    len_args = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    arg_frame = [MagicMock()] * len_args
    arg_headers = ['TIME'] * len_args

    cut = Kalman.__new__(Kalman)
    forced_get_error_return_value = True
    mocker.patch.object(cut, 'current_attribute_chunk_get_error', return_value=forced_get_error_return_value)

    # Act
    result = cut.frame_diagnosis(arg_frame, arg_headers)

    # Assert
    assert result == []

def test_Kalman_frame_diagnosis_returns_list_of_all_elements_in_headers_arg_when_current_attribute_chunk_get_error_always_returns_true_and_args_not_empty_and_headers_does_not_contain_strings_matching_time_str(mocker):
    # Arrange
    len_args = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    arg_frame = []
    arg_headers = []
    for i in range(len_args):
        arg_frame.append(MagicMock())
        arg_headers.append(str(MagicMock()))

    cut = Kalman.__new__(Kalman)
    forced_get_error_return_value = True
    mocker.patch.object(cut, 'current_attribute_chunk_get_error', return_value=forced_get_error_return_value)

    # Act
    result = cut.frame_diagnosis(arg_frame, arg_headers)

    # Assert
    assert result == arg_headers

def test_Kalman_frame_diagnosis_returns_expected_sublist_of_headers_when_headers_contains_strings_matching_time_str_and_the_result_of_current_attribute_chunk_get_error_is_not_constant(mocker):
    # Arrange
    len_args = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    arg_frame = []
    arg_headers = []
    for i in range(len_args):
        arg_frame.append(MagicMock())
        arg_headers.append(str(MagicMock()))

    num_time_strings = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    for i in range(num_time_strings):
        rand_index = pytest.gen.randint(0, len_args) # random index in arg_frame
        arg_frame.append(MagicMock()) # ordering of frame does not matter
        arg_headers.insert(rand_index, 'time')
        len_args += 1

    cut = Kalman.__new__(Kalman)
    expected_result = []
    forced_get_error_side_effect = []
    for i in range(len_args):
        coin_flip = pytest.gen.randint(0, 1) # random int, either 0 or 1
        if coin_flip == 0 or arg_headers[i] == 'time':
            forced_get_error_side_effect.append(False)
        else:
            forced_get_error_side_effect.append(True)
            expected_result.append(arg_headers[i])
    mocker.patch.object(cut, 'current_attribute_chunk_get_error', side_effect=forced_get_error_side_effect)

    # Act
    result = cut.frame_diagnosis(arg_frame, arg_headers)

    # Assert
    assert result == expected_result