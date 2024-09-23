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
import numpy as np

from plugins.kalman import kalman_plugin
from plugins.kalman.kalman_plugin import Plugin as Kalman

# test init
def test_Kalman__init__initializes_variables_using_both_given_and_default_arguments_and_creates_filter_when_only_given_required_arguments(mocker):
    # Arrange
    arg_name = MagicMock()
    arg_headers = MagicMock()
    arg_headers.__len__.return_value = pytest.gen.randint(1, 10)
    fake_kf = MagicMock()
    forced_diag_return_value = MagicMock()
    forced_array_return_value = MagicMock()
    expected_default_window_size = 15
    expected_default_residual_threshold = 1.5
    expected_state_transition = [[1,1],[0,1]]
    expected_process_noise = forced_diag_return_value
    expected_observation_model = forced_array_return_value
    expected_observation_noise = 1.0

    mocker.patch(kalman_plugin.__name__ + '.simdkalman.KalmanFilter', return_value=fake_kf)
    mocker.patch(kalman_plugin.__name__ + '.np.diag', return_value=forced_diag_return_value)
    mocker.patch(kalman_plugin.__name__ + '.np.array', return_value=forced_array_return_value)

    cut = Kalman.__new__(Kalman)
    parent_of_cut = cut.__class__.__mro__[1]

    mocker.patch.object(parent_of_cut, "__init__", return_value=MagicMock())

    # Act
    cut.__init__(arg_name, arg_headers)

    # Assert
    assert parent_of_cut.__init__.call_count == 1
    assert parent_of_cut.__init__.call_args_list[0].args == (arg_name, arg_headers)
    assert len(cut.frames) == len(arg_headers)
    assert all(isinstance(frame, list) and len(frame) == 0 for frame in cut.frames)
    assert cut.component_name == arg_name
    assert cut.headers == arg_headers
    assert cut.window_size == expected_default_window_size
    assert cut.residual_threshold == expected_default_residual_threshold
    assert cut.kf is fake_kf
    assert kalman_plugin.simdkalman.KalmanFilter.call_count == 1
    assert kalman_plugin.simdkalman.KalmanFilter.call_args_list[0].args == ()
    assert kalman_plugin.simdkalman.KalmanFilter.call_args_list[0].kwargs == {
        'state_transition':expected_state_transition,
        'process_noise':expected_process_noise,
        'observation_model':expected_observation_model,
        'observation_noise':expected_observation_noise,
    }

def test_Kalman__init__initializes_variables_using_given_arguments_and_creates_filter_when_given_required_and_optional_arguments_with_window_size_greater_than_2(mocker):
    # Arrange
    arg_name = MagicMock()
    arg_headers = MagicMock()
    arg_headers.__len__.return_value = pytest.gen.randint(1, 10)
    arg_window_size = pytest.gen.randint(3, 20)
    arg_residual_threshold = MagicMock()
    fake_kf = MagicMock()
    forced_diag_return_value = MagicMock()
    forced_array_return_value = MagicMock()
    expected_state_transition = [[1,1],[0,1]]
    expected_process_noise = forced_diag_return_value
    expected_observation_model = forced_array_return_value
    expected_observation_noise = 1.0

    mocker.patch(kalman_plugin.__name__ + '.simdkalman.KalmanFilter', return_value=fake_kf)
    mocker.patch(kalman_plugin.__name__ + '.np.diag', return_value=forced_diag_return_value)
    mocker.patch(kalman_plugin.__name__ + '.np.array', return_value=forced_array_return_value)

    cut = Kalman.__new__(Kalman)
    parent_of_cut = cut.__class__.__mro__[1]
    mocker.patch.object(parent_of_cut, "__init__", return_value=MagicMock())

    # Act
    cut.__init__(arg_name, arg_headers, arg_window_size, arg_residual_threshold)

    # Assert
    assert parent_of_cut.__init__.call_count == 1
    assert parent_of_cut.__init__.call_args_list[0].args == (arg_name, arg_headers)
    assert len(cut.frames) == len(arg_headers)
    assert all(isinstance(frame, list) and len(frame) == 0 for frame in cut.frames)
    assert cut.component_name == arg_name
    assert cut.headers == arg_headers
    assert cut.window_size == arg_window_size
    assert cut.residual_threshold == arg_residual_threshold
    assert cut.kf is fake_kf
    assert kalman_plugin.simdkalman.KalmanFilter.call_count == 1
    assert kalman_plugin.simdkalman.KalmanFilter.call_args_list[0].args == ()
    assert kalman_plugin.simdkalman.KalmanFilter.call_args_list[0].kwargs == {
        'state_transition':expected_state_transition,
        'process_noise':expected_process_noise,
        'observation_model':expected_observation_model,
        'observation_noise':expected_observation_noise,
    }

def test_Kalman__init__raises_error_when_given_window_size_is_less_than_3(mocker):
    # Arrange
    arg_name = MagicMock()
    arg_headers = MagicMock()
    arg_window_size = pytest.gen.randint(-1, 2)
    arg_residual_threshold = MagicMock()

    cut = Kalman.__new__(Kalman)

    expected_exception_message = \
        f"Kalman plugin unable to operate with window size < 3: given {arg_window_size}"

    # Act
    with pytest.raises(RuntimeError) as e_info:
        cut.__init__(arg_name, arg_headers, arg_window_size, arg_residual_threshold)

    # Assert
    assert e_info.match(expected_exception_message)

# test update
def test_Kalman_update_not_providing_low_level_data_issues_warning(mocker):
    # Arrange
    cut = Kalman.__new__(Kalman)
    num_fake_data_points = pytest.gen.randint(1, 10)
    cut.frames = [[] for _ in range(num_fake_data_points)]
    cut.window_size = 1 # smallest value without popping starting at none

    mocker.patch(kalman_plugin.__name__ + '.print_msg')

    # Act
    cut.update()

    # Assert
    kalman_plugin.print_msg.call_count = 1


def test_Kalman_update_with_initially_empty_frames(mocker):
    # Arrange
    cut = Kalman.__new__(Kalman)
    num_fake_data_points = pytest.gen.randint(1, 10)
    cut.frames = [[] for _ in range(num_fake_data_points)]
    cut.window_size = 1 # smallest value without popping starting at none

    arg_low_level_data = [pytest.gen.uniform(-10, 10) for _ in range(num_fake_data_points)]
    mocker.patch(kalman_plugin.__name__ + '.floatify_input', return_value=arg_low_level_data)

    # Act
    cut.update(arg_low_level_data)

    # Assert
    assert kalman_plugin.floatify_input.call_count == 1
    assert kalman_plugin.floatify_input.call_args_list[0].args == (arg_low_level_data,)
    assert len(cut.frames) == num_fake_data_points
    for i, frame in enumerate(cut.frames):
        assert frame == [arg_low_level_data[i]]

def test_Kalman_update_with_existing_data_in_frames_but_less_than_full_window_size(mocker):
    # Arrange
    cut = Kalman.__new__(Kalman)
    num_headers = pytest.gen.randint(1, 10)
    existing_data_points = pytest.gen.randint(1, 4)
    cut.window_size = pytest.gen.randint(
        existing_data_points + 1, 
        existing_data_points + 5
    ) # arbitrary 1-5 extra points left in sliding window

    cut.frames = [
        [pytest.gen.uniform(-10, 10) for _ in range(existing_data_points)]
        for _ in range(num_headers)
    ]

    arg_low_level_data = [pytest.gen.uniform(-10, 10) for _ in range(num_headers)]
    mocker.patch(kalman_plugin.__name__ + '.floatify_input', return_value=arg_low_level_data)

    # Act
    cut.update(arg_low_level_data)

    # Assert
    assert len(cut.frames) == num_headers
    for i, frame in enumerate(cut.frames):
        assert len(frame) == existing_data_points + 1
        assert frame[-1] == arg_low_level_data[i]
        assert frame[:-1] == cut.frames[i][:-1]  # Check that previous data is preserved

def test_Kalman_update_with_full_window_size(mocker):
    # Arrange
    cut = Kalman.__new__(Kalman)
    num_headers = pytest.gen.randint(1, 10)
    cut.window_size = 5
    existing_data_points = cut.window_size  # Fill the window

    original_frames = [
        [pytest.gen.uniform(-10, 10) for _ in range(existing_data_points)]
        for _ in range(num_headers)
    ]
    cut.frames = [frame.copy() for frame in original_frames]  # Create a copy for the cut object

    arg_low_level_data = [pytest.gen.uniform(-10, 10) for _ in range(num_headers)]
    mocker.patch(kalman_plugin.__name__ + '.floatify_input', return_value=arg_low_level_data)

    # Act
    cut.update(arg_low_level_data)

    # Assert
    assert len(cut.frames) == num_headers
    for i, frame in enumerate(cut.frames):
        assert len(frame) == cut.window_size  # Length should still be window_size
        assert frame[-1] == arg_low_level_data[i]  # New data point should be at the end
        assert frame[:-1] == original_frames[i][1:]  # Check that data shifted correctly

# test render reasoning

def test_Kalman_render_reasoning_returns_empty_list_when_all_values_below_threshold(mocker):
    # Arrange
    cut = Kalman.__new__(Kalman)
    num_headers = pytest.gen.randint(2, 5)  # Random number of headers between 2 and 5
    cut.frames = [list(range(pytest.gen.randint(3, 7))) for _ in range(num_headers)]  # Random frame sizes
    cut.headers = [f"header{i}" for i in range(num_headers)]
    cut.residual_threshold = pytest.gen.uniform(1.0, 5.0)  # Random threshold between 1.0 and 5.0
    
    # All residuals are below the threshold
    forced_residuals = np.array([cut.residual_threshold - 0.1 - i * 0.1 for i in range(num_headers)])
    mocker.patch.object(cut, '_generate_residuals', return_value=forced_residuals)

    # Act
    result = cut.render_reasoning()

    # Assert
    assert len(result) == 0  # The result should be an empty list
    assert cut._generate_residuals.call_count == 1
    assert cut._generate_residuals.call_args_list[0].args == ()  # No arguments passed to _generate_residuals

    # Additional check to ensure all residuals are indeed below the threshold
    assert all(residual < cut.residual_threshold for residual in forced_residuals)

def test_Kalman_render_reasoning_returns_values_above_threshold(mocker):
    # Arrange
    cut = Kalman.__new__(Kalman)
    num_headers = pytest.gen.randint(3, 6)  # Random number of headers between 3 and 5
    cut.frames = [list(range(pytest.gen.randint(3, 7))) for _ in range(num_headers)]  # Random frame sizes
    cut.headers = [f"header{i}" for i in range(num_headers)]
    cut.residual_threshold = pytest.gen.uniform(1.0, 5.0)  # Random threshold between 1.0 and 5.0
    
    # Randomly select between 1 and num_headers-1 headers to fail
    num_failing = pytest.gen.randint(1, num_headers-1)
    failing_indices = np.random.choice(num_headers, num_failing, replace=False)
    expected_result = [cut.headers[i] for i in failing_indices]
    
    # Create forced_residuals with values above threshold for failing headers
    forced_residuals = np.array([cut.residual_threshold - 0.1] * num_headers)  # Initialize all below threshold
    for i in failing_indices:
        forced_residuals[i] = cut.residual_threshold + 0.1 + i * 0.1  # Set failing headers above threshold
    
    mocker.patch.object(cut, '_generate_residuals', return_value=forced_residuals)

    # Act
    result = cut.render_reasoning()

    # Assert
    assert set(result) == set(expected_result)  # Check if the result contains all expected failing headers
    assert cut._generate_residuals.call_count == 1
    assert cut._generate_residuals.call_args_list[0].args == ()  # No arguments passed to _generate_residuals

def test_Kalman_render_reasoning_handles_all_values_above_threshold(mocker):
    # Arrange
    cut = Kalman.__new__(Kalman)
    num_headers = pytest.gen.randint(2, 5)  # Random number of headers between 2 and 5
    cut.frames = [list(range(pytest.gen.randint(3, 7))) for _ in range(num_headers)]  # Random frame sizes
    cut.headers = [f"header{i}" for i in range(num_headers)]
    cut.residual_threshold = pytest.gen.uniform(1.0, 5.0)  # Random threshold between 1.0 and 5.0
    
    # All residuals are above the threshold
    forced_residuals = np.array([cut.residual_threshold + 0.1 + i * 0.1 for i in range(num_headers)])
    mocker.patch.object(cut, '_generate_residuals', return_value=forced_residuals)

    # Act
    result = cut.render_reasoning()

    # Assert
    assert set(result) == set(cut.headers)  # The result should contain all headers
    assert cut._generate_residuals.call_count == 1
    assert cut._generate_residuals.call_args_list[0].args == ()  # No arguments passed to _generate_residuals

# test _predict
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

# test _generate_residuals
def test_Kalman__generate_residuals_with_frame_data_length_less_than_3(mocker):
    # Arrange
    cut = Kalman.__new__(Kalman)
    num_headers = pytest.gen.randint(2, 5)
    short_frame_length = pytest.gen.randint(1, 2)  # Generate frames with 1 or 2 items
    cut.frames = [[pytest.gen.uniform(-10, 10) for _ in range(short_frame_length)] for _ in range(num_headers)]

    # Act
    result = cut._generate_residuals()

    # Assert
    assert np.array_equal(result, np.zeros(num_headers))
    assert len(result) == len(cut.frames)

def test_Kalman__generate_residuals_with_sufficient_frame_data_length(mocker):
    # Arrange
    cut = Kalman.__new__(Kalman)
    num_headers = pytest.gen.randint(2, 5)
    frame_length = pytest.gen.randint(3, 7)
    cut.frames = [[pytest.gen.uniform(-10, 10) for _ in range(frame_length)] for _ in range(num_headers)]
    
    mock_predict_result = MagicMock()
    mock_predict_result.observations.mean = np.array([[pytest.gen.uniform(-100, 100)] for _ in range(num_headers)])
    mocker.patch.object(cut, '_predict', return_value = mock_predict_result)
    
    expected_residuals = np.abs(np.subtract([frame[-1] for frame in cut.frames], 
                                            [pred[0] for pred in mock_predict_result.observations.mean]))
   
    expected_initial_value = np.array([[[frame[0]], [0]] for frame in cut.frames])
    # Act
    result = cut._generate_residuals()

    # Assert
    assert cut._predict.call_count == 1
    assert cut._predict.call_args[0][0] == [frame[1:-1] for frame in cut.frames]
    assert cut._predict.call_args[0][1] == 1
    np.testing.assert_array_almost_equal(cut._predict.call_args[0][2], expected_initial_value)
    np.testing.assert_array_almost_equal(result, expected_residuals)