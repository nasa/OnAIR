# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test tflite_model Plugin Functionality """
import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np

# mock dependencies of tflite_model plugin
import sys
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.lite'] = MagicMock()

from plugins.tflite_model.tflite_model_plugin import Plugin as tflite_model_Plugin

def generate_random_float_model_details():
    np.random.seed(0) # setting the seed for now
    n_inputs = np.random.randint(1,4)
    n_outputs = np.random.randint(1,4)
    
    input_details = [
        {
            'index': i,
            'shape': np.random.randint(1,20,size=4), # input shape is a 4-d tensor with randomly sized dims.
            'dtype': np.float32
        }
        for i in range(0, n_inputs) ]
    
    output_details = [
        {
            'index': i,
            'shape': np.random.randint(1,20,size=4), # input shape is a 4-d tensor with randomly sized dims.
            'dtype': np.float32
        }
        for i in range(0, n_outputs)]
    return input_details, output_details
    
@patch('plugins.tflite_model.tflite_model_plugin.tflite')
def test_tflite_plugin_init_creates_base_plugin_and_tflite_interpreter(mock_tflite):
    # Arrange
    cut = tflite_model_Plugin.__new__(tflite_model_Plugin)
    arg_names = [MagicMock()]
    arg_headers = [MagicMock()]
    
    # Act
    cut.__init__(arg_names, arg_headers)
    
    # Assert
    assert isinstance(cut, tflite_model_Plugin)
    mock_tflite.Interpreter.assert_called_once_with(ANY)
    cut.interpreter.allocate_tensors.assert_called_once()
    cut.interpreter.get_input_details.assert_called_once()
    cut.interpreter.get_output_details.assert_called_once()
    assert cut.input_details == cut.interpreter.get_input_details()
    assert cut.output_details == cut.interpreter.get_output_details()
    assert cut.input_tensors is None
    assert cut.output_tensors is None
    
    
def test_tflite_plugin_update_sets_input_data_and_output_data():
    # Arrange
    cut = tflite_model_Plugin.__new__(tflite_model_Plugin)
    cut.input_tensors, cut.output_tensors = None, None
    cut.input_details, cut.output_details = generate_random_float_model_details()
    n_inputs = len(cut.input_details)
    n_outputs = len(cut.output_details)
    cut.interpreter = MagicMock()
    
    # Act
    cut.update()
    
    # Assert
    assert len(cut.input_tensors) == n_inputs
    assert cut.interpreter.set_tensor.call_count == n_inputs
    for i in range(0, n_inputs):
        assert cut.interpreter.set_tensor.call_args_list[i].args == (cut.input_details[i]['index'], cut.input_tensors[i])
    assert cut.interpreter.invoke.call_count == 1
    assert len( cut.output_tensors) == n_outputs
    assert cut.interpreter.get_tensor.call_count == n_outputs
    for i in range(0, n_outputs):
        assert cut.interpreter.get_tensor.call_args_list[i].args == (cut.output_details[i]['index'],)
        
def test_tflite_plugin_render_reasoning_returns_output_tensors():
    # Arrange
    cut = tflite_model_Plugin.__new__(tflite_model_Plugin)
    cut.output_tensors = MagicMock()
    
    # Act & Assert
    assert cut.render_reasoning() is cut.output_tensors
    
def test_tflite_plugin_generate_random_input_is_correct():
    # Arrange
    cut = tflite_model_Plugin.__new__(tflite_model_Plugin)
    cut.input_details, _ = generate_random_float_model_details()
    
    # Act
    input_tensors = cut.generate_random_input()
    
    # Assert
    assert isinstance(input_tensors, tuple)
    assert len(cut.input_details) == len(input_tensors)
    for i in range(len(cut.input_details)):
        expected_shape = cut.input_details[i]['shape']
        actual_shape = input_tensors[i].shape
        assert np.array_equal(expected_shape, actual_shape)
    