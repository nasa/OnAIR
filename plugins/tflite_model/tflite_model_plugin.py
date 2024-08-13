# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin
import numpy as np

# tflite_runtime is the bare minimum required to run tflite models
try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    import tensorflow.lite as tflite

class Plugin(AIPlugin):
    def __init__(self, _name, _headers):
        super().__init__(_name, _headers)
        # Your model here
        model_path = r"tflite_models\mobilebert-tflite-default-v1\1.tflite"

        # Load and initialize the model
        self.interpreter = tflite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()        
        self.input_tensors = None
        self.output_tensors = None
    
    def update(self,low_level_data=[], high_level_data={}):
        """
        Given streamed data point, system should update internally
        """
        self.input_tensors = self.generate_random_input()
        
        # set inputs - this should handle models with multiple inputs
        for model_input, tensor in zip(self.input_details, self.input_tensors):
            self.interpreter.set_tensor(model_input['index'], tensor)
            
        self.interpreter.invoke()
        
        outputs = []
        for model_output in self.output_details:
            tensor = self.interpreter.get_tensor(model_output['index'])
            outputs.append(tensor)
            
        self.output_tensors = tuple(outputs)

    def render_reasoning(self):
        """
        System should return its diagnosis
        """
        return self.output_tensors
        
            
    def generate_random_input(self) -> tuple:
        """
        Generates random tensors to be used as the input of a tflite model.

        Returns:
            tuple of np.ndarray
        """
        tensors = []
        for model_input in self.input_details:
            rand_tensor = np.random.rand(*model_input['shape'])
            rand_tensor = rand_tensor.astype(model_input['dtype'])
            tensors.append(rand_tensor)
        return tuple(tensors)
    