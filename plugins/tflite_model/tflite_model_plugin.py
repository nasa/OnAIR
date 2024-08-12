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

try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    import tensorflow.lite as tflite
else:
    raise ModuleNotFoundError("tflite_runtime or tensorflow modules not found")

class Plugin(AIPlugin):
    def __init__(self, _name, _headers):
        super().__init__(_name, _headers)
        # your model goes here
        model_path = r"tflite_models\yolo-v5-tflite-tflite-tflite-model-v1\1.tflite"
        
        self.interpreter = tflite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    
    def update(self,low_level_data=[], high_level_data={}):
        """
        Given streamed data point, system should update internally
        """
        pass

    def render_reasoning(self):
        """
        System should return its diagnosis
        """
        pass
    
    def generate_random_input(self):
        pass
    
    
