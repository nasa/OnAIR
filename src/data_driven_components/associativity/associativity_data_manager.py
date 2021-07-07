"""
This class processes the raw sensor data frames into a format
that is usable by the associativity algorithm

Author: Chris Trombley
June 10th 2021
"""

import numpy as np
import math
import keras
import os

from src.data_driven_components.curve_characterizer.curve_characterizer import CurveCharacterizer

class AssociativityDataManager:
    def __init__(self, _headers=[]):
        self.window_size = 20
        self.headers = _headers
        self.frames = [['0.0']*len(_headers) for i in range(self.window_size)]
        self.frame_id = 0
        self.categorical_data = []
        self.curve_characterizer = CurveCharacterizer(os.environ['RUN_PATH'] + '/data/')
        
    def add_frame(self, frame):
        self.frames.append(frame)
        self.frames.pop(0)
        self.frame_id += 1
        self.convert_frames_to_categorical()

    def convert_frames_to_categorical(self):
        categorical_data_record = []
        for sensor_idx in range(len(self.headers)):
            individual_sensor_stream = []

            for frame_number in range(self.window_size): 
                individual_sensor_stream.append(float(self.frames[frame_number][sensor_idx]))
            reformatted_stream = np.array(individual_sensor_stream).reshape(1, self.window_size, 1)
            prediction = self.curve_characterizer.predict(reformatted_stream) ## do we need to do midpt preprocessing first?
            categorization = (str(self.headers[sensor_idx]) + ' ' + str(prediction)).replace(' ', '_')
            categorical_data_record.append(categorization)
            
        self.categorical_data.append(categorical_data_record)

    def get_data(self):
        if len(self.categorical_data) > 0:
            return self.categorical_data    
        else:    
            return -1




