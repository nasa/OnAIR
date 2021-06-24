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
    def __init__(self, prepModel=False):
        self.window_size = 20
        self.frame_id = 0
        self.datum_id = 0
        self.frames = []
        self.sensor_data = []
        self.categorical_data = []
        self.categorical_data_record = []
        self.sensor_idx_mapping = {3: 'counter_3', 4: 'Voltage', 5: 'Current', 6: 'counter', 7: 'internal temp', 8: 'extenal temp', 9: 'counter_2', 10: 'lat', 11: 'long', 12: 'alt'} ##this could be better
        self.class_index_mapping = {0: 'linear increasing', 1: 'linear decrease', 2: 'constant', 3: 'sin'}

        self.curve_characterizer = CurveCharacterizer(os.environ['RUN_PATH'] + '/data/', prepModel)

        # self.model_path = os.environ['RUN_PATH'] + '/util/data_generation_scripts/models'
        # self.model = keras.models.load_model(self.model_path)

    def convert_frames_to_categorical(self):
        for sensor_idx in range(len(self.frames[0])):
            self.categorical_data_record = []
            self.input_data = []

            for data_row in range(len(self.frames)): ##assumes all are the same size
                print(self.frames[data_row])

                if sensor_idx < 4 or sensor_idx == 6 or sensor_idx == 9: ##hacky - change
                    continue
                else:
                    self.input_data.append(float(self.frames[data_row][sensor_idx]))
                    delta = 0
                    if delta == 0:
                        self.categorical_data_record.append(str(self.sensor_idx_mapping[sensor_idx]) + ' constant')    
                    elif delta > 0:
                        self.categorical_data_record.append(str(self.sensor_idx_mapping[sensor_idx]) + ' increasing')
                    elif delta < 0:
                        self.categorical_data_record.append(str(self.sensor_idx_mapping[sensor_idx]) + ' decreasing')
            
            # print('-'*30)

            if sensor_idx >= 4 and sensor_idx != 6 and sensor_idx != 9: ##hacky - change 
                input_data_np = np.array(self.input_data).reshape(1, 20, 1)
                preds = self.curve_characterizer.predict(input_data_np) ## do we need to do midpt preprocessing first?
                max_pre_prob = max(preds.tolist())
                max_pre_prob_idx = preds.tolist().index(max_pre_prob)

            self.categorical_data.append(self.categorical_data_record)
        
    def add_frame(self, frame):
        self.frames.append(frame)
        self.frame_id += 1
        print(len(self.frames))
        if int(self.frame_id / self.window_size) >= 1:
            self.convert_frames_to_categorical()
            
    def get_data(self):
        if len(self.categorical_data) > 0:
            return self.categorical_data    
        else:    
            return -1