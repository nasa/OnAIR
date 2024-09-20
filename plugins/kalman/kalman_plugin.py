# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import simdkalman
import numpy as np
from onair.data_handling.parser_util import floatify_input
from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin

class Plugin(AIPlugin):
    def __init__(self, name, headers, window_size=15, residual_threshold=1.5):
        """
        :param headers: (int) length of time agent examines
        :param window_size: (int) size of time window to examine
        :param residual_threshold: (float) threshold of residual above which is considered a fault
        """
        super().__init__(name, headers)
        self.component_name = name
        self.headers = headers
        self.window_size = window_size
        self.residual_threshold = residual_threshold
        self.frames = [[] for _ in range(len(headers))]

        self.kf = simdkalman.KalmanFilter(
        state_transition = [[1,1],[0,1]],      # matrix A
        process_noise = np.diag([0.1, 0.1]),   # Q
        observation_model = np.array([[1,0]]), # H
        observation_noise = 1.0)               # R

    def update(self, low_level_data):
        """
        :param low_level_data: (list of data points) input sequence of len (input_dim)
        :return: None
        """
        frame = floatify_input(low_level_data)

        for i, value in enumerate(frame):
            self.frames[i].append(value)
            if len(self.frames[i]) > self.window_size:
                self.frames[i].pop(0)

    def render_reasoning(self):
        """
        :return: diagnosis
        """
        residuals = self._generate_residuals()
        residuals_above_thresh = residuals > self.residual_threshold
        broken_attributes = []
        for attribute_index in range(len(self.frames)):
            if residuals_above_thresh[attribute_index] and not self.headers[attribute_index].upper() == 'TIME':
                broken_attributes.append(self.headers[attribute_index])
        return broken_attributes

    def _predict(self, subframe, forward_steps, initial_val):
        '''
        :param subframe: (list of list of floats) data for kalman filter prediction
        :param forward_steps: (int) number of forward predictions to make
        :param initial_val: (list of floats) initial value for kalman filter
        :return: predicted values
        '''
        self.kf.smooth(subframe, initial_value = initial_val)
        predictions =  self.kf.predict(subframe, forward_steps)
        return predictions

    def _generate_residuals(self):
        '''
        Predicts last observation in frame based on all previous observations in frame
        :return: (list of floats) residuals based on difference between last observation and KF-smoothed prediction
        '''
        # length of frame must be greater than 2 for valid initial and last value and data for KF to smooth
        if len(self.frames[0]) > 2:
            # generate initial values for frame, use first value for each attribute
            initial_val = np.zeros((len(self.frames), 2, 1))
            for i in range(len(self.frames)):
                initial_val[i] = np.array([[self.frames[i][0], 0]]).transpose()
            predictions = self._predict([data[1:-1] for data in self.frames], 1, initial_val)
            actual_next_obs = [data[-1] for data in self.frames]
            pred_mean = [pred for attr in predictions.observations.mean for pred in attr]
            residuals = np.abs(np.subtract(pred_mean, actual_next_obs))
        else:
            # return residual of 0 for frames less than or equal to 2
            residuals = np.zeros((len(self.frames),))
        return residuals