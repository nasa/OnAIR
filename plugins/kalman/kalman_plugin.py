# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"
"""
This module contains the Plugin class for detecting faults using Kalman filtering.

The Plugin class is derived from AIPlugin and uses a Kalman filter to predict
future values based on a sliding window of observations. It calculates residuals
between predicted and actual values to identify potential faults in the system.

The module is part of NASA's On-Board Artificial Intelligence Research (OnAIR) Platform.
"""
import simdkalman
import numpy as np
from onair.data_handling.parser_util import floatify_input
from onair.src.util.print_io import print_msg
from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin

class Plugin(AIPlugin):
    """
    A plugin for detecting faults using Kalman filtering.

    This plugin uses a Kalman filter to predict future values based on a sliding window
    of observations. It calculates residuals between predicted and actual values to
    identify potential faults in the system.

    The plugin maintains a separate Kalman filter for each data attribute and uses
    a specified threshold to determine if a residual indicates a fault.

    Attributes:
        component_name (str): The name of the plugin component.
        headers (list): List of descriptive header names for low_level_data.
        window_size (int): Size of the sliding window for observations.
        residual_threshold (float): Threshold above which a residual is considered a fault.
        frames (list): List of lists containing the sliding window data for each attribute.
        kf (simdkalman.KalmanFilter): The Kalman filter used for predictions.
    """

    def __init__(self, name, headers, window_size=15, residual_threshold=1.5):
        """
        Initialize the Plugin class.

        Parameters
        ----------
        name : str
            The name of the plugin.
        headers : list
            List of header names for the data attributes.
        window_size : int, optional
            Size of the time window to examine (default is 15).
        residual_threshold : float, optional
            Threshold of residual above which is considered a fault (default is 1.5).

        Returns
        -------
        None
        """
        if window_size < 3:
            raise RuntimeError(
                f"Kalman plugin unable to operate with window size < 3: given {window_size}"
            )

        super().__init__(name, headers)
        self.component_name = name
        self.headers = headers
        self.window_size = window_size
        self.residual_threshold = residual_threshold
        self.frames = [[] for _ in range(len(headers))]

        self.kf = simdkalman.KalmanFilter(
            # matrix
            # A
            state_transition=[[1, 1], [0, 1]],
            # Q
            process_noise=np.diag([0.1, 0.1]),
            # H
            observation_model=np.array([[1, 0]]),
            # R
            observation_noise=1.0,
        )

    def update(self, low_level_data=None, _high_level_data=None):
        """
        Update the frames with new low-level data.

        This method converts the input data to float type, appends it to the
        corresponding frames, and maintains the window size by removing older
        data points if necessary.

        Parameters
        ----------
        low_level_data : list
            Input sequence of data points with length equal header dimensions.

        Returns
        -------
        None
        """
        if low_level_data is None:
            print_msg("Kalman plugin requires low_level_data but received None.", ['FAIL'])
        else:
            frame = floatify_input(low_level_data)

            for i, value in enumerate(frame):
                self.frames[i].append(value)
                if len(self.frames[i]) > self.window_size:
                    self.frames[i].pop(0)

    def render_reasoning(self):
        """
        Generate a list of attributes that show fault-like behavior based on residual analysis.

        This method calculates residuals using the Kalman filter predictions and compares
        them against a threshold. Attributes with residuals exceeding the threshold are
        considered to show fault-like behavior, except for the 'TIME' attribute.

        Returns
        -------
        list of str
            A list of attribute names that show fault-like behavior (i.e., have residuals
            above the threshold). The 'TIME' attribute is excluded from this list even
            if its residual is above the threshold.

        Notes
        -----
        The residual threshold is defined by the `residual_threshold` attribute of the class.
        """
        residuals = self._generate_residuals()
        residuals_above_thresh = residuals > self.residual_threshold
        broken_attributes = []
        for attribute_index in range(len(self.frames)):
            if (
                residuals_above_thresh[attribute_index]
                and not self.headers[attribute_index].upper() == "TIME"
            ):
                broken_attributes.append(self.headers[attribute_index])
        return broken_attributes

    def _predict(self, subframe, forward_steps, initial_val):
        """
        Provide predicted future values using Kalman filter.

        Parameters
        ----------
        subframe : list of list of float
            Data for Kalman filter prediction.
        forward_steps : int
            Number of forward predictions to make.
        initial_val : list of float
            Initial value for Kalman filter.

        Returns
        -------
        predictions : object
            Predicted values from the Kalman filter with fields for states and observations.
        """
        self.kf.smooth(subframe, initial_value=initial_val)
        predictions = self.kf.predict(subframe, forward_steps)
        return predictions

    def _generate_residuals(self):
        """
        Predict last observation in frame based on all previous observations in frame.

        This method uses a Kalman filter to predict the last observation in each frame
        based on all previous observations. It then calculates the residuals as the
        absolute difference between the predicted and actual last observations.

        Returns
        -------
        residuals : numpy.ndarray
            Residuals based on the difference between the last observation and
            the Kalman filter-smoothed prediction. If the frame length is 2 or less,
            returns an array of zeros.

        Notes
        -----
        The current size of each frame must be greater than 2 for valid initial and last values,
        and for the Kalman filter to have sufficient data for smoothing.
        """
        if len(self.frames[0]) > 2:
            # generate initial values for frame, use first value for each attribute
            initial_val = np.zeros((len(self.frames), 2, 1))
            for i, frame in enumerate(self.frames):
                initial_val[i] = np.array([[frame[0], 0]]).transpose()
            predictions = self._predict(
                [data[1:-1] for data in self.frames], 1, initial_val
            )
            actual_next_obs = [data[-1] for data in self.frames]
            pred_mean = [
                pred for attr in predictions.observations.mean for pred in attr
            ]
            residuals = np.abs(np.subtract(pred_mean, actual_next_obs))
        else:
            # return residual of 0 for frames size less than or equal to 2
            residuals = np.zeros((len(self.frames),))
        return residuals
