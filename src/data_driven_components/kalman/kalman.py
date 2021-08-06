import simdkalman
import numpy as np

class Kalman():
    def __init__(self):
        self.kf = simdkalman.KalmanFilter(
        state_transition = [[1,1],[0,1]],        # matrix A
        process_noise = np.diag([0.1, 0.01]),    # Q
        observation_model = np.array([[1,0]]),   # H
        observation_noise = 1.0)                 # R

    # Gets mean of values
    def mean(self, values):
        return sum(values)/len(values)

    # Gets absolute value residual from actual and predicted value
    def residual(self, predicted, actual):
        return abs(float(actual) - float(predicted))

    #Gets standard deviation of data
    def std_dev(self, data):
        return np.std(data)

    # Takes in the kf being used, the data, how many prediction "steps" it will make, and an optional initial value
    # Gives a prediction values based on given parameters
    def predict(self, data, forward_steps, inital_val = None):
        for i in range(len(data)):
            data[i] = float(data[i]) # Makes sure all the data being worked with is a float
        if(inital_val != None):
            smoothed = self.kf.smooth(data, initial_value = [float(inital_val),0]) # If there's an initial value, smooth it along that value
        else:
            smoothed =  self.kf.smooth(data) # If not, smooth it however you like
        predicted =  self.kf.predict(data, forward_steps) # Make a prediction on the smoothed data
        return predicted

    def predictions_for_given_data(self, data):
        returned_data = []
        initial_val = data[0]
        for item in range(len(data)-1):
            predicted = self.predict(data[0:item+1], 1, initial_val)        
            actual_next_state = data[item+1]
            pred_mean = predicted.observations.mean
            returned_data.append(pred_mean)
        if(len(returned_data) == 0): # If there's not enough data just set it to 0
            returned_data.append(0)
        return returned_data
    
    # Get data, make predictions, and then find the errors for these predictions 
    def generate_residuals_for_given_data(self, data):
        residuals = []
        initial_val = data[0]
        for item in range(len(data)-1):
            predicted = self.predict(data[0:item+1], 1, initial_val)        
            actual_next_state = data[item+1]
            pred_mean = predicted.observations.mean
            residual_error = float(self.residual(pred_mean, actual_next_state))
            residuals.append(residual_error)
        if(len(residuals) == 0): # If there are no residuals because the data is of length 1, then just say residuals is equal to [0]
            residuals.append(0)
        return residuals

    #Info: takes a chunk of data of n size. Walks through it and gets residual errors.
    #Takes the mean of the errors and determines if they're too large overall in order to determine whether or not there's a chunk in said error. 
    def current_attribute_chunk_get_error(self, data):
        residuals = self.generate_residuals_for_given_data(data)
        mean_residuals = abs(self.mean(residuals))
        if (abs(mean_residuals) < 1.5):
                return False
        return True
    
    def frame_diagnose(self, frame, headers):
        kal_broken_attributes = []
        for attribute_index in range(len(frame)):
            error = self.current_attribute_chunk_get_error(frame[attribute_index])
            if error and not headers[attribute_index].upper() == 'TIME':
                kal_broken_attributes.append(headers[attribute_index])
        return kal_broken_attributes