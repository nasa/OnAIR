import simdkalman
import numpy as np

###TODO: properly calculate process noise (Q) and observation noise (R)##
kf = simdkalman.KalmanFilter(
state_transition = [[1,1],[0,1]],        # matrix A
process_noise = np.diag([0.1, 0.01]),    # Q
observation_model = np.array([[1,0]]),   # H
observation_noise = 1.0)                 # R

#Returns the Kalman Filter created in this file
def return_KF():
    return kf

# Takes in the kf being used, the data, how many prediction "steps" it will make, and an optional initial value
# Gives a prediction values based on given parameters
def predict(kf, data, forward_steps, inital_val = None):
    for i in range(len(data)):
        data[i] = float(data[i]) # Makes sure all the data being worked with is a float
    if(inital_val != None):
        smoothed = kf.smooth(data, initial_value = [float(inital_val),0]) # If there's an initial value, smooth it along that value
    else:
        smoothed = kf.smooth(data) # If not, smooth it however you like
    predicted = kf.predict(data, forward_steps) # Make a prediction on the smoothed data
    return predicted

# Get data, make predictions, and then find the errors for these predictions 
def generate_residuals_for_given_data(kf, data):
    residuals = []
    initial_val = data[0]
    for item in range(len(data)-1):
        predicted = predict(kf, data[0:item+1], 1, initial_val)        
        actual_next_state = data[item+1]
        pred_mean = predicted.observations.mean
        residual_error = float(residual(pred_mean, actual_next_state))
        residuals.append(residual_error)
    if(len(residuals) == 0): # If there are no residuals because the data is of length 1, then just say residuals is equal to [0]
        residuals.append(0)
    return residuals

# Gets mean of values
def mean(values):
    return sum(values)/len(values)

# Gets absolute value residual from actual and predicted value
def residual(predicted, actual):
    return abs(float(actual) - float(predicted))

#Info: Potential static method for modular use.
#Purpose: Takes kalman filter model, past data (of any size, atm), a singular new data point, and some residuals (not required)
# then returns if there's an error in the new incoming data point 
def current_attribute_get_error(kf, previous_data, new_data, residuals = None):
    #If the data presented doesn't come with residuals generate some including past residuals 
    if(residuals == None or len(residuals) == 0):
        residuals = generate_residuals_for_given_data(kf, previous_data)
    #Predict next state
    predicted = predict(kf, previous_data, 1)
    pred_mean = predicted.observations.mean
    #Get error, then add that to running errors
    residual_error = float(residual(pred_mean, np.array(new_data)))
    residuals.append(residual_error)
    #Take the mean of all error
    mean_val = abs(mean(residuals))
    if (residuals[-1] - mean_val < 1 ):
        return False, residuals
    return True

#Info: takes a chunk of data of n size. Walks through it and gets residual errors.
#Takes the mean of the errors and determines if they're too large overall in order to determine whether or not there's a chunk in said error. 
def current_attribute_chunk_get_error(kf, data):
    residuals = generate_residuals_for_given_data(kf, data)
    mean_residuals = abs(mean(residuals))
    if (mean_residuals < 1.5):
        return False
    return True
