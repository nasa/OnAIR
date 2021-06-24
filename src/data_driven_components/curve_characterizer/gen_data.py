import random 
import math 
import os
import numpy as np
import sys 

from src.data_driven_components.curve_characterizer.file_io import * 

# LINEAR INCREASE LABEL: 1
def linear_increase(num_samples, frame_size=10):
    x_data = []
    y_data = []

    for i in range(num_samples):
        sample = []
        epsilon = random.randint(1,100)
        delta = math.floor(i + (i + epsilon)/2)
        for j in range(frame_size):
            sample.append(float(i + j*delta))
        x_data.append(sample)
        y_data.append(1)

    return x_data, y_data

# LINEAR DECREASE LABEL: 2
def linear_decrease(num_samples, frame_size=10):
    x_data = []
    y_data = []

    for i in range(num_samples):
        sample = []
        epsilon = random.randint(1,100)
        delta = math.floor(i + (i + epsilon)/2)
        for j in range(frame_size):
            sample.append(float(i + j*delta))
        sample.reverse()
        x_data.append(sample)
        y_data.append(2)
        
    return x_data, y_data


# SINUSOIDAL LABEL: 3
def sinusoidal(num_samples, frame_size=10):
    x_data = []
    y_data = []

    for i in range(num_samples):
        sample = []
        epsilon = random.randint(10,20) #epsilon between 1 and 10
        for j in range(frame_size):
            behavior = j%3 - 1
            sample.append(float(i + behavior*epsilon))
        x_data.append(sample)
        y_data.append(3)

    return x_data, y_data

# FLAT LABEL: 4
def flat(num_samples, frame_size=10):
    x_data = []
    y_data = []

    for constant in range(num_samples):
        sample = []

        epsilons = []
        epsilons.append(random.randint(1,6))
        epsilons.append(random.random())

        should_disrupt = random.randint(0,1)

        disprupt_values = []
        disprupt_values.append(random.randint(0,9))
        disprupt_values.append(random.randint(0,9))
        
        for j in range(frame_size):
            if (j in disprupt_values) and (should_disrupt == 1):
                ep_size = random.randint(0,1)
                sample.append(float(constant + epsilons[ep_size]))
            else:
                sample.append(float(constant))
        
        x_data.append(sample)
        y_data.append(4)

    return x_data, y_data

def gen_data(samples, frame_size=10):
    x_data = []
    y_data = []

    x,y = linear_increase(samples, frame_size)
    x_data += x
    y_data += y

    x,y = linear_decrease(samples, frame_size)
    x_data += x
    y_data += y

    x,y = sinusoidal(samples, frame_size)
    x_data += x
    y_data += y

    x,y = flat(samples, frame_size)
    x_data += x
    y_data += y

    return x_data, y_data







