"""
Utility file for translating data into different formats
"""

import numpy as np
import copy 

def prep_apriori_training_data(data, window_size):
    batch_data = copy.deepcopy(data)
    extra_frames = len(batch_data)%window_size
    batch_data = batch_data[:len(batch_data)-extra_frames]

    batch_data = [floatify_input(elem) for elem in batch_data]
    num_batches = int(len(batch_data)/window_size)
    num_features = len(batch_data[0])
    batch_data = np.array(batch_data).reshape(num_batches, window_size, num_features)
    batch_data = list(batch_data)

    return batch_data

def floatify_input(_input, remove_str=False):
    floatified = []
    for i in _input:
        if type(i) == str:
            try:
                x = float(i)
                floatified.append(x)
            except:
                try:
                    x = i.replace('-', '').replace(':', '').replace('.', '')
                    floatified.append(float(x))
                except:
                    if remove_str == False:
                        floatified.append(0.0)
                    else:
                        continue
                    continue
        else:
            floatified.append(float(i))
    return floatified
