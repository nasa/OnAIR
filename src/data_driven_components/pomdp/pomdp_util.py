## Nick Pellegrino
## pomdp_util.py

import os
import csv
import random
import copy
import ast
import torch
from src.data_driven_components.vae.vae_model import VAEModel


## Load All The Data
def mass_load_data(folder, lookback, filetype=".csv"):
    data = []
    config, dict_config = load_config(os.path.join(folder, "config.csv"))
    for file in os.listdir(folder):
        if file.find(filetype) != -1 and file != "config.csv":
            file_data = load_data(os.path.join(folder, file))
            for i in range(1+lookback, len(file_data)):
                new_point = {}
                for k in range(len(config[0])):
                    new_point[config[0][k]] = []
                for j in range(lookback):
                    frame = file_data[i-lookback+j]
                    for k in range(len(config[0])):
                        new_point[config[0][k]].append(frame[k])
                data.append(new_point)
    return dict_config, data

# Dictionary-Sort Data
def dict_sort_data(dict_config, data):
    output_data = []
    for time_chunk in data:
        # time_chunk is a list of N frames for some N > 0
        attributes = [[] for i in range(len(time_chunk[0]))]
        for frame in time_chunk:
            for i in range(len(frame)):
                attributes[i].append(frame[i])
        new_point = {}
        index_for_key = 0
        for key in dict_config:
            new_point[key] = attributes[index_for_key]
            index_for_key += 1
        output_data.append(new_point)
    return output_data

def dict_to_3d_tensor(data):
    #Get window size
    window_size = 0
    for key in data[0]:
        window_size = len(data[0][key])
        break
    headers = []
    if len(data) > 0:
        headers = list(data[0].keys())
    list_of_all_data_frames = []
    for i in range(len(data)):
        inner_frame = []
        for _ in range(window_size):
            data_frame = []
            for key in data[i]:
                data_frame.append(data[i][key].pop(0))
            inner_frame.append(data_frame)
        list_of_all_data_frames.append(inner_frame)
    return list_of_all_data_frames, headers, window_size

## Load data from a .csv file
def load_data(file_path, delimiter=',', quotechar='\"'):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        data = []
        for row in reader:
            data.append(row)
        return data

## Save data to a .csv file
def save_data(file_path, data, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL):
    with open(file_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter, quotechar=quotechar, quoting=quoting)
        for row in data:
            writer.writerow(row)

def split_headers(headers, depth):
    master_list = copy.deepcopy(headers)
    arrayed_headers = []
    for i in range(len(headers)):
        arrayed_headers.append([headers[i]])
    for i in range(1, depth):
        arrayed_headers = split_headers_helper(arrayed_headers, master_list, i)
    return arrayed_headers

def split_headers_helper(arrayed_headers, master_list, current_depth):
    new_headers = []
    for i in range(len(arrayed_headers)):
        z = master_list.index(arrayed_headers[i][len(arrayed_headers[i])-1]) + 1
        for j in range(z, len(master_list)):
            new_header = copy.deepcopy(arrayed_headers[i])
            new_header.append(master_list[j])
            new_headers.append(new_header)
    return new_headers

# data should be split by lookback at this point
def stratified_sampling(config, data, print_on=True):
    error_data = []
    no_error_data = []
    label, label_key = check_label(config)
    if label:
        for i in range(len(data)):
            error = False
            for j in range(len(data[i][label_key])):
                if data[i][label_key][j] == '1':
                    error = True
                    break
            if error:
                error_data.append(data[i])
            else:
                no_error_data.append(data[i])
    else:
        label_list = get_vae_error_over_each_point(data)
        for i in range(len(data)):
            if label_list[i]:
                error_data.append(data[i])
            else:
                no_error_data.append(data[i])
    random.shuffle(error_data)
    random.shuffle(no_error_data)
    min_len = len(error_data)
    if min_len % 2 != 0:
        min_len -= 1
    if len(no_error_data) < min_len:
        min_len = len(no_error_data)
    output_data = []
    for i in range(min_len):
        output_data.append(error_data[i])
        output_data.append(no_error_data[i])
    if ((print_on) and (output_data == [])):
        print("WARNING!!! Not enough diverse data for stratified sampling, defaulting to unsampled data.")
        print("This will lead to suboptimal training.")
        output_data = data
    return output_data

## Load the config
def load_config(config_path):
    try:
        config = load_data(config_path)
    except:
        print("Error: config.csv not found at " + config_path)
        exit()
    dict_config = {}
    for i in range(len(config[0])):
        dict_config[config[0][i]] = [config[1][i], config[2][i], config[3][i]]
    return config, dict_config

def load_config_from_txt(config_path):
    #Config should be in the format {Header:[data, lower_thresh, upper_thresh, index associated with header]}
    #The index associated with the header will be set in the pomdp.py class
    config_dictionary = {}
    joined_config_path = os.path.join(os.environ['RUN_PATH'], config_path)
    config = open(joined_config_path,"r")
    config_text = config.read().split("\n")
    for attribute in config_text:
        split_attribute = attribute.split(" ")
        if(len(split_attribute) == 3):
            third_attribute = ast.literal_eval(split_attribute[2])
            data_type = ''
            lower_thresh = None
            upper_thresh = None
            if (len(third_attribute) == 2 and third_attribute[1] == "LABEL"):
                data_type = 'label'
                lower_thresh = 0
                upper_thresh = 1
            elif (len(third_attribute) == 2 and third_attribute[1] == "TIME"):
                data_type = 'time'
            elif (len(third_attribute) > 2 and third_attribute[0] == "FEASIBILITY"):
                data_type = 'data'
                lower_thresh = third_attribute[1]
                upper_thresh = third_attribute[len(third_attribute)-1]
            else:
                data_type = 'ignore'
            if upper_thresh != None and lower_thresh !=None:
                config_dictionary[split_attribute[0]] = [data_type, lower_thresh, upper_thresh]
            else:
                config_dictionary[split_attribute[0]] = [data_type, '', '']
    return config_dictionary

def check_label(config):
    label = False
    label_key = "Colomar"
    for key in config:
        if config[key][0] == "label":
            label = True
            label_key = key
            break
    return label, label_key

def get_vae_error_over_all_data(data):
    data = copy.deepcopy(data)
    tensor_data, headers, window_size = dict_to_3d_tensor(data)
    data_to_pass = torch.tensor(tensor_data, dtype=torch.float)
    VAE = VAEModel(headers, window_size)
    VAE.apriori_training(data_to_pass) # check for model first, if it doesnt exist break
    error = VAE.model(data_to_pass)
    if error.item() > 0.0000001:
        return True
    return False

def get_vae_error_over_each_point(data):
    data = copy.deepcopy(data)
    tensor_data, headers, window_size = dict_to_3d_tensor(data)
    data_to_pass = torch.tensor(tensor_data, dtype=torch.float)
    VAE = VAEModel(headers, window_size)
    VAE.apriori_training(data_to_pass) # check for model first, if it doesnt exist break
    label_list = []
    for data_point in tensor_data:
        data_point = [data_point]
        data_point_to_pass = torch.tensor(data_point, dtype=torch.float)
        #Use VAE to populate label_list
        error = VAE.model(data_point_to_pass)
        if error.item() > 0.0000001:
            label_list.append(True)
        else:
            label_list.append(False)
    return label_list

## data_train = list of frames, with headers and labels as described in a POMDP's self.config
def split_by_lookback(data_train, lookback):
    new_data = []
    for i in range(lookback, len(data_train)+1):
        new_data.append(data_train[i-lookback:i])
    return new_data

# Takes in a list of numbers/data [1,2,3...] where each number is associated with a header in the same index position
# The frame of data should be in the format {'Header' : [x, x2, x3...] (for size of window)}
def list_to_dictionary_with_headers(list_of_numbers, headers, dictionary, window_size):

    for h in range(len(headers)):
        #Example: If header = 'Time' and time is not in dictionary
        if headers[h] in dictionary:
            dictionary[headers[h]].append(list_of_numbers[h])
            if(len(dictionary[headers[h]])>window_size): # If the frame of data is greater than the window size, it pops the first element of the list
                dictionary[headers[h]].pop(0)
        else:
            dictionary[headers[h]] = []
            dictionary[headers[h]].append(list_of_numbers[h])
            if(len(dictionary[headers[h]])>window_size):
                dictionary[headers[h]].pop(0)
    return dictionary
