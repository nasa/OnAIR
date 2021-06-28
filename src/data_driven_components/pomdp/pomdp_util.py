## Nick Pellegrino
## pomdp_util.py

import os
import csv
import random
import copy

## Load All The Data
def mass_load_data(folder, lookback, filetype=".csv"):
    data = []
    config, dict_config = load_config(folder)
    for file in os.listdir(folder):
        if file.find(filetype) != -1 and file != "config.csv":
            file_data = load_data(folder + file)
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

def stratified_sampling(config, data):
    error_data = []
    no_error_data = []
    label_key = check_label(config)
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
    random.shuffle(error_data)
    random.shuffle(no_error_data)
    min_len = len(error_data)
    if min_len % 2 != 0:
        min_len -= 1
    if len(no_error_data) < min_len:
        min_len = len(no_error_data)
    training_data = []
    testing_data = []
    for i in range(min_len-1):
        training_data.append(error_data[i])
        training_data.append(no_error_data[i])
        i += 1
        testing_data.append(error_data[i])
        testing_data.append(no_error_data[i])
    return training_data, testing_data

def stratified_sampling_threes(config, data):
    error_data = []
    no_error_data = []
    label_key = check_label(config)
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
    random.shuffle(error_data)
    random.shuffle(no_error_data)
    min_len = len(error_data)
    while min_len % 3 != 0:
        min_len -= 1
    if len(no_error_data) < min_len:
        min_len = len(no_error_data)
    training_data = []
    kappa_testing_data = []
    forest_testing_data = []
    for i in range(min_len-2):
        training_data.append(error_data[i])
        training_data.append(no_error_data[i])
        i += 1
        kappa_testing_data.append(error_data[i])
        kappa_testing_data.append(no_error_data[i])
        i += 1
        forest_testing_data.append(error_data[i])
        forest_testing_data.append(no_error_data[i])
    return training_data, kappa_testing_data, forest_testing_data

## Load the config
def load_config(folder):
    try:
        config = load_data(folder + "config.csv")
    except:
        print("Error: Please put a config.csv into the data folder!")
        exit()
    dict_config = {}
    for i in range(len(config[0])):
        dict_config[config[0][i]] = [config[1][i], config[2][i], config[3][i]]
    return config, dict_config

def check_label(config):
    label_key = "Colomar"
    for key in config:
        if config[key][0] == "label":
            label_key = key
            break
    if label_key == "Colomar":
        print("Error: No label column found in config.csv!")
        exit()
    return label_key

## data_train = list of frames, with headers and labels as described in a POMDP's self.config
def split_by_lookback(data_train, lookback):
    new_data = []
    for i in range(lookback, len(data_train)+1):
        new_data.append(data_train[i-lookback:i])
    return new_data
