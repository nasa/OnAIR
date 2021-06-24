from pandas import read_csv
from numpy import dstack
import numpy as np
import os
import shutil 

def setup_folders(data_path):
    try:
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
        if not os.path.isdir(data_path + '/train/'):  
            os.mkdir(data_path + '/train/')
        if not os.path.isdir(data_path + '/test/'):  
            os.mkdir(data_path + '/test/')
    except:
        pass


def breakdown_folders(data_path):
    try:
        shutil.rmtree(data_path)
    except:
        pass

def write_to_file(x_data, y_data, exp_dir, train_or_test='train'):
    x_file = open(exp_dir + train_or_test + "/x_" + train_or_test + ".txt", "w")
    for element in x_data:
        datapt = str(element).replace('[', '  ').replace(']', '\n').replace(',', ' ')
        x_file.write(datapt)
    x_file.close()

    y_file = open(exp_dir + train_or_test + "/y_" + train_or_test + ".txt", "w")
    for element in y_data:
        y_file.write(str(element) + "\n")
    y_file.close()

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values
 
# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    loaded = dstack(loaded)
    return loaded
 
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/'
    filenames = list()
    filenames += ['x_'+group+'.txt']
    X = load_group(filenames, filepath)
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y
 
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    trainX, trainy = load_dataset_group('train', prefix)
    testX, testy = load_dataset_group('test', prefix)
    trainy = trainy - 1
    testy = testy - 1
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    # print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy
 

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical