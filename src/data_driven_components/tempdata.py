### Non permanent file TODO: delete this
### TODO: if this is in a pull request, delete it

### Just waiting for official RAISR data loading

import csv
import os

import vae
import torch
from vae import TimeseriesDataset, VAE, train, VAEExplainer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import shap
from torch import nn
import matplotlib.pyplot as plt
from captum.attr import KernelShap, DeepLift, DeepLiftShap

shap.initjs()

## Load All The Data
def massLoadData(folder, lookback):
    data = []
    headers = []
    for file in os.listdir(folder):
        if file.find(".csv") != -1:
            fileData = loadData(folder + file)
            if headers == []:
                headers = fileData[0]
            for i in range(1+lookback, len(fileData)):
                newPoint = []
                for j in range(lookback):
                    newPoint.append(fileData[i-lookback+j])
                data.append(newPoint)
    return headers, data

## Load data from a .csv file
def loadData(filepath, delimiter=',', quotechar='\"'):
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        data = []
        for row in reader:
            data.append(row)
        return data

if __name__ == "__main__":
    basePath = os.path.dirname('/Users/gabriel/Desktop/NASA/RAISR-2.0/src/data_driven_components')
    headers, data = massLoadData(os.path.join(basePath, "data/data_physics_generation/No_Errors/"), 1)
    data = np.array(data)
    data = data[:,:,1:8].astype(np.float)
    data = torch.tensor(data)
    data = data.float()
    headers = headers[1:8]
    transform = lambda x: x.float()
    train_dataset = TimeseriesDataset(data)
    train_dataloader = DataLoader(train_dataset, batch_size=1000)
    vae = VAE(input_dim=7, seq_len=1, z_units=3)
    train(vae, {'train': train_dataloader}, epochs=2, phases=["train"], checkpoint=True)