"""
LSTM-VAE for streamed satellite telemetry fault diagnosis
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from src.data_driven_components.vae.vae_train import train
from src.data_driven_components.vae.vae_diagnosis import VAEExplainer

import matplotlib.pyplot as plt

#import shap
# shap.initjs() # TODO deal with viz

class AE(nn.Module):
    def __init__(self, headers=[], window_size=10, 
                       z_units=5, hidden_units=100):
        """
        LSTM-VAE class for anomaly detection and diagnosis
        Make sure seq_len is always the same, TODO: accept any seq_len
        :param headers: (string list) list of headers for each input feature
        :param window_size: (int) number of data points in our data sequence
        :param z_units: (int) dimensions of our latent space gaussian representation
        :param hidden_units: (int) dimension of our hidden_units
        """
        super(AE, self).__init__()
        self.headers = headers
        self.window_size = window_size

        self.hidden_dim = hidden_units
        self.z_units = z_units
        self.input_dim = len(headers)
        self.seq_len = self.window_size

        # batch_first = true makes output tensor of size (batch, seq, feature).
        self.norm = nn.BatchNorm1d(self.seq_len)
        self.enc1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.enc2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.z_units,
            num_layers=1,
            batch_first=True
        )

        self.dec1 = nn.LSTM(
            input_size=self.z_units,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.dec2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.input_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.input_dim, self.input_dim) # TODO: add activation, maybe relu?

    def encoder(self, x):
        """
        :param x: (Tensor) input sequence of shape (batch_size, seq_len, input_dim)
        :return: (average, logvar) each of shape (batch_size, 1, z_units)
        """
        #x = x.reshape((1, self.seq_len, self.input_dim)) # TODO: should move to data processing

        # LSTM output is tuple (output, (hidden state, cell state))
        x = self.norm(x)
        self.input = x
        x, (_, _) = self.enc1(x)
        x, (_, _) = self.enc2(x)
        
        return x

    def decoder(self, x):
        """
        :param x: (Tensor) sampling sequence of shape (batch_size, seq_len, z_units)
        :return: (Tensor) output sequence of shape (batch_size, seq_len, input_dim)
        """
        x, _ = self.dec1(x)
        x, _ = self.dec2(x)

        return self.output_layer(x)

    def forward(self, x):
        """
        :param x: (Tensor) input sequence of shape (batch_size, seq_len, input_dim)
        :return: (Tensor, Float) tuple of output sequence of shape (batch_size, seq_len, input_dim)
                and mse reconstruction error
        """
        batch_size = x.shape[0]
        x = self.encoder(x)
        x = self.decoder(x)
        self.output = x
        mse = nn.functional.mse_loss(self.input, self.output, reduction='none')
        mse = torch.sum(mse, dim=(1,2))
        return mse

    def reconstruction(self):
        """
        Return the autoencoders reconstruction
        :return: (Tensor) reconstructed tensor of shape (batch_size, seq_len, input_dim)
        """
        return self.output

    def loss(self):#, x, mu, logvar, output):
        """
        mse (reconstruction error) between input and output 
        :param x: (Tensor) input sequence of shape (batch_size, seq_len, input_dim)
        :param output: mse+KL divergence loss
        """
        x = self.input
        output = self.output
        num_batches = x.shape[0]
        mse_loss = nn.functional.mse_loss(x, output)
        mse_loss = torch.mean(mse_loss, dim=0)

        return mse_loss

class TimeseriesDataset(Dataset):
    def __init__(self, data, transform = None):
        """
        Timeseries dataset class, contains sequential data and applies transformation before returning datum
        :param data: (Arraylike) 3D data container, first dimension is datum, second is time, third is features
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        if self.transform:
            datum = self.transform(datum)
        return datum



# if __name__ == "__main__":
#     data = range(30)
#     data = [[list(data)]] # serrated shape

#     data2 = [1]*30
#     data2 = [[data2]] # uniform

#     transform = lambda x: torch.tensor(x).float()
#     train_dataset = TimeseriesDataset(data, transform)
#     train_dataloader = DataLoader(train_dataset, batch_size=1)

#     test_dataset = TimeseriesDataset(data2, transform)
#     test_dataloader = DataLoader(test_dataset, batch_size=1)

#     print("Creating VAE...")
#     vae = VAE(input_dim=30, seq_len=1, z_units=5)
#     print("Successfuly created VAE")

#     train(vae, {'train': train_dataloader}, phases=["train"])

    """
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-1)
    vae.train(True)
    print('Begin training')
    for i in range(500):
        for x in train_dataloader:
            bass = x
            vae(x)
            loss = vae.loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vae.train(False)
    print('Begin testing')
    for x in test_dataloader:
        vae(x)
        e = VAEExplainer(vae)
        print(e.shap(x, bass))
        e.viz()
    """

