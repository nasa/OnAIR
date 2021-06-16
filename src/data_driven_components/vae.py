"""
LSTM-VAE for streamed satellite telemetry fault diagnosis
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from captum.attr import KernelShap
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap
import functools
import os
from datetime import datetime 
from torch.utils.tensorboard import SummaryWriter

shap.initjs() # TODO deal with viz

class VAE(nn.Module):
    def __init__(self, input_dim=30, seq_len=15, z_units=5):
        """
        LSTM-VAE class for anomaly detection and diagnosis
        Make sure seq_len is always the same, TODO: accept any seq_len
        :param input_dim: (int) number of input features
        :param seq_len: (int) number of data points in our data sequence
        :param z_units: (int) dimensions of our latent space gaussian representation
        """
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = z_units * 2 # good rule of thumb
        self.z_units = z_units
        self.seq_len = seq_len # TODO: maybe detect automatically from data?

        # batch_first = true makes output tensor of size (batch, seq, feature).
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

        self.mulinear = nn.Linear(self.z_units, self.z_units)
        self.logvarlinear = nn.Linear(self.z_units, self.z_units)

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
        x, (_, _) = self.enc1(x)
        _, (hidden_state, _) = self.enc2(x)
        mu = self.mulinear(hidden_state)
        logvar = self.logvarlinear(hidden_state)
        
        return (mu, logvar)

    def reparametrize(self, mu, logvar):
        """
        :param mu: (Tensor) Latent space average tensor of shape (batch_size, 1, z_units)
        :param logvar: (Tensor) Logvariance of representation, shape (batch_size, 1, z_units)
        :return: (Tensor) Sampling from our Gaussian (batch_size, 1, z_units)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

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
        self.input = x
        mu, logvar = self.encoder(x)
        self.mu = mu
        self.logvar = logvar
        x = self.reparametrize(mu, logvar)
        x = x.repeat(1,self.seq_len,1)
        x = self.decoder(x)
        self.output = x
        mse = nn.functional.mse_loss(self.input, self.output)
        return mse

    def reconstruction(self):
        """
        Return the autoencoders reconstruction
        :return: (Tensor) reconstructed tensor of shape (batch_size, seq_len, input_dim)
        """
        return self.output

    def loss(self):#, x, mu, logvar, output):
        """
        Combination of mse (reconstruction error) between input and output and 
            KL-divergence of N(0,1) and N(mu, logvar)
        :param x: (Tensor) input sequence of shape (batch_size, seq_len, input_dim)
        :param mu: (Tensor) Latent space average tensor of shape (batch_size, 1, z_units)
        :param logvar: (Tensor) Logvariance of representation, shape (batch_size, 1, z_units)
        :param output: mse+KL divergence loss
        """
        x = self.input
        mu = self.mu
        logvar = self.logvar
        output = self.output
        num_batches = x.shape[0]
        mse_loss = nn.functional.mse_loss(x, output)
        mse_loss = torch.mean(mse_loss, dim=0)
        kldivergence_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        kldivergence_loss = torch.sum(kldivergence_loss, dim=0)
        # TODO: double check batch size averaging
        return mse_loss + kldivergence_loss

class VAEExplainer():
    def __init__(self, vae, n_features=30, n_samples=200):
        """
        Takes in vae model to explain.
        :param vae: (VAE) vae model
        :param n_features: (optional int) number of features for a sequence input, defaults to 30
        :param n_samples: (optional int) number of times to evaluate model, defaults to 200
        """
        self.explainer = KernelShap(vae)
        self.n_features = n_features
        self.n_samples = n_samples
    
    def shap(self, input, baseline):
        """
        Calculate shapley values for a given input as compared to baseline
        :param input: (Tensor) input shape (batch_size, seq_len, input_dim)
        :param baseline: (Tensor) baseline sample shape (batch_size, seq_len, input_dim)
        """
        self.input = input
        self.shap_values = self.explainer.attribute(input, baseline, n_samples=self.n_samples)
        return self.shap_values

    def viz(self):
        """
        Return values to visualize previously calculated shapley values
        To plot, call shap.force_plot(0, shap_values, data, data_names)
        :return: (shap_values, data, data_names) shap_values array of shape (n_features,) with shapley
                value for each feature, data array of shape (n_features,) with data of each feature, data_names array (n_features,) with name of each feature
        """
        shap_values = self.shap_values.detach().numpy().reshape((self.n_features))
        data = self.input.detach().numpy().reshape((self.n_features))
        data_names = list(range(self.n_features))
        return (shap_values, data, data_names)

def train(vae, loaders, epochs=20, lr=1e-1, checkpoint=False, phases=["train", "val"]):
    """
    Training loop util
    :param loaders: {train: train_loader, val: val_loader} data loaders in dictionary
    :param epochs: (optional int) number of epochs to train for, defaults to 20
    :param lr: (optional float) learning rate, defaults to 1e-1
    :param checkpoint: (optional bool) save model to directory, defaults to False
    :param phases: (string list) phases in training, defaults to ["train", "val"],
                each phase should have a corresponding data loader
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"runs")

    e = datetime.now()

    writer = SummaryWriter(os.path.join(checkpoint_dir, "{}-{}-{}_{}:{}:{}".format(e.day, e.month, e.year, e.hour, e.minute, e.second)))

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch_counter in tqdm(range(epochs)):
        for phase in phases:
            if phase == "train":
                vae.train(True)
            else:
                vae.train(False)

            running_loss = 0.0

            for x in loaders[phase]:
                if phase == "train":
                    vae(x)
                    loss = vae.loss()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        vae(x)
                        loss = vae.loss()

                writer.add_scalar('Loss/' + phase, loss, epoch_counter)

                running_loss += loss

            avg_loss = running_loss / len(loaders[phase])

        if checkpoint:
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            torch.save({
                'epoch': epoch_counter,
                'state_dict': vae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }, os.path.join(checkpoint_dir, checkpoint_name))

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

if __name__ == "__main__":
    data = range(30)
    data = [[list(data)]] # serrated shape

    data2 = [1]*30
    data2 = [[data2]] # uniform

    transform = lambda x: torch.tensor(x).float()
    train_dataset = TimeseriesDataset(data, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1)

    test_dataset = TimeseriesDataset(data2, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    print("Creating VAE...")
    vae = VAE(input_dim=30, seq_len=1, z_units=5)
    print("Successfuly created VAE")

    train(vae, {'train': train_dataloader}, phases=["train"])

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

