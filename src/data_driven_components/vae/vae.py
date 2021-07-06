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

def loadVAE(path, headers=[], window_size=10, 
                       z_units=5, hidden_units=100):
    """
    Loads VAE from previous save if possible, if not, creates a new LSTM-VAE
        with given parameters
    Make sure seq_len is always the same, TODO: accept any seq_len
    :param path: (string) path of vae save relative to src directory
    :param headers: (string list) list of headers for each input feature
    :param window_size: (int) number of data points in our data sequence
    :param z_units: (int) dimensions of our latent space gaussian representation
    :param hidden_units: (int) dimension of our hidden_units
    """
    try:
        checkpoint_path = os.path.join(os.environ['SRC_ROOT_PATH'], 'src/data_driven_components/vae/models/checkpoint_latest.pth.tar')
        model = VAE(headers, window_size, z_units, hidden_units)
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        return model
    except:
        return VAE(headers, window_size, z_units, hidden_units)



#import shap
# shap.initjs() # TODO deal with viz

class VAE(nn.Module):
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
        super(VAE, self).__init__()
        self.headers = headers
        self.window_size = window_size
        self.frames = [[0.0]*len(headers) for i in range(self.window_size)]

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

    ########################################################################
    #### RAISR FUNCTIONS ####
    def apriori_training(self, data_train):
        """
        :param data_train: (list of lists) input sequence of shape (batch_size, seq_len, input_dim)
        :return: None
        """

        return

        try:
            self.load_state_dict(torch.load(os.path.join(os.environ['SRC_ROOT_PATH'], 'src/data_driven_components/vae/models/checkpoint_latest.pth.tar')))
        except:
            _batch_size = len(data_train[0])
            _input_dim = len(data_train[0][0])

            transform = lambda x: torch.tensor(x).float()
            train_dataset = TimeseriesDataset(data_train, transform)
            train_dataloader = DataLoader(train_dataset, batch_size=_batch_size)
            train(self, {'train': train_dataloader}, phases=["train"], checkpoint=True)

    def update(self, frame):
        """
        :param frame: (list of floats) input sequence of len (input_dim)
        :return: None
        """
        self.frames.append(frame)
        self.frames.pop(0)

    def render_diagnosis(self):
        """
        :param None
        :return: 1; stub
        """
        seq = [self.frames]
        transform = lambda x: torch.tensor(x).float()
        prediction_dataset = TimeseriesDataset(seq, transform)
        dataloader = DataLoader(prediction_dataset, batch_size=1)
        # for x in dataloader:
        #     self(x)
        #     print("******** predicted ********")
        #     e = VAEExplainer(self, self.headers)
        #     # print(e.shap(x, bass))
        #     e.viz()

        return 1
    ########################################################################


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
        _, (hidden_state, _) = self.enc2(x)
        hidden_state = torch.transpose(hidden_state, 0, 1) # Hidden state has batch in middle
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
        batch_size = x.shape[0]
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

        return mse_loss + kldivergence_loss

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

