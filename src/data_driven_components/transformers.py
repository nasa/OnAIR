import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch
from torch import nn
from vae import *
import numpy as np
from tqdm import tqdm

class TransformerModel(nn.Module):

    def __init__(self, input_dim=30, seq_len=15, z_units=5, hidden_units=100, att_heads=3, hidden_layers=2):
        """
        Vanilla-ish transformer model, see https://arxiv.org/pdf/1706.03762.pdf figure 1
        Note: pytorch 1.7 does not support batch_first, make sure to pass data (seq_len, batch, features)
        :param input_dim: (optional int) Number of input features, default 100
        :param seq_len: (int) number of data points in our data sequence
        :param z_units: (optional int) Size of latent distribution (posterior), default 5
        :param hidden_unit: (optional int) Size of transformer feedforwards, default 100
        :param att_head: (optional int) Number of attention heads
        :param hidden_layers: (optional int) Depth of transformer encoder/decoder, default 1
        """
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.input_dim = input_dim
        self.z_units = z_units

        self.norm = nn.BatchNorm1d(seq_len)

        self.pos_encoder = PositionalEncoding(input_dim)

        encoder_layers = TransformerEncoderLayer(input_dim, att_heads, dim_feedforward=hidden_units)
        self.transformer_encoder = TransformerEncoder(encoder_layers, hidden_layers)

        self.mulinear = nn.Linear(self.input_dim, self.z_units)
        self.logvarlinear = nn.Linear(self.input_dim, self.z_units)

        self.latentencoder = nn.Linear(self.z_units, self.input_dim)

        decoder_layers = TransformerDecoderLayer(input_dim, att_heads, dim_feedforward=hidden_units)
        self.transformer_decoder = TransformerDecoder(decoder_layers, hidden_layers)

        self.out_linear = nn.Linear(input_dim, input_dim) # Single output, TODO: add reconstruction error output

    def generate_square_subsequent_mask(self, seq_len):
        """
        Prevent peak-ahead in decoder. Masked tokens are -inf, unmasked are 0.
        :param seq_len: (int) Sequence length # Since this is an autoencoder this'll be input seq_len
        """
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_right_shift_data(self, x):
        """
        Equivalent to adding start token to sequence, decoder input must be right shifted to avoid
            information leak
        :param x: (Tensor) sequence to augment, must be of shape (seq_len, batch_size, input_dim)
        :return: (Tensor) tensor of shape (seq_len+1, batch_size, input_dim)
        """
        start_token = torch.Tensor([0]*self.input_dim).unsqueeze(0).unsqueeze(0)
        start_token = start_token.repeat(1,x.shape[1], 1)
        return torch.cat((start_token, x))

    def forward(self, x, x_mask, target_mask):
        """
        Assume x: (seq_len, batch_size, input_dim)
        """
        #x = x.transpose(0,1) # put batch first for batchnorm
        x = self.norm(x)
        x = x.transpose(0,1) # put batch second for transformer encoder
        self.input = x

        x = x * math.sqrt(self.input_dim) # scale up
        x = self.pos_encoder(x)

        target = self.generate_right_shift_data(x)

        encoded = self.transformer_encoder(x, x_mask)
        encoded = torch.sum(encoded, 0) # 1nd dimension is seq len since we cant do batch first
        encoded = encoded.unsqueeze(0)

        mu = self.mulinear(encoded)
        logvar = self.logvarlinear(encoded)
        self.mu = mu
        self.logvar = logvar
        sample = self.reparametrize(mu, logvar)

        latentEncoded = self.latentencoder(sample) # upsample from R^z_dim -> R^input_dim

        decoded = self.transformer_decoder(target, latentEncoded, target_mask)
        output = self.out_linear(decoded)
        output = output[:-1,:,:]
        self.reconstruction = output
        mse = nn.functional.mse_loss(self.input, self.reconstruction)
        return mse

    def reparametrize(self, mu, logvar):
        """
        :param mu: (Tensor) Latent space average tensor of shape (batch_size, 1, z_units)
        :param logvar: (Tensor) Logvariance of representation, shape (batch_size, 1, z_units)
        :return: (Tensor) Sampling from our Gaussian (batch_size, 1, z_units)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss(self):
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
        output = self.reconstruction

        num_batches = x.shape[1]
        mse_loss = nn.functional.mse_loss(x, output)
        mse_loss = torch.mean(mse_loss, dim=0)
        kldivergence_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        kldivergence_loss = torch.sum(kldivergence_loss, dim=0)
        return mse_loss #+ kldivergence_loss

class PositionalEncoding(nn.Module):

    def __init__(self, input_dim, dropout=0.1, max_len=5000):
        """
        Generate positional encoding for each element in the sequence, flunctuating sin and cosine wave that 
            encode position
        :param input_dim: (int) number of input features
        :param dropout: (float) dropout for encoding
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-math.log(10000.0) / input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)

        if input_dim % 2: # if odd remove last term
            pe[:, 1::2] = torch.cos(position * div_term)[:,:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x) # Better for training robust representations

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

class TransformerExplainer():
    def __init__(self, vae, headers, n_features=7, seq_len=1, n_samples=200):
        """
        Takes in vae model to explain.
        :param vae: (VAE) vae model
        :param headers: (string list) ordered list of headers, must have n_features elements
        :param n_features: (optional int) number of features for a sequence input, defaults to 30
        :param seq_len: (optional int) number of sequence components per input
        :param n_samples: (optional int) number of times to evaluate model, defaults to 200
        """
        self.explainer = KernelShap(vae)
        self.headers = headers
        self.n_features = n_features
        self.seq_len = seq_len
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

    def makeLongHeaders(self):
        """
        Make sequential headers from single header list
        """
        long_header = []
        for t in range(self.seq_len):
            long_header += [str(t) + '_' + h for h in self.headers]
        return long_header

    def viz(self, average=False):
        """
        Return values to visualize previously calculated shapley values
        To plot, call shap.force_plot(0, shap_values, data, data_names)
        :param average: (bool) if seq_len > 1, whether to average data and shap over sequence length
        :return: (shap_values, data, data_names) shap_values array of shape (n_features,) with shapley
                value for each feature, data array of shape (n_features,) with data of each feature, data_names array (n_features,) with name of each feature
        """
        if self.seq_len == 1:
            # Point data
            shap_values = self.shap_values.detach().numpy().reshape((self.n_features))
            data = self.input.detach().numpy().reshape((self.n_features))
            data_names = self.headers
        elif average:
            # Averaging timeseries
            shap_values = self.shap_values.detach().numpy().reshape((self.seq_len, self.n_features)).sum(axis=0)/self.seq_len
            data = self.input.detach().numpy().reshape((self.seq_len, self.n_features)).sum(axis=0)/self.seq_len
            data_names = self.headers
        else:
            # Timeseries data we don't want to average
            shap_values = self.shap_values.detach().numpy().reshape((self.seq_len*self.n_features))
            data = self.input.detach().numpy().reshape((self.seq_len*self.n_features))
            data_names = self.makeLongHeaders()

        return (shap_values, data, data_names)

def train(vae, seq_len, loaders, epochs=20, lr=1e-1, checkpoint=False, phases=["train", "val"]):
    """
    Training loop util
    :param vae: Transformer autoencoder
    :param seq_len: (int) length of sequence for mask
    :param loaders: {train: train_loader, val: val_loader} data loaders in dictionary
    :param epochs: (optional int) number of epochs to train for, defaults to 20
    :param lr: (optional float) learning rate, defaults to 1e-1
    :param checkpoint: (optional bool) save model to directory, defaults to False
    :param phases: (string list) phases in training, defaults to ["train", "val"],
                each phase should have a corresponding data loader
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"runs")

    e = datetime.now()
    run_dir = os.path.join(checkpoint_dir, "{}-{}-{}_{}:{}:{}".format(e.day, e.month, e.year, e.hour, e.minute, e.second))

    writer = SummaryWriter(run_dir)
    print("Starting training, see run at", run_dir)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    mask = vae.generate_square_subsequent_mask(seq_len)
    tar_mask = vae.generate_square_subsequent_mask(seq_len+1)

    for epoch_counter in tqdm(range(epochs)):
        for phase in phases:
            if phase == "train":
                vae.train(True)
            else:
                vae.train(False)

            running_loss = 0.0

            for x in loaders[phase]:
                if phase == "train":
                    vae(x, mask, tar_mask)
                    loss = vae.loss()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        vae(x, mask, tar_mask)
                        loss = vae.loss()

                running_loss += loss

            avg_loss = running_loss / len(loaders[phase])

            writer.add_scalar('Loss/' + phase, avg_loss, epoch_counter)

        if checkpoint:
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            torch.save({
                'epoch': epoch_counter,
                'state_dict': vae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }, os.path.join(run_dir, checkpoint_name))

if __name__ == "__main__":
    data = range(30)
    data = [[list(data), list(data)]] # serrated shape

    data2 = [1]*30
    data2 = [[data2]] # uniform

    transform = lambda x: torch.tensor(x).float()
    train_dataset = TimeseriesDataset(data, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1)

    test_dataset = TimeseriesDataset(data2, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1)


    transformer = TransformerModel(seq_len=2)

    train(transformer, 2, {"train": train_dataloader}, phases=["train"])

    #print(transformer(next(iter(train_dataloader)), transformer.generate_square_subsequent_mask(2), transformer.generate_square_subsequent_mask(3)))
