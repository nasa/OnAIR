"""
LSTM-VAE for streamed satellite telemetry fault diagnosis
Gabriel Rasskin @ NASA
"""
import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, z_units=5, input_dim=30, seq_len=15):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = z_units * 2 # good rule of thumb
        self.z_units = z_units
        self.seq_len = seq_len

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
        x = x.reshape((1, self.seq_len, self.input_dim)) # TODO: should move to data processing
        # LSTM output is tuple (output, (hidden state, cell state))
        x, (_, _) = self.enc1(x)
        _, (hidden_state, _) = self.enc2(x)
        # hidden state of last lstm is latent representation
        return hidden_state.reshape((self.input_dim, self.z_units)) # TODO: cleanup dims

    def decoder(self, x):
        # TODO: deal with x reshaping
        x, (_, _) = self.dec1(x)
        x, (_, _) = self.dec2(x)
        # TODO: deal with x reshaping
        return self.output_layer(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

# TODO: add loss criterion bce and recontruction error

if __name__ == "__main__":
    print("Creating VAE...")
    vae = VAE()
    # TODO: training loop
    print("Successfuly created VAE")
