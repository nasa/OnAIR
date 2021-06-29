""" Test VAE Functionality """
import os
import unittest

from src.data_driven_components.vae.vae import VAE, TimeseriesDataset
from src.data_driven_components.vae.vae_train import train
from src.data_driven_components.vae.vae_diagnosis import findThreshold, VAEExplainer
from torch.utils.data import DataLoader
import torch

class TestVAE(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.headers = ['a','b','c','d','e','f','g','h','i','j',
                        'k','l','m','n','o','p','q','r','s','t',
                        'u','v','w','x','y','z','1','2','3','4']
    def test_init_vae(self):
        vae = VAE(headers=self.headers, z_units=5)
        self.assertEqual(vae.input_dim, 30)
        self.assertEqual(vae.z_units, 5)

    def test_train_vae(self):
        data = range(30)
        data = [[list(data)]] # serrated shape

        data2 = [1]*30
        data2 = [[data2]] # uniform

        transform = lambda x: torch.tensor(x).float()
        train_dataset = TimeseriesDataset(data, transform)
        train_dataloader = DataLoader(train_dataset, batch_size=1)

        test_dataset = TimeseriesDataset(data2, transform)
        test_dataloader = DataLoader(test_dataset, batch_size=1)

        vae = VAE(headers=self.headers, window_size=1, z_units=5)

        train(vae, {'train': train_dataloader}, phases=["train"], logging=False)

    def test_threshold_vae(self):
        data = range(30)
        data = [[list(data)]] # serrated shape

        data2 = [1]*30
        data2 = [[data2]] # uniform

        transform = lambda x: torch.tensor(x).float()
        train_dataset = TimeseriesDataset(data, transform)

        vae = VAE(headers=self.headers, window_size=1, z_units=5)
        findThreshold(vae, train_dataset, 0.2)
    
    def test_shapley(self):
        vae = VAE(headers=self.headers, window_size=1, z_units=5)
        VAEExplainer(vae, [])

if __name__ == '__main__':
    unittest.main()