import os
import unittest

from src.data_driven_components.transformer.transformer_model import TransformerModel, TimeseriesDataset
from src.data_driven_components.vae.vae import TimeseriesDataset
from src.data_driven_components.vae.vae_train import train
from src.data_driven_components.vae.vae_diagnosis import VAEExplainer
from torch.utils.data import DataLoader
import torch

class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_init_transformer(self):
        transformer = TransformerModel(list(range(30)), 1)
        self.assertEqual(transformer.has_baseline, False)

    def test_train_transformer(self):
        data = range(30)
        data = [[list(data)]] # serrated shape

        data2 = [1]*30
        data2 = [[data2]] # uniform

        transform = lambda x: torch.tensor(x).float()
        train_dataset = TimeseriesDataset(data, transform)
        train_dataloader = DataLoader(train_dataset, batch_size=1)

        test_dataset = TimeseriesDataset(data2, transform)
        test_dataloader = DataLoader(test_dataset, batch_size=1)

        transformer = TransformerModel(list(range(30)), 1)

        self.mask = transformer.model.generate_square_subsequent_mask(1)
        self.tar_mask = transformer.model.generate_square_subsequent_mask(2)

        train(transformer.model, {"train": train_dataloader}, 1, phases=["train"], logging=False,forward=lambda x: transformer.model(x, self.mask, self.tar_mask), print_on=False)

    def test_shapley(self):
        transformer = TransformerModel(list(range(30)), 2)
        VAEExplainer(transformer, [])

if __name__ == '__main__':
    unittest.main()
