import os
import unittest

from src.data_driven_components.transformers import TransformerModel, TimeseriesDataset, train, TransformerExplainer
from torch.utils.data import DataLoader
import torch

class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_init_transformer(self):
        transformer = TransformerModel(seq_len=1)
        self.assertEqual(transformer.model_type, 'Transformer')
        self.assertEqual(transformer.z_units, 5)

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

        transformer = TransformerModel(seq_len=1)

        train(transformer, 1, {"train": train_dataloader}, phases=["train"], logging=False)
    
    def test_shapley(self):
        transformer = TransformerModel(seq_len=2)
        TransformerExplainer(transformer, [])

if __name__ == '__main__':
    unittest.main()