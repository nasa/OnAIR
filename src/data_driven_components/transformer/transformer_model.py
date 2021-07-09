from src.data_driven_components.vae.vae import TimeseriesDataset
from src.data_driven_components.transformer.transformer import Transformer
from src.data_driven_components.vae.vae_train import train
from src.data_driven_components.data_learners import DataLearner
import os
import torch
from torch.utils.data import DataLoader, Dataset

class TransformerModel(DataLearner):

    def __init__(self, headers, window_size, z_units=5, hidden_units=100, 
            path='src/data_driven_components/transformer/models/checkpoint_latest.pth.tar'):
        """
        :param headers: (string list) list of headers for each input feature
        :param window_size: (int) number of data points in our data sequence
        :param z_units: (int) dimensions of our latent space gaussian representation
        :param hidden_units: (int) dimension of our hidden_units
        :param path: (string) path of vae save relative to src directory
        """
        self.path = path
        self.headers = headers
        self.window_size = window_size
        self.model = Transformer(len(headers), window_size, z_units, hidden_units)
        self.frames = [[0.0]*len(headers) for i in range(self.window_size)]

    def apriori_training(self, data_train):
        """
        Given data, system should learn any priors necessary for realtime diagnosis.
        :param data_train: (Tensor) shape (batch_size, seq_size, feat_size)
        # TODO: double check sizes
        """
        try:
            self.model.load_state_dict(torch.load(os.path.join(os.environ['SRC_ROOT_PATH'], self.path)))
        except:
            _batch_size = len(data_train[0])
            _input_dim = len(data_train[0][0])

            transform = lambda x: torch.tensor(x).float()
            train_dataset = TimeseriesDataset(data_train, transform)
            train_dataloader = DataLoader(train_dataset, batch_size=_batch_size)

            train(self.model, {'train': train_dataloader}, phases=["train"], checkpoint=True)

    def update(self, frame):
        """
        :param frame: (list of floats) input sequence of len (input_dim)
        :return: None
        """
        self.frames.append(frame)
        self.frames.pop(0)

    def render_diagnosis(self):
        """
        System should return its diagnosis
        """
        pass
