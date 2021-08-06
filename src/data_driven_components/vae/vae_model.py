import os

import torch
import numpy as np
import copy
from src.data_driven_components.data_learners import DataLearner
from src.data_driven_components.vae.vae import VAE, TimeseriesDataset
from src.data_driven_components.vae.autoencoder import AE
from src.data_driven_components.vae.vae_train import train
from src.data_driven_components.vae.vae_diagnosis import VAEExplainer, findThreshold
from torch.utils.data import DataLoader, Dataset
from src.util.config import get_config
import numpy as np


class VAEModel(DataLearner):

    def __init__(self, headers, window_size, z_units=5, hidden_units=100, 
            path=os.path.join('src','data_driven_components','vae','models','checkpoint_latest.pth.tar')):
        """
        :param headers: (string list) list of headers for each input feature
        :param window_size: (int) number of data points in our data sequence
        :param z_units: (int) dimensions of our latent space gaussian representation
        :param hidden_units: (int) dimension of our hidden_units
        :param path: (string) path of vae save relative to src directory
        """
        self.path = path
        self.headers = []
        self.ignore_headers = []
        for i, h in enumerate(headers):
            if not h.upper() in ['LABEL_ERROR_STATE', 'TIME']:
                self.ignore_headers.append(i)
                self.headers.append(h)

        #self.headers = headers
        self.window_size = window_size
        if get_config()['VAE'].getboolean('Autoencoder', True):
            self.model = AE(self.headers, window_size, z_units, hidden_units)
        else:
            self.model = VAE(self.headers, window_size, z_units, hidden_units)
        self.frames = [[0.0]*len(headers) for i in range(self.window_size)]
        self.explainer = VAEExplainer(self.model, self.headers, len(self.headers), self.window_size)
        self.has_baseline = False
        self.counter = 0
        self.baseline = []
        self.baseline_every = get_config()['VAE'].getint('TakeBaselineEvery', 20)
        self.max_baselines = get_config()['VAE'].getint('MaxBaselines', 5)

    def load_model(self, path):
        """
        :param path: (string) path from RUN_PATH
        """
        self.model.load_state_dict(torch.load(os.path.join(os.environ['RUN_PATH'], path)))
        
    def apriori_training(self, data_train):
        """
        Given data, system should learn any priors necessary for realtime diagnosis.
        :param data_train: (Tensor) shape (batch_size, seq_size, feat_size)
        # TODO: double check sizes
        """
        _batch_size = len(data_train[0])
        _input_dim = len(data_train[0][0])

        transform = lambda x: (torch.tensor(x).float())[:,self.ignore_headers]
        train_dataset = TimeseriesDataset(data_train, transform)
        train_dataloader = DataLoader(train_dataset, batch_size=_batch_size)

        train(self.model, {'train': train_dataloader}, phases=["train"], checkpoint=True, logging=True)

        self.explainer.updateModel(self.model)

    def find_threshold(self, data_train):
        _batch_size = len(data_train[0])

        transform = lambda x: (torch.tensor(x).float())[:,self.ignore_headers]
        train_dataset = TimeseriesDataset(data_train, transform)
        train_dataloader = DataLoader(train_dataset, batch_size=_batch_size)

        return findThreshold(self.model, train_dataloader, 0)

    def update(self, frame, status):
        """
        :param frame: (list of floats) input sequence of len (input_dim)
        :param status: (int) 0 for red, 1 yellow, 2 green, 3 no data
        :return: None
        """
        self.frames.append(frame)
        self.frames.pop(0)

        if status == 2 and (self.counter % self.baseline_every) == 0:
            self.baseline.append(copy.deepcopy(self.frames))
            self.has_baseline = True
            if len(self.baseline) > self.max_baselines:
                self.baseline.pop(0)

        self.counter += 1
        """
        if status == 2:
            self.baseline = self.frames
            self.has_baseline = True
        """

    def render_diagnosis(self):
        """
        System should return its diagnosis, do not run unless model is loaded
        """

        self.explainer = VAEExplainer(self.model, self.headers, len(self.headers), self.window_size)
        transformation = lambda x: (torch.tensor(x).float())[:,self.ignore_headers]

        data = transformation(self.frames).unsqueeze(0)

        shap = np.zeros(len(self.headers))

        if self.has_baseline:
            if len(self.baseline) > 1:
                baseline = torch.stack([transformation(b) for b in self.baseline])
            else:
                baseline = torch.stack([transformation(self.baseline[0])]*2)
            self.explainer.shap(data, baseline)
            vae_diagnosis = self.explainer.viz(True)
            shap += vae_diagnosis[0]
            data_vals = list(vae_diagnosis[1])
            hdrs = list(vae_diagnosis[2])
        else:
            baseline = torch.stack([torch.zeros_like(data)]*2).squeeze(1)
            self.explainer.shap(data, baseline)
            vae_diagnosis = self.explainer.viz(True)
            shap = vae_diagnosis[0]
            data_vals = list(vae_diagnosis[1])
            hdrs = list(vae_diagnosis[2])

        if get_config()['VAE'].getboolean('UseAbsoluteShap', True):
            shap = list(np.abs(shap))
        else:
            shape = list(shap)

        ordered_shapleys, ordered_headers = zip(*sorted(zip(shap, hdrs), reverse=True))
        return (ordered_headers, [x / np.linalg.norm(ordered_shapleys) for x in ordered_shapleys])
