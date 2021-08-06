from torch.utils.data import DataLoader, Dataset
from captum.attr import KernelShap, GradientShap, DeepLiftShap, DeepLift
from src.data_driven_components.vae.viz import isNotebook

if isNotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def findThreshold(vae, dataloader, error_margin):
    """
    Finds fault threshold based on upper bound for reconstruction error and a percent margin
    :param vae: (VAE) model to evaluate on
    :param dataset: (Dataset) pytorch dataloader containing normal data
    :param error_margin: (float) the error margin as a float, for example, to add 20% to the threshold
                        this should be set to 0.2
    """
    #dataloader = DataLoader(dataset, batch_size=1)
    reconstruction_error = 0
    for d in tqdm(dataloader, disable=False):
        if (x := vae(d).sum()) > reconstruction_error:
            reconstruction_error = x

    return reconstruction_error * (1+error_margin)

class VAEExplainer():
    def __init__(self, vae, headers, n_features=7, seq_len=1, n_samples=5):
        """
        Takes in vae model to explain.
        :param vae: (VAE) vae model
        :param headers: (string list) ordered list of headers, must have n_features elements
        :param n_features: (optional int) number of features for a sequence input, defaults to 30
        :param seq_len: (optional int) number of sequence components per input
        :param n_samples: (optional int) number of times to evaluate model, defaults to 200
        """
        self.createExplainer(vae)
        self.headers = headers
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_samples = n_samples

    def createExplainer(self, vae):
        self.vae = vae
        self.explainer = DeepLiftShap(vae)

    def updateModel(self, vae):
        """
        Update Kernel model
        :param vae: (VAE model) new model to update to
        """
        self.createExplainer(vae)
    
    def shap(self, input, baseline):
        """
        Calculate shapley values for a given input as compared to baseline
        :param input: (Tensor) input shape (batch_size, seq_len, input_dim)
        :param baseline: (Tensor) baseline sample shape (batch_size, seq_len, input_dim)
        """
        #print(input.shape, baseline.shape, self.vae(input), self.vae(baseline))
        self.input = input
        self.shap_values = self.explainer.attribute(self.input, baseline)#, n_samples=self.n_samples)
        if self.vae(input).sum() > self.vae(baseline).sum():
            return self.shap_values
        else:
            return -self.shap_values

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