
from src.data_driven_components.pomdp.pomdp_util import mass_load_data, stratified_sampling
from src.data_driven_components.pomdp.ppo import PPO
from src.data_driven_components.data_learners import DataLearner
import os

class PPOModel(DataLearner):

    def __init__(self):
        """
        :param headers: (string list) list of headers for each input feature
        :param window_size: (int) number of data points in our data sequence
        :param z_units: (int) dimensions of our latent space gaussian representation
        :param hidden_units: (int) dimension of our hidden_units
        :param path: (string) path of vae save relative to src directory
        """
        self.frames = []
        data_path = os.path.join(os.environ['SRC_ROOT_PATH'], 'src/data/raw_telemetry_data/data_physics_generation/Errors')
        dict_config, data = mass_load_data(data_path, lookback=15)
        data = stratified_sampling(dict_config, data)
        training_data = data[:int(len(data)*(0.7))]
        testing_data = data[int(len(data)*(0.7)):]

        model_path = os.path.join(os.environ['SRC_ROOT_PATH'], 'src/data_driven_components/pomdp/models')
        config_path = os.path.join(os.environ['SRC_ROOT_PATH'], 'src/data/raw_telemetry_data/data_physics_generation/Errors/config.csv')
        agent = PPO('ppo_train', model_path, config_path=config_path)
        agent.train_ppo(training_data, testing_data, 1090)

    def apriori_training(self, data_train):
        """
        Given data, system should learn any priors necessary for realtime diagnosis.
        :param data_train: (Tensor) shape (batch_size, seq_size, feat_size)
        # TODO: double check sizes
        """
        pass

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