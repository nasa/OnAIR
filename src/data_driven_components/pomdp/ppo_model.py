
from src.data_driven_components.pomdp.pomdp_util import mass_load_data, stratified_sampling
from src.data_driven_components.pomdp.ppo import PPO
from src.data_driven_components.data_learners import DataLearner
import os

class PPOModel(DataLearner):

    def __init__(self, window_size, config_path='src/data/raw_telemetry_data/data_physics_generation/Errors/config.csv', model_path='src/data_driven_components/pomdp/models'):
        """
        :param window_size: (int) length of time agent examines
        :param config_path: (optional String) path to PPO config
        :param model_path: (optional String) path to saved model
        """
        self.frames = []
        model_path = os.path.join(os.environ['SRC_ROOT_PATH'], model_path)
        config_path = os.path.join(os.environ['SRC_ROOT_PATH'], config_path)

        self.agent = PPO('ppo_train', model_path, config_path=config_path)

    def apriori_training(self, data):
        """
        Given data, system should learn any priors necessary for realtime diagnosis.
        :param data_train: (Tensor) shape (batch_size, seq_size, feat_size)
        # TODO: double check sizes
        """
        #data_path = os.path.join(os.environ['SRC_ROOT_PATH'], 'src/data/raw_telemetry_data/data_physics_generation/Errors')
        #dict_config, data = mass_load_data(data_path, lookback=15)
        data = stratified_sampling(dict_config, data)
        training_data = data[:int(len(data)*(0.7))]
        testing_data = data[int(len(data)*(0.7)):]

        agent.train_ppo(training_data, testing_data, 1090)

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