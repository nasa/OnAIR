
from src.data_driven_components.pomdp.pomdp_util import mass_load_data, stratified_sampling, split_by_lookback, dict_sort_data
from src.data_driven_components.pomdp.ppo import PPO
from src.data_driven_components.data_learners import DataLearner
import os

class PPOModel(DataLearner):

    def __init__(self, window_size, config_path='src/data/raw_telemetry_data/data_physics_generation/Errors/config.csv'):
        """
        :param window_size: (int) length of time agent examines
        :param config_path: (optional String) path to PPO config
        """
        self.frames = []
        self.window_size = window_size

        self.agent = PPO(config_path=config_path)

    def apriori_training(self, data, use_stratified=True):
        #data_path = os.path.join(os.environ['SRC_ROOT_PATH'], 'src/data/raw_telemetry_data/data_physics_generation/Errors')
        #dict_config, data = mass_load_data(data_path, self.window_size)
        split_data = split_by_lookback(data, self.window_size)
        data_train = dict_sort_data(self.agent.config, split_data)
        if use_stratified:
            split_data_train = stratified_sampling(self.agent.config, data_train)
        #Data should be in the format of { Time : [ 0, 1, 2] , Voltage : [5, 5, 5] } at this point
        self.agent.train_ppo(split_data_train, batch_size=1090)

    def update(self, frame):
        """
        :param frame: (list of floats) input sequence of len (input_dim)
        :return: None
        """

        #A stub for once config is integrated with initialization 

        #Use self.agent.config to find the headers in the pomdp
        #single_frame = {}
        #for h in range(len(self.agent.headers)):
        #   single_frame[self.agent.headers[h]] = frame[self.agent.config[self.agent.headers[h]][3]]
        #self.frames.append(single_frame)
        
        self.frames.append(frame)
        if(len(self.frames)>self.window_size):
            self.frames.pop(0)

    def render_diagnosis(self):
        """
        System should return its diagnosis
        """
        pass