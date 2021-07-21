
from src.data_driven_components.pomdp.pomdp_util import mass_load_data, stratified_sampling, split_by_lookback, dict_sort_data, list_to_dictionary_with_headers
from src.data_driven_components.pomdp.ppo import PPO
from src.data_driven_components.data_learners import DataLearner
import os


class PPOModel(DataLearner):

    def __init__(self, headers, window_size):
        """
        :param headers: (int) length of time agent examines
        :param window_size: (int) size of time window to examine
        """
        self.frames = {}
        self.headers = headers
        self.window_size = window_size
        self.agent = PPO()

    def apriori_training(self, data, use_stratified=True):
        """
        :param data: (3D array) first dim data points, second time frames and third features so (batch_size, window_size, input_dim)
        """
        #split_data = split_by_lookback(data, self.window_size)
        data_train = dict_sort_data(self.agent.config, data)
        if use_stratified:
            data_train = stratified_sampling(self.agent.config, data_train)
        #Data should be in the format of { Time : [ 0, 1, 2] , Voltage : [5, 5, 5] } at this point
        batch_size = int(len(data_train)/50) if int(len(data_train)/50) > 0 else 1
        self.agent.train_ppo(data_train, batch_size=batch_size)

    def update(self, frame):
        """
        :param frame: (list of floats) input sequence of len (input_dim)
        :return: None
        """
        self.frames = list_to_dictionary_with_headers(frame, self.headers, self.frames, self.window_size)

    ####################################################################################
    def render_diagnosis(self):
        """
        System should return its diagnosis
        """

        #A stub for once ppo is fully integrated
        #reward, correct, actions, states[-1] = self.agent.diagnose_frames(self.frames)
        # potentially the the library-less runner?
        pass

    # def render_diagnosis(self):
    #     """
    #     System should return its diagnosis
    #     """
    #     info = self.agent.diagnose_frames(self.frames) 
    #     #A stub for once ppo is fully integrated
    #     #reward, correct, actions, states = self.agent.diagnose_frames(self.frames)
    #     # potentially the the library-less runner?
    #     return info

    ####################################################################################
    
