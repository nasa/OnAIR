###############
## ppo_runner.py
###############

#### related files ####
import src.data_driven_components.pomdp.observation as observation
import src.data_driven_components.pomdp.pomdp_util as pomdp_util
from src.data_driven_components.pomdp.pomdp import POMDP
########################
#### libraries ####
import pickle
import copy
import numpy as np
########################

class PPO_Runner(POMDP):
    def __init__(self, name, path, config_path=""):
        super().__init__(name, path, config_path)
        self.load_PPO()

    ###---### Load PPO model ###---###
    def load_PPO(self):
        self.load_model()
        self.weights = self.load_weights_helper()

    def load_weights_helper(self):
        return pickle.load(open(self.path + "ppo_weights_" + str(self.name) + ".pkl","rb"))
    ###---### ###---### ###---###  ###---###

    """Get an action to take from actor network"""
    def action(self, state):
        '''
        Input:
        state - current state

        Output:
        action - the action index chosen (aka which action it will take)
        action_logprob - the logprobability of choosing this action from this state
        '''
        if isinstance(state, list) or isinstance(state, np.ndarray): # Make sure the state is a list
            state = np.array(state).tolist()

        # Feed Forwards
        action_probs = copy.deepcopy(state)
        action_probs = np.matmul(self.weights[0], action_probs) # nn.Linear(self.state_dim, 64),
        action_probs = self.relu(action_probs + self.weights[1]) # nn.ReLU(),
        action_probs = np.matmul(self.weights[2], action_probs) # nn.Linear(64, 64),
        action_probs = self.relu(action_probs + self.weights[3]) # nn.ReLU(),
        action_probs = np.matmul(self.weights[4], action_probs) # nn.Linear(64, self.act_dim),
        action_probs = self.softmax(action_probs + self.weights[5]) # nn.Softmax(dim=-1))

        # Categorical Sample: Choose An Action Probabilistically
        action = self.categorical_sample(action_probs)
        return action

    ###---### Testing ###---###

    def test_instance(self, data_point):
        #Initializing running variables
        total_reward = 0
        run_time = 0
        done = False
        correct = False
        #Initialize diagnosis information
        actions = []
        states = []
        #Get first state
        obs = self.states[self.get_starting_state()]
        self.current_state_index = self.get_starting_state()
        states.append(obs)
        obs = observation.floatify_state(obs)
        obs = self.state_flatten_preprocess(obs)
        while(not done):
            run_time += 1
            action = self.action(obs)
            actions.append(self.actions[action])
            reward, done = self.take_action(action, data_point, False)
            total_reward += reward
            obs = self.states[self.current_state_index]
            states.append(obs)
            obs = observation.floatify_state(obs)
            obs = self.state_flatten_preprocess(obs)
            if run_time >= self.run_limit:
                done = True
                total_reward += (self.rewards[1]-100)
                correct = False
        if reward == self.rewards[0]:
            correct = True
        else:
            correct = False
        return total_reward, correct, actions, states

    def diagnose_frames(self, time_chunk):
        # Time_chunk should be in the form
        # { Attribute : [List of data points for attribute of size lookback]}
        reward, correct, actions, states = self.test_instance(time_chunk)
        return reward, correct, actions, states

    ###---### ###---### ###---###  ###---###


    ###---### Helper Functions ###---###

    def state_flatten_preprocess(self, state):
        state = np.array(state)
        states = state.flatten()
        return states

    def relu(self, data):
        new_data = []
        for i in range(len(data)):
            new_data.append(self.relu_helper(data[i]))
        return new_data

    def relu_helper(self, num):
        if num > 0:
        	return num
        else:
        	return 0

    def softmax(self, data):
        e_x = np.exp(data - np.max(data))
        return e_x / e_x.sum()

    def categorical_sample(self, probabilities):
        return (probabilities.cumsum(-1) >= np.random.uniform(size=probabilities.shape[:-1])[..., None]).argmax(-1)

    ###---### ###---### ###---###  ###---###

if __name__ == "__main__":
    dict_config, data = pomdp_util.mass_load_data('RAISR-2.0\\src\\data\\raw_telemetry_data\\data_physics_generation\\Errors\\', lookback=15)
    agent = PPO_Runner('ppo_train', "RAISR-2.0\\src\\data_driven_components\\pomdp\\models\\", config_path='RAISR-2.0\\src\\data\\raw_telemetry_data\\data_physics_generation\\Errors\\config.csv')
    print(agent.diagnose_frames(data[0]))
