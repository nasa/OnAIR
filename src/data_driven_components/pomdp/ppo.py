###############
## ppo.py
###############

#### related files ####
import observation
import pomdp_util
from pomdp import POMDP
########################
#### libraries ####
import os
from tqdm import tqdm
import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from math import exp
########################

class PPO(POMDP):
    def __init__(self, name, path, config_path="", epsilon = 0.2, epochs = 30, learning_rate_actor = 0.0005, learning_rate_critic = 0.001, discount = 0.99):
        super().__init__(name, path, config_path)
        self.epsilon = epsilon
        self.discount = discount
        self.epochs = epochs
        self.state_dim = len(observation.floatify_state(self.states[self.get_starting_state()]))
        self.act_dim = len(self.actions)
        # The Actor : ouputs probabilities of actions from one state
        self.actor = nn.Sequential(
                            nn.Linear(self.state_dim, 64),
                            nn.ReLU(),
                            nn.Linear(64, 64),
                            nn.ReLU(),
                            nn.Linear(64, self.act_dim),
                            nn.Softmax(dim=-1))
        # The Critic : determines state value (V) from input state
        self.critic = nn.Sequential(
                        nn.Linear(self.state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1))
        #Load old actor and critic information
        self.actor.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(self.critic.state_dict())
        #Create optimizer
        self.optimizer = Adam([
                        {'params': self.actor.parameters(), 'lr': learning_rate_actor},
                        {'params': self.critic.parameters(), 'lr': learning_rate_critic}
                    ])
        self.MseLoss = nn.MSELoss()
    
    ###---### Save and Load PPO model ###---###
    def save_PPO(self):
        pickle.dump(self.get_save_data(),open(self.path + "pomdp_model_" + str(self.name) + ".pkl","wb"))
        torch.save(self.actor.state_dict(), self.path + "pomdp_model_" + str(self.name) + "_actor_policy_state_dict.pt")
        torch.save(self.critic.state_dict(), self.path + "pomdp_model_" + str(self.name) + "_critic_policy_state_dict.pt")
   
    def load_PPO(self, path = ""):
        self.load_model()
        self.actor.load_state_dict(torch.load(self.path + "pomdp_model_" + str(self.name) + "_actor_policy_state_dict.pt"))
        self.critic.load_state_dict(torch.load(self.path + "pomdp_model_" + str(self.name) + "_critic_policy_state_dict.pt"))    
    ###---### ###---### ###---###  ###---###
    
    """Gets state values and the action probabilities"""                 
    def evaluate(self, state, action):
        '''
        Input:
        state - current state 
        action - action taken at state

        Output:
        V - the "state value" as determined by the critic
        logprobs - the logprobability of choosing this action from this state
        dist_entropy - the entropy of the distribution to better improve loss function
        '''
        if isinstance(state, list) or isinstance(state, np.ndarray): #Make sure the state is a tensor
            state = torch.tensor(state,  dtype=torch.float)
        V = self.critic(state).squeeze()
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        logprobs = dist.log_prob(action)    
        dist_entropy = dist.entropy()    
        return V, logprobs, dist_entropy

    """Get an action to take from actor network"""
    def action(self, state):
        '''
        Input:
        state - current state 

        Output:
        action - the action index chosen (aka which action it will take)
        action_logprob - the logprobability of choosing this action from this state
        '''
        if isinstance(state, list) or isinstance(state, np.ndarray): #Make sure the state is a tensor
            state = torch.tensor(state,  dtype=torch.float)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().numpy().item(), action_logprob.detach()

    """Calculate discounted rewards"""
    def discounted_rewards(self, rewards):
        discounted_rewards = []
        for run_through_reward in reversed(rewards):
            discounted_reward = 0 
            for reward in reversed(run_through_reward):
                discounted_reward = reward + discounted_reward * self.discount
                discounted_rewards.insert(0, discounted_reward)
        return discounted_rewards
    
    """Get loss as calculated in PPO clip formulas """
    def get_loss(self, ratios, advantage, state_values, rewards, dist_entropy):
        # Part 1 of formula: r(theta)*advantage            
        value1 = ratios * advantage
        # Part 2 of formula: clip(rations, 1 - epsilon, 1 + epsilon)*advantage
        value2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantage
        # Calculate actor and critic losses.
        #Loss^Clip(theta) - coefficient1(SquaredErrorValuesLoss(theta)) + coefficient2(EntropyBonus)
        loss = -torch.min(value1, value2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
        return loss

    ###---### Training ###---###

    def train_ppo(self, train_data, test_data, batch_size):
        iterations = int(len(train_data)/batch_size)
        timestep = []
        accuracy = []
        rewards = []
        for iteration in range(iterations):
            print("####### Batch " + str(iteration + 1) + "/" + str(iterations) +" ####### \n")
            # Walk through batch, get states, actions, log probabilites, and discounted rewards
            old_observed, old_actions, old_log_probs, disc_rewards = self.walk_through_batch(train_data[(batch_size*iteration):(batch_size*(iteration+1))])
            self.train_update_step(old_observed, old_actions, old_log_probs, disc_rewards)
            reward_accuracy, correct_accuracy = self.test(test_data)
            timestep.append(iteration)
            accuracy.append(correct_accuracy)
            rewards.append(reward_accuracy) 
            self.plot_graph(timestep, rewards, "Batch #", "Avg. Rewards")
            self.plot_graph(timestep, accuracy, "Batch #", "Avg. Accuracy")    
            self.save_PPO()               

    def train_update_step(self, old_observed, old_actions, old_log_probs, disc_rewards):
        old_observed = torch.tensor(old_observed, dtype=torch.float)
        old_actions = torch.tensor(old_actions, dtype=torch.float)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float)
        disc_rewards = torch.tensor(disc_rewards, dtype=torch.float)
        #Normalize rewards
        disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + 1e-7)
        for k in range(self.epochs):
            state_values, curr_log_probs, dist_entropy = self.evaluate(old_observed, old_actions)
            A_k = disc_rewards - state_values.detach() # The advantage at this current step (what kind of reward does a state value associate with)
            ratios = torch.exp(curr_log_probs - old_log_probs) # Get ratio r(theta), we can simply subtract because we're taking probabilities in log form
            loss = self.get_loss(ratios, A_k, state_values, disc_rewards, dist_entropy) # Get loss
            #Gradient
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def walk_through_batch(self, data):
        total_states = []
        total_actions = []
        total_rewards = []
        total_prob = []

        for data_point in tqdm(range(len(data))):
            done = False
            run_time = 0 #Variable to keep track of how many decisions it's making
            run_through_rewards = []
            obs = self.states[self.get_starting_state()]
            obs = observation.floatify_state(obs)
            obs = self.state_flatten_preprocess(obs)
            while(not done):
                run_time += 1
                total_states.append(obs) #Add current state to total states
                action, log_prob = self.action(obs) #Get the action and its probability distribution
                reward, done = self.take_action(action, data[data_point])
                obs = self.states[self.current_state_index]
                obs = observation.floatify_state(obs)
                obs = self.state_flatten_preprocess(obs)
                total_actions.append(action)                
                total_prob.append(log_prob)
                if run_time >= self.run_limit:
                    done = True
                    run_through_rewards.append(self.rewards[1]-1) #reward incorrect is rewards[1]
                else:
                    run_through_rewards.append(reward)
            total_rewards.append(run_through_rewards)    
        total_rewards = self.discounted_rewards(total_rewards)
        return total_states, total_actions, total_prob, total_rewards                   
    
    ###---### ###---### ###---###  ###---###


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
        states.append(obs)
        obs = observation.floatify_state(obs)
        obs = self.state_flatten_preprocess(obs)
        while(not done):
            run_time += 1            
            action, log_prob = self.action(obs)
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

    def test(self, data):
        correct_sum = 0
        reward_sum = 0
        for data_point_index in tqdm(range(len(data))):
            reward, correct, _, _ = self.test_instance(data[data_point_index])
            reward_sum += reward
            if correct:
                correct_sum += 1
        return reward_sum/len(data), correct_sum/len(data)
    
    def diagnose_frames(self, time_chunk):
        # Time_chunk should be in the form 
        # { Attribute : [List of data points for attribute of size lookback]}
        reward, correct, actions, states = test_instance(time_chunk)
        return reward, correct, actions, states

    ###---### ###---### ###---###  ###---###


    ###---### Helper Functions ###---###

    def state_flatten_preprocess(self, state):
        state = np.array(state)
        states = state.flatten()
        return states

    ###---### ###---### ###---###  ###---###

if __name__ == "__main__":
    dict_config, data = pomdp_util.mass_load_data('RAISR-2.0\\src\\data\\raw_telemetry_data\\data_physics_generation\\Errors\\', lookback=15)
    data = pomdp_util.stratified_sampling(dict_config, data)
    training_data = data[:int(len(data)*(0.7))]
    testing_data = data[int(len(data)*(0.7)):]
    agent = PPO('ppo_train', "RAISR-2.0\\src\\data_driven_components\\pomdp\\models\\", config_path='RAISR-2.0\\src\\data\\raw_telemetry_data\\data_physics_generation\\Errors\\config.csv')
    agent.train_ppo(training_data, testing_data, 1090)
