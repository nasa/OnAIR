###############
## ppo.py
###############

#### related files ####
import pomdp
import observation
########################
#### libraries ####
import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from math import exp
########################

class PPO(POMDP):
    def __init__(self, state_dimensions, action_dimensions, epsilon = 0.2, epochs = 30, learning_rate_actor = 0.0005, learning_rate_critic = 0.001, discount = 0.99):
        super().__init__()
        self.epsilon = epsilon
        self.discount = discount
        self.epochs = epochs
        self.state_dim = state_dimensions
        self.act_dim = action_dimensions
        # The Actor : ouputs probabilities of actions from one state
        self.actor = nn.Sequential(
                            nn.Linear(state_dimensions, 64),
                            nn.ReLU(),
                            nn.Linear(64, 64),
                            nn.ReLU(),
                            nn.Linear(64, action_dimensions),
                            nn.Softmax(dim=-1))
        # The Critic : determines state value (V) from input state
        self.critic = nn.Sequential(
                        nn.Linear(state_dimensions, 64),
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
    def save(self):
        pickle.dump([self.states, self.quality_values, self.actions, self.alpha, self.discount, self.epsilon, self.telemetry_headers, self.reportable_states, self.run_limit, self.rewards],open(self.path + "pomdp_model_" + str(self.name) + ".pkl","wb"))
        torch.save(self.actor.state_dict(), self.path + "pomdp_model_" + str(self.name) + "_actor_policy_state_dict.pt")
        torch.save(self.critic.state_dict(), self.path + "pomdp_model_" + str(self.name) + "_critic_policy_state_dict.pt")
   
    def load(self, path = ""):
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

