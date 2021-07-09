## Nick Pellegrino
## pomdp.py
import os
import random
import pickle
import src.data_driven_components.pomdp.observation as observation
import src.data_driven_components.pomdp.reward as reward
import numpy as np
import ast
import src.data_driven_components.pomdp.pomdp_util as util
from src.util.config import get_config
import matplotlib.pyplot as plt
from tqdm import tqdm

class POMDP:
    # Name is used when saving the model to file
    # Path is the directory you want models to be saved in
    # Actions = list of strings, i.e. ["north", "south", "stop"]
    # my_rewards = parallel list to actions, i.e. [-1, -1, 100]
    # Alpha = learning rate, generally set between 0 and 1
        # 0.0 = q-values are never updated
        # 0.1 = agent will learn slowly
        # 0.9 = agent will learn quickly
    # Discount = determines how much the agents care about my_rewards in the distant future
        # = 1, agent doesn't care about the future
        # 0 < discount < 1, agent wants to get to big my_rewards as fast as possible
        # = 0, agent will only learn about actions that yield an immediate my_reward
    # Epsilon = exploratory rate, generally set between 0 and 1
        # 0.1 = 10% chance to take a random action during training
    def __init__(self, name="pomdp", path="models/", config_path="", print_on=False, save_me=True, reportable_states=['no_error', 'error'], alpha=0.01, discount=0.8, epsilon=0.2, run_limit=-1, reward_correct=100, reward_incorrect=-100, reward_action=-1):
        self.name = name
        base_path = ""
        if (os.path.dirname(__file__) != ""):
            base_path = os.path.dirname(__file__) + "/"
        self.path = base_path + path
        self.print_on = print_on
        self.save_me = save_me
        self.answer = 0
        try:
            self.load_model()
        except:
            self.states = []
            self.quality_values = []
            self.actions = []
            self.alpha = alpha
            self.discount = discount
            self.epsilon = epsilon
            config_path = get_config()['DEFAULT']['TelemetryMetadataFilePath']
            config_path = config_path + ast.literal_eval(get_config()['DEFAULT']['MetaFiles'])[0]
            self.config = util.load_config_from_txt(config_path)
            self.reportable_states = reportable_states
            self.headers = []
            index = 0
            for key in self.config:
                if self.config[key][0] == "data":
                    self.headers.append(key)
                    self.actions.append("view_" + key)
                    self.config[key].append(index)
                    index += 1
            if run_limit == -1:
                self.run_limit = len(self.actions)+1
            else:
                self.run_limit = run_limit
            for r in self.reportable_states:
                self.actions.append("report_" + r)
            self.rewards = [reward_correct, reward_incorrect, reward_action]
            self.kappa = 0 # Cohen's Kappa
            self.confusion_matrix = [1, 1, 1, 1]
            self.save_model()
        self.current_state_index = self.get_starting_state()
        self.total_reward = 0
        self.correct = False
        self.run_time = 0
        self.states_examined = [] # This is just for the recursive examination filter method

    def save_model(self):
        if self.save_me:
            pickle.dump(self.get_save_data(),open(self.path + "pomdp_model_" + str(self.name) + ".pkl","wb"))

    def get_save_data(self):
        return [self.states, self.quality_values, self.actions, self.alpha, self.discount, self.epsilon, self.config, self.reportable_states, self.run_limit, self.rewards, self.kappa, self.confusion_matrix, self.headers]

    def get_current_state(self):
        return self.states[self.current_state_index]

    def load_with_save_data(self, data):
        self.states = data[0]
        self.quality_values = data[1]
        self.actions = data[2]
        self.alpha = data[3]
        self.discount = data[4]
        self.epsilon = data[5]
        self.config = data[6]
        self.reportable_states = data[7]
        self.run_limit = data[8]
        self.rewards = data[9]
        self.kappa = data[10]
        self.confusion_matrix = data[11]
        try:
            self.headers = data[12]
        except:
            self.headers = []
            for key in self.config:
                if self.config[key][0] == "data":
                    self.headers.append(key)

    def load_model(self):
        data = pickle.load(open(self.path + "pomdp_model_" + str(self.name) + ".pkl","rb"))
        self.load_with_save_data(data)

    def save_new_state(self, state):
        if state not in self.states:
            self.states.append(state)
            quality_val = []
            for i in range(len(self.actions)):
                quality_val.append(0)
            self.quality_values.append(quality_val)
        return self.states.index(state)

    def get_starting_state(self):
        state = observation.get_starting_state(self.config)
        return self.save_new_state(state)

    def get_observation(self, state_index, action_index, data):
        if self.actions[action_index].find("view") != -1:
            new_state = observation.get_observation(self.states[state_index], self.actions[action_index], data, self.config)
            return self.save_new_state(new_state)
        return state_index # Report actions lead a state to itself, then end the training round

    def get_reward(self, action_index, data):
        my_reward, self.answer = reward.get_reward(self.actions[action_index], data, self.rewards, self.config)
        return my_reward

    # This is where the magic happens for the Q-Learning
    def update_quality_values(self, state_index, action_index, new_state_index, my_reward):
        sample = my_reward + (self.discount * self.quality_values[new_state_index][self.best_action_index(new_state_index)])
        self.quality_values[state_index][action_index] = ((1 - self.alpha) * self.quality_values[state_index][action_index]) + (self.alpha * sample)

    def new_training(self, data):
        self.total_reward = 0
        self.run_time = 0
        self.current_state_index = self.get_starting_state()
        self.correct = False
        done = False
        if self.print_on:
            print("\n-=-=-=- New Training Run -=-=-=-")
            print("\nStarting State:", self.states[self.current_state_index])
        while not done:
            self.run_time += 1
            action_index = self.best_action_index(self.current_state_index)
            old_state_index = self.current_state_index
            done = self.take_action(action_index, data)
            if self.run_time > self.run_limit:
                done = True
                self.total_reward += self.rewards[1]
                self.update_quality_values(old_state_index, action_index, self.current_state_index, self.rewards[1])

    ## data_train = list of frames, with headers and labels as described in self.config
    def apriori_training(self, data_train, data_test=[], lookback=15, batch_size=250, use_stratified=True):
        split_data_train = util.split_by_lookback(data_train, lookback)
        split_data_test = util.split_by_lookback(data_test, lookback)

        split_data_train = util.dict_sort_data(self.config, split_data_train)
        split_data_test = util.dict_sort_data(self.config, split_data_test)

        if use_stratified:
            split_data_train = util.stratified_sampling(self.config, split_data_train)

        avg_rewards = []
        avg_accuracies = []
        batches = []
        batch_num = 0

        for i in tqdm(range(len(split_data_train))):
            self.new_training(split_data_train[i])
            if i % batch_size == 0:
                if split_data_test != []:
                    avg_reward, avg_accuracy = self.mass_test(split_data_test)
                    avg_rewards.append(avg_reward)
                    avg_accuracies.append(avg_accuracy)
                    batches.append(batch_num)
                    batch_num += 1
                    self.plot_graph(batches, avg_rewards, "Batch #", "Avg. Rewards")
                    self.plot_graph(batches, avg_accuracies, "Batch #", "Avg. Accuracy")
                self.save_model()
        self.save_model()

    def mass_train(self, training_data, testing_data, batch_size, test_while_running=True, kappa_test=False):
        avg_rewards = []
        avg_accuracies = []
        batches = []
        batch_num = 0
        for i in tqdm(range(len(training_data))):
            self.new_training(training_data[i])
            if i % batch_size == 0:
                if test_while_running:
                    if len(testing_data) >= i+batch_size:
                        avg_reward, avg_accuracy = self.mass_test(testing_data[i:i+batch_size])
                    else:
                        avg_reward, avg_accuracy = self.mass_test(testing_data[i:])
                    avg_rewards.append(avg_reward)
                    avg_accuracies.append(avg_accuracy)
                    batches.append(batch_num)
                    batch_num += 1
                    self.plot_graph(batches, avg_rewards, "Batch #", "Avg. Rewards")
                    self.plot_graph(batches, avg_accuracies, "Batch #", "Avg. Accuracy")
                self.save_model()
        if kappa_test:
            self.calculate_kappa(testing_data)
        self.save_model()

    def calculate_confusion_matrix(self, testing_data):
        TP = 1 # true positive = there was an error, agent correct
        FN = 1 # false negative = there was an error, agent incorrect
        FP = 1 # false positive = there was no error, agent incorrect
        TN = 1 # true negative = there was no error, agent correct
        for i in range(len(testing_data)):
            _ = self.run_test(testing_data[i])
            if self.correct:
                if self.answer == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if self.answer == 1:
                    FN += 1
                else:
                    FP += 1
        self.confusion_matrix = [TP, FN, FP, TN]

    def calculate_kappa(self, testing_data):
        self.calculate_confusion_matrix(testing_data)
        TP = self.confusion_matrix[0]
        FN = self.confusion_matrix[1]
        FP = self.confusion_matrix[2]
        TN = self.confusion_matrix[3]
        sum = TP + FN + FP + TN
        p_o = (TP + TN) / sum # total accuracy
        p_yes = ((TP + FN)/sum) * ((TP + FP)/sum) # probability there is error and agent is correct
        p_no = ((FP + TN)/sum) * ((FN + TN)/sum) # probability there is no error and agent is correct
        p_e = p_yes + p_no # probability of random agreement
        self.kappa = (p_o - p_e)/(1 - p_e)

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_kappa(self):
        return self.kappa

    def mass_test(self, testing_data):
        reward_sum = 0
        accuracy_sum = 0
        for i in range(len(testing_data)):
            _ = self.run_test(testing_data[i])
            reward_sum += self.total_reward
            if self.correct:
                accuracy_sum += 1
        return reward_sum/len(testing_data), accuracy_sum/len(testing_data)

    def diagnose_frames(self, time_chunk):
        data_dictionary = util.dict_sort_data(self.config, [time_chunk])[0]
        return self.run_test(data_dictionary)

    def run_test(self, data):
        self.total_reward = 0
        self.run_time = 0
        self.correct = False
        self.current_state_index = self.get_starting_state()
        done = False
        action_index = 0
        if self.print_on:
            print("\nAgent's Actions:", self.actions)
            print("\n-=-=-=- New Testing Run -=-=-=-")
            print("\nStarting State:", self.states[self.current_state_index])
        while not done:
            self.run_time += 1
            action_index = self.best_action_index(self.current_state_index, random_chance=False, testing=True)
            done = self.take_action(action_index, data, training_run=False)
            if (not done) and (self.run_time > self.run_limit):
                done = True
                self.total_reward += self.rewards[1]
                return "over_runtime_limit"
        return self.actions[action_index]

    def get_answer(self):
        return self.answer

    def best_action_index(self, state_index, random_chance=True, testing=False, full_list=False):
        # Only when testing: Report error for unknown states
        if testing and (self.quality_values[state_index] == [0 for i in range(len(self.quality_values[state_index]))]):
            return self.actions.index("report_error")
        if random_chance and (random.random() < self.epsilon):
            return random.randint(0, len(self.actions)-1)
        best_action_indexes = [0]
        for i in range(1, len(self.quality_values[state_index])):
            if self.quality_values[state_index][i] > self.quality_values[state_index][best_action_indexes[0]]: # New highest Q-Value
                best_action_indexes = [i] # Reset the list to just the best action index
            elif self.quality_values[state_index][i] == self.quality_values[state_index][best_action_indexes[0]]: # Ties highest Q-Value
                best_action_indexes.append(i) # Add tied best action index to the list
        if not full_list:
            return random.choice(best_action_indexes) # Break ties randomly
        return best_action_indexes

    def take_action(self, action_index, data, training_run=True):
        old_state_index = self.current_state_index
        new_state_index = self.get_observation(old_state_index, action_index, data)
        my_reward = self.get_reward(action_index, data)
        if training_run:
            self.update_quality_values(old_state_index, action_index, new_state_index, my_reward)
        self.current_state_index = new_state_index
        if self.print_on:
            print("\nAction Taken:", self.actions[action_index])
            print("New State:", self.states[new_state_index])
            print("Reward:", my_reward)
        self.total_reward += my_reward
        if self.actions[action_index].find("report") != -1:
            if my_reward == 100:
                self.correct = True
            else:
                self.correct = False
            return my_reward, True
        return my_reward, False

    def set_print_on(self, new_print_on):
        self.print_on = new_print_on

    def plot_graph(self, x, y, x_title, y_title):
        plt.clf() # Clear Previous Graph
        plt.plot(x, y)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.savefig(self.path + 'graph_' + self.name + "_" + y_title + '.png')

    def show_examination(self):
        for i in range(len(self.states)):
            print("\nState #" + str(i) + ":", self.states[i])
            print("Best Action:", self.actions[self.best_action_index(i, random_chance=False)])

    def show_examination_filtered(self):
        state_index = self.get_starting_state()
        self.show_examination_filtered_helper(state_index, 0)
        self.states_examined = []

    def show_examination_filtered_helper(self, state_index, depth):
        best_action_indexes = self.best_action_index(state_index, random_chance=False, full_list=True)
        actions = []
        for best_action_index in best_action_indexes:
            actions.append(self.actions[best_action_index])
        if len(actions) != 1:
            print("\nDepth:", depth, "\nState:", self.states[state_index], "\nBest Actions:", actions)
        else:
            print("\nDepth:", depth, "\nState:", self.states[state_index], "\nBest Action:", actions[0])
        if state_index in self.states_examined:
            print("REPEAT STATE DETECTED!")
        else:
            self.states_examined.append(state_index)
            possible_next_states = []
            for best_action_index in best_action_indexes:
                possible_next_states += observation.get_possible_branches(self.states[state_index], self.actions[best_action_index], self.config)
            depth += 1
            for possible_next_state in possible_next_states:
                try:
                    self.show_examination_filtered_helper(self.states.index(possible_next_state), depth)
                except:
                    print("\nDepth:", depth, "\nState:", possible_next_state, "\nBest Action: report_error")
                    print("UNSEEN STATE DETECTED!")
