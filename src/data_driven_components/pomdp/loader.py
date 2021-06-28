## Nick Pellegrino
## loader.py for POMDP and Forest objects

from pomdp import POMDP
from forest import Forest
import os
import csv
import random
import pomdp_util as util

def main():
    train_pomdp("test", 15, data)
    test_pomdp("test", 15, data)
    examine_pomdp("test", "data/")

def train_pomdp(name, lookback, data_path, save_path='models/', alpha=0.01, discount=0.8, epsilon=0.2, reward_correct=100, reward_incorrect=-500, reward_action=-1):
    config, data, base_path = get_data(lookback, data_path)
    training_data, testing_data = util.stratified_sampling(config, data)
    agent = POMDP(name, base_path + save_path, config, alpha=alpha, discount=discount, epsilon=epsilon, reward_correct=reward_correct, reward_incorrect=reward_incorrect, reward_action=reward_action)
    agent.mass_train(training_data, testing_data, 250)
    return agent

def test_pomdp(name, lookback, data_path, save_path='models/'):
    config, data, base_path = get_data(lookback, data_path)
    _, testing_data = util.stratified_sampling(config, data)
    agent = POMDP(name, base_path + save_path, config)
    agent.set_print_on(True)
    data_point = random.choice(testing_data)
    print(data_point)
    agent.run_test(data_point)

def examine_pomdp(name, data_path, save_path='models/'):
    config, base_path = get_config(data_path)
    agent = POMDP(name, base_path + save_path, config)
    agent.set_print_on(True)
    agent.show_examination_filtered()

def get_data(lookback, data_path):
    base_path = ""
    if (os.path.dirname(__file__) != ""):
        base_path = os.path.dirname(__file__) + "/"
    config, data = util.mass_load_data(base_path + data_path, lookback)
    return config, data, base_path

def get_config(data_path):
    base_path = ""
    if (os.path.dirname(__file__) != ""):
        base_path = os.path.dirname(__file__) + "/"
    config = util.load_config(base_path + data_path)
    return config, base_path

if __name__ == '__main__':
    main()
