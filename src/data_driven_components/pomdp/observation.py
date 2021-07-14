## Hayley Owens & Nick Pellegrino
## NASA GSFC
## observation.py

import copy
import numpy as np
from src.data_driven_components.pomdp.kalman_filter import return_KF, current_attribute_chunk_get_error


# Current tools we're using, semi-hardcoded to allow more tools being added and more generalization
# Tool_Name : Index for states
OBSERVATION_TOOLS = {"THRESH" : 0, "KAL" : 1}

# -=-=-=- State Related Functions -=-=-=- #
# This not only gives the starting state, but also the general state configuration for the whole POMDP
def get_starting_state(config):
    state = []
    for h in config: # For each property ['VOLTAGE', 'CURRENT', 'THRUST', 'ALTITUDE', 'ACCELERATION', 'TEMPERATURE', 'SCIENCE_COLLECTION']...
        if(config[h][0] == 'data'):
            observation_state_list = ["?"] * len(OBSERVATION_TOOLS)
            state.append(observation_state_list) # ...the state remembers: [broke threshold, or didn't, or unknown; kalman filter approves, or finds suspicious, or unknown]
    return state

def floatify_state(state):
    new_state = []
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] == '?':
                new_state += [1, 0, 0]
            elif state[i][j].find("STABLE") != -1:
                new_state += [0, 1, 0]
            else:
                new_state += [0, 0, 1]
    return new_state

# -=-=-=- -=-=-=- -=-=-=- -=-=-=- -=-=-=- #

# State = Whatever format you defined in the above function get_starting_state()
# Action = "view_X" where X is every header in the data from VOLTAGE to SCIENCE_COLLECTION
# Data = ['Time', 'VOLTAGE', 'CURRENT', 'THRUST', 'ALTITUDE', 'ACCELERATION', 'TEMPERATURE', 'SCIENCE_COLLECTION', '[LABEL]: ERROR_STATE']
# IMPORTANT: Do not use '[LABEL]: ERROR_STATE' in your observation, as it's unrealistic the POMDP would be able to observe the supervised labels
def get_observation(state, action, data, config):
    new_state = copy.deepcopy(state)
    new_state = update_by_threshold(new_state, action, data, config)
    new_state = update_by_kalman(new_state, action, data, config)
    return new_state

# -=-=-=- get_observation's Observation Functions -=-=-=- #

def update_by_threshold(state, action, data, config):
    '''
        State - Current array of current state
        Action - string formatted like "view_VOLTAGE" to check particular attribute values
        Data - {VOLTAGE: [x..], CURRENT: [x..], etc.} - Dictionary of attributes with list of data
    '''
    answer = "STABLE" # Start with the assumption the no errors occur for the given property during the time chunk
    attribute = strip_view_prefix(action) # Take "view_" out of the action string to get specific attribute being viewed
    for i in range(len(data[attribute])): # Loop through the entire time chunk, and change answer to 1 if any errors are found
        lower_thresh_val, upper_thresh_val = get_attribute_threshold(attribute, config)
        if float(data[attribute][i]) < lower_thresh_val or float(data[attribute][i]) > upper_thresh_val:
            answer = "BROKEN"
            break
    # Update the state with the new information
    state = update_state(state, attribute, "THRESH", answer, config)
    return state

def update_by_kalman(state, action, data, config):
    '''
        State - Current array of current state
        Action - string formatted like "view_VOLTAGE" to check particular attribute values
        Data - {VOLTAGE: [x..], CURRENT: [x..], etc.} - Dictionary of attributes with list of data
    '''
    answer = "STABLE" # Start with the assumption the no errors occur for the given property during the time chunk
    attribute = strip_view_prefix(action) # Take "view_" out of the action string to get specific attribute being viewed
    if (current_attribute_chunk_get_error(return_KF(), data[attribute])):
            answer = "BROKEN"
    # Update the state with the new information
    state = update_state(state, attribute, "KAL", answer, config)
    return state

# -=-=-=- get_observation's Helper Functions -=-=-=- #

# Takes out the "view_" portion of action to get specific attribute being viewed
def strip_view_prefix(action):
    #Takes "view_" out of action
    action = action.replace("view_", "")
    #Makes sure the attribute is capitalized (which is how the dictionary keys should be)
    action.upper()
    return action

# Goes into specific run instances data configs and gets threshold of current attribute
def get_attribute_threshold(attribute, config):
    lower_bound = config[attribute][1]
    upper_bound = config[attribute][2]
    return float(lower_bound), float(upper_bound)

# Takes in the current state, the attribute being adjusted, the tool used to adjust it, and the answer of what it's being adjusted to
# then returns the updated state.
def update_state(state, attribute, observation_tool_used, answer, config):
    index = get_index(attribute, config)
    state[index][OBSERVATION_TOOLS[observation_tool_used]] = attribute + "_" + observation_tool_used + "_" +  answer
    return state

# Gets index of header from config file
def get_index(attribute, config):
    return config[attribute][3]

def get_possible_branches(state, action, config):
    if action.find("report") != -1:
        return []
    attribute = strip_view_prefix(action) # Take "view_" out of the action string to get specific attribute being viewed
    index = get_index(attribute, config)
    statuses = ["STABLE", "BROKEN"]
    # Make copy of current states
    permutation_number = len(OBSERVATION_TOOLS) * len(statuses)
    states = []
    for i in range(permutation_number):
        states.append(copy.deepcopy(state))
    for state_index in range(len(states)):
        # If the state is at the first index, it should be entirely stable across every tool
        if state_index == 0:
            for tool in OBSERVATION_TOOLS:
                states[state_index][index][OBSERVATION_TOOLS[tool]] = attribute + "_" + tool + "_" + "STABLE"
        # If the state is at the last index, it should be entirely broken across every tool
        elif state_index == (len(states)-1):
            for tool in OBSERVATION_TOOLS:
                states[state_index][index][OBSERVATION_TOOLS[tool]] = attribute + "_" + tool + "_" + "BROKEN"
        # If the state is in between first and last index, it should be alternate each tool
        # Ex:
        #       stable broke
        #       broke stable
        else:
            for tool in OBSERVATION_TOOLS:
                n = OBSERVATION_TOOLS[tool] % len(statuses) # Figures out which index is alternating
                if state_index % len(statuses) == 0: #If the state has made a full pass through the length of statuses, time to begin alternating
                    n = abs(n-1)
                states[state_index][index][OBSERVATION_TOOLS[tool]] = attribute + "_" + tool + "_" + statuses[n]
                
    return states
