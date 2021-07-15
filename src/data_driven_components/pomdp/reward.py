## Nick Pellegrino
## NASA GSFC
## reward.py

import src.data_driven_components.pomdp.pomdp_util as util
# action = "view_X" where X is VOLTAGE, CURRENT, etc. or "report_error" or "report_no_error"
# data = ['Time', 'VOLTAGE', 'CURRENT', 'THRUST', 'ALTITUDE', 'ACCELERATION', 'TEMPERATURE', 'SCIENCE_COLLECTION', '[LABEL]: ERROR_STATE']
# rewards = [rewCor, rewWro, rewAct]
def get_reward(action, data, rewards, config):
    # For view actions: Reward -1 to encourage quicker reporting
    if action.find("view") != -1:
        return rewards[2], -1
    # For report actions: Check if POMDP was correct or not, and reward accordingly
    answer = 0
    label, label_key = util.check_label(config, data)
    if label:
        for i in range(len(data[label_key])):
            if data[label_key][i] == '1': # If there is an error anywhere in this chunk of time
                answer = 1 # Then there's an error
                break
    else:
        error = util.get_vae_error_over_all_data([data])
        if error:
            answer = 1
    if action == "report_error" and answer == 1:
        return rewards[0], answer
    elif action == "report_no_error" and answer == 0:
        return rewards[0], answer
    return rewards[1], answer
