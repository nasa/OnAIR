## This file is an example of how to use the pomdp class, which should be helpful during integration

from pomdp import POMDP # import the POMDP class

# The POMDP has three functionalities: train, test, and examine. All three are explained in sudo functions below:

# Make a new POMDP and train it
def train_pomdp():
    # Here's the parameters you need to train:
    save_path = "blahblahblah/data_driven_componenets/pomdp/models/" # This is the path models & graphs will be saved to
    name = "name_of_model" # Every unique model needs a different name.
                           # If you make a POMDP with a name of a model found at the save_path...
                           # ...then POMDP will load that model instead of making a fresh one
    config_path = "blahblahblah/raw_telemetry_data/data_physics_generation/Errors/config.csv" # Path to config.csv, which we've left in the same folder as the data for now
    data_train = [] # A list of frames, as you described to us earlier.
    data_test = [] # A list of frames, as you described to us earlier. Optional (can be left as [] if you don't want accuracy graphs during training).

    # Here's how you create a POMDP to train:
    agent = POMDP(name, save_path, config_path=config_path)

    # Here's how you train:
    agent.apriori_training(data_train, data_test=data_test)

# Diagnose errors using trained POMDP
def test_pomdp():
    # Here's the parameters you need to run the pomdp:
    save_path = "blahblahblah/data_driven_componenets/pomdp/models/" # This is the path the model was saved to
    name = "name_of_model" # Every unique model needs a different name.
                           # If you make a POMDP with a name of a model found at the save_path...
                           # ...then POMDP will load that model instead of making a fresh one
    data_test = [] # A list of frames, as you described to us earlier. Optional (can be left as [] if you don't want accuracy graphs during training).

    # Here's how you create a POMDP that's already been trained and saved:
    agent = POMDP(name, save_path)
        # Note that if a POMDP was trained (see train_pomdp() in this file) with the same name and save_path, then agent is going to load that old save.
        # This is what you want when your goal is to run the pomdp on some data. You should've trained it already if you're at this point.

    agent.set_print_on(True) # Doing this makes the agent print out its "thought-process" during diagnose_frames
                             # If you don't do this, then by default the agent will simply return the final decision

    # Here's how you test:
        # The POMDP needs a list of more than one frame for run_test. For example, you could send in data_test[0:15] and it'll receive a list of 15 frames.
        # The POMDP will consider all frames sent in. The POMDP will say there's an error if something goes wrong in any of those sent in frames.
        # Sending in more than one frame at a time is important for Kalman Filter to work correctly.
        # If you send in a single frame, for example data_test[0], then Kalman Filter will break because it's meant to look at data over time.
        # Unlike last year, where the LSTM had a hardcoded lookback, the POMDP doesn't care how many frames you send for it to make a prediciton on.
        # It just needs to be more than one.
    output = agent.diagnose_frames(data_test[n:m]) # Where n and m are integers, m > n, 0 <= n < len(data_test), 0 < m <= len(data_test)

    output == "report_error" # This means the POMDP thinks data_test[n:m] contains an error.
    output == "report_no_error" # This means the POMDP thinks data_test[n:m] does not contain an error.
    output == "over_runtime_limit" # This means the POMDP entered a loop and never answered the question. If you ever get this, it might not have trained well.

# View the mind of a trained POMDP
def examine_pomdp():
    # Here's the parameters you need to run the pomdp:
    save_path = "blahblahblah/data_driven_componenets/pomdp/models/" # This is the path the model was saved to
    name = "name_of_model" # Every unique model needs a different name.
                           # If you make a POMDP with a name of a model found at the save_path...
                           # ...then POMDP will load that model instead of making a fresh one

    # Here's how you create a POMDP that's already been trained and saved:
    agent = POMDP(name, save_path)
        # Note that if a POMDP was trained (see train_pomdp() in this file) with the same name and save_path, then agent is going to load that old save.
        # This is what you want when your goal is to run the pomdp on some data. You should've trained it already if you're at this point.

    # Running this will print out the "decision tree" behind a trained Q-Learning POMDP
    agent.show_examination_filtered() # <-- This is the essense of the agent's explainability
