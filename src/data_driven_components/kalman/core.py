from src.data_driven_components.generic_intelligence_construct import GenericIntelligenceConstruct
from src.data_driven_components.kalman.kalman_plugin import Kalman

class AIPlugIn(GenericIntelligenceConstruct):
    def __init__(self, name, headers, window_size=3):
        """
        :param headers: (int) length of time agent examines
        :param window_size: (int) size of time window to examine
        """
        self.frames = []
        self.component_name = name
        self.headers = headers
        self.window_size = window_size
        self.agent = Kalman()

    def apriori_training(self):
        pass

    def update(self, frame):
        """
        :param frame: (list of floats) input sequence of len (input_dim)
        :return: None
        """
        for data_point_index in range(len(frame)):
            if len(self.frames) < len(frame): # If the frames variable is empty, append each data point in frame to it, each point wrapped as a list
                # This is done so the data can have each attribute grouped in one list before being passed to kalman
                # Ex: [[1:00, 1:01, 1:02, 1:03, 1:04, 1:05], [1, 2, 3, 4, 5]]
                self.frames.append([frame[data_point_index]]) 
            else:
                self.frames[data_point_index].append(frame[data_point_index])
                if len(self.frames[data_point_index]) > self.window_size: # If after adding a point to the frame, that attribute is larger than the window_size, take out the first element
                    self.frames[data_point_index].pop(0) 

    def render_diagnosis(self):
        """
        System should return its diagnosis
        """

        #A stub for once ppo is fully integrated
        broken_attributes = self.agent.frame_diagnose(self.frames, self.headers)
        # potentially the the library-less runner?
        return broken_attributes