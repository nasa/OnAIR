# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import numpy as np
import redis
from time import sleep
import json

from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin

TARGET_ALT = 5 # What altitude should the drone maintain during operations?
POSITION_TOLERANCE = 0.2 # How close must the drone get to a target location for operations purposes?
HOME_COORDS = [0.0, 0.0] # Where is the drone's home coordinates? May be different than origin in some cases
DRONE_NUMBER = 1 # Which drone is this? Used to publish commands to the right channel

def xy_position_difference(pos1, pos2): # Calculate the difference between XY positions, used for position tolerance comparisons
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

class Plugin(AIPlugin):
    def __init__(self, name, headers):
        """
        Initialize Redis connection and set target altitude, position tolerance, and home coordinates
        """
        # Redis initialization 
        super().__init__(name, headers)
        pool = redis.ConnectionPool(host="localhost", port=6379, password="")
        self.r = redis.Redis(connection_pool=pool, charset="utf-8", decode_responses=True)
        self.pub_channel = 'mavlink_cmd'


        self.target_alt = TARGET_ALT * -1
        self.home = HOME_COORDS

        self.state = 'standby' # Initial drone operating state
        
        self.target_position = None

    def send_cmd(self, cmd): # Command publishing to Redis
        serialized_cmd = json.dumps(cmd)
        self.r.publish(f'edp_drone_{DRONE_NUMBER}_command', serialized_cmd)

    def update(self,low_level_data=[], high_level_data={}):
        """
        Given streamed data point, system should update internally
        """              
        print(f'\nCOMPLEX REASONER sees high level data as: {high_level_data}')
        self.drone_position = high_level_data['vehicle_rep']['KAST']['pose']
        pass

    def render_reasoning(self):
        """
        Basic state machine implemented to take off, move, pause, return, and land.
        """
        rnd_xy = np.random.uniform(-5,5, size=2)

        if self.target_position == None:
            self.target_position = [rnd_xy[0], rnd_xy[1], 0.5] # XYZ
            self.send_cmd(self.target_position)
        else:
            position_err = xy_position_difference(self.drone_position, self.target_position)
            if position_err < POSITION_TOLERANCE:
                self.target_position = [rnd_xy[0], rnd_xy[1], 1.0]
                self.send_cmd(self.target_position)
            else:
                print(f'Traveling to new target position {self.target_position}')
        
