# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

from datetime import datetime

#import numpy as np
from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin

class Plugin(AIPlugin):
    def __init__(self, name, headers):
        super().__init__(name, headers)

        # Init some basic parameters, like number of entries for a single file
        self.first_frame = True
        self.lines_per_file = 10
        self.lines_current = 0
        self.current_buffer = [] # List of telemetry points
        self.filename_preamble = "csv_out_"
        self.filename = ""

        # Plugin should write each frame of data out to a .csv file
        # low_level_data is optional. Should use the headers provided in __init__
        # high level data is coming from different sources: list in headers

        # simplest mode is just always writing out to a file
        # live move writes out headers? then n lines to a temp file (or internal buffer), then writes to a new file with date/time filename

    def update(self,low_level_data=[], high_level_data={}):
        """
        Given streamed data point, system should update internally
        """

        if (self.first_frame):
            # TODO: Should look at vehicle_rep, learning_systems, and planning_systems
            for plugin in high_level_data['learning_systems']:
                self.headers.append(str(plugin))

        self.current_buffer = []

        # Add low level data
        for telem_point in low_level_data:
            self.current_buffer.append(str(telem_point))

        # Add high level data
        # TODO: Should look at vehicle_rep, learning_systems, and planning_systems
        for plugin in high_level_data['learning_systems']:
            plugin_output = high_level_data['learning_systems'].get(plugin, "empty")
            # Assume each plugin is outputting a list
            for telem_point in plugin_output:
                self.current_buffer.append(str(telem_point))

    def render_reasoning(self):
        """
        System should return its diagnosis
        """

        if (self.first_frame):
            # Create file- TODO: make separate function
            date_stamp = datetime.today().strftime('%j_%H_%M')
            self.file_name = self.filename_preamble + date_stamp + ".csv"

            # Write out to file
            with open(self.file_name, 'a') as file:
                delimiter = ','
                file.write(delimiter.join(self.headers) + '\n')
            self.first_frame = False
        pass

        # Write out to file
        with open(self.file_name, 'a') as file:
            delimiter = ','
            file.write(delimiter.join(self.current_buffer) + '\n')
        self.current_buffer = []
        self.lines_current += 1

        if (self.lines_per_file != 0 and self.lines_per_file == self.lines_current):
            # Create new file
            date_stamp = datetime.today().strftime('%j_%H_%M')
            self.file_name = self.filename_preamble + date_stamp + ".csv"

            self.lines_current = 0
