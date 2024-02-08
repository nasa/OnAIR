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
        print("csv_output_plugin.py:init name: " + name + "\theaders: " + str(headers))
        super().__init__(name, headers)

        self.headers_line = ""
        for header in headers:
            self.headers_line += header + ','

        # Init some basic parameters, like number of entries for a single file
        self.first_frame = True
        self.lines_before_output = 10
        self.lines_current = 0
        self.current_buffer = [] # List of strings, each string is a line for the .csv
        self.filename_preamble = "csv_out_"
        self.filename = ""

        # Plugin should write each frame of data out to a .csv file
        # low_level_data is optional. Should use the headers provided in __init__
        # high level data is coming from different sources: list in headers

        # simplest mode is just always writting out to a file
        # live move writes out headers? then n lines to a temp file (or internal buffer), then writes to a new file with date/time filename

    def update(self,low_level_data=[], high_level_data={}):
        """
        Given streamed data point, system should update internally
        """
        print("csv_output_plugin:update")
        print("low_level_data: " + str(low_level_data))

        # TODO: Need to get headers for high level data
        if (self.first_frame):
            for plugin in high_level_data:
                self.headers_line += str(plugin) + ","
            self.headers_line = self.headers_line[:-1]

        new_line = ""

        # Add low level data
        for telem_point in low_level_data:
            new_line += str(telem_point) + ","

        # Add high level data
        # a bit trickier since it a dictionary. One entry per plugin and I don't know what is inside
        for plugin_data in high_level_data:
            # Assume each plugin is outputing a list
            for telem_point in high_level_data:
                new_line += str(telem_point) + ","

        # remove trailing comma, save to buffer
        self.current_buffer.append(new_line[:-1])

    def render_reasoning(self):
        """
        System should return its diagnosis
        """

        if (self.first_frame):
            # Create file- TODO: make separate function
            date_stamp = datetime.today().strftime('%j_%H_%M')
            self.file_name = self.filename_preamble + date_stamp + ".csv"

            self.current_buffer.insert(0, self.headers_line)

            self.first_frame = False
        pass

        if (self.lines_before_output == 0):
            # Write out to file
            with open(self.file_name, 'a') as file:
                for line in self.current_buffer:
                    file.write(line + '\n')
                    print("Wrote out: " + line + '\n')
            self.current_buffer = []
        elif (self.lines_before_output != 0 and self.lines_before_output == self.lines_current):
            # Create new file
            print("Create new file")
            date_stamp = datetime.today().strftime('%j_%H_%M')
            self.file_name = self.filename_preamble + date_stamp + ".csv"
            with open(self.file_name, 'w') as file:
                for line in self.current_buffer:
                    file.write(line + '\n')
                    print("Wrote out: " + line + '\n')

            self.current_buffer = []
            self.lines_current = 0
        else:
            print("Nothing to write this go...") # TODO: should still write out, otherwise everything is batched until the end
            self.lines_current += 1
            pass
