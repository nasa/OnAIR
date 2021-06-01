"""
DataSource class
Helper class to iterate through data
"""


class DataSource:
    def __init__(self, data=[]):
        self.index = 0
        self.data = data

    # Get the data at self.index and increment the index
    def get_next(self):
        self.index = self.index + 1
        return self.data[self.index - 1]

    # Return whether or not the index has finished traveling through the data
    def has_more(self):
        return self.index < len(self.data)

    # Return whether or not there is data 
    def has_data(self):
        for timestep in self.data:
            for ss in timestep.keys():
                if timestep[ss]['data'] != []:
                    return True
