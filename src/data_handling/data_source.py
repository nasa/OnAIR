"""
DataSource class
Helper class to iterate through data
"""

class DataSource:
    def __init__(self, data=[]):
        self.index = 0
        self.data = data
        if data != []:
            self.data_dimension = len(data[0])
        else:
            self.data_dimension = 0

    # Get the data at self.index and increment the index
    def get_next(self):
        self.index = self.index + 1
        return self.data[self.index - 1]

    # Return whether or not the index has finished traveling through the data
    def has_more(self):
        return self.index < len(self.data)

    # Return whether or not there is data 
    def has_data(self):
        if self.data == []:
            return False

        empty_step = ['-']*(self.data_dimension - 1)
        for timestep in self.data:
            if timestep[1:] != empty_step: # Dont count the time data stamp
                return True
        return False
