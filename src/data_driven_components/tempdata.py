### Non permanent file TODO: delete this

### Just waiting for official RAISR data loading

import csv

## Load All The Data
def massLoadData(folder, lookback):
    data = []
    headers = []
    for file in os.listdir(folder):
        if file.find(".csv") != -1:
            fileData = loadData(folder + file)
            if headers == []:
                headers = fileData[0]
            for i in range(1+lookback, len(fileData)):
                newPoint = []
                for j in range(lookback):
                    newPoint.append(fileData[i-lookback+j])
                data.append(newPoint)
    return headers, data

## Load data from a .csv file
def loadData(filepath, delimiter=',', quotechar='\"'):
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        data = []
        for row in reader:
            data.append(row)
        return data