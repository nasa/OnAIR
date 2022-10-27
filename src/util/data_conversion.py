"""
data_conversion.py
Utility file to perform conversions for supervised learning, and beyond
"""
import numpy as np 

classes = {'RED' : 0,
        'YELLOW' : 1,
         'GREEN' : 2,
           '---' : 3}

def floatify_input(_input, remove_str=False):
    floatified = []
    for i in _input:
        if type(i) is str:
            try:
                x = float(i)
                floatified.append(x)
            except:
                try:
                    x = i.replace('-', '').replace(':', '').replace('.', '')
                    floatified.append(float(x))
                except:
                    if remove_str == False:
                        floatified.append(0.0)
                    else:
                        continue
                    continue
        else:
            floatified.append(float(i))
    return floatified

def status_to_oneHot(status):
    if isinstance(status, np.ndarray):
        return status
    one_hot = [0.0, 0.0, 0.0, 0.0]
    one_hot[classes[status]] = 1.0
    return list(one_hot)
