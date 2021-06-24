import matplotlib.pyplot as plt
import numpy as np

colors = {'LINEAR INCREASE' :'b', 
          'LINEAR DECREASE' : 'r', 
          'SINUSOIDAL' : 'm',
          'FLAT' : 'c'}


def plot_results(prediction_sample, points, color_assignments):
    # Data for plotting
    t = np.arange(len(prediction_sample))
    s = prediction_sample.flatten()
    fig, ax = plt.subplots()
    ax.plot(t, s)

    for i in range(len(points)):
        calibrated_pt = points[i]
        plt.axvspan(calibrated_pt[0], calibrated_pt[1], facecolor=colors[color_assignments[i]])

    ax.set(xlabel='Time Step', ylabel='Sensor Reading',
           title='Testing the Characterizer')
    ax.grid()
    plt.show()

    return
