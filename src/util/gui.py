import networkx as nwx
import matplotlib.pyplot as plt
import numpy as np

colors = {'LINEAR INCREASE' :'b', 
          'LINEAR DECREASE' : 'r', 
          'SINUSOIDAL' : 'm',
          'FLAT' : 'c'}
          
def create_and_view_graph(save_graph=False):
    associativity_graph = nwx.Graph()
    associativity_graph.add_nodes_from(self.graph_nodes)
    associativity_graph.add_edges_from(self.graph_edges_with_weights)
    edge_labelsa = nwx.get_edge_attributes(associativity_graph, 'weight')
    pos = nwx.spring_layout(associativity_graph)
    nwx.draw(associativity_graph, pos, with_labels = True)
    nwx.draw_networkx_edge_labels(associativity_graph, pos, edge_labels = edge_labelsa)

    if save_graph == True:
        plt.savefig('pathtofig')

    plt.show()

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
