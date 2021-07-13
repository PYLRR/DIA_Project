import environment.graph as graph
import numpy as np
import time

g = graph.Graph()
np.random.seed(int(time.time()))
max_output = g.nbNodes
## Build a random connection matrix undirected
connection_matrix = g.generateConnectivityMatrix()

print(connection_matrix)
print(g.activation_probabilities_matrix)
g.changeTransitionProbabilities2([1, 1,1,1,1], g.activation_probabilities_matrix)
print(g.prob_matrix)
g.display()