import environment.graph as graph
from environment.BaseEnvironment import BaseEnvironment
import numpy as np
import time

g = graph.Graph()
np.random.seed(1)
max_output = g.nbNodes
## Build a random connection matrix undirected
connection_matrix = g.generateConnectivityMatrix()

print(connection_matrix)
# print(g.activation_probabilities_matrix)
g.changeTransitionProbabilities2([0.5, 0.5, 0.5, 0.5, 0.5], g.activation_probabilities_matrix)

be = BaseEnvironment(graph=g, seeds=[15])
n_episodes = 10
active_nodes = [g.nbNodes]
for i in range(n_episodes):
    print("Active: ", be.active_nodes)
    clicked = be.round()
    print("Clicked: ", clicked)
# print(g.prob_matrix)
g.display()