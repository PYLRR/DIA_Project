import numpy as np


class BaseEnvironment:
    def __init__(self, graph, seeds):

        self.graph = graph
        self.active_nodes = seeds

    # In this function we do a round
    # some nodes will click and some nodes activated
    # out the nodes who clicked in this round
    def round(self):
        clicked = []
        new_active_nodes = []

        for a in self.active_nodes:
            # just a uniform random variable
            r = 1 - np.random.uniform()
            for j in range(len(self.graph.prob_matrix[a])):
                if self.graph.prob_matrix[a, j] > r:
                    # append influenced nodes
                    new_active_nodes.append(j)
                    if a >= self.graph.nbNodes:
                        # node clicked
                        clicked.append(a - self.graph.nbNodes)

        self.active_nodes.extend(new_active_nodes)
        self.active_nodes = list(set(self.active_nodes))
        return list(set(clicked))





