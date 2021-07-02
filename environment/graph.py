import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import warnings


class Graph:
    def __init__(self, nbNodes, minSocialInfluProb=0.0, maxSocialInfluProb=0.1, clickProbability=0.4,
                 stochasticitySeed=0):
        # with that, every execution will lead to the same random generated values
        np.random.seed(stochasticitySeed)

        self.nbNodes = nbNodes

        # we create exactly nbNodes+5 nodes, so that we have 1 extra node per category
        # (it will be the node representing click probability)
        self.prob_matrix = np.random.uniform(minSocialInfluProb, maxSocialInfluProb,
                                             (self.nbNodes + 5, self.nbNodes + 5))

        # attributes a category for each node. Category of node i is categories[i] (out from 0 to 4)
        self.categories = np.random.randint(0, 5, self.nbNodes)

        # creates the click probability between the extra node per category and the nodes of this category
        for j in range(5):
            for i in range(self.nbNodes):  # links extra nodes-normal nodes
                self.prob_matrix[i][self.nbNodes + j] = 0.0
                if self.categories[i] == j:
                    self.prob_matrix[self.nbNodes + j][i] = clickProbability
                else:
                    self.prob_matrix[self.nbNodes + j][i] = 0.0
            for i in range(5):  # links extra nodes-extra nodes
                self.prob_matrix[self.nbNodes + i][self.nbNodes + j] = 0.0

    # this method displays the graph using the networkx library
    # it is not very beautiful but it allows to have a look
    def display(self):
        G = nx.DiGraph()
        G.add_nodes_from(range(self.nbNodes + 5))
        for i in range(self.nbNodes + 5):
            for j in range(self.nbNodes + 5):
                if self.prob_matrix[i][j] != 0:
                    G.add_edge(i, j)

        # colors differently the extra nodes
        color_map = [(lambda i: 'red' if i >= self.nbNodes else 'green')(i) for i in range(self.nbNodes + 5)]
        # labels differently the extra nodes
        labeldict = {}
        for i in range(self.nbNodes):
            labeldict[i] = str(i+5)
        for i in range(5):
            labeldict[self.nbNodes+i] = str(i)

        # drawing of the graph
        # ignore the deprecation warnings (because it's internal to networkx)
        warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
        pos = nx.spring_layout(G, k=1, iterations=20)
        nx.draw_networkx(G, pos, node_color=color_map, labels=labeldict, with_labels=True)
        plt.show()


graph = Graph(15)
graph.display()
