import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import warnings
import environment.auctionHouse as auctionHouse

MIN_SOCIAL_INFLUENCE_PROB = 0.0
MAX_SOCIAL_INFLUENCE_PROB = 0.1

N_NODES = 15


# class used to compute and then access the global graph of the project
# The idea of the graph is to have a number of nodes n, and to add a fictious
# node to each node, resulting in a graph of nodes cardinality of 2n.
# Edge weight from fictious to its corresponding real node is click probability (click prob * view prob)
# Edge weight from real node to other fictious nodes is the influence probability (random)
class Graph:
    # clickProbabilities : array (nbCategories) representing the final click probability of each category
    def __init__(self, clickProbabilities, nbNodes=N_NODES, stochasticitySeed=0,
                 minSocialInfluProb=MIN_SOCIAL_INFLUENCE_PROB, maxSocialInfluProb=MAX_SOCIAL_INFLUENCE_PROB):
        # with that, every execution will lead to the same random generated values
        np.random.seed(stochasticitySeed)

        self.nbNodes = nbNodes

        # attributes a category for each node. Category of node i is categories[i] (out from 0 to 4)
        self.categoriesPerNode = np.random.randint(0, auctionHouse.NB_CATEGORIES, self.nbNodes)
        self.nodePerCategory = [np.where(self.categoriesPerNode == i)[0] for i in range(auctionHouse.NB_CATEGORIES)]

        # we create exactly nbNodes*2 nodes, so that we have 1 extra node per node
        # (it will be the node enabling to separate click probability from social influence)
        self.prob_matrix = np.zeros((self.nbNodes * 2, self.nbNodes * 2), float)
        for i in range(self.nbNodes * 2):  # line
            for j in range(self.nbNodes * 2):  # column
                if i == j + self.nbNodes:  # red node i' to green node i
                    self.prob_matrix[i, j] = clickProbabilities[self.categoriesPerNode[j]]
                if i < self.nbNodes <= j != i + self.nbNodes:  # green node i to red node j' (i!=j)
                    self.prob_matrix[i, j] = np.random.uniform(minSocialInfluProb, maxSocialInfluProb)

    # this method displays the graph using the networkx library
    # it is not very beautiful but it allows to have a look
    def display(self):
        G = nx.DiGraph()
        # add the nodes, numerated from 0 to nbNodes*2-1
        G.add_nodes_from(range(len(self.prob_matrix)))

        # add the edges with the good color
        for i in range(len(self.prob_matrix)):
            for j in range(len(self.prob_matrix)):
                if self.prob_matrix[i][j] != 0:  # an edge exist between i and j
                    if i < self.nbNodes:  # i is green, we set its edge black
                        G.add_edge(i, j, color='k')
                    else:  # i is red, we set its edge red too
                        G.add_edge(i, j, color='r')

        # colors differently the extra nodes
        color_map_nodes = [(lambda i: 'r' if i >= self.nbNodes else 'g')(i) for i in range(len(self.prob_matrix))]

        # colors differently arrows from extra nodes
        color_map_edges = [G[u][v]['color'] for u, v in G.edges()]

        # labels differently the extra nodes
        labeldict = {}
        for i in range(self.nbNodes):
            labeldict[i] = str(i)
            labeldict[i + self.nbNodes] = str(i) + "'"

        # ignore the deprecation warnings (because it's internal to networkx)
        warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

        # drawing of the graph. These specific parameters are to have a wide spread graph
        pos = nx.spring_layout(G, k=1, iterations=20)
        nx.draw_networkx(G, pos, node_color=color_map_nodes, edge_color=color_map_edges,
                         labels=labeldict, with_labels=True)
        plt.show()
