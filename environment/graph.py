import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import warnings
import environment.auctionHouse as auctionHouse

MIN_SOCIAL_INFLUENCE_PROB = 0.0
MAX_SOCIAL_INFLUENCE_PROB = 0.5

N_NODES = 15


# class used to compute and then access the global graph of the project
# The idea of the graph is to have a number of nodes n, and to add a fictious
# node to each node, resulting in a graph of nodes cardinality of 2n.
# Edge weight from fictious to its corresponding real node is click probability (click prob * view prob)
# Edge weight from real node to other fictious nodes is the influence probability (random)
class Graph:
    # clickProbabilities : array (nbCategories) representing the final click probability of each category
    def __init__(self, nbNodes=N_NODES):

        self.nbNodes = nbNodes

        # attributes a category for each node. Category of node i is categories[i] (out from 0 to 4)
        self.categoriesPerNode = np.random.randint(0, auctionHouse.NB_CATEGORIES, self.nbNodes)
        self.nodePerCategory = [np.where(self.categoriesPerNode == i)[0] for i in range(auctionHouse.NB_CATEGORIES)]

        # we create exactly nbNodes*2 nodes, so that we have 1 extra node per node
        # (it will be the node enabling to separate click probability from social influence)
        self.prob_matrix = np.zeros((self.nbNodes * 2, self.nbNodes * 2), float)

        # no seed by default
        self.seeds = np.zeros(self.nbNodes * 2)

    def changeTransitionProbabilities(self, clickProbabilities, minSocialInfluProb=MIN_SOCIAL_INFLUENCE_PROB,
                                      maxSocialInfluProb=MAX_SOCIAL_INFLUENCE_PROB):
        for i in range(self.nbNodes * 2):  # line
            for j in range(self.nbNodes * 2):  # column
                if i == j + self.nbNodes:  # red node i' to green node i
                    self.prob_matrix[i, j] = clickProbabilities[self.categoriesPerNode[j]]
                if i < self.nbNodes <= j != i + self.nbNodes:  # green node i to red node j' (i!=j)
                    self.prob_matrix[i, j] = np.random.uniform(minSocialInfluProb, maxSocialInfluProb)

    # updates transition probabilities and seeds from the results of an auction
    def updateFromAuctionResult(self, auctionResults, ad_quality):
        # wonArray[i]=j if jth slot won for category i (the learning advertiser is the advertiser 0)
        learningAdvertiserWonAuctions = np.full(auctionHouse.NB_CATEGORIES, -1)
        wonCategories = []
        for i in range(auctionHouse.NB_CATEGORIES):
            for j in range(auctionHouse.NB_SLOTS_PER_CATEGORY):
                if auctionResults[i, j] == 0:  # the learning advertiser won slot j for category i
                    learningAdvertiserWonAuctions[i] = j
                    wonCategories.append(i)

        # get click probability of each category knowing the slot we have
        clickProb = []
        for slot in learningAdvertiserWonAuctions:
            if slot == -1:
                # we didnt win any slot for this category
                # this click probability is then only useful for cascade
                # and for this purpose we assume we have the worst slot
                slotProminence = auctionHouse.SLOT_PROMINENCES[auctionHouse.NB_SLOTS_PER_CATEGORY - 1]
            else:
                slotProminence = auctionHouse.SLOT_PROMINENCES[slot]
            clickProb.append(slotProminence * ad_quality)
        self.changeTransitionProbabilities(clickProb)

        # Seeds update

        # no seed by default
        self.seeds = np.zeros(self.nbNodes * 2)
        for category in wonCategories:
            for node in self.nodePerCategory[category]:
                self.seeds[node + self.nbNodes] = 1  # add the corresponding fictious node in seeds

        # we return learningAdvertiserWonAuctions as it can be used by the learner
        # contains the slot won for each cat (or -1 if no one)
        return learningAdvertiserWonAuctions

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
