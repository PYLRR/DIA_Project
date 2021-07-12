import numpy as np
import Q1.MonteCarlo as MonteCarlo
import environment.auctionHouse as auctionHouse
import environment.graph as graph


class LinearMabEnvironment:
    def __init__(self, n_arms):
        self.dim = auctionHouse.NB_CATEGORIES
        self.arms = \
            np.random.random_integers(0, auctionHouse.MAX_BID, size=(n_arms, self.dim))
        self.graph = graph.Graph()

        # bids of other advertisers
        self.bids = np.random.randint(0, auctionHouse.MAX_BID + 1,
                                      (auctionHouse.NB_ADVERTISERS, auctionHouse.NB_CATEGORIES))

        # the environment knows the ad quality of the ads
        self.adQualitiesVector = np.clip(np.random.normal(0.5, 0.1, auctionHouse.NB_ADVERTISERS), 0.1, 0.9)
        # the environment also knows the click value for each advertiser
        self.valuesOfClick = np.clip(np.random.normal(2, 0.5, auctionHouse.NB_ADVERTISERS), 0.5, 5.0)

        self.history = []

        self.learningAdvertiserWonAuctions = []
        # keep track of history of auctions
        self.learningAdvertiserWonAuctionsHistory = []

    def round(self, pulled_arm, nbAdvertiser=0):
        # set bids of learning advertiser to pulled arm value
        self.bids[0] = pulled_arm

        # gets winner of each category
        winners = auctionHouse.runAuction(self.bids)

        self.learningAdvertiserWonAuctions = \
            self.graph.updateFromAuctionResult(winners, self.adQualitiesVector[nbAdvertiser])

        self.learningAdvertiserWonAuctionsHistory.append(self.learningAdvertiserWonAuctions)

        # run exactly 1 simulation (it's not really a montecarlo but the method does the job)
        historyDataset, activations = MonteCarlo.run(
            self.graph, self.graph.seeds, 1)

        self.history.append(historyDataset[0])

        # now compute reward
        gain = 0
        for i in range(self.graph.nbNodes):
            cat = self.graph.categoriesPerNode[i]
            slot = self.learningAdvertiserWonAuctions[cat]
            profit = self.valuesOfClick[nbAdvertiser]
            cost = auctionHouse.computeVCG(0, self.bids, self.adQualitiesVector, cat, slot)
            gain += activations[i] * (profit - cost)

        return gain
