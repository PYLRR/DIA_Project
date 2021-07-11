import numpy as np
import Q1.MonteCarlo as MonteCarlo
import environment.auctionHouse as auctionHouse
import environment.graph as graph


class LinearMabEnvironment:
    def __init__(self, n_arms):
        self.dim = auctionHouse.NB_CATEGORIES
        self.theta = np.random.dirichlet(np.ones(self.dim), size=1)
        self.arms_features = \
            np.random.random_integers(0, auctionHouse.MAX_BID, size=(n_arms, self.dim))
        self.p = np.zeros(n_arms)
        for i in range(0, n_arms):
            self.p[i] = np.dot(self.theta, self.arms_features[i])
        self.graph = graph.Graph()

        # bids of other advertisers
        self.bids = np.random.randint(0, auctionHouse.MAX_BID + 1,
                                      (auctionHouse.NB_ADVERTISERS, auctionHouse.NB_CATEGORIES))

        # the environment knows the ad quality of the ads
        self.adQualitiesVector = np.random.random(auctionHouse.NB_ADVERTISERS)
        # the environment also knows the click value for each advertiser
        self.valuesOfClick = np.full(auctionHouse.NB_ADVERTISERS, 2)

        self.history = []

    def round(self, pulled_arm, nbAdvertiser=0):
        # set bids of learning advertiser to pulled arm value
        self.bids[0] = pulled_arm

        # gets winner of each category
        winners = auctionHouse.runAuction(self.bids)

        learningAdvertiserWonAuctions = \
            self.graph.updateFromAuctionResult(winners, self.adQualitiesVector[nbAdvertiser])

        # run exactly 1 simulation (it's not really a montecarlo but the method does the job)
        self.history = MonteCarlo.run(
            self.graph, self.graph.seeds, 1)

        # now compute reward
        gain = 0
        for i in range(self.graph.nbNodes):
            cat = self.graph.categoriesPerNode[i]
            slot = learningAdvertiserWonAuctions[cat]
            profit = self.valuesOfClick[nbAdvertiser]
            cost = auctionHouse.computeVCG(0, self.bids, self.adQualitiesVector, cat, slot)
            gain += self.history[i] * (profit - cost)

        return gain
