import numpy as np
import math
import environment.graph as graph
import environment.auctionHouse as auctionHouse
import MonteCarlo as MonteCarlo
import matplotlib.pyplot as plt

# prob to click on the ad knowing we looked at it
AD_QUALITY = 0.7

np.random.seed(0)

### BIDS/AUCTIONS
# randomize the bids, may be improved later
bids = np.random.randint(0, auctionHouse.MAX_BID + 1,
                         (auctionHouse.NB_ADVERTISERS, auctionHouse.NB_CATEGORIES))

# gets winner of each category
winners = auctionHouse.runAuction(bids)

# wonArray[i]=j if jth slot won for category i (the learning advertiser is the advertiser 0)
learningAdvertiserWonAuctions = np.full(auctionHouse.NB_CATEGORIES, -1)
wonCategories = []
for i in range(auctionHouse.NB_CATEGORIES):
    for j in range(auctionHouse.NB_SLOTS_PER_CATEGORY):
        if winners[i, j] == 0:  # the learning advertiser won slot j for category i
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
    clickProb.append(slotProminence * AD_QUALITY)

### GRAPH
graph = graph.Graph(clickProb, stochasticitySeed=0)
graph.display()  # just disable that in case there are problems with networkx lib

# compute the seeds (all the fictious nodes connected to a won category node)
seeds = np.zeros(graph.nbNodes * 2)
for category in wonCategories:
    for node in graph.nodePerCategory[category]:
        seeds[node + graph.nbNodes] = 1  # add the corresponding fictious node in seeds

### MONTECARLO
x = []
y = []
i = 0
for n_episodes in range(1, 1000, 10):
    activationProbabilities = MonteCarlo.run(graph, seeds, n_episodes, 15)
    # avg nb of activated nodes is just the sum of the probabilities of activation of each node (not the fictious ones)
    averageNbOfActivatedNodes = np.sum(activationProbabilities[:graph.nbNodes])
    x.append(n_episodes)
    y.append(averageNbOfActivatedNodes)

plt.plot(x,y)
plt.show()

# value stabilized for 250 iterations, we can use it
optimalNb = 250

# computation of estimation fiability bound
precision_sigma = 0.05
precision = math.sqrt(math.log(np.count_nonzero(seeds)) * math.log(1 / precision_sigma) / optimalNb)
print("\nprecision of pActivation per node with p >", 100 * (1 - precision_sigma), "% : ", precision)
