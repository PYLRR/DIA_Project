import numpy as np
import math
import environment.graph as graph
import environment.auctionHouse as auctionHouse
import Q1.MonteCarlo as MonteCarlo

# reward when a click happens
VALUE_OF_CLICK = 2
# prob to click on the ad knowing we looked at it
AD_QUALITY = 0.7

np.random.seed(0)

# Ad qualities of all advertisers (0 is the learning one)
AdQualitiesVector = np.clip(np.random.normal(0.5, 0.1, auctionHouse.NB_ADVERTISERS), 0.1, 0.9)

### GRAPH
graph = graph.Graph()

### BIDS/AUCTIONS
# randomize the bids, may be improved later
bids = np.random.randint(0, auctionHouse.MAX_BID + 1,
                         (auctionHouse.NB_ADVERTISERS, auctionHouse.NB_CATEGORIES))
bids[0] = np.zeros(auctionHouse.NB_CATEGORIES)  # set bids of current advertiser to 0

previousReward = 0  # when bids are 0, the reward will be 0
currentImprovedCategory = 0  # to jump to 0 at next iteration
nbOfTurnsWithoutImprovement = 0  # if it reaches NB_CATEGORIES, the greedy algorithm stops
stabilized = False
while not stabilized:
    bids[0, currentImprovedCategory] += 1

    # if this bid reached the maximum, it is considered as a no-improvement
    if bids[0, currentImprovedCategory] > auctionHouse.MAX_BID:
        bids[0, currentImprovedCategory] -= 1
        nbOfTurnsWithoutImprovement += 1
        if nbOfTurnsWithoutImprovement >= auctionHouse.NB_CATEGORIES:
            break
        currentImprovedCategory = (currentImprovedCategory + 1) % auctionHouse.NB_CATEGORIES
        continue

    # gets winner of each category
    winners = auctionHouse.runAuction(bids)
    learningAdvertiserWonAuctions = graph.updateFromAuctionResult(winners, AD_QUALITY)

    ### MONTECARLO
    activationProbabilities = MonteCarlo.run(graph, graph.seeds, 2000)[1]

    # compute gain
    gain = 0
    for i in range(graph.nbNodes):
        cat = graph.categoriesPerNode[i]
        slot = learningAdvertiserWonAuctions[cat]
        profit = VALUE_OF_CLICK
        cost = auctionHouse.computeVCG(0, bids, AdQualitiesVector, cat, slot)
        gain += activationProbabilities[i] * (profit - cost)

    if gain >= previousReward:  # we improved
        previousReward = gain
        nbOfTurnsWithoutImprovement = 0
    else:  # we didnt't improve : reset last modification
        bids[0, currentImprovedCategory] -= 1
        nbOfTurnsWithoutImprovement += 1
        if nbOfTurnsWithoutImprovement >= auctionHouse.NB_CATEGORIES:
            break

    # change category from which we improve bid
    currentImprovedCategory = (currentImprovedCategory + 1) % auctionHouse.NB_CATEGORIES

print(bids)
