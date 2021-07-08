import numpy as np
import math
import environment.graph as graph
import environment.auctionHouse as auctionHouse
import Q1.MonteCarlo as MonteCarlo

# reward when a click happens
VALUE_OF_CLICK = 2
# prob to click on the ad knowing we looked at it
AD_QUALITY = 0.7
# Ad qualities of all advertisers (0 is the learning one)
AdQualitiesVector = np.random.random(auctionHouse.NB_ADVERTISERS)

np.random.seed(0)

### GRAPH
graph = graph.Graph()

### BIDS/AUCTIONS
# randomize the bids, may be improved later
bids = np.random.randint(0, auctionHouse.MAX_BID + 1,
                         (auctionHouse.NB_ADVERTISERS, auctionHouse.NB_CATEGORIES))
bids[0] = np.zeros(auctionHouse.NB_CATEGORIES)  # set bids of current advertiser to 0

previousReward = 0 # when bids are 0, the reward will be 0
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

    graph.changeTransitionProbabilities(clickProb)

    # compute the seeds (all the fictious nodes connected to a won category node)
    seeds = np.zeros(graph.nbNodes * 2)
    for category in wonCategories:
        for node in graph.nodePerCategory[category]:
            seeds[node + graph.nbNodes] = 1  # add the corresponding fictious node in seeds

    ### MONTECARLO
    activationProbabilities = MonteCarlo.run(graph, seeds, 2000)
    # avg nb of activated nodes is just the sum of the probabilities of activation of each node (not the fictious ones)
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