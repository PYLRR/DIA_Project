import numpy as np
import math
import environment.graph as graph
import environment.auctionHouse as auctionHouse
import Q1.MonteCarlo as MonteCarlo
import matplotlib.pyplot as plt

# reward when a click happens
VALUE_OF_CLICK = 3
# prob to click on the ad knowing we looked at it
AD_QUALITY = 0.7

np.random.seed(0)

# Ad qualities of all advertisers (0 is the learning one)
AdQualitiesVector = np.clip(np.random.normal(0.5, 0.1, auctionHouse.NB_ADVERTISERS), 0.1, 0.9)
AdQualitiesVector[0] = AD_QUALITY

### GRAPH
graph = graph.Graph()

# randomize the bids, may be improved later
bids = np.random.randint(0, auctionHouse.MAX_BID + 1,
                         (auctionHouse.NB_ADVERTISERS, auctionHouse.NB_CATEGORIES))

rewardHistory = []  # to later draw curve of evolution of reward
rewardHistoryWithoutRollbacks = []
labels = []  # to annotate points

### BIDS/AUCTIONS
# sets bids of learning advertiser to 0
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

    rewardHistory.append(gain)
    rewardHistoryWithoutRollbacks.append(gain)

    if gain >= previousReward:  # we improved
        previousReward = gain
        nbOfTurnsWithoutImprovement = 0
        labels.append(str(currentImprovedCategory) + "+")
    else:  # we didnt't improve : reset last modification
        bids[0, currentImprovedCategory] -= 1
        nbOfTurnsWithoutImprovement += 1
        labels.append(str(currentImprovedCategory) + "-")


        if nbOfTurnsWithoutImprovement >= auctionHouse.NB_CATEGORIES:
            break

    # change category from which we improve bid
    currentImprovedCategory = (currentImprovedCategory + 1) % auctionHouse.NB_CATEGORIES

### GRAPH/OUTPUTS

n = len(rewardHistory)
print(bids)
print(previousReward)

# Best reward obtained in fct of time
x = range(n)
plt.xlabel('iterations')
plt.ylabel('average reward')
plt.title('Evolution of reward with the greedy algorithm iterations')
plt.plot(x, rewardHistory, 'k-')
for i in range(len(labels)):
    if labels[i][-1] == "+":  # increase of reward
        plt.annotate(labels[i][:-1], xy=(i, rewardHistory[i]), xytext=(i, rewardHistory[i] + 0.02),
                     color='green')
    else:  # decrease of reward
        plt.annotate(labels[i][:-1], xy=(i, rewardHistory[i]), xytext=(i, rewardHistory[i] + 0.02),
                     color='red')
plt.show()

# Regret in fct of steps
n2 = len(rewardHistoryWithoutRollbacks)
x2 = range(n2)
maxRewards = np.full(n2, previousReward)
y2 = np.cumsum(rewardHistoryWithoutRollbacks)
ymax = np.cumsum(maxRewards)
plt.xlabel('iterations')
plt.ylabel('regret')
plt.title('Evolution of regret with the greedy algorithm iterations')
plt.plot(x2, ymax - y2, 'k-')
plt.show()
