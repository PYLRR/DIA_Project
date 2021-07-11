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

### GRAPH
graph = graph.Graph()
graph.updateFromAuctionResult(winners, AD_QUALITY)
graph.display()  # just disable that in case there are problems with networkx lib

### MONTECARLO
x = []
y = []
meanNbOfActivated = 0
maxNbIterations = 1000
for n_episodes in range(1, maxNbIterations, 1):
    activationProbabilities = MonteCarlo.run(graph, graph.seeds, n_episodes)
    # avg nb of activated nodes is just the sum of the probabilities of activation of each node (not the fictious ones)
    averageNbOfActivatedNodes = np.sum(activationProbabilities[:graph.nbNodes])
    x.append(n_episodes)
    y.append(averageNbOfActivatedNodes)
    meanNbOfActivated += averageNbOfActivatedNodes

# now compute bounds of the estimation
yup = []
ydown = []
meanNbOfActivated /= maxNbIterations
precision_sigma = 0.05
for i in range(1, maxNbIterations, 1):
    # computation of estimation fiability bound
    precision = math.sqrt(math.log(np.count_nonzero(graph.seeds)) * math.log(1 / precision_sigma) / i)
    yup.append(meanNbOfActivated + precision*math.sqrt(graph.nbNodes)/2)
    ydown.append(meanNbOfActivated - precision*math.sqrt(graph.nbNodes)/2)


plt.xlabel('number of iterations')
plt.ylabel('average number of activated nodes')
plt.title('Estimations of average number of activated nodes when iterations vary')
plt.plot(x,y, 'k-')
plt.plot(x,yup, 'r-')
plt.plot(x,ydown, 'r-')
plt.axis([0, maxNbIterations, 0, 2*meanNbOfActivated])
plt.show()

# value precise enough for n iterations
optimalNb = 250

# computation of estimation fiability bound
precision_sigma = 0.05
precision = math.sqrt(math.log(np.count_nonzero(graph.seeds)) * math.log(1 / precision_sigma) / optimalNb)
print("\nprecision of pActivation per node with p >", 100 * (1 - precision_sigma), "% : ", precision)
