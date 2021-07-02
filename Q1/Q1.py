import numpy as np
import math
import environment.graph as graph


# Single Monte-Carlo run
def simulate_episode(init_prob_matrix, initial_active_nodes, n_steps_max):
    prob_matrix = init_prob_matrix.copy()
    history = np.array([initial_active_nodes])
    active_nodes = initial_active_nodes
    newly_active_nodes = active_nodes

    t = 0
    while t < n_steps_max and np.sum(newly_active_nodes) > 0:
        p = (prob_matrix.T * active_nodes).T
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
        prob_matrix = prob_matrix * ((p != 0) == activated_edges)
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
        active_nodes = np.array(active_nodes + newly_active_nodes)
        history = np.concatenate((history, [newly_active_nodes]), axis=0)
        t += 1
    return history


np.random.seed(0)

NB_ADVERTISERS = 3
NB_CATEGORIES = 5
NB_SLOTS = 6  # currently unused
n_nodes = 15
n_episodes = 1000  # nb of Monte-Carlo we run

# create a graph (with extra nodes per category to model click probability)
graph = graph.Graph(n_nodes, stochasticitySeed=0)
graph.display()  # just display that in case there are problems with networkx lib

# randomize the bids for now, may be improved later
bids = []
for i in range(NB_ADVERTISERS):  # advertisers
    # a bid for each category, belonging to {0,1,2,3,4}
    bids.append(np.random.randint(0, 5, NB_CATEGORIES))
# gets winner of each category
winners = []
for j in range(NB_CATEGORIES):
    maxi = 0
    for i in range(NB_ADVERTISERS):
        if bids[i][j] > bids[maxi][j]:
            maxi = i
    winners.append(maxi)

# now get prepared for the Monte-Carlo
dataset = []  # will contain histories of Monte-Carlo runs
# 1 for category won by us (advertiser 0) 0 for others
seedsArray = []
for i in range(n_nodes + NB_CATEGORIES):
    if i >= n_nodes and winners[i - n_nodes] == 0:
        seedsArray.append(1)
    else:
        seedsArray.append(0)
seeds = np.array(seedsArray)

# simulations
for e in range(n_episodes):
    dataset.append(simulate_episode(graph.prob_matrix, seeds, n_steps_max=15))

# let's count the activations of each node
scores = np.zeros(n_nodes + NB_CATEGORIES)
for history in dataset:
    unactivated_nodes = list(range(n_nodes + NB_CATEGORIES))
    for state in history:
        for i in unactivated_nodes:
            if state[i] == 1:
                scores[i] += 1
                unactivated_nodes.remove(i)
scores /= n_episodes
print("\nestimated probabilities of activation : \n", scores)

# computation of estimation fiability
precision_sigma = 0.05
precision = math.sqrt(math.log(np.count_nonzero(seeds)) * math.log(1/precision_sigma) / n_episodes)
print("\nprecision per node with p >",100*(1-precision_sigma),"% : ",precision)
