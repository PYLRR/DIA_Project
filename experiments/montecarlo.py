import numpy as np
import math

# see first video of exercise of social influence topic
def simulate_episode(init_prob_matrix, initial_active_nodes, n_steps_max):
  prob_matrix = init_prob_matrix.copy()
  n_nodes = prob_matrix.shape[0]
  history = np.array([initial_active_nodes])
  active_nodes = initial_active_nodes
  newly_active_nodes = active_nodes

  t=0
  while t<n_steps_max and np.sum(newly_active_nodes)>0:
    p = (prob_matrix.T * active_nodes).T
    activated_edges = p>np.random.rand(p.shape[0], p.shape[1])
    prob_matrix = prob_matrix * ((p!=0)==activated_edges)
    newly_active_nodes = (np.sum(activated_edges,axis=0)>0) * (1 - active_nodes)
    active_nodes = np.array(active_nodes + newly_active_nodes)
    history = np.concatenate((history, [newly_active_nodes]),axis=0)
    t+=1
  return history

# with that, every execution will lead to the same random generated values
np.random.seed(0) 

n_nodes = 5
n_episodes = 5000 # nb of Monte-Carlo runs
prob_matrix = np.random.uniform(0.0,0.1,(n_nodes,n_nodes))
dataset = [] # will contain histories of Monte-Carlo runs

# seeds we're testing with Monte-Carlo
seeds = np.random.binomial(1, 0.4, size=(n_nodes))

print("probability matrix : \n",prob_matrix)
print("\nroots : \n",seeds)

# simulations
for e in range(n_episodes):
  dataset.append(simulate_episode(prob_matrix, seeds, n_steps_max=15))

# let's count the activations of each node
scores = np.zeros(n_nodes)
for history in dataset:
  unactivated_nodes = list(range(n_nodes))
  for state in history:
    for i in unactivated_nodes:
      if state[i]==1:
        scores[i]+=1
        unactivated_nodes.remove(i)
scores /= n_episodes
print("\nestimated probabilities of activation : \n",scores)

# computation of estimation fiability
precision_sigma = 0.05
precision = n_nodes * math.sqrt(math.log(seeds.shape[0]) * math.log(1/precision_sigma) / n_episodes)
print("\nprecision with p >",100*(1-precision_sigma),"% : ",precision)

