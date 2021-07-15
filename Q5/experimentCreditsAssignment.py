import numpy as np
from Q4.MabEnvironment import MabEnvironment
from Q4.UcbLearner import UcbLearner
from Q4.TSLearner import TSLearner
import environment.auctionHouse as auctionHouse
import matplotlib.pyplot as plt

# NOTE : Q5 does not have a code answer. This is some experiment to estimate activation probabilities,
# it shows that classical credit assignments is likely to fail according to our model.

np.random.seed(1)

n_arms = 15
n_experiments = 1000
lin_ucb_rewards_per_experiment = []

env = MabEnvironment(n_arms=n_arms)
learner = UcbLearner(env.arms)

for i in range(n_experiments):
    arm = learner.pull_arm()
    reward = env.round(learner.arms[arm])
    learner.update(arm, reward)

bestArm = np.argmax(learner.meanRewardPerArm)
bestReward = learner.meanRewardPerArm[bestArm]
print("best arm : " + str(bestArm) + " with a mean reward of " + str(bestReward))
print("bids of the best arm : " + str(learner.arms[bestArm]))


# in : matrix of history of activations (n_experiments x length_sim x nbNodes)
# out : estimation of the activation probabilities
# in the graph, activations are like the following :
#       Activation           Click
# node i -> fictitious node j -> node j
#
# We can estimate the activation probabilities with the credit assignment method
def estimate_activ_p(dataset, graph, node_index):
    n_nodes = graph.nbNodes * 2
    estimated_prob = np.ones(n_nodes) * 1.0 / (n_nodes - 1)
    credits = np.zeros(n_nodes)
    occurr_v_active = np.zeros(n_nodes)
    n_episodes = len(dataset)
    for episode in dataset:
        idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)
        if len(idx_w_active) > 0 and idx_w_active[0] > 0:
            active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
            credits += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)
        for v in range(0, n_nodes):
            if v != node_index:
                idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                if len(idx_w_active) > 0 and (idx_v_active < idx_w_active or len(idx_w_active) == 0):
                    occurr_v_active[v] += 1
    estimated_prob = credits / occurr_v_active
    estimated_prob = np.nan_to_num(estimated_prob)
    return estimated_prob


tab = np.ones((env.graph.nbNodes, env.graph.nbNodes*2))
for node in range(env.graph.nbNodes, env.graph.nbNodes * 2):
    tab[node-env.graph.nbNodes] = np.array(estimate_activ_p(env.history, env.graph, node))

print(tab)
print(env.graph.prob_matrix)
# very different results, because of non independencies
