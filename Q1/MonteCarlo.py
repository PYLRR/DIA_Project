import numpy as np

# Single Monte-Carlo run
def simulate_episode(init_prob_matrix, initial_active_nodes, n_steps_max):
    prob_matrix = init_prob_matrix.copy()
    history = np.array([initial_active_nodes])
    active_nodes = initial_active_nodes
    newly_active_nodes = active_nodes

    t = 0
    while t < n_steps_max and np.sum(newly_active_nodes) > 0:
        p = (prob_matrix.T * active_nodes).T
        rnd = np.random.rand(p.shape[0], p.shape[1])
        activated_edges = p > rnd
        prob_matrix = prob_matrix * ((p != 0) == activated_edges)
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
        active_nodes = np.array(active_nodes + newly_active_nodes)
        history = np.concatenate((history, [newly_active_nodes]), axis=0)
        t += 1
    return history


def run(graph,seeds, n_episodes):
    dataset = []  # will contain histories of Monte-Carlo runs

    # simulations
    for e in range(n_episodes):
        dataset.append(simulate_episode(graph.prob_matrix, seeds, n_steps_max=15))

    # let's count the activations of each node
    scores = np.zeros(graph.nbNodes * 2)
    for history in dataset:
        unactivated_nodes = list(range(graph.nbNodes * 2))
        for state in history:
            for i in unactivated_nodes:
                if state[i] == 1:
                    scores[i] += 1
                    unactivated_nodes.remove(i)
    scores /= n_episodes
    return scores