estimated_prob = np.ones() * 1.0 / (n_nodes - 1)
credits = np.zeros(n_nodes)
occurr_v_active = np.zeros(n_nodes)
n_episodes = len(dataset)
for episode in dataset:
    idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)
    if len(idx_w_active) > 0 and idx_w_active > 0:
        active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
        credits += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)
    for v in range(0, n_nodes):
        if (v != node_index):
            idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
            if len(idx_w_active) > 0 and (idx_v_active < idx_w_active or len(idx_w_active) == 0):
                occurr_v_active[v] += 1
estimated_prob = credits / occurr_v_active
estimated_prob = np.nan_to_num(estimated_prob)