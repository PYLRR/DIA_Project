import numpy as np


class SlidingTSLearner:
    def __init__(self, arms):
        self.arms = arms
        self.dim = arms.shape[1]
        self.collected_rewards = []
        self.pulled_arms = []

        self.t = 0

        self.windowSize = 30
        self.history = np.full((len(self.arms), self.windowSize), np.nan)

        self.age = [] # will contain age of each recorded reward
        self.window = [] # will contain rewards
        for i in range(len(self.arms)):
            self.window.append([])
            self.age.append([])

        # we need history of means and times played to compute gaussian parameters
        self.meanRewardPerArm = np.zeros(len(self.arms))

        # 2 parameters of the gaussian
        self.tau = np.ones(len(self.arms))
        self.mu = np.ones(len(self.arms))

    def pull_arm(self):
        return np.argmax(np.random.normal(self.mu, 1 / self.tau))

    def computeMuFromHistory(self, arm_idx):
        mu = 1
        tau = 1

        mean = 0
        for t in range(len(self.window[arm_idx])):
            mean = np.mean(self.window[arm_idx][:t+1])
            mu = (tau * mu + 1 * self.window[arm_idx][t]) / (tau + 1)
            tau = tau + 1

        self.mu[arm_idx] = mu
        self.tau[arm_idx] = tau
        self.meanRewardPerArm[arm_idx] = mean

    def update_estimation(self, arm_idx, reward):
        self.window[arm_idx].append(reward)
        self.age[arm_idx].append(self.t)

        # recompute parameters of each arm
        for arm in range(len(self.arms)):
            if len(self.age[arm]) > 0 and self.t-self.age[arm][0] > self.windowSize:
                self.window[arm] = self.window[arm][1:]
                self.age[arm] = self.age[arm][1:]
            self.computeMuFromHistory(arm)

    def update(self, arm_idx, reward):
        self.t += 1
        if arm_idx not in self.pulled_arms:
            self.pulled_arms.append(arm_idx)
        self.collected_rewards.append(reward)
        self.update_estimation(arm_idx, reward)
