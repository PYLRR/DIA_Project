import numpy as np


class TSLearner:
    def __init__(self, arms):
        self.arms = arms
        self.dim = arms.shape[1]
        self.collected_rewards = []
        self.pulled_arms = []

        self.meanRewardPerArm = np.zeros(len(self.arms))
        self.timesPlayed = np.zeros(len(self.arms))

        # 2 parameters of the gaussian
        self.tau = np.ones(len(self.arms))
        self.mu = np.ones(len(self.arms))

        self.t = 0

    def pull_arm(self):
        return np.argmax(np.random.normal(self.mu[:],1/self.tau[:]))

    def update_estimation(self, arm_idx, reward):
        self.timesPlayed[arm_idx] += 1
        n = self.timesPlayed[arm_idx]
        # running mean
        self.meanRewardPerArm[arm_idx] = (1-1/n)*self.meanRewardPerArm[arm_idx] + (1/n)*reward

        tau = self.tau[arm_idx]
        mu = self.mu[arm_idx]
        self.tau[arm_idx] = self.tau[arm_idx] + 1
        self.mu[arm_idx] = (tau*mu + 1*reward)/(tau+1)

    def update(self, arm_idx, reward):
        self.t += 1
        if arm_idx not in self.pulled_arms:
            self.pulled_arms.append(arm_idx)
        self.collected_rewards.append(reward)
        self.update_estimation(arm_idx, reward)

