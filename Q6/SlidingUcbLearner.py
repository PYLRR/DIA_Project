import numpy as np


class SlidingUcbLearner:
    def __init__(self, arms):
        self.arms = arms
        self.dim = arms.shape[1]
        self.collected_rewards = []
        self.pulled_arms = []

        self.meanRewardPerArm = np.zeros(len(self.arms))
        self.timesPlayed = np.zeros(len(self.arms))
        self.t = 0

        self.windowSize = 50
        self.history = np.full((len(self.arms), self.windowSize), np.nan)
        self.index = 0  # index where we do the next insert in the window

    def compute_ucbs(self):
        ucbs = np.zeros(len(self.arms))
        for arm in range(len(self.arms)):
            mean = self.meanRewardPerArm[arm]
            na = self.timesPlayed[arm]
            bonus = np.inf if na == 0 else \
                (2 * np.log(self.t) / na) ** 0.5
            ucbs[arm] = mean + bonus
        return ucbs

    def pull_arm(self):
        ucbs = self.compute_ucbs()
        return np.argmax(ucbs)

    def update_estimation(self, arm_idx, reward):
        self.timesPlayed[arm_idx] += 1

        self.history[:, self.index] = np.nan # set no value to other arms
        self.history[arm_idx,self.index] = reward
        self.index = (self.index + 1) % self.windowSize

        # recompute mean and n of each arm
        for arm in range(len(self.arms)):
            relevantHistoryValues = ~np.isnan(self.history[arm])
            mean = np.mean(self.history[arm,relevantHistoryValues])
            self.meanRewardPerArm[arm] = mean

            n = np.count_nonzero(relevantHistoryValues)
            self.timesPlayed[arm] = n

    def update(self, arm_idx, reward):
        self.t += 1
        if arm_idx not in self.pulled_arms:
            self.pulled_arms.append(arm_idx)
        self.collected_rewards.append(reward)
        self.update_estimation(arm_idx, reward)