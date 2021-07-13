import numpy as np


class SlidingTSLearner:
    def __init__(self, arms):
        self.arms = arms
        self.dim = arms.shape[1]
        self.collected_rewards = []
        self.pulled_arms = []

        self.t = 0

        self.windowSize = 50
        self.history = np.full((len(self.arms), self.windowSize), np.nan)
        self.index = 0  # index where we do the next insert in the window

        # we need history of means and times played to compute gaussian parameters
        self.meanRewardPerArm = np.zeros((len(self.arms), self.windowSize))
        self.timesPlayed = np.zeros((len(self.arms), self.windowSize))

        # 2 parameters of the gaussian
        self.tau = np.ones((len(self.arms), self.windowSize))
        self.mu = np.ones((len(self.arms), self.windowSize))

    def pull_arm(self):
        previousIdx = (self.index - 1) % self.windowSize
        return np.argmax(np.random.normal(self.mu[:, previousIdx], 1 / self.tau[:, previousIdx]))

    def computeMuFromHistory(self, arm_idx):
        # if window not ful dont iterate over full array
        if self.t < self.windowSize:
            oldest = self.windowSize - 1
            nbIterations = self.t
        else:
            oldest = self.index
            nbIterations = self.windowSize

        # special variable used to ignore cases of the array where history has nan
        # (this means the arm wasn't played at this step)
        malusNan = 0

        # we consider that at oldest observation, we had a 1
        self.mu[arm_idx, oldest] = 1
        for t in range(nbIterations):
            next = (oldest + t + 1) % self.windowSize
            if np.isnan(self.history[arm_idx, next]):
                malusNan += 1
                continue
            previous = (oldest + t - malusNan) % self.windowSize
            malusNan = 0

            n = self.timesPlayed[arm_idx, next]
            mean = self.meanRewardPerArm[arm_idx, next]
            tau = self.tau[arm_idx, previous]
            mu = self.mu[arm_idx, previous]
            self.mu[arm_idx, next] = (tau * mu + n * mean) / (tau + n)

    def update_estimation(self, arm_idx, reward):
        prevI = (self.index - 1) % self.windowSize
        self.timesPlayed[arm_idx, self.index] = self.timesPlayed[arm_idx, prevI] + 1

        self.history[:, self.index] = np.nan  # set no value to other arms
        self.history[arm_idx, self.index] = reward

        # recompute parameters of each arm
        for arm in range(len(self.arms)):
            relevantHistoryValues = ~np.isnan(self.history[arm])
            mean = np.mean(self.history[arm, relevantHistoryValues])
            self.meanRewardPerArm[arm, self.index] = mean

            n = np.count_nonzero(relevantHistoryValues)
            self.timesPlayed[arm, self.index] = n

            self.computeMuFromHistory(arm)
            # tau is 1 + nb of times the arm was played (iteratively done in Q4)
            self.tau[arm, self.index] = n + 1

        self.index = (self.index + 1) % self.windowSize

    def update(self, arm_idx, reward):
        self.t += 1
        if arm_idx not in self.pulled_arms:
            self.pulled_arms.append(arm_idx)
        self.collected_rewards.append(reward)
        self.update_estimation(arm_idx, reward)

    def getBestArm(self):
        best = np.argmax(self.meanRewardPerArm[:, self.index])
        return best, self.meanRewardPerArm[best, self.index]
