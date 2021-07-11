import numpy as np


class LinUcbLearner:
    def __init__(self, arms_features):
        self.arms = arms_features
        self.dim = arms_features.shape[1]
        self.collected_rewards = []
        self.pulled_arms = []
        self.c = 2.0
        self.M = np.identity(self.dim)
        self.b = np.atleast_2d(np.zeros(self.dim)).T
        self.theta = np.dot(np.linalg.inv(self.M), self.b)

        self.meanRewardPerArm = np.zeros(len(self.arms))
        self.timesPlayed = np.zeros(len(self.arms))
        self.t = 0

    def compute_ucbs(self):
        self.theta = np.dot(np.linalg.inv(self.M), self.b)
        ucbs = np.zeros(len(self.arms))
        for arm in range(len(self.arms)):
            mean = self.meanRewardPerArm[arm]
            na = self.timesPlayed[arm]
            bonus = np.inf if na == 0 else\
                (2 * np.log(self.t) / na) ** 0.5
            ucbs[arm] = mean + bonus
        return ucbs

    def pull_arm(self):
        # all arms have been pulled, we now use ucbs
        ucbs = self.compute_ucbs()
        return np.argmax(ucbs)

    def update_estimation(self, arm_idx, reward):
        arm = np.atleast_2d(self.arms[arm_idx]).T
        self.M += np.dot(arm, arm.T)
        self.b += reward * arm

        self.timesPlayed[arm_idx] += 1
        n = self.timesPlayed[arm_idx]
        # running mean
        self.meanRewardPerArm[arm_idx] = (1-1/n)*self.meanRewardPerArm[arm_idx] + (1/n)*reward

    def update(self, arm_idx, reward):
        self.t += 1
        if arm_idx not in self.pulled_arms:
            self.pulled_arms.append(arm_idx)
        self.collected_rewards.append(reward)
        self.update_estimation(arm_idx, reward)

