import numpy as np
from LinearMabEnvironment import LinearMabEnvironment
from LinUcbLearner import LinUcbLearner

n_arms = 15
T = 100
n_experiments = 50
lin_ucb_rewards_per_experiment = []

env = LinearMabEnvironment(n_arms=n_arms)
learner = LinUcbLearner(env.arms_features)

for i in range(n_experiments):
    arm = learner.pull_arm()
    reward = env.round(arm)
    learner.update(arm, reward)

bestArm = np.argmax(learner.meanRewardPerArm)
print("best arm : "+str(bestArm)+" with a mean reward of "+str(learner.meanRewardPerArm[bestArm]))
print("bids of the best arm : "+str(learner.arms[bestArm]))
