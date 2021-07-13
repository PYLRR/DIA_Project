import numpy as np
from NonStationaryMabEnvironment import NonStationaryMabEnvironment
from SlidingUcbLearner import SlidingUcbLearner
from SlidingTSLearner import SlidingTSLearner
import environment.auctionHouse as auctionHouse
import matplotlib.pyplot as plt

np.random.seed(1)

n_arms = 15
n_experiments = 10000
lin_ucb_rewards_per_experiment = []

env = NonStationaryMabEnvironment(n_arms=n_arms)
learner = SlidingUcbLearner(env.arms)

for i in range(n_experiments):
    arm = learner.pull_arm()
    reward = env.round(learner.arms[arm])
    learner.update(arm, reward)

bestArm = np.argmax(learner.meanRewardPerArm)
bestReward = learner.meanRewardPerArm[bestArm]
print("best arm : " + str(bestArm) + " with a mean reward of " + str(bestReward))
print("bids of the best arm : " + str(learner.arms[bestArm]))


# in : matrix of history of activations (n_experiments x length_sim x nbNodes)
# out : estimation of the ad_quality
# in the graph, activations are like the following :
#       Activation           Click
# node i -> fictitious node j -> node j
#
# Assuming activation is known, and that click probability is slot_prominence*ad_quality
# we can estimate the click probability and divide it by slot_prominence to get ad_quality
def estimate_ad_quality(dataset, graph, learningAdvertiserWonAuctionsHistory, considerOnlyFirstSlot=True):
    n_nodes = env.graph.nbNodes
    # when some node activates, we consider its ancestor and give it specific credit for our node slot
    # this separation of ancestors is done because the slot prominence will be different between 2 slots (while same ad)
    # Notice that when considerOnlyFirstSlot is true, we only consider the first slot.
    activationsForGivenSlot = np.zeros(auctionHouse.NB_SLOTS_PER_CATEGORY)
    fictitiousActivationsForGivenSlot = np.zeros(auctionHouse.NB_SLOTS_PER_CATEGORY)

    episodeCount = 0
    for episode in dataset:
        learningAdvertiserWonAuctions = learningAdvertiserWonAuctionsHistory[episodeCount]
        idx_w_active = np.argwhere(episode[:, :graph.nbNodes] == 1).reshape(
            -1)  # find active nodes, excluding fictitious ones

        for i in range(0, len(idx_w_active), 2):
            slot = learningAdvertiserWonAuctions[graph.categoriesPerNode[idx_w_active[i + 1]]]
            if slot == -1:
                slot = auctionHouse.NB_SLOTS_PER_CATEGORY - 1  # last slot if no one allocated
            # we don't need to browse for previously activated nodes.
            # indeed, a node only activates if its fictitious node was previously activated. So we know it has been.
            # it will be an addition of 1 as we have only 1 ancestor
            activationsForGivenSlot[slot] += 1

        # we will now get the activations of fictitious nodes
        idx_v_active = np.argwhere(episode[:, graph.nbNodes:] == 1).reshape(
            -1)  # find fictitious nodes, excluding active ones

        for i in range(0, len(idx_v_active), 2):
            slot = learningAdvertiserWonAuctions[graph.categoriesPerNode[idx_v_active[i + 1] - graph.nbNodes]]
            if slot == -1:
                slot = auctionHouse.NB_SLOTS_PER_CATEGORY - 1  # last slot if no one allocated
            fictitiousActivationsForGivenSlot[slot] += 1

        episodeCount += 1

    maxSlotToConsider = auctionHouse.NB_SLOTS_PER_CATEGORY - 1
    if considerOnlyFirstSlot:
        maxSlotToConsider = 0

    totalEstimation = 0
    divider = 0
    for slot in range(maxSlotToConsider + 1):
        if fictitiousActivationsForGivenSlot[slot] == 0:
            clickProbabilityForCurrentSlot = 0
        else:
            clickProbabilityForCurrentSlot = activationsForGivenSlot[slot] / fictitiousActivationsForGivenSlot[slot]
            divider += fictitiousActivationsForGivenSlot[slot]
        totalEstimation += fictitiousActivationsForGivenSlot[slot] * clickProbabilityForCurrentSlot / \
                           auctionHouse.SLOT_PROMINENCES[slot]
    totalEstimation /= divider

    return totalEstimation, divider


print("-------------")

estimation, n = estimate_ad_quality(env.history, env.graph, env.learningAdvertiserWonAuctionsHistory)

print("estimation of ad quality with first slot considered : " + str(estimation) + " vs real : " + str(
    env.adQualitiesVector[0]))
d = 1.96 * (estimation * (1 - estimation)) ** 0.5 / n ** 0.5
print("Confidence interval with 95% : [" + str(estimation - d) + ";" + str(estimation + d) + "]")

print("-------------")

estimation, n = estimate_ad_quality(env.history, env.graph, env.learningAdvertiserWonAuctionsHistory, False)
print("estimation of ad quality with all slots considered : " + str(estimation) + " vs real : " + str(
    env.adQualitiesVector[0]))
d = 1.96 * (estimation * (1 - estimation)) ** 0.5 / n ** 0.5
print("Confidence interval with 95% : [" + str(estimation - d) + ";" + str(estimation + d) + "]")


meanReward = [learner.collected_rewards[0]]
for i in range(1, n_experiments):
    reward = learner.collected_rewards[i]
    meanReward.append(i * meanReward[i - 1] / (i + 1) + reward / (i + 1))

### PLOTS
# Plot mean reward
x = range(n_experiments)
ymax = np.full(n_experiments, bestReward)

plt.xlabel('number of iterations')
plt.ylabel('mean reward')
plt.title('Mean reward in function of iterations')
plt.plot(x, meanReward, 'k-')
plt.plot(x,ymax, 'r-')
plt.show()

# Plot cum reward
y = np.cumsum(learner.collected_rewards)
ymaxCum = np.cumsum(ymax)
plt.xlabel('number of iterations')
plt.ylabel('cumulated reward')
plt.title('Cumulated reward in function of iterations')
plt.plot(x, y, 'k-')
plt.plot(x,ymaxCum, 'r-')
plt.show()

# Plot regret
y = np.cumsum(learner.collected_rewards)
ymaxCum = np.cumsum(ymax)
plt.xlabel('number of iterations')
plt.ylabel('estimated regret')
plt.title('Estimated regret in function of iterations')
plt.plot(x, ymaxCum-y, 'k-')
plt.show()