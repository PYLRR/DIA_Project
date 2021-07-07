# this file represents the social network that acts as an ad publisher

import numpy as np

NB_ADVERTISERS = 20
NB_CATEGORIES = 5
NB_SLOTS_PER_CATEGORY = 6
MAX_BID = 4

# array (nbSlotsPerCategory) representing the view prob of each slot
SLOT_PROMINENCES = [0.5, 0.45, 0.4, 0.35, 0.3, 0.2]


# input : np matrix (nbAdvertisers,nbCategories) representing the bid of each advertiser to each category
# output : np matrix (nbCategories,nbSlotsPerCategory) representing the affectation of each slot of each category
def runAuction(bids):
    res = np.zeros((NB_CATEGORIES, NB_SLOTS_PER_CATEGORY))

    # shuffle bids so that when we have some equal values, the choice of advertiser is random
    shuffledBids = np.copy(bids)
    np.random.shuffle(shuffledBids)

    for i in range(NB_CATEGORIES):
        # sort advertisers in decreasing bid order for this category
        sorted_advertisers = shuffledBids[np.argsort(shuffledBids[:, i])][::-1]
        for j in range(NB_SLOTS_PER_CATEGORY):
            # set the jth best bid of this category to the slot j
            try:
                res[i, j] = np.where(np.all(bids == sorted_advertisers[j], axis=1))[0][0]
            except:  # triggered if nbAdvertisers < nbSlotsPerCategory
                res[i, j] = -1
    return res


def computeVCG(numberOfConcernedAdvertiser, bids, adQualitiesVector, concernedCategory, concernedSlot):
    # compute Ya
    sBar = runAuction(bids)
    Ya = 0
    for i in range(NB_CATEGORIES):
        for j in range(NB_SLOTS_PER_CATEGORY):
            advertiser = int(sBar[i, j])
            if advertiser != -1 and advertiser != numberOfConcernedAdvertiser: # check advertiser exists and is not us
                Ya += SLOT_PROMINENCES[j] * adQualitiesVector[advertiser] * bids[advertiser, i]

    # compute Xa
    bidsWithoutAdvertiser = np.delete(bids, numberOfConcernedAdvertiser, numberOfConcernedAdvertiser)
    smax = runAuction(bidsWithoutAdvertiser)
    Xa = 0
    for i in range(NB_CATEGORIES):
        for j in range(NB_SLOTS_PER_CATEGORY):
            advertiser = int(smax[i, j])
            if advertiser != -1: # check advertiser exists
                if advertiser >= numberOfConcernedAdvertiser:
                    advertiser+=1 # change rank to avoid picking our bid
                Xa += SLOT_PROMINENCES[j] * adQualitiesVector[advertiser] * bids[advertiser, i]

    # compute total
    slotPro = SLOT_PROMINENCES[concernedSlot]
    bid = bids[numberOfConcernedAdvertiser, concernedCategory]
    return (Xa-Ya)/(slotPro*bid)


#np.random.seed(0)

#bids = [[0, 4, 2, 3, 1], [4, 4, 2, 1, 0], [3, 2, 0, 0, 1], [4, 3, 2, 1, 0], [0, 1, 2, 3, 4], [2, 2, 2, 2, 2],
#        [0, 1, 0, 0, 0]]
#print(runAuction(np.array(bids)))
#print(computeVCG(0 ,np.array(bids),[0.2,0.3,0.4,0.5,0.2,0.4,0.3],1,1))
