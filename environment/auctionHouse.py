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
    for i in range(NB_CATEGORIES):
        # sort advertisers in decreasing bid order for this category
        sorted_advertisers = bids[np.argsort(bids[:, i])][::-1]
        for j in range(NB_SLOTS_PER_CATEGORY):
            # set the jth best bid of this category to the slot j
            try:
                res[i, j] = np.where(np.all(bids == sorted_advertisers[j], axis=1))[0][0]
            except: # triggered if nbAdvertisers < nbSlotsPerCategory
                res[i, j] = -1
    return res
