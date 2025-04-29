
import numpy as np
from random import randint


# -------------------------------------------------
# -------- Scheduler evaluation parameters --------
# -------------------------------------------------
STEP = 0.025
NWEIGHTS = (1-0.5)/STEP
FID_WEIGHTS = np.arange(0.5, 1.01, STEP)
#FID_WEIGHTS = [1]

CIRC_CUT_FACTOR = 2 #This means that each random benchmark will be halved in terms of size and copied this number of times

# -------------------------------------------------
# ------- End to end evaluation parameters --------
# -------------------------------------------------
FID_WEIGHT = 0.9
UTIL_WEIGHT = 0
SIZE_TO_REACH = 6
BUDGET = 3

# -------------------------------------------------
# -------- Circuit generator parameters -----------
# -------------------------------------------------
CIRC_NUMBER = 5

#There are 8 users on the dataset
NCLIENTS = 5

TIME_DEVIATION = (NCLIENTS/8)*0.22 #This might not be correct but should be close
TIME_MEAN = (NCLIENTS/8)*3.5
#TIME_DEVIATION = 0.25
#TIME_MEAN = 3.5


#Circuit shots is defined by an uniform distribution
SHOTS_DEVIATION = 2048
SHOTS_MEAN = 8192

#Circuit width is defined by an uniform distribution
WIDTH_MIN = 8
WIDTH_MAX = 24