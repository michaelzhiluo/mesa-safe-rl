"""
Constants associated with the Maze env.
"""

HORIZON = 50

SHAPED_REWARD = True
SHAPING_GAMMA = 0.99

NOISE_SCALE = 0.05
AIR_RESIST = 0.2

MAX_FORCE = 0.3
HARD_MODE = True
FAILURE_COST = 0
VALUE_FUNC_MODE = {"value": 1,
                   "opt_value": 2,
                   "reward_value": 3,
                   "no_value": -1}

IMAGES = True
GOAL_THRESH = 5e-2