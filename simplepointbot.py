"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise. State
representation is (x, y). Action representation is (dx, dy).
"""

import os
import pickle

import os.path as osp
import numpy as np
# import matplotlib.pyplot as plt
from gym import Env
from gym import utils
from gym.spaces import Box


"""
Constants associated with the PointBot env.
"""

START_POS = [-50, 0]
END_POS = [0, 0]
GOAL_THRESH = 1.
START_STATE = START_POS
GOAL_STATE = END_POS

MAX_FORCE = 1
HORIZON = 100

NOISE_SCALE = 0.05
AIR_RESIST = 0.2

HARD_MODE = False
FAILURE_COST = 100


class Obstacle:
    def __init__(self, boundsx, boundsy, penalty=100):
        self.boundsx = boundsx
        self.boundsy = boundsy
        self.penalty = 1


    def __call__(self, x):
        return (self.boundsx[0] <= x[0] <= self.boundsx[1] and self.boundsy[0] <= x[1] <= self.boundsy[1]) * self.penalty

class ComplexObstacle(Obstacle):

    def __init__(self, bounds):
        self.obs = []
        for boundsx, boundsy in bounds:
            self.obs.append(Obstacle(boundsx, boundsy))

    def __call__(self, x):
        return np.max([o(x) for o in self.obs])


OBSTACLE = [
        [[-100, 150], [5, 10]],
        [[-100, 150],[-10, -5]]]
        
OBSTACLE = ComplexObstacle(OBSTACLE)



def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)

def teacher_action(state, goal):
    disp = np.subtract(goal, state)
    disp[disp > MAX_FORCE] = MAX_FORCE
    disp[disp < -MAX_FORCE] = -MAX_FORCE
    return disp



class SimplePointBot(Env, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.A = np.eye(2)
        self.B = np.eye(2)
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(2) * MAX_FORCE, np.ones(2) * MAX_FORCE)
        self.observation_space = Box(-np.ones(2) * np.float('inf'), np.ones(2) * np.float('inf'))
        self._max_episode_steps = HORIZON
        self.obstacle = OBSTACLE

    def step(self, a):
        a = process_action(a)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a)
        cur_cost = self.step_cost(self.state, a)
        self.cost.append(cur_cost)
        self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.done = HORIZON <= self.time

        return self.state, cur_cost, self.done, {
                "constraint": self.obstacle(self.state),
                "reward": cur_cost,
                "state": old_state,
                "next_state": next_state,
                "action": a}

    def reset(self):
        self.state = START_STATE + np.random.randn(2)
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        return self.state

    def _next_state(self, s, a):
        if self.obstacle(s):
            return s
        return self.A.dot(s) + self.B.dot(a) + NOISE_SCALE * np.random.randn(len(s))

    def step_cost(self, s, a):
        if HARD_MODE:
            return int(np.linalg.norm(np.subtract(GOAL_STATE, s)) < GOAL_THRESH)
        return -np.linalg.norm(np.subtract(GOAL_STATE, s))

    def values(self):
        return np.cumsum(np.array(self.cost)[::-1])[::-1]

    def sample(self):
        """
        samples a random action from the action space.
        """
        return np.random.random(2) * 2 * MAX_FORCE - MAX_FORCE

    def plot_trajectory(self, states=None):
        if states == None:
            states = self.hist
        states = np.array(states)
        plt.scatter(states[:,0], states[:,2])
        plt.show()

    # Returns whether a state is stable or not
    def is_stable(self, s):
        return np.linalg.norm(np.subtract(GOAL_STATE, s)) <= GOAL_THRESH



if __name__ == '__main__':
    env = SimplePointBot()
    obs = env.reset()
    env.step([1,1])

    for i in range(HORIZON-1):
        env.step([0,0])
    # env.plot_trajectory()

