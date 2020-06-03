"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise. State
representation is (x, y). Action representation is (dx, dy).
"""

import os
import pickle

import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from gym import Env
from gym import utils
from gym.spaces import Box


"""
Constants associated with the PointBot env.
"""
START_STATE = [0, 0, 0]
TARGET_X = 50
ROAD_WIDTH = 4

MAX_FORCE = 0.25
MAX_VEL = 1.5
HORIZON = 100

NOISE_SCALE = 0.05
DIST_THRESH = 0.5


def collision(state):
    return not -ROAD_WIDTH <= state[1] <= ROAD_WIDTH


def process_action(a):
    if a[0] < -MAX_FORCE:
        a[0] = -MAX_FORCE
    elif a[0] > MAX_FORCE:
        a[0] = MAX_FORCE
    return a

def clip_vel(s):
    if s[2] < -MAX_VEL:
        s[2] = -MAX_VEL
    elif s[2] > MAX_VEL:
        s[2] = MAX_VEL
    return s


class DubinsCar(Env, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.done = self.time = self.state = None
        self.reward = []
        self.horizon = HORIZON
        self.action_space = Box(-np.array([MAX_FORCE, np.pi]), np.array([MAX_FORCE, np.pi]))
        self.observation_space = Box(-np.ones(3) * np.float('inf'), np.ones(3) * np.float('inf'))
        self._max_episode_steps = HORIZON
        self.target_x = TARGET_X
        self.transition_function = get_random_transitions

    def step(self, a):
        a = process_action(a)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a)
        cur_rew = self.step_reward(self.state, a)
        self.reward.append(cur_rew)
        self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        constraint = collision(next_state)
        self.done = HORIZON <= self.time or -cur_rew < DIST_THRESH

        return self.state, cur_rew, self.done, {
                "constraint": constraint,
                "reward": cur_rew,
                "state": old_state,
                "next_state": next_state,
                "action": a}

    def reset(self):
        self.state = np.array(START_STATE)
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        return self.state

    def _next_state(self, s, a, override=False):
        if collision(s):
            # print("obs", s, a)
            return s
        else:
            new_state = np.copy(s)
            new_state = new_state + np.array([s[2] * np.cos(a[1]), s[2] * np.sin(a[1]), a[0]])
            new_state = clip_vel(new_state)
        return new_state

    def step_reward(self, s, a):
        return -np.abs(s[0] - TARGET_X)

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
        plt.scatter(states[:,0], states[:,1])
        plt.savefig('car_traj.png')

    def teacher(self, sess=None):
        return SimplePointBotTeacher()

    def expert_action(self, s):
        a = [1, 0]
        # print(s)
        if s[1] > 3:
            a[1] = -np.pi/2
        elif s[1] < -3:
            a[1] = np.pi/2
        elif s[0] > TARGET_X:
            a[1] = -np.pi
        if np.abs(s[0] - TARGET_X) > 2:
            a[0] = 2 - s[2]
        else:
            a[0] = 0.5 * np.abs(s[0] - TARGET_X) - s[2]

        return np.array(a)

def get_random_transitions(num_transitions, task_demos=False, save_rollouts=False):
    env = DubinsCar()
    transitions = []
    rollouts = []
    done = False
    collisions = 0
    for i in range(int(0.7*num_transitions//10)):
        rollouts.append([])
        if np.random.random() < 0.5:
            state = np.array([np.random.uniform(0, TARGET_X), np.random.uniform(ROAD_WIDTH*7/8, ROAD_WIDTH), np.random.uniform(-MAX_VEL, MAX_VEL)])
        else:
            state = np.array([np.random.uniform(0, TARGET_X), np.random.uniform(-ROAD_WIDTH, -ROAD_WIDTH*7/8), np.random.uniform(-MAX_VEL, MAX_VEL)])
        for j in range(10):
            action = env.action_space.sample()
            next_state = env._next_state(state, action)
            constraint = collision(next_state)
            collisions += constraint
            reward = env.step_reward(state, action)
            transitions.append((state, action, constraint, next_state, done))
            rollouts[-1].append((state, action, constraint, next_state, done))
            state = next_state

    for i in range(int(0.3*num_transitions//10)):
        rollouts.append([])
        state = env.reset()
        for j in range(10):
            action = env.expert_action(state)
            next_state = env._next_state(state, action)
            constraint = collision(next_state)
            collisions += constraint
            reward = env.step_reward(state, action)
            transitions.append((state, action, constraint, next_state, done))
            rollouts[-1].append((state, action, constraint, next_state, done))
            state = next_state

    print("Num collisions: ", collisions)
    if save_rollouts:
        return rollouts
    else:
        return transitions


if __name__ == '__main__':
    env = DubinsCar()
    obs = env.reset()
    env.step([1,1])

    for i in range(5):
        env.step([1,.1])
    for i in range(HORIZON-1):
        s,r,d,i = env.step(env.expert_action(env.state))
        print("REWARD: ", r)
    env.plot_trajectory()
    assert(False)

    get_random_transitions(10000)

    # teacher = env.teacher()
    # teacher.generate_demonstrations(1000)
    # env.plot_trajectory()
