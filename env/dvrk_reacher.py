import os
import pickle
import time

import os.path as osp
import numpy as np
# import matplotlib.pyplot as plt
from gym import Env
from gym import utils
from gym.spaces import Box

from BA_exp.dvrkMotionBridgeP import dvrkMotionBridgeP
from BA_exp.dvrkKinematics import dvrkKinematics
from BA_exp.ZividCapture import ZividCapture
import BA_exp.utils as U

GOAL_THRESH = 1.
START_STATE = [0.07, 0.07, -0.13]
GOAL_STATE = [0.09, 0.07, -0.13]
MAX_FORCE = 0.005
HORIZON = 20

BOUNDSX = [0.03, 0.13]
BOUNDSY = [0.03, 0.13]
BOUNDSZ = [-0.17, -0.09]

def process_action(s, a):
    a_proc = np.clip(a, -MAX_FORCE, MAX_FORCE)
    a_proc[2] = 0
    if s[0] < BOUNDSX[0]:
        a_proc[0] = MAX_FORCE
    elif s[0] > BOUNDSX[1]:
        a_proc[0] = -MAX_FORCE
        print("HIT BOUND")

    if s[1] < BOUNDSY[0]:
        a_proc[1] = MAX_FORCE
    elif s[1] > BOUNDSY[1]:
        a_proc[1] = -MAX_FORCE
        print("HIT BOUND")

    if s[2] < BOUNDSZ[0]:
        a_proc[2] = MAX_FORCE
    elif s[2] > BOUNDSZ[1]:
        a_proc[2] = -MAX_FORCE
        print("HIT BOUND")
    return a_proc


class DVRK_Reacher(Env, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(3) * MAX_FORCE, np.ones(3) * MAX_FORCE)
        self.observation_space = Box(-np.ones(3) * np.float('inf'), np.ones(3) * np.float('inf'))
        self._max_episode_steps = HORIZON
        self.goal = GOAL_STATE
        self.dvrk = dvrkMotionBridgeP()
        self.zivid = ZividCapture(initialize=True)

    def step(self, a):
        a = process_action(self.state, a)
        old_state = self._state
        next_state = self._execute_action(a)
        cur_cost = self.step_cost(self.state, a)
        self.cost.append(cur_cost)
        self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.done = cur_cost > -0.01
        print(self.time, old_state, next_state, a, cur_cost)

        return self.state, cur_cost, self.done, {
                "constraint": 0,
                "reward": cur_cost,
                "state": old_state,
                "next_state": next_state,
                "action": a}

    def reset(self):
        # self.dvrk.set_joint(joint1=joint, jaw1=jaw)
        pos = [0.07, 0.07, -0.13]   # position in (m)
        rot = [0.0, 0.0, 0.0]   # Euler angles
        jaw = [0*np.pi/180.]    # jaw angle in (rad)
        quat = U.euler_to_quaternion(rot, unit='deg')   # convert Euler angles to quaternion
        self.dvrk.set_pose(pos1=pos, rot1=quat, jaw1=jaw)
        time.sleep(0.5)

        self.state = self._state
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        return self.state

    def _execute_action(self, a):
        target_pos = self._state + a
        quat = U.euler_to_quaternion([0, 0, 0], unit='deg')   # convert Euler angles to quaternion
        jaw = [0*np.pi/180.]    # jaw angle in (rad)
        self.dvrk.set_pose(pos1=target_pos, rot1=quat, jaw1=jaw)
        time.sleep(0.5)
        return self._state

    @property
    def _state(self):
        return np.array(self.dvrk.get_pose()[0])

    def step_cost(self, s, a):
        return -np.linalg.norm(np.subtract(GOAL_STATE, s)) * 1000

    def values(self):
        return np.cumsum(np.array(self.cost)[::-1])[::-1]

    def sample(self):
        """
        samples a random action from the action space.
        """
        return self.action_space.sample()
