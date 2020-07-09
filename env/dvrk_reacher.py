import os
import pickle
import time

import os.path as osp
import numpy as np
# import matplotlib.pyplot as plt
from gym import Env
from gym import utils
from gym.spaces import Box
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

from BA_exp.dvrkMotionBridgeP import dvrkMotionBridgeP
from BA_exp.dvrkKinematics import dvrkKinematics
from BA_exp.ZividCapture import ZividCapture
import BA_exp.utils as U

from obstacle import Obstacle3D
import pickle

GOAL_THRESH = 1.
START_STATE = [0.06, 0.07, -0.13]
GOAL_STATE = [0.09, 0.07, -0.13]
SUBGOAL_STATE_1 = [0.08, 0.03, -0.13]
SUBGOAL_STATE_2 = [0.08, -0.03, -0.13]
MAX_FORCE = 0.005 * 100
HORIZON = 20

BOUNDSX = [0.03, 0.13]
BOUNDSY = [0.03, 0.13]
BOUNDSZ = [-0.17, -0.09]

OBS_X = (7.7, 8.2)
OBS_Y = (6.9, 7.1)
OBS_Z = (-17, -9)

CAUTION_X = (7.2, 8.7)
CAUTION_Y = (6.4, 7.6)
CAUTION_Z = (-17, -9)

OBSTACLE = Obstacle3D(OBS_X, OBS_Y, OBS_Z)
CAUTION_ZONE = Obstacle3D(CAUTION_X, CAUTION_Y, CAUTION_Z)
GT_RECOVERY = True

def process_obs(obs):
    im = np.transpose(obs, (2, 0, 1))
    return im

def process_action(s, a):
    a_proc = np.clip(a, -MAX_FORCE, MAX_FORCE)
    # a_proc[2] = 0
    if s[0] < BOUNDSX[0] * 100:
        a_proc[0] = MAX_FORCE
    elif s[0] > BOUNDSX[1] * 100:
        a_proc[0] = -MAX_FORCE
        print("HIT BOUND")

    if s[1] < BOUNDSY[0] * 100:
        a_proc[1] = MAX_FORCE
    elif s[1] > BOUNDSY[1] * 100:
        a_proc[1] = -MAX_FORCE
        print("HIT BOUND")

    if s[2] < BOUNDSZ[0] * 100:
        a_proc[2] = MAX_FORCE
    elif s[2] > BOUNDSZ[1] * 100:
        a_proc[2] = -MAX_FORCE
        print("HIT BOUND")
    return a_proc


class DVRK_Reacher(Env, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(3) * MAX_FORCE, np.ones(3) * MAX_FORCE)
        self._max_episode_steps = HORIZON
        self.goal = GOAL_STATE
        self.dvrk = dvrkMotionBridgeP()
        self.zivid = ZividCapture(initialize=True)
        self.obstacle = OBSTACLE
        self.caution_zone = CAUTION_ZONE
        self.pointbot_dynamics = False
        self.ring = False

        self.image_obs = False
        if self.image_obs:
            self.observation_space = (30, 64, 3)
        else:
            self.observation_space = Box(-np.ones(3) * np.float('inf'), np.ones(3) * np.float('inf'))

    def step(self, a):
        a = process_action(self._low_dim_state, a)

        # Implement recovery in env
        if GT_RECOVERY and self.caution_zone(self._low_dim_state):
            print("CALLED RECOVERY")
            a = process_action(self._low_dim_state, self.safe_action(self._low_dim_state))

        old_state = np.copy(self.state)
        if self.obstacle(self._low_dim_state):
            next_state = np.copy(self.state)
        else:
            next_state = self._execute_action(a)
        cur_cost = self.step_cost(self._low_dim_state, a) # QUESTION: is _low_dim_state the next state? Because thats not ideal right then this is C(s', a) instead of C(s, a)
        self.cost.append(cur_cost)
        self.time += 1
        self.hist.append(self.state)
        constraint = self.obstacle(self._low_dim_state)
        constraint = 0
        print(self.time, self._low_dim_state, a, cur_cost)
        print("CONSTRAINT: ", constraint)
        self.done = cur_cost > -0.01 * 100 or constraint

        return self.state, cur_cost, self.done, {
                "constraint": constraint,
                "reward": cur_cost,
                "state": old_state,
                "next_state": next_state,
                "action": a}

    def safe_action(self, state):
        a = np.zeros(3)
        if state[0] < OBS_X[0]:
            a[0] = -MAX_FORCE
        elif state[0] > OBS_X[1]:
            a[0] = MAX_FORCE

        if state[1] < OBS_Y[0]:
            a[1] = -MAX_FORCE
        elif state[1] > OBS_Y[1]:
            a[1] = MAX_FORCE

        if state[2] < OBS_Z[0]:
            a[2] = -MAX_FORCE
        elif state[2] > OBS_Z[1]:
            a[2] = MAX_FORCE

        return a

    def reset(self):
        if self.pointbot_dynamics:
            self.state = np.array(START_STATE) * 100
            if self.ring:
                theta = np.random.uniform(0, 2*np.pi, 1)
                self.state = (np.array(GOAL_STATE) + 0.03 * np.array([np.cos(theta[0]), np.sin(theta[0]), 0])) * 100 
        else:
            pos = START_STATE   # position in (m)
            if self.ring:
                theta = np.random.uniform(0, 2*np.pi, 1)
                pos = GOAL_STATE + 0.03 * np.array([np.cos(theta[0]), np.sin(theta[0]), 0])   # position in (m)
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
        if self.pointbot_dynamics:
            self.state = self.state + a
            return self.state
        target_pos = (self._low_dim_state + a)/100
        quat = U.euler_to_quaternion([0, 0, 0], unit='deg')   # convert Euler angles to quaternion
        jaw = [0*np.pi/180.]    # jaw angle in (rad)
        self.dvrk.set_pose(pos1=target_pos, rot1=quat, jaw1=jaw)
        time.sleep(0.5)
        self.state = self._state
        return self._state

    @property
    def _state(self):

        if self.image_obs:
            img_color, img_depth, img_point = self.zivid.capture_3Dimage(color='RGB')    # 7~10 fps
            img_color = resize(img_color, (img_color.shape[0] // 30, img_color.shape[1] // 30), 
                anti_aliasing=True)
            return img_color
            # zivid.display_rgb(img_color)
            # zivid.display_depthmap(img_point)
            # zivid.display_pointcloud(img_point, img_color)
        return np.array(self.dvrk.get_pose()[0]) * 100

    @property
    def _low_dim_state(self):
        if self.pointbot_dynamics:
            return np.array(self.state)
        return np.array(self.dvrk.get_pose()[0]) * 100
    

    def step_cost(self, s, a):
        return -np.linalg.norm(np.subtract(np.array(GOAL_STATE) * 100, s))

    def values(self):
        return np.cumsum(np.array(self.cost)[::-1])[::-1]

    def sample(self):
        """
        samples a random action from the action space.
        """
        return self.action_space.sample()

if __name__ == "__main__":
    # TODO: make demos vary in z as well...also need to add a Qsafe visualization
    # Collect demos
    env = DVRK_Reacher()
    demo_transitions = []
    demo_transitions_lowdim = []
    num_demos = 500

    for ep in range(num_demos):
        if ep % 10 == 0:
            print("EPISODE: ", ep)
        state = env.reset()
        lowdim_state = env._low_dim_state
        print("RESET!")
        done = False
        episode_steps = 0
        SUBGOAL_STATE = SUBGOAL_STATE_1 if np.random.random() < 0.5 else SUBGOAL_STATE_2
        # Randomize z:
        SUBGOAL_STATE[2] = np.random.uniform(100*BOUNDSZ[0], 100*BOUNDSZ[1])
        while not done:
            # Sample a point inside the obstacle and try to hit it
            obs_point = np.array([np.random.uniform(*OBS_X), np.random.uniform(*OBS_Y), np.random.uniform(*OBS_Z)])
            # Aim at it with probability 0.8, otherwise do random things
            # a = MAX_FORCE*(obs_point - env._low_dim_state)/(np.linalg.norm(obs_point - env._low_dim_state))
            if episode_steps < 4:
                a = MAX_FORCE*(np.array(SUBGOAL_STATE)*100 - env._low_dim_state)/(np.linalg.norm(np.array(SUBGOAL_STATE)*100 - env._low_dim_state))
            else:
                a = MAX_FORCE*(np.array(GOAL_STATE)*100 - env._low_dim_state)/(np.linalg.norm(np.array(GOAL_STATE)*100 - env._low_dim_state))
            a += MAX_FORCE*np.random.randn(len(a))
            action = process_action(env._low_dim_state, a)
            next_state, reward, done, info = env.step(action)
            next_lowdim_state = env._low_dim_state
            constraint = info['constraint']
            mask = float(not done)
            demo_transitions.append((process_obs(state), action, constraint, process_obs(next_state), mask))
            demo_transitions_lowdim.append((lowdim_state, action, constraint, next_lowdim_state, mask))
            state = next_state
            lowdim_state = next_lowdim_state
            episode_steps += 1

            if episode_steps == env._max_episode_steps:
                done = True

    pickle.dump({"images": demo_transitions, "lowdim": demo_transitions_lowdim}, 
                open(os.path.join("demos/dvrk_reach", "constraint_demos.pkl"), "wb"))


