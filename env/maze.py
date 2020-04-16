import os
import pickle

import os.path as osp
import numpy as np
import tensorflow as tf
from gym import Env
from gym import utils
from gym.spaces import Box
from mujoco_py import load_model_from_path, MjSim

from .maze_const import *
import cv2

def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)


def get_random_transitions(num_transitions):
    env = MazeNavigation()
    transitions = []
    for i in range(num_transitions):
        if i %(num_transitions//100) == 0:
            state = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        constraint = info['constraint']
        transitions.append((state, action, constraint, next_state, done))
        state = next_state
    return transitions

class MazeNavigation(Env, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'simple_maze.xml')
        self.sim = MjSim(load_model_from_path(filename))
        self.horizon = HORIZON
        self._max_episode_steps = self.horizon
        self.transition_function = get_random_transitions
        self.steps = 0
        self.images = not GT_STATE
        self.action_space = Box(-MAX_FORCE*np.ones(2), MAX_FORCE*np.ones(2))
        self.transition_function = get_random_transitions
        obs = self._get_obs()
        # print("OBS", obs.shape)
        # print("OBS", np.max(obs), np.min(obs))
        # cv2.imwrite('maze.jpg', 255*obs)
        # assert(False)
        self.dense_reward = DENSE_REWARD

        if self.images:
            self.observation_space = obs.shape
        else:
            self.observation_space = Box(-0.3, 0.3, shape=obs.shape)

        self.gain = 1.05
        self.goal = np.zeros((2,))
        # self.goal[0] = np.random.uniform(0.15, 0.27)
        # self.goal[1] = np.random.uniform(-0.27, 0.27)
        self.goal[0] = 0.25
        self.goal[1] = 0
        

    def disable_images(self):
        self.images = False

    def step(self, action):
        action = process_action(action)
        self.sim.data.qvel[:] = 0
        self.sim.data.ctrl[:] = action
        cur_obs = self._get_obs()
        for _ in range(500):
          self.sim.step()
        obs = self._get_obs()
        self.sim.data.qvel[:] = 0
        self.steps +=1 
        constraint = int(self.sim.data.ncon > 3)
        self.done = self.steps >= self.horizon or constraint
        if not self.dense_reward:
            reward = - (self.get_distance_score() > GOAL_THRESH).astype(float)
        else:
            reward = -self.get_distance_score()

        info = {"constraint": constraint,
                "reward": reward,
                "state": cur_obs,
                "next_state": obs,
                "action": action}

        return obs, reward, self.done, info
      
    def _get_obs(self):
        #joint poisitions and velocities
        state = np.concatenate([self.sim.data.qpos[:].copy(), self.sim.data.qvel[:].copy()])
        
        if not self.images:
          return state[:2] # State is just (x, y) now

        #get images
        ims = self.sim.render(64, 64, camera_name= "cam0")
        return ims/255

    def reset(self, difficulty='h'):
        if difficulty is None:
          self.sim.data.qpos[0] = np.random.uniform(-0.27, 0.27)
        elif difficulty == 'e':
          self.sim.data.qpos[0] = np.random.uniform(0.15, 0.27)
        elif difficulty == 'm':
          self.sim.data.qpos[0] = np.random.uniform(-0.15, 0.15)
        elif difficulty == 'h':
          self.sim.data.qpos[0] = np.random.uniform(-0.27, -0.15)
        self.sim.data.qpos[1] = np.random.uniform(-0.27, 0.27)
        self.steps = 0

        # self.sim.data.qpos[0] = 0.25
        # self.sim.data.qpos[1] = 0
        # print(self._get_obs())
        # print("GOT HERE")
        # assert(False)

        # Randomize wal positions
        w1 = -0.15#np.random.uniform(-0.2, 0.2)
        w2 = 0.15 #np.random.uniform(-0.2, 0.2)
    #     print(self.sim.model.geom_pos[:])
    #     print(self.sim.model.geom_pos[:].shape)
        self.sim.model.geom_pos[5, 1] = 0.25 + w1
        self.sim.model.geom_pos[7, 1] = -0.25 + w1
        self.sim.model.geom_pos[6, 1] = 0.25 + w2
        self.sim.model.geom_pos[8, 1] = -0.25 + w2
        # print("RESET!", self._get_obs())
        return self._get_obs()

    def get_distance_score(self):
        """
        :return:  mean of the distances between all objects and goals
            """
        d = np.sqrt(np.mean((self.goal - self.sim.data.qpos[:])**2))
        return d

    # TODO: implement noise_std, demo_quality, right now these are ignored      
    def expert_action(self, noise_std=0, demo_quality='high'):
        st = self.sim.data.qpos[:]
        # print(st)
        if st[0] <= -0.151:
          delt = (np.array([-0.15, -0.17]) - st)
        elif st[0] <= 0.149:
          delt = (np.array([0.15, 0.17]) - st)
        # elif st[1] < 0.25:
        #   delt = (np.array([0.25, 0]) - st)
        else:
          delt = (np.array([self.goal[0], self.goal[1]]) - st)
        act = self.gain*delt

        return act

class MazeTeacher(object):

    def __init__(self, mode):
        self.env = MazeNavigation(mode=mode)
        self.demonstrations = []
        self.default_noise = 0.2

    # all get_rollout functions for all envs should have a noise parameter
    def get_rollout(self, noise_param_in=None, mode="eps_greedy"):
        if mode == "eps_greedy":
            if noise_param_in is None:
                noise_param = 0
            else: 
                noise_param = noise_param_in

        elif mode == "gaussian_noise":
            if noise_param_in is None:
                noise_param = 0
            else:
                noise_param = noise_param_in

        obs = self.env.reset(difficulty='h')
        O, A, cost_sum, costs = [obs], [], 0, []

        noise_idx = np.random.randint(int(2 * HORIZON / 4))
        for i in range(HORIZON):
            action = self.env.expert_action()

            if i < noise_idx:
                if mode == "eps_greedy":
                    assert(noise_param <= 1)
                    if np.random.random() < noise_param:
                        action = self.env.action_space.sample()
                    else:
                        if np.random.random() < self.default_noise:
                            action = self.env.action_space.sample()

                elif mode == "gaussian_noise":
                    action = (np.array(action) +  np.random.normal(0, noise_param + self.default_noise, self.env.action_space.shape[0])).tolist()
                else:
                    print("Invalid Mode!")
                    assert(False)

            A.append(action)
            obs, cost, done, info = self.env.step(action)
            O.append(obs)
            cost_sum += cost
            costs.append(cost)
            if done:
                break

        values = np.cumsum(costs[::-1])[::-1]

        print(cost_sum)
        print(-HORIZON)

        if int(cost_sum) == -HORIZON:
            print("FAILED")
            # return self.get_rollout(noise_param_in)

        # cv2.imwrite('maze.jpg', 255*obs)
        # assert(False)

        print("obs", O)

        return {
            "obs": np.array(O),
            "noise": noise_param,
            "actions": np.array(A),
            "reward_sum": -cost_sum,
            "rewards": -np.array(costs),
            "values": -np.array(values)
        }

    
