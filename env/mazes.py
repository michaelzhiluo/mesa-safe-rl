import os
import pickle
import matplotlib.pyplot as plt

import os.path as osp
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
from mujoco_py import load_model_from_path, MjSim

from .maze_const import *
from .maze import MazeNavigation, get_random_transitions
import cv2


class Maze1Navigation(MazeNavigation):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'simple_maze_1.xml')
        self.sim = MjSim(load_model_from_path(filename))
        self.horizon = HORIZON
        self._max_episode_steps = self.horizon
        self.transition_function = get_random_transitions
        self.steps = 0
        self.images = not GT_STATE
        self.action_space = Box(-MAX_FORCE * np.ones(2),
                                MAX_FORCE * np.ones(2))
        self.transition_function = get_random_transitions
        obs = self._get_obs()
        if False:
            self.reset()
            ob = self._get_obs(images=True)
            cv2.imwrite('runs/maze.jpg', 255*ob)
            exit()
        self.dense_reward = DENSE_REWARD

        if self.images:
            self.observation_space = obs.shape
        else:
            self.observation_space = Box(-0.3, 0.3, shape=obs.shape)

        self.gain = 1.05
        self.goal = np.zeros((2, ))
        self.goal[0] = 0.25
        self.goal[1] = 0

    def reset(self, difficulty='h', check_constraint=True, demos=False,
              pos=()):
        if len(pos):
            self.sim.data.qpos[0] = pos[0]
            self.sim.data.qpos[1] = pos[1]
        else:
            if difficulty is None:
                self.sim.data.qpos[0] = np.random.uniform(-0.27, 0.27)
            elif difficulty == 'e':
                self.sim.data.qpos[0] = np.random.uniform(0.14, 0.22)
            elif difficulty == 'm':
                self.sim.data.qpos[0] = np.random.uniform(-0.04, 0.04)
            elif difficulty == 'h':
                self.sim.data.qpos[0] = np.random.uniform(-0.22, -0.13)
            self.sim.data.qpos[1] = np.random.uniform(-0.22, 0.22)

        self.steps = 0

        self.sim.forward()
        # print("RESET!", self._get_obs())
        constraint = int(self.sim.data.ncon > 3)
        if constraint and check_constraint:
            if not len(pos):
                self.reset(difficulty)
        return self._get_obs()


class Maze2Navigation(Maze1Navigation):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'simple_maze_2.xml')
        self.sim = MjSim(load_model_from_path(filename))
        self.horizon = HORIZON
        self._max_episode_steps = self.horizon
        self.transition_function = get_random_transitions
        self.steps = 0
        self.images = not GT_STATE
        self.action_space = Box(-MAX_FORCE * np.ones(2),
                                MAX_FORCE * np.ones(2))
        self.transition_function = get_random_transitions
        obs = self._get_obs()
        if False:
            self.reset()
            ob = self._get_obs(images=True)
            cv2.imwrite('runs/maze.jpg', 255*ob)
            exit()
        self.dense_reward = DENSE_REWARD

        if self.images:
            self.observation_space = obs.shape
        else:
            self.observation_space = Box(-0.3, 0.3, shape=obs.shape)

        self.gain = 1.05
        self.goal = np.zeros((2, ))
        self.goal[0] = 0.25
        self.goal[1] = 0


class Maze3Navigation(Maze1Navigation):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'simple_maze_3.xml')
        self.sim = MjSim(load_model_from_path(filename))
        self.horizon = HORIZON
        self._max_episode_steps = self.horizon
        self.transition_function = get_random_transitions
        self.steps = 0
        self.images = not GT_STATE
        self.action_space = Box(-MAX_FORCE * np.ones(2),
                                MAX_FORCE * np.ones(2))
        self.transition_function = get_random_transitions
        obs = self._get_obs()
        if False:
            self.reset()
            ob = self._get_obs(images=True)
            cv2.imwrite('runs/maze.jpg', 255*ob)
            exit()
        self.dense_reward = DENSE_REWARD

        if self.images:
            self.observation_space = obs.shape
        else:
            self.observation_space = Box(-0.3, 0.3, shape=obs.shape)

        self.gain = 1.05
        self.goal = np.zeros((2, ))
        self.goal[0] = 0.25
        self.goal[1] = 0

class Maze4Navigation(Maze1Navigation):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'simple_maze_4.xml')
        self.sim = MjSim(load_model_from_path(filename))
        self.horizon = HORIZON
        self._max_episode_steps = self.horizon
        self.transition_function = get_random_transitions
        self.steps = 0
        self.images = not GT_STATE
        self.action_space = Box(-MAX_FORCE * np.ones(2),
                                MAX_FORCE * np.ones(2))
        self.transition_function = get_random_transitions
        obs = self._get_obs()
        if False:
            self.reset()
            ob = self._get_obs(images=True)
            cv2.imwrite('runs/maze.jpg', 255*ob)
            exit()
        self.dense_reward = DENSE_REWARD

        if self.images:
            self.observation_space = obs.shape
        else:
            self.observation_space = Box(-0.3, 0.3, shape=obs.shape)

        self.gain = 1.05
        self.goal = np.zeros((2, ))
        self.goal[0] = 0.25
        self.goal[1] = 0

    def reset(self, difficulty='h', check_constraint=True, demos=False,
              pos=()):
        if len(pos):
            self.sim.data.qpos[0] = pos[0]
            self.sim.data.qpos[1] = pos[1]
        else:
            if difficulty is None:
                self.sim.data.qpos[0] = np.random.uniform(-0.27, 0.27)
            elif difficulty == 'e':
                self.sim.data.qpos[0] = np.random.uniform(0.14, 0.22)
            elif difficulty == 'm':
                self.sim.data.qpos[0] = np.random.uniform(-0.04, 0.04)
            elif difficulty == 'h':
                self.sim.data.qpos[0] = np.random.uniform(-0.22, -0.13)
            self.sim.data.qpos[1] = np.random.uniform(-0.22, 0.22)

        self.steps = 0
        # Randomize wal positions
        w1 = -0.08  #np.random.uniform(-0.2, 0.2)
        w2 = 0.08  #np.random.uniform(-0.2, 0.2)
        #     print(self.sim.model.geom_pos[:])
        #     print(self.sim.model.geom_pos[:].shape)
        self.sim.model.geom_pos[5, 1] = 0.4 + w2
        self.sim.model.geom_pos[7, 1] = -0.25 + w2
        self.sim.model.geom_pos[6, 1] = 0.5 + w1
        self.sim.model.geom_pos[8, 1] = -0.25 + w1

        self.sim.model.geom_pos[9, 1] = 0.45
        self.sim.model.geom_pos[10, 1] = -0.25

        self.sim.forward()
        # print("RESET!", self._get_obs())
        constraint = int(self.sim.data.ncon > 3)
        if constraint and check_constraint:
            if not len(pos):
                self.reset(difficulty)
        return self._get_obs()


class Maze5Navigation(Maze1Navigation):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'simple_maze_5.xml')
        self.sim = MjSim(load_model_from_path(filename))
        self.horizon = HORIZON
        self._max_episode_steps = self.horizon
        self.transition_function = get_random_transitions
        self.steps = 0
        self.images = not GT_STATE
        self.action_space = Box(-MAX_FORCE * np.ones(2),
                                MAX_FORCE * np.ones(2))
        self.transition_function = get_random_transitions
        obs = self._get_obs()
        if False:
            self.reset()
            ob = self._get_obs(images=True)
            cv2.imwrite('runs/maze.jpg', 255*ob)
            exit()
        self.dense_reward = DENSE_REWARD

        if self.images:
            self.observation_space = obs.shape
        else:
            self.observation_space = Box(-0.3, 0.3, shape=obs.shape)

        self.gain = 1.05
        self.goal = np.zeros((2, ))
        self.goal[0] = 0.25
        self.goal[1] = 0

    def reset(self, difficulty='h', check_constraint=True, demos=False,
              pos=()):
        if len(pos):
            self.sim.data.qpos[0] = pos[0]
            self.sim.data.qpos[1] = pos[1]
        else:
            if difficulty is None:
                self.sim.data.qpos[0] = np.random.uniform(-0.27, 0.27)
            elif difficulty == 'e':
                self.sim.data.qpos[0] = np.random.uniform(0.14, 0.22)
            elif difficulty == 'm':
                self.sim.data.qpos[0] = np.random.uniform(-0.04, 0.04)
            elif difficulty == 'h':
                self.sim.data.qpos[0] = np.random.uniform(-0.22, -0.13)
            self.sim.data.qpos[1] = np.random.uniform(-0.22, 0.22)

        self.steps = 0
        # Randomize wal positions
        w1 = -0.08  #np.random.uniform(-0.2, 0.2)
        w2 = 0.08  #np.random.uniform(-0.2, 0.2)
        #     print(self.sim.model.geom_pos[:])
        #     print(self.sim.model.geom_pos[:].shape)
        self.sim.model.geom_pos[5, 1] = 0.4
        self.sim.model.geom_pos[6, 1] = 0.4
        self.sim.model.geom_pos[7, 1] = -0.25
        self.sim.model.geom_pos[8, 1] = -0.25

        self.sim.model.geom_pos[9, 1] = 0.45
        self.sim.model.geom_pos[10, 1] = -0.20

        self.sim.forward()
        # print("RESET!", self._get_obs())
        constraint = int(self.sim.data.ncon > 3)
        if constraint and check_constraint:
            if not len(pos):
                self.reset(difficulty)
        return self._get_obs()

class Maze6Navigation(Maze1Navigation):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'simple_maze_6.xml')
        self.sim = MjSim(load_model_from_path(filename))
        self.horizon = HORIZON
        self._max_episode_steps = self.horizon
        self.transition_function = get_random_transitions
        self.steps = 0
        self.images = not GT_STATE
        self.action_space = Box(-MAX_FORCE * np.ones(2),
                                MAX_FORCE * np.ones(2))
        self.transition_function = get_random_transitions
        obs = self._get_obs()
        if False:
            self.reset()
            ob = self._get_obs(images=True)
            cv2.imwrite('runs/maze.jpg', 255*ob)
            exit()
        self.dense_reward = DENSE_REWARD

        if self.images:
            self.observation_space = obs.shape
        else:
            self.observation_space = Box(-0.3, 0.3, shape=obs.shape)

        self.gain = 1.05
        self.goal = np.zeros((2, ))
        self.goal[0] = 0.25
        self.goal[1] = 0