"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise and air resistance
proportional to its velocity. State representation is (x, vx, y, vy). Action
representation is (fx, fy), and mass is assumed to be 1.
"""

import os
import pickle

import os.path as osp
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
from mujoco_py import load_model_from_path, MjSim

from maze_const import *
import cv2

def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)

class MazeNavigation(Env, utils.EzPickle):

    def __init__(self, mode=1):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.mode = mode
        
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'simple_maze.xml')
        self.sim = MjSim(load_model_from_path(filename))
        self.horizon = HORIZON
        
        self.steps = 0
        self.images = IMAGES
        self.dense_reward = not HARD_MODE
        self.action_space = Box(-MAX_FORCE*np.ones(2), MAX_FORCE*np.ones(2))
        obs = self._get_obs()
        self._max_episode_steps = HORIZON

        if self.images:
            self.observation_space = obs.T.shape
        else:
            self.observation_space = Box(-0.3, 0.3, shape=obs.shape, dtype='float32')

        self.reset()

    def step(self, action):
        action = process_action(action)
        self.sim.data.qvel[:] = 0
        self.sim.data.ctrl[:] = action
        for _ in range(500):
          self.sim.step()
        obs = self._get_obs()
        self.sim.data.qvel[:] = 0
        self.steps +=1 
        self.done = self.steps >= self.horizon
        if not self.dense_reward:
            reward = - (self.get_distance_score() > GOAL_THRESH).astype(float)
        else:
            reward = -self.get_distance_score()
        if SHAPED_REWARD:
            reward = -(self.get_distance_score() > GOAL_THRESH).astype(float)
            if reward:
                reward = shaped_reward_function(self.sim.data.qpos[:])
            else:
                print("goal reached!")
        return obs, reward, self.done, {}
      
    def _get_obs(self):
        #joint poisitions and velocities
        state = np.concatenate([self.sim.data.qpos[:].copy(), self.sim.data.qvel[:].copy()])
        
        if not self.images:
          return state[:2] # State is just (x, y) now

        #get images
        ims = self.sim.render(64, 64, camera_name= "cam0").T
        return ims/255

    def reset(self, difficulty='m'):
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

        self.goal = np.zeros((2,))
        self.goal[0] = 0.25 #np.random.uniform(0.15, 0.27)
        self.goal[1] = -0.25#-0.25 #np.random.uniform(-0.27, 0.27)

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

def shaped_reward_function(pos):
    if pos[0] < -0.15:
        return -4
    elif pos[0] < 0.15 and pos[1] < 0:
        return -3
    elif pos[0] < 0.15:
        return -2
    else:
        return -1

      
if __name__ == '__main__':
    import moviepy.editor as mpy

    def npy_to_gif(im_list, filename, fps=4):
        clip = mpy.ImageSequenceClip(im_list, fps=fps)
        clip.write_gif(filename + '.gif')


    env = MazeNavigation()
    env.reset()
    ims = []
    for i in range(10):
        action = env.action_space.sample()
        s, r, _, _ = env.step(action)
        print(s, r)
        im = env.sim.render(64, 64, camera_name= "cam0")
        ims.append(im)
    npy_to_gif(ims, "out")
