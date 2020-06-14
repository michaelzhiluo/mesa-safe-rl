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
import cv2

def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)

def process_obs(obs):
    im = np.transpose(obs, (2, 0, 1))
    return im

def get_random_transitions(num_transitions, images=False, save_rollouts=False, task_demos=False):
    env = MazeNavigation()
    transitions = []
    num_constraints = 0
    total = 0
    rollouts = []

    for i in range(1*num_transitions//2):
        if i % 20 == 0:
            sample = np.random.uniform(0, 1, 1)[0]
            if sample < 0.2: # maybe make 0.2 to 0.3
                mode = 'e'
            elif sample < 0.4:
                mode = 'm'
            else:
                mode = 'h'
            state = env.reset(mode, check_constraint=False)
            rollouts.append([])

            if images:
                im_state = env.sim.render(64, 64, camera_name= "cam0")
                im_state = process_obs(im_state)
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if images:
            im_next_state = env.sim.render(64, 64, camera_name= "cam0")
            im_next_state = process_obs(im_next_state)
        constraint = info['constraint']

        rollouts[-1].append((state, action, constraint, next_state, done))
        transitions.append((state, action, constraint, next_state, done))

        total += 1
        num_constraints += int(constraint)
        state = next_state
        if images:
            im_state = im_next_state

    for i in range(1*num_transitions//2):
        if i % 20 == 0:
            sample = np.random.uniform(0, 1, 1)[0]
            if sample < 0.2: # maybe make 0.2 to 0.3
                mode = 'e'
            elif sample < 0.6:
                mode = 'm'
            else:
                mode = 'h'
            state = env.reset(mode, check_constraint=False)
            rollouts.append([])

            if images:
                im_state = env.sim.render(64, 64, camera_name= "cam0")
                im_state = process_obs(im_state)
        action = env.expert_action()
        next_state, reward, done, info = env.step(action)
        if images:
            im_next_state = env.sim.render(64, 64, camera_name= "cam0")
            im_next_state = process_obs(im_next_state)
        constraint = info['constraint']

        rollouts[-1].append((state, action, constraint, next_state, done))
        transitions.append((state, action, constraint, next_state, done))

        total += 1
        num_constraints += int(constraint)
        state = next_state
        if images:
            im_state = im_next_state

    print("data dist", total, num_constraints)
    if save_rollouts:
        return rollouts
    else:
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

    def step(self, action):
        action = process_action(action)
        self.sim.data.qvel[:] = 0
        self.sim.data.ctrl[:] = action
        cur_obs = self._get_obs()
        constraint = int(self.sim.data.ncon > 3)
        if not constraint:
            for _ in range(500):
              self.sim.step()
        obs = self._get_obs()
        self.sim.data.qvel[:] = 0
        self.steps +=1 
        constraint = int(self.sim.data.ncon > 3)
        self.done = self.steps >= self.horizon or constraint or (self.get_distance_score() < GOAL_THRESH)
        if not self.dense_reward:
            reward = - (self.get_distance_score() > GOAL_THRESH).astype(float)
        else:
            reward = -self.get_distance_score()
            # if self.get_distance_score() < GOAL_THRESH:
            #     reward += 10

        info = {"constraint": constraint,
                "reward": reward,
                "state": cur_obs,
                "next_state": obs,
                "action": action}

        return obs, reward, self.done, info
      
    def _get_obs(self, images=False):
        if images:
            return self.sim.render(64, 64, camera_name= "cam0")
        #joint poisitions and velocities
        state = np.concatenate([self.sim.data.qpos[:].copy(), self.sim.data.qvel[:].copy()])
        
        if not self.images and not images:
          return state[:2] # State is just (x, y) now

        #get images
        ims = self.sim.render(64, 64, camera_name= "cam0")
        return ims/255

    def reset(self, difficulty='h', check_constraint=True):
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

        # self.sim.data.qpos[0] = 0.25
        # self.sim.data.qpos[1] = 0
        # print(self._get_obs())
        # print("GOT HERE")
        # assert(False)

        # Randomize wal positions
        w1 = -0.08#np.random.uniform(-0.2, 0.2)
        w2 = 0.08 #np.random.uniform(-0.2, 0.2)
    #     print(self.sim.model.geom_pos[:])
    #     print(self.sim.model.geom_pos[:].shape)
        self.sim.model.geom_pos[5, 1] = 0.4 + w1
        self.sim.model.geom_pos[7, 1] = -0.25 + w1
        self.sim.model.geom_pos[6, 1] = 0.4 + w2
        self.sim.model.geom_pos[8, 1] = -0.25 + w2
        self.sim.forward()
        # print("RESET!", self._get_obs())
        constraint = int(self.sim.data.ncon > 3)
        if constraint and check_constraint:
            self.reset(difficulty)
        #     # self.render()
        #     im = self.sim.render(64, 64, camera_name= "cam0")
        #     print('aaa',self.sim.data.ncon, self.sim.data.qpos, im.sum())
        #     plt.imshow(im)
        #     plt.show()
        #     plt.pause(0.1)
            # assert 0
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
          delt = (np.array([-0.15, -0.125]) - st)
        elif st[0] <= 0.149:
          delt = (np.array([0.15, 0.125]) - st)
        # elif st[1] < 0.25:
        #   delt = (np.array([0.25, 0]) - st)
        else:
          delt = (np.array([self.goal[0], self.goal[1]]) - st)
        act = self.gain*delt

        return act

class MazeTeacher(object):

    def __init__(self):
        self.env = MazeNavigation()
        self.demonstrations = []
        self.default_noise = 0

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
        constraints_violated = 0

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
            print("CON", info['constraint'])
            print("STATE", obs)
            print("DONE", done)
            constraints_violated += info['constraint']
            O.append(obs)
            cost_sum += cost
            costs.append(cost)
            if done:
                break

        values = np.cumsum(costs[::-1])[::-1]

        print(cost_sum)
        print(len(O))
        print("CONSTRAINTS: ", constraints_violated)

        if int(cost_sum) == -HORIZON:
            print("FAILED")
            # return self.get_rollout(noise_param_in)

        cv2.imwrite('maze.jpg', 255*obs)
        assert(False)

        print("obs", O)

        return {
            "obs": np.array(O),
            "noise": noise_param,
            "actions": np.array(A),
            "reward_sum": -cost_sum,
            "rewards": -np.array(costs),
            "values": -np.array(values)
        }

if __name__ == "__main__": 
    teacher = MazeTeacher()
    reward_sum_completed = []
    constraint_sat = 0
    for i in range(1000):
        rollout_stats = teacher.get_rollout()
        print("Iter: ", i)
        print(rollout_stats['reward_sum'])
        print(len(rollout_stats['rewards']))
        ep_len = len(rollout_stats['rewards'])
        diff = HORIZON - ep_len
        if ep_len == HORIZON:
            constraint_sat += 1
        reward_sum_completed.append(rollout_stats['reward_sum'] + diff * rollout_stats['rewards'][-1])

    print("completed reward sum", np.mean(reward_sum_completed), np.std(reward_sum_completed), constraint_sat)


    
