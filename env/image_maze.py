import os
import pickle
import matplotlib.pyplot as plt

import os.path as osp
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
from mujoco_py import load_model_from_path, MjSim
import moviepy.editor as mpy
from .maze_const_images import *
import cv2


def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)


def process_obs(obs):
    im = np.transpose(obs, (2, 0, 1))
    return im


def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


def get_random_transitions(num_transitions,
                           images=False,
                           save_rollouts=False,
                           task_demos=False):
    env = MazeImageNavigation()
    transitions = []
    num_constraints = 0
    total = 0
    rollouts = []
    obs_seqs = []
    ac_seqs = []
    constraint_seqs = []

    for i in range(int(0.7 * num_transitions)):
        if i % 500 == 0:
            print("DEMO: ", i)
        if i % 20 == 0:
            sample = np.random.uniform(0, 1, 1)[0]
            if sample < 0.4:  # maybe make 0.2 to 0.3
                mode = 'e'
            else:
                mode = 'm'
            state = env.reset(mode, check_constraint=False)
            if not GT_STATE:
                state = process_obs(state)
            rollouts.append([])
            obs_seqs.append([state])
            ac_seqs.append([])
            constraint_seqs.append([])

        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        if not GT_STATE:
            next_state = process_obs(next_state)

        constraint = info['constraint']

        rollouts[-1].append((state, action, constraint, next_state, not done))
        obs_seqs[-1].append(next_state)
        constraint_seqs[-1].append(constraint)
        ac_seqs[-1].append(action)
        transitions.append((state, action, constraint, next_state, not done))

        total += 1
        num_constraints += int(constraint)
        state = next_state
        if images:
            im_state = im_next_state

    for i in range(int(0.3 * num_transitions)):
        if i % 500 == 0:
            print("DEMO: ", i)
        if i % 20 == 0:
            sample = np.random.uniform(0, 1, 1)[0]
            if sample < 0.4:  # maybe make 0.2 to 0.3
                mode = 'e'
            else:
                mode = 'm'
            state = env.reset(mode, check_constraint=False)
            if not GT_STATE:
                state = process_obs(state)
            rollouts.append([])
            obs_seqs.append([state])
            ac_seqs.append([])
            constraint_seqs.append([])

        action = env.expert_action()
        next_state, reward, done, info = env.step(action)

        if not GT_STATE:
            next_state = process_obs(next_state)

        constraint = info['constraint']

        rollouts[-1].append((state, action, constraint, next_state, not done))
        obs_seqs[-1].append(next_state)
        constraint_seqs[-1].append(constraint)
        ac_seqs[-1].append(action)
        transitions.append((state, action, constraint, next_state, not done))

        total += 1
        num_constraints += int(constraint)
        state = next_state
        if images:
            im_state = im_next_state

    print("data dist", total, num_constraints)
    rollouts = np.array(rollouts)

    for i in range(len(ac_seqs)):
        ac_seqs[i] = np.array(ac_seqs[i])
    for i in range(len(obs_seqs)):
        obs_seqs[i] = np.array(obs_seqs[i])
    for i in range(len(constraint_seqs)):
        constraint_seqs[i] = np.array(constraint_seqs[i])
    ac_seqs = np.array(ac_seqs)
    obs_seqs = np.array(obs_seqs)
    constraint_seqs = np.array(constraint_seqs)
    print("ACS SHAPE", ac_seqs.shape)
    print("OBS SHAPE", obs_seqs.shape)
    print("CONSTRAINT SHAPE", constraint_seqs.shape)

    if save_rollouts:
        return rollouts
    else:
        return transitions, obs_seqs, ac_seqs, constraint_seqs


class MazeImageNavigation(Env, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'simple_maze_images.xml')
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
        # print("OBS", obs.shape)
        # print("OBS", np.max(obs), np.min(obs))
        # cv2.imwrite('maze.jpg', 255*obs)
        # assert(False)
        self.dense_reward = DENSE_REWARD

        if self.images:
            self.observation_space = obs.shape
        else:
            self.observation_space = Box(-0.3, 0.3, shape=obs.shape)

        self.gain = 5
        self.goal = np.zeros((2, ))
        # self.goal[0] = np.random.uniform(0.15, 0.27)
        # self.goal[1] = np.random.uniform(-0.27, 0.27)
        self.goal[0] = 0.25
        self.goal[1] = 0.25

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
        self.steps += 1
        constraint = int(self.sim.data.ncon > 3)
        self.done = self.steps >= self.horizon or (self.get_distance_score() <
                                                   GOAL_THRESH) or constraint
        if not self.dense_reward:
            reward = -(self.get_distance_score() > GOAL_THRESH).astype(float)
        else:
            reward = -self.get_distance_score()
            # if self.get_distance_score() < GOAL_THRESH:
            #     reward += 10

        info = {
            "constraint": constraint,
            "reward": reward,
            "state": cur_obs,
            "next_state": obs,
            "action": action
        }

        return obs, reward, self.done, info

    def _get_obs(self, images=False):
        if images:
            return cv2.resize(
                self.sim.render(64, 64, camera_name="cam0")[20:64, 20:64],
                (64, 64),
                interpolation=cv2.INTER_AREA)
        #joint poisitions and velocities
        state = np.concatenate(
            [self.sim.data.qpos[:].copy(), self.sim.data.qvel[:].copy()])

        if not self.images:
            return state[:2]  # State is just (x, y) now

        #get images
        ims = cv2.resize(
            self.sim.render(64, 64, camera_name="cam0")[20:64, 20:64],
            (64, 64),
            interpolation=cv2.INTER_AREA)
        return ims

    def reset(self, difficulty='m', check_constraint=True, pos=()):
        if len(pos):
            self.sim.data.qpos[0] = pos[0]
            self.sim.data.qpos[1] = pos[1]
        else:
            if difficulty == 'e':
                self.sim.data.qpos[0] = np.random.uniform(0.15, 0.22)
            elif difficulty == 'm':
                self.sim.data.qpos[0] = np.random.uniform(-0.04, 0.04)
            self.sim.data.qpos[1] = np.random.uniform(0.0, 0.22)
        self.steps = 0

        # self.sim.data.qpos[0] = 0.25
        # self.sim.data.qpos[1] = 0
        # print(self._get_obs())
        # print("GOT HERE")
        # assert(False)

        # Randomize wal positions
        #     w1 = -0#np.random.uniform(-0.2, 0.2)
        #     w2 = 0 #np.random.uniform(-0.2, 0.2)
        # #     print(self.sim.model.geom_pos[:])
        # #     print(self.sim.model.geom_pos[:].shape)
        #     self.sim.model.geom_pos[5, 1] = 0.4 + w1
        #     self.sim.model.geom_pos[7, 1] = -0.25 + w1
        #     self.sim.model.geom_pos[6, 1] = 0.4 + w2
        #     self.sim.model.geom_pos[8, 1] = -0.25 + w2
        w1 = -0  #np.random.uniform(-0.2, 0.2)
        w2 = 0.08  #np.random.uniform(-0.2, 0.2)
        #     print(self.sim.model.geom_pos[:])
        #     print(self.sim.model.geom_pos[:].shape)
        self.sim.model.geom_pos[5, 1] = 0.25 + w1
        self.sim.model.geom_pos[7, 1] = -0.25 + w1
        self.sim.model.geom_pos[6, 1] = 0.35 + w2
        self.sim.model.geom_pos[8, 1] = -0.25 + w2

        self.sim.forward()
        # print("RESET!", self._get_obs())
        constraint = int(self.sim.data.ncon > 3)
        if constraint and check_constraint:
            if not len(pos):
                self.reset(difficulty, pos=pos)
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
        # print("STATE", st)
        if st[0] <= 0.149:
            delt = (np.array([0.15, 0.125]) - st)
        else:
            delt = (np.array([self.goal[0], self.goal[1]]) - st)
        act = self.gain * delt

        return act


class MazeImageTeacher(object):
    def __init__(self):
        self.env = MazeImageNavigation()
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

        obs = self.env.reset(difficulty='m')
        O, A, cost_sum, costs = [obs], [], 0, []
        constraints_violated = 0
        im_list = [self.env._get_obs(images=True)]

        noise_idx = np.random.randint(int(2 * HORIZON / 4))
        for i in range(HORIZON):
            action = self.env.expert_action()

            if i < noise_idx:
                if mode == "eps_greedy":
                    assert (noise_param <= 1)
                    if np.random.random() < noise_param:
                        action = self.env.action_space.sample()
                    else:
                        if np.random.random() < self.default_noise:
                            action = self.env.action_space.sample()

                elif mode == "gaussian_noise":
                    action = (np.array(action) + np.random.normal(
                        0, noise_param + self.default_noise,
                        self.env.action_space.shape[0])).tolist()
                else:
                    print("Invalid Mode!")
                    assert (False)

            A.append(action)
            obs, cost, done, info = self.env.step(action)
            print("CON", info['constraint'])
            # print("STATE", obs)
            # print("DONE", done)
            constraints_violated += info['constraint']
            O.append(obs)
            im_list.append(self.env._get_obs(images=True))
            cost_sum += cost
            costs.append(cost)
            if done:
                break

        values = np.cumsum(costs[::-1])[::-1]

        print(cost_sum)
        print(len(O))
        print("CONSTRAINTS: ", constraints_violated)
        print("FINAL COST: ", cost)
        if int(cost_sum) == -HORIZON:
            print("FAILED")
            # return self.get_rollout(noise_param_in)

        npy_to_gif(im_list, 'image_maze')
        assert (False)

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
    teacher = MazeImageTeacher()
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
        reward_sum_completed.append(rollout_stats['reward_sum'] +
                                    diff * rollout_stats['rewards'][-1])

    print("completed reward sum", np.mean(reward_sum_completed),
          np.std(reward_sum_completed), constraint_sat)
