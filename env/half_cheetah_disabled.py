from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import gym
from gym import utils
from learning_to_adapt.envs.mujoco_env import MujocoEnv
from learning_to_adapt.utils.serializable import Serializable
from gym.utils import seeding

HORIZON = 1000

def transition_function(num_transition, discount = 0.99):
    env = HalfCheetahEnv()
    transitions = []
    rollouts = []
    done = True
    steps =0
    while True:
        if done:
            steps =0
            if len(rollouts):
                mc_reward =0
                for transition in rollouts[::-1]:
                    mc_reward = transition[2] + discount * mc_reward
                    transition.append(mc_reward)
                transitions.extend(rollouts)
                if len(transitions) > num_transition:
                    break
            # Reset
            state = env.reset()
            rollouts = []

        action = env.action_space.sample()
        next_state, reward, _, info = env.step(action)
        steps +=1
        constraint = info['constraint']
        done = steps == 1000
        rollouts.append([state, action, constraint, next_state,
                             not constraint])
        state = next_state

    return transitions

class HalfCheetahEnv(MujocoEnv, Serializable, utils.EzPickle):
    metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50
    }
    def __init__(self, task='cripple', reset_every_episode=False):
        self.no_task = True
        self.task = None
        Serializable.quick_init(self, locals())
        self.cripple_mask = None
        self.first = True
        self._max_episode_steps = 1000
        self.task = task#'cripple'
        self.crippled_leg = 0
        self.prev_torso = None
        self.prev_qpos = None
        self.transition_function = transition_function
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets", "half_cheetah_disabled.xml"))
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.dt = self.model.opt.timestep

        self.act_high = self.action_space.high[0]
        self.act_low = self.action_space.low[0]

        self.cripple_mask = np.ones(self.action_space.shape)
        #self.lmao = np.array([0.69248867, 0.75095502, 0.54999795, 0.82621723, 0.9406307, 0.88019476])
        self.reward_range = (-np.inf, np.inf)
        utils.EzPickle.__init__(self, locals())

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_current_obs(self):
        if self.prev_qpos == None:
            self.prev_qpos = self.get_body_com("torso")[:1]
        self.dt = self.model.opt.timestep
        if self.no_task:
            return np.concatenate([
                self.model.data.qpos.flatten()[1:],
                self.model.data.qvel.flat,
                self.get_body_com("torso").flat[1:],
            ])
        else:
            return np.concatenate([
                self.model.data.qpos.flatten()[1:],
                self.model.data.qvel.flat,
                self.get_body_com("torso").flat[1:],
                (self.get_body_com("torso")[:1] - self.prev_qpos)/self.dt, #reward
                self.check_catastrophe(), #catastrophe indicator
            ])

    def check_catastrophe(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            name_set = set()
            name_set.add(self.model.geom_names[contact.geom1])
            name_set.add(self.model.geom_names[contact.geom2])
            if 'floor' in name_set and 'head' in name_set:
                return [1] 
        return [0] 

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action, early_stop = True):
        obs = self.get_current_obs()
        # Clip
        action = np.clip(action, -1.0, 1.0)
        action = self.cripple_mask * action
        #action = self.lmao * action
        self.prev_qpos = self.get_body_com("torso").flat[:1]
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        self.steps+=1
        done = False
        catastrophe = self.check_catastrophe()[0] #next_obs[-1] == 1
        if early_stop:
            done = catastrophe
        info = {
            "constraint": catastrophe,
            "reward": reward,
            "action": action,
            "state": obs,
            "next_state": next_obs,
        }
        if catastrophe and self.mode == 'test':
            done = True
        return next_obs, reward, done, info

    def reward(self, obs, action, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action), axis=1)
        forward_reward = (next_obs[:, 0] - obs[:, 0])/self.dt
        reward = forward_reward - ctrl_cost
        return reward

    def reset_mujoco(self, init_state=None):
        super(HalfCheetahEnv, self).reset_mujoco(init_state=init_state)

    def reset_task(self, value=None):
        value = 5
        if self.first:
            self.first = False
            return
        if self.task == 'cripple':
            crippled_joint = value if value is not None else np.random.randint(1, self.action_dim)
            self.cripple_mask = np.ones(self.action_space.shape)
            self.cripple_mask[crippled_joint] = 0
            geom_idx = self.model.geom_names.index(self.model.joint_names[crippled_joint+3])
            geom_rgba = self._init_geom_rgba.copy()
            geom_rgba[geom_idx, :3] = np.array([1, 0, 0])
            self.model.geom_rgba = geom_rgba

        elif self.task is None:
            pass

        else:
            raise NotImplementedError

        self.model.forward()

    def reset(self, mode='train'):
        self.steps=0
        self.prev_qpos = None
        self.mode = mode
        if mode == 'train':
            self.reset_task(value=np.random.randint(1, self.action_dim - 1))
        else:
            self.reset_task(value=self.action_dim - 1)
        return MujocoEnv.reset(self)

    def close(self):
        self.stop_viewer()

