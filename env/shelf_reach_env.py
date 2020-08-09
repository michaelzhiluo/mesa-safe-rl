'''
Built on cartrgipper implementation in
https://github.com/SudeepDasari/visual_foresight
'''

import numpy as np
import moviepy.editor as mpy
import copy
import os.path as osp
from env.base_mujoco_env import BaseMujocoEnv
import matplotlib.pyplot as plt
from gym.spaces import Box
import os

FIXED_ENV = False
DENSE_REWARD = True
GT_STATE = True
EARLY_TERMINATION = True


def no_rot_dynamics(prev_target_qpos, action):
    target_qpos = np.zeros_like(prev_target_qpos)
    target_qpos[:] = action[:] + prev_target_qpos[:]
    # target_qpos[4] = action[3]
    return target_qpos


def clip_target_qpos(target, lb, ub):
    target[:len(lb)] = np.clip(target[:len(lb)], lb, ub)
    return target


class ShelfRotEnv(BaseMujocoEnv):
    def __init__(self):
        parent_params = super()._default_hparams()
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        self.reset_xml = os.path.join(envs_folder,
                                      'cartgripper_assets/shelf_reach.xml')
        super().__init__(self.reset_xml, parent_params)
        self._adim = 5
        self.substeps = 500
        self.low_bound = np.array([-0.4, -0.4, -0.05])
        self.high_bound = np.array([0.4, 0.4, 0.15])
        self.ac_high = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        self.ac_low = -self.ac_high
        self.action_space = Box(self.ac_low, self.ac_high)
        self._previous_target_qpos = None
        self.target_height_thresh = 0.03
        self.object_fall_thresh = -0.03
        self.obj_y_dist_range = np.array([0.05, 0.2])
        self.obj_x_range = np.array([-0.25, -0.1])
        self.randomize_objects = not FIXED_ENV
        self.dense_reward = DENSE_REWARD
        self.gt_state = GT_STATE
        self._max_episode_steps = 25

        if self.gt_state:
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(27, ))
        else:
            self.observation_space = (48, 64, 3)
        self.reset()

    def set_reward_type(self, dense_reward):
        self.dense_reward = dense_reward

    def render(self):
        return super().render()[:, ::-1].copy().squeeze(
        )  # cartgripper cameras are flipped in height dimension

    def reset(self):
        self._reset_sim(self.reset_xml)

        #clear our observations from last rollout
        self._last_obs = None

        state = self.sim.get_state()
        pos = np.copy(state.qpos[:])
        pos[6:] = self.object_reset_poses().ravel()
        state.qpos[:] = pos
        self.sim.set_state(state)

        self.sim.forward()

        self._previous_target_qpos = copy.deepcopy(
            self.sim.data.qpos[:5].squeeze())
        self._previous_target_qpos[-1] = self.low_bound[-1]

        if self.gt_state:
            return pos
        else:
            return self.render()

    def step(self, action):
        position = self.position
        action = np.clip(action, self.ac_low, self.ac_high)
        target_qpos = self._next_qpos(action)
        if self._previous_target_qpos is None:
            self._previous_target_qpos = target_qpos

        for st in range(self.substeps):
            alpha = st / (float(self.substeps) - 1)
            self.sim.data.ctrl[:] = alpha * target_qpos + (
                1. - alpha) * self._previous_target_qpos
            self.sim.step()

        self._previous_target_qpos = target_qpos
        constraint = self.topple_check()
        reward = self.reward_fn()

        if EARLY_TERMINATION:
            done = (constraint > 0) or (reward > 0)
        else:
            done = False

        if done and reward > 0:
            reward = 5

        info = {
            "constraint": constraint,
            "reward": reward,
            "state": position,
            "next_state": self.position,
            "action": action
        }

        if self.gt_state:
            return self.position, reward, done, info
        else:
            return self.render(), reward, done, info

    def topple_check(self, debug=False):
        quat = self.object_poses[:, 3:]
        phi = np.arctan2(
            2 *
            (np.multiply(quat[:, 0], quat[:, 1]) + quat[:, 2] * quat[:, 3]),
            1 - 2 * (np.power(quat[:, 1], 2) + np.power(quat[:, 2], 2)))
        theta = np.arcsin(2 * (np.multiply(quat[:, 0], quat[:, 2]) -
                               np.multiply(quat[:, 3], quat[:, 1])))
        psi = np.arctan2(
            2 * (np.multiply(quat[:, 0], quat[:, 3]) + np.multiply(
                quat[:, 1], quat[:, 2])),
            1 - 2 * (np.power(quat[:, 2], 2) + np.power(quat[:, 3], 2)))
        euler = np.stack([phi, theta, psi]).T[:, :2] * 180. / np.pi
        if debug:
            return np.abs(euler).max() > 15 or np.isnan(euler).sum() > 0, euler
        return np.abs(euler).max() > 15 or np.isnan(euler).sum() > 0

    @property
    def jaw_width(self):
        pos = self.position
        return 0.08 - (pos[4] - pos[5])  # 0.11 might be slightly off

    def set_y_range(self, bounds):
        self.obj_y_dist_range[0] = bounds[0]
        self.obj_y_dist_range[1] = bounds[1]

    def expert_action(self, t, noise_std=0.0):
        cur_pos = self.position[:3]
        cur_pos[1] += 0.05  # compensate for length of jaws
        target_obj_pos = self.object_poses[1][:3]
        ac = np.zeros(5)
        if t < 3 and np.abs(cur_pos[0] - target_obj_pos[0] - 0.2) > 0.01:
            ac[0] = -(cur_pos[0] - target_obj_pos[0] - 0.2)
            # print(cur_pos[0] - target_obj_pos[0])
            ac[1] = -(cur_pos[1] - target_obj_pos[1] - 0.2)
        if t < 3:
            ac[3] = -0.1
        elif t < 5:
            ac[1] = -0.1
            ac[0] = -0.02
        elif t < 8:
            ac[3] = -0.1
        elif t < 14:
            ac[0] = -0.02
            ac[1] = -0.02
        elif t < 17:
            ac[4] = 0.06
        elif t < 20:
            ac[2] = 0.05

        return ac + np.random.randn(self._adim) * noise_std

    def get_demonstration(self, noise_std=0.0):
        self.reset()
        i = 0
        im_list = []
        done = False
        while not done:
            ac = self.expert_action(i, noise_std=noise_std)
            ns, r, done, info = self.step(ac)
            im_list.append(env.render().squeeze())
            done = done or i == self._max_episode_steps
            i += 1
        npy_to_gif(im_list, "out")

    def reward_fn(self):
        if not self.dense_reward:
            return (self.target_object_height >
                    self.target_height_thresh).astype(float)
        else:
            lift_reward = (self.target_object_height >
                           self.target_height_thresh).astype(float)
            # if lift_reward == 1:
            #     print("lifted")
            cur_pos = self.position[:2]
            cur_pos[1] += 0.05  # compensate for length of jaws
            target_obj_pos = self.object_poses[1][:2]
            action = np.zeros(self._adim)
            delta = target_obj_pos - cur_pos
            ee_reward = -np.linalg.norm(delta)
            if ee_reward > -0.03:
                ee_reward = 0.
            return lift_reward + 0.1 * ee_reward

    def object_reset_poses(self):
        new_poses = np.zeros((3, 7))
        new_poses[:, 3] = 1
        if self.randomize_objects == True:
            x = np.random.uniform(self.obj_x_range[0], self.obj_x_range[1])
            y1 = np.random.randn() * 0.05
            y0 = y1 - np.random.uniform(self.obj_y_dist_range[0],
                                        self.obj_y_dist_range[1])
            y2 = y1
            new_poses[0, 0:2] = np.array([y0, x])
            new_poses[1, 0:2] = np.array([y1, x])
            new_poses[2, 0:2] = np.array([y2, x + 0.2])
        else:
            x = np.mean(self.obj_x_range)
            y1 = 0.
            y0 = y1 - np.mean(self.obj_y_dist_range)
            new_poses[0, 0:2] = np.array([y0, x])
            new_poses[1, 0:2] = np.array([y1, x])
            new_poses[2, 0:2] = np.array([y1, x + 0.2])
        return new_poses

    @property
    def position(self):
        return np.copy(self.sim.get_state().qpos[:])

    @property
    def object_poses(self):
        pos = self.position
        num_objs = (self.position.shape[0] - 6) // 7
        poses = []
        for i in range(num_objs):
            poses.append(np.copy(pos[i * 7 + 6:(i + 1) * 7 + 6]))
        return np.array(poses)

    @property
    def target_object_height(self):
        return self.object_poses[1, 2] - 0.072

    def _next_qpos(self, action):
        assert action.shape[0] == self._adim, action
        target = no_rot_dynamics(self._previous_target_qpos, action)
        target = clip_target_qpos(target, self.low_bound, self.high_bound)
        return target


def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


if __name__ == '__main__':
    env = ShelfRotEnv()
    env.get_demonstration()
