'''
All cartgripper env modules built on cartrgipper implementation in
https://github.com/SudeepDasari/visual_foresight
'''

import copy

import cv2
import numpy as np

from dmbrl.env.cartgripper_env.cartgripper_rot_grasp import CartgripperRotGraspEnv
from dmbrl.env.util.action_util import no_rot_dynamics, clip_target_qpos
from dmbrl.env.cartgripper_env.util.sensor_util import is_touching
from gym.spaces import Box

ENV_PARAMS = {}


def zangle_to_quat(zangle):
    """
    :param zangle in rad
    :return: quaternion
    """
    return np.array([np.cos(zangle / 2), 0, 0, np.sin(zangle / 2)])


def quat_to_zangle(quat):
    """
    :param quat: quaternion with only
    :return: zangle in rad
    """
    theta = np.arctan2(2 * quat[0] * quat[3], 1 - 2 * quat[3]**2)
    return np.array([theta])


# repository specific params


class TallCartgripperEnv(CartgripperRotGraspEnv):
    def __init__(self, env_params={}, reset_state=None):
        assert 'mode_rel' not in env_params, "Autograsp sets mode_rel"
        params = copy.deepcopy(ENV_PARAMS)

        new_params = copy.deepcopy(env_params)

        for k in new_params:
            params[k] = new_params[k]

        if 'autograsp' in params:
            ag_dict = params.pop('autograsp')
            for k in ag_dict:
                params[k] = ag_dict[k]

        super().__init__(params, reset_state)
        self._adim = 4
        self._goal_reached, self._ground_zs = False, None
        self.unwrapped = self
        self.ac_low_bound = np.array([-0.25, -0.25, -0.25, -0.25])
        self.ac_high_bound = np.array([0.25, 0.25, 0.25, 0.25])
        self.action_space = Box(self.ac_low_bound, self.ac_high_bound)
        self.goal_image_shape = (48, 64, 3)
        self.observation_space = self.goal_image_shape
        # self.goal_image = None

    def _default_hparams(self):
        ag_params = {
            'x_range': 0.06,
            'y_range': 0.06,
            # 'x_range': 0.12,
            # 'y_range': 0.12,
            # 'default_y': 0.,
            'default_theta': 0.,
            'no_motion_goal': False,
            'reopen': False,
            'zthresh': -0.06,
            'touchthresh': 0.0,
            'lift_height': 0.01,
            'pos_lower_bound': np.array([-0.2, -0.15]),
            'pos_upper_bound': np.array([0.2, 0.15])
        }

        parent_params = super()._default_hparams()
        parent_params.set_hparam('finger_sensors', True)
        parent_params.set_hparam('ncam', 2)
        for k in ag_params:
            parent_params.add_hparam(k, ag_params[k])
        return parent_params

    def _init_dynamics(self):
        self._goal_reached = False
        self._gripper_closed = False
        self._ground_zs = self._last_obs['object_poses_full'][:, 2].copy()

    def _next_qpos(self, action):
        # print("action", action)
        assert action.shape[0] == self._adim
        gripper_z = self._previous_target_qpos[2]
        z_thresh = self._hp.zthresh
        delta_z_cond = np.amax(self._last_obs['object_poses_full'][:, 2] -
                               self._ground_zs) > 0.01

        target, self._gripper_closed = no_rot_dynamics(
            self._previous_target_qpos, action, self._gripper_closed,
            gripper_z, z_thresh, self._hp.reopen, delta_z_cond)
        target = clip_target_qpos(target, self._hp.pos_lower_bound,
                                  self._hp.pos_upper_bound)
        return target

    def _post_step(self):
        #if np.amax(self._last_obs['object_poses_full'][:, 2] - self._ground_zs) > 0.05:
        self._goal_reached = True

    def cost_fn(self, obs):
        # NOTE: obs_cost_fn takes in a processed obs right now
        return np.sum((obs - self.goal_image)**2)

    def has_goal(self):
        return True

    def is_stable(self, obs):
        return self._goal_reached

    def get_object_poses(self, idx=None):
        if idx is None:
            return self._last_obs['object_poses_full']
        return self._last_obs['object_poses_full'][idx]

    def get_object_mask(self, idx, obs):
        pose = self.get_object_poses(idx)

    def get_grasp_action(self, idx=1, noise_std=0.00, drop=False):
        position = np.copy(self.sim.get_state().qpos[:])
        obj_position = self.get_object_poses(1)[:3]
        control = np.zeros(self._adim)
        gain = 1
        if np.abs(position[1] - 0.15) > 0.02 and np.abs(
                position[0] - obj_position[0]) > 0.04:
            print(0)
            control[1] = 0.15 - position[1]
        elif np.abs(position[0] - obj_position[0]) > 0.04:
            print(1)
            control[0] = obj_position[0] - position[0]
        elif np.abs(position[1] - obj_position[1]) > 0.02:
            print(2)
            control[1] = obj_position[1] - position[1]
        elif drop:
            control[3] = 0.01
            noise_std = 0.03
        else:
            print(3)
            control[3] = 0.1
            control[2] = 0.015
            control[:-2] += np.random.randn(self._adim - 2) * 0.01
        # print(position[:3], obj_position, control)
        return control * gain + np.random.randn(self._adim) * noise_std

    def _create_pos(self):
        object_poses = super()._create_pos()
        positions = []
        for i in range(self.num_objects):
            object_poses[i][0] = np.random.uniform(-self._hp.x_range,
                                                   self._hp.x_range)
            # object_poses[i][1] = np.random.uniform(-self._hp.y_range, self._hp.y_range)
            object_poses[i][1] = -0.12
            object_poses[i][3:] = zangle_to_quat(self._hp.default_theta)
            while len(positions) > 0 and \
                    np.linalg.norm(np.array(positions) - np.array([object_poses[i][0], object_poses[i][1]]), axis=1).min() < 0.03:
                object_poses[i][0] = np.random.uniform(-self._hp.x_range,
                                                       self._hp.x_range)
                # object_poses[i][1] = np.random.uniform(-self._hp.y_range, self._hp.y_range)
                object_poses[i][1] = -0.12
            positions.append((object_poses[i][0], object_poses[i][1]))
        return object_poses

    def goal_reached(self):
        return self._goal_reached

    def generate_goal_image(self):
        self.reset(randomize_objects=False)
        actions = np.tile(np.array([0, 0, -0.02, 0]), (5, 1))
        actions = np.vstack(
            [np.tile(np.array([0.02, 0, 0, 0]), (5, 1)), actions])
        for ac in actions:
            obs = self.step(ac)
        # im = obs[0]['images'][0]
        im = self.render()[0]
        target_img_height, target_img_width, _ = self.goal_image_shape
        im = cv2.resize(
            im, (target_img_width, target_img_height),
            interpolation=cv2.INTER_AREA)
        self.goal_image = im
        print("GOAL_IMAGE", self.goal_image.shape)
        self.reset(randomize_objects=False)
        import scipy.misc
        scipy.misc.imsave("goal_image.jpg", im)
        return im

    def get_armpos(self, object_pos):
        xpos0 = super().get_armpos(object_pos)
        xpos0[3] = 0
        xpos0[4:6] = [0.05, -0.05]

        return xpos0

    def topple_check(self, debug=False):
        quat = self.get_object_poses()[:, 3:]
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

    def true_cost(self):
        return -(self.get_object_poses()[1, 2] > 0.08).astype(int)

    @staticmethod
    def get_real_state(env, sim_state, actions=[]):
        env.set_state(sim_state)
        env.sim.forward()
        for action in actions:
            env.step(action)
        im = env.render()[0]
        state = env.sim.get_state()
        return im

    @staticmethod
    def get_object_masks_from_sim_state(env, sim_state):
        im_list = []
        num_objects = (len(sim_state.qpos) - 6) // 7
        original_qpos = sim_state.qpos.copy()
        for i in range(num_objects):
            new_qpos = original_qpos.copy()
            new_qpos[:3] = [1.5, 0.2, 0.05]
            for j in range(num_objects):
                if i == j:
                    continue
                else:
                    obj_idx = j * 7 + 6
                    new_qpos[obj_idx:obj_idx + 3] = [1.2, 0.2, 0.05]
            sim_state.qpos[:] = new_qpos
            env.sim.set_state(sim_state)
            env.sim.forward()
            im = env.render()[0]
            im_list.append(im)
        sim_state.qpos[:] = original_qpos
        masks = []
        for im in im_list:
            # mask = np.zeros_like(hsv)
            # for i in range(3):
            #     mask[:,:, i] = hsv[:,:,0] > 40
            # frame = np.multiply(hsv, mask)
            mask = hsv[:, :, 0] > 40
            masks.append(mask)
        return masks

    @staticmethod
    def get_mask(im):
        # import matplotlib.pyplot as plt

        # plt.imshow(im)
        # plt.show()

        mask = np.logical_and(im[:, :, 0] > 65, im[:, :, 1] < 65)
        mask = np.logical_and(mask, im[:, :, 2] < 40)
        mask1 = mask.copy()

        mask = np.logical_and(im[:, :, 0] > 60, im[:, :, 1] > 40)
        mask = np.logical_and(mask, im[:, :, 2] < im[:, :, 1] / 2)
        mask2 = mask.copy()

        # mask = np.expand_dims((mask1), axis=-1)
        mask = np.stack((mask1, mask2), axis=-1)

        # mask = np.logical_and(np.max(im, axis=2) - np.min(im, axis=2) > 10, np.abs(im.max(2) - 82 ) > 10)
        # mask = np.logical_and(im[:,:,0] > 60, mask)

        # plt.imshow(mask2.squeeze())
        # plt.show()

        # mask2 = np.maximum(np.stack((mask1, mask1, mask1), axis=-1), np.stack((mask2, mask2, mask2), axis=-1))
        # im2 = np.multiply(im, mask2)
        # print(im2.shape)

        # plt.imshow(im2)
        # plt.show()

        # assert len(mask.shape) == 3, mask.shape

        return mask
