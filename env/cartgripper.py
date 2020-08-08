'''
All cartgripper env modules built on cartrgipper implementation in
https://github.com/SudeepDasari/visual_foresight
'''

import copy

import cv2
import numpy as np

from dmbrl.env.cartgripper_env.cartgripper_rot_grasp import CartgripperRotGraspEnv
from dmbrl.env.util.action_util import autograsp_dynamics
from dmbrl.env.cartgripper_env.util.sensor_util import is_touching

ENV_PARAMS = {
    'num_objects': 2,
    'object_mass': 0.5,
    'friction': 1.0,
    'finger_sensors': True,
    'minlen': 0.03,
    'maxlen': 0.1,
    'object_object_mindist': 0.18,
    'cube_objects': True,
    'arm_start_lifted': False,
    'randomize_initial_pos': False,
    'init_pos': np.array([0.3, 0.2, 0]),
    'autograsp': {
        'zthresh': -0.06,
        'touchthresh': 0.0,
        'reopen': True
    }
}

# repository specific params


class AutograspCartgripperEnv(CartgripperRotGraspEnv):
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
        self.goal_image = np.zeros((48, 64, 3))

    def _default_hparams(self):
        ag_params = {
            'no_motion_goal': False,
            'reopen': False,
            'zthresh': -0.06,
            'touchthresh': 0.0,
            'lift_height': 0.01
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
        print("action", action)
        assert action.shape[0] == self._adim
        gripper_z = self._previous_target_qpos[2]
        z_thresh = self._hp.zthresh
        delta_z_cond = np.amax(self._last_obs['object_poses_full'][:, 2] -
                               self._ground_zs) > 0.01

        target, self._gripper_closed = autograsp_dynamics(
            self._previous_target_qpos, action, self._gripper_closed,
            gripper_z, z_thresh, self._hp.reopen, delta_z_cond)
        return target

    def _post_step(self):
        if np.amax(self._last_obs['object_poses_full'][:, 2] -
                   self._ground_zs) > 0.05:
            self._goal_reached = True

    def cost_fn(self, obs):
        # NOTE: obs_cost_fn takes in a processed obs right now
        return np.sum((obs - self.goal_image)**2)

    def has_goal(self):
        return True

    def is_stable(self, obs):
        return self._goal_reached

    def generate_goal_image(self):
        self.reset(randomize_objects=False)
        actions = np.tile(np.array([0, 0, -0.02, 0]),
                          (5, 1)) + np.random.randn(5, 4) * 0.01
        actions = np.vstack([
            np.tile(np.array([0, -0.02, 0, 0]),
                    (10, 1)) + np.random.randn(10, 4) * 0.01, actions
        ])
        for ac in actions:
            obs = self.step(ac)
        im = obs[0]['images'][0]
        target_img_height, target_img_width, _ = self.goal_image.shape
        im = cv2.resize(
            im, (target_img_width, target_img_height),
            interpolation=cv2.INTER_AREA)
        self.goal_image = im
        print("GOAL_IMAGE", self.goal_image.shape)
        self.reset(randomize_objects=False)
        import scipy.misc
        scipy.misc.imsave("goal_image.jpg", im)
        return im
