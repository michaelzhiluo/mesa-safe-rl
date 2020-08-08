import copy

import numpy as np

from .cartgripper_rot_grasp import CartgripperRotGraspEnv
from dmbrl.env.util.action_util import autograsp_dynamics
from dmbrl.env.cartgripper_env.util.sensor_util import is_touching

env_params = {
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
        params = copy.deepcopy(env_params)

        if 'autograsp' in params:
            ag_dict = params.pop('autograsp')
            for k in ag_dict:
                params[k] = ag_dict[k]

        super().__init__(params, reset_state)
        self._adim = 4
        self._goal_reached, self._ground_zs = False, None

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
        ag_params = copy.deepcopy({**ag_params, **env_params})
        return parent_params

    def _init_dynamics(self):
        self._goal_reached = False
        self._gripper_closed = False
        self._ground_zs = self._last_obs['object_poses_full'][:, 2].copy()

    def _next_qpos(self, action):
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

    def has_goal(self):
        return True

    def goal_reached(self):
        return self._goal_reached
