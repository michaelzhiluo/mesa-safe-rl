from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco import mujoco_env
import numpy as np
import os

# TODO: update this function
def get_random_transitions(num_transitions):
    env = CliffCheetahEnv()
    state = env.reset()
    transitions = []
    for i in range(num_transitions):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        constraint = False # TODO: fix this
        transitions.append((state, action, constraint, next_state, done))
        state = next_state
    return transitions

def tolerance(x, bounds, margin):
    '''Returns 1 when x is within the bounds, and decays sigmoidally
    when x is within a certain margin outside the bounds.
    We've copied the function from [1] to reduce dependencies.
    [1] Tassa, Yuval, et al. "DeepMind Control Suite." arXiv preprint
    arXiv:1801.00690 (2018).
    '''
    (lower, upper) = bounds
    if lower <= x <= upper:
        return 0
    elif x < lower:
        dist_from_margin = lower - x
    else:
        assert x > upper
        dist_from_margin = x - upper
    loss_at_margin = 0.95
    w = np.arctanh(np.sqrt(loss_at_margin)) / margin
    s = np.tanh(w * dist_from_margin)
    return s*s


class CliffCheetahEnv(HalfCheetahEnv):
    def __init__(self):
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder,
                                    'assets/cliff_cheetah.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_filename, 5)
        self.transition_function = get_random_transitions
        self._max_episode_steps = 1000

    def step(self, a):
        curr_state = self._get_obs()
        (s, _, done, info) = super(CliffCheetahEnv, self).step(a)

        r = self._get_rewards(s, a)[0]
        # TODO: actually implement constraint
        info = {"constraint": False,
                "reward": r,
                "state": curr_state,
                "next_state": s,
                "action": a}

        return s, r, done, info

    def _get_obs(self):
        '''Modified to include the x coordinate.'''
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])

    def _get_rewards(self, s, a):
        (x, z, theta) = s[:3]
        xvel = s[9]
        # Reward the forward agent for running 9 - 11 m/s.
        forward_reward = (1.0 - tolerance(xvel, (9, 11), 7))
        theta_reward = 1.0 - tolerance(theta,
                                       bounds=(-0.05, 0.05),
                                       margin=0.1)
        # Reward the reset agent for being at the origin, plus
        # reward shaping to be near the origin and upright.
        reset_reward = 0.8 * (np.abs(x) < 0.5) + 0.1 * (1 - 0.2 * np.abs(x)) + 0.1 * theta_reward
        return (forward_reward, reset_reward)

if __name__ == '__main__':
    import time
    env = CliffCheetahEnv()
    env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        next_state, reward, done, info  = env.step(action)
        print("STATE", next_state)
        print("ACTION", action)
        # env.render()
        time.sleep(0.01)