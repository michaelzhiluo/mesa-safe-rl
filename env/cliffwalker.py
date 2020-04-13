from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco import mujoco_env
import numpy as np
import os

# TODO: update this function
def get_random_transitions(num_transitions):
    env = CliffWalkerEnv()
    state = env.reset()
    transitions = []
    for i in range(num_transitions):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        constraint = False # TODO: fix this
        transitions.append((state, action, constraint, next_state))
        state = next_state
    return transitions

def huber(x, p):
    return np.sqrt(x*x + p*p) - p

class CliffWalkerEnv(Walker2dEnv):
    def __init__(self):
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder,
                                    'assets/cliff_walker.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_filename, 5)
        self.transition_function = get_random_transitions
        self._max_episode_steps = np.inf

    def step(self, a):
        curr_state = self._get_obs()
        (s, _, done, info) = super(CliffWalkerEnv, self).step(a)
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
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[:], np.clip(qvel, -10, 10)]).ravel()

    def _get_rewards(self, s, a):
        x = s[0]
        print("X", x)
        running_vel = s[9] - 2.0
        torso_height = s[1]
        is_standing = float(torso_height > 1.2)
        is_falling = float(torso_height < 0.7)
        run_reward = np.clip(1 - 0.2 * huber(running_vel, p=0.1), 0, 1)
        stand_reward = np.clip(0.25 * torso_height +
                               0.25 * is_standing +
                               0.5 * (1 - is_falling), 0, 1)
        control_reward = np.clip(1 - 0.05 * np.dot(a, a), 0, 1)
        reset_location_reward = 0.8 * (np.abs(x) < 0.5) + 0.2 * (1 - 0.2 * np.abs(x))
        forward_reward = 0.5 * run_reward + 0.25 * stand_reward + 0.25 * control_reward
        reset_reward = 0.5 * reset_location_reward + 0.25 * stand_reward + 0.25 * control_reward
        return (forward_reward, reset_reward)

if __name__ == '__main__':
    import time
    env = CliffWalkerEnv()
    env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        next_state, reward, done, info  = env.step(action)
        # print("STATE", next_state)
        # print("ACTION", action)
        # env.render()
        time.sleep(0.01)

