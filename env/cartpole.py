import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
import os
from filelock import FileLock
import xml.etree.ElementTree

HORIZON = 200


def transition_function(num_transitions, length=None, discount=0.8):
    env = CartPoleEnv(init_length = length)
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
                if len(transitions) > num_transitions:
                    break
            # Reset
            state = env.reset()
            rollouts = []

        action = env.action_space.sample()
        next_state, reward, _, info = env.step(action)
        steps +=1
        constraint = info['constraint']
        done = steps == 30
        rollouts.append([state, action, constraint, next_state,
                             not constraint])
        state = next_state

    return transitions

class CartPoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, init_length=None, no_task=True):
        print("Carpole Length: ", init_length)
        self.no_task = no_task
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.xml_location = '%s/assets/cartpole.xml' % dir_path
        self.mode = 'train'
        self.test_domain = 1.0
        self.domain_low = 0.4
        self.domain_high = 0.8
        self.fixed = False
        self.pendulum_length = 0.6
        if init_length:
            self.pendulum_length = init_length
            self.fixed = True
        self.set_length(self.pendulum_length)
        self.steps = 0
        self._max_episode_steps = HORIZON
        self.max_episode_steps = HORIZON
        self.transition_function = transition_function
        mujoco_env.MujocoEnv.__init__(self, self.xml_location, 2)

    def step(self, a, early_stop=True):
        self.steps +=1
        cur_ob = self._get_obs()
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, self.pendulum_length]))) / (self.pendulum_length ** 2)
        )
        catastrophe = (np.abs(ob[1]) > np.pi/2) or (np.abs(ob[0]) >= 2.4)
        info = {}
        constraint = False
        if catastrophe:
            if not self.no_task:
                ob[-1] = 1
            constraint = True

        notdone = np.isfinite(ob).all() and not (catastrophe) #and self.mode == 'test')
        done = (not notdone) or self.steps >=200
        if not early_stop:
            done = False
        info = {
            "constraint": constraint,
            "reward": reward,
            "action": a,
            "state": cur_ob,
            "next_state": ob,
        }
        return ob, reward, done, info

    def reset_model(self):
        if not hasattr(self, "pendulum_length"):
            self.pendulum_length = self.np_random.uniform(self.domain_low, self.domain_high)
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        original_obs = np.concatenate([self.data.qpos, self.data.qvel]).ravel()
        if self.no_task:
            return original_obs
        curr_obs = np.concatenate([original_obs, [self.pendulum_length, 0]], axis=-1)
        return curr_obs

    def _get_ee_pos(self, x):
        x0, theta = x[0], x[1]
        return np.array([
            x0 + self.pendulum_length * np.sin(theta),
            self.pendulum_length * np.cos(theta)
        ])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def set_length(self, length):
        lock = FileLock(self.xml_location + '.lock')  # concurrency protection
        with lock:
            et = xml.etree.ElementTree.parse(self.xml_location)
            et.find('worldbody').find('body').find('body').find('geom').set('fromto',
                                                                            "0 0 0 0.001 0 %0.3f" % length)  # changing size of pole
            et.write(self.xml_location)
            self.model = mujoco_py.load_model_from_path(self.xml_location)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data

    def reset(self, mode='train'):
        self.steps = 0
        if mode == 'train' and not self.fixed:
            self.pendulum_length = self.np_random.uniform(self.domain_low, self.domain_high)
            self.set_length(self.pendulum_length)
        elif self.mode != 'test' and mode == 'test' and not self.fixed: #starting adaptation
            self.pendulum_length = self.test_domain
            self.mode = mode
            self.set_length(self.test_domain)
        mujoco_env.MujocoEnv.reset(self)
        return self._get_obs()

    def get_image(self):
        return self.render(mode='rgb_array', width=150, height=150)

'''
env = CartPoleEnv()
env.reset()

for i in range(1000):
    print("fuckery")
    env.step(0)

haha = env.render(mode='rgb_array', width=256, height=256)
from PIL import Image
im = Image.fromarray(haha)
im.save("your_file.jpeg")

env.reset()
'''