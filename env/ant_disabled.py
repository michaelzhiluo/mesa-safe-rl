import numpy as np
from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.envs.mujoco_env import MujocoEnv
import os
from gym.utils import seeding

HORIZON = 1000

def transition_function(num_transition, discount = 0.99):
    env = AntEnv()
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

class AntEnv(MujocoEnv,  Serializable):

    def __init__(self, task='cripple', reset_every_episode=True):
        Serializable.quick_init(self, locals())
        self.cripple_mask = None
        self.reset_every_episode = reset_every_episode
        self._max_episode_steps = HORIZON
        self.first = True
        self.task = task
        self.transition_function = transition_function
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets", "ant.xml"))
        task = None if task == 'None' else task

        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.dt = self.model.opt.timestep

        assert task in [None, 'cripple']

        self.crippled_leg = 0
        self.steps = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_current_obs(self):
        obs =  np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

        obs = np.concatenate([obs, # Reward
            [self.get_body_comvel("torso")[0] + 0.05],
            # Catastrophe
            self.check_catastrophe(),])

        return obs


    def check_catastrophe(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            name_set = set()
            name_set.add(self.model.geom_names[contact.geom1])
            name_set.add(self.model.geom_names[contact.geom2])
            if 'floor' in name_set and 'torso_geom' in name_set:
                return [1]
        return [0] 

    def step(self, action, early_stop=True):
        obs = self.get_current_obs()
        if self.cripple_mask is not None:
            action = self.cripple_mask * action
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        lb, ub = self.action_space.low[0], self.action_space.high[0]
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling)) # 0
        self.steps += 1
        contact_cost = 0
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        done = False
        next_obs = self.get_current_obs()
        catastrophe = self.check_catastrophe()[0]
        if early_stop:
            done = catastrophe
        info = {
            "constraint": catastrophe,
            "reward": reward,
            "action": action,
            "state": obs,
            "next_state": next_obs,
        }
        return next_obs, reward, done, info

    def reset_mujoco(self, init_state=None):
        super(AntEnv, self).reset_mujoco(init_state=init_state)
        if self.reset_every_episode and not self.first:
            self.reset_task()
        if self.first:
            self.first = False

    def reset(self, mode='train'):
        self.steps = 0
        self.mode = mode
        return MujocoEnv.reset(self)

    '''
    our "front" is in +x direction, to the right side of screen
    LEG 4 (they call this back R)
    action0: front-right leg, top joint 
    action1: front-right leg, bottom joint
    
    LEG 1 (they call this front L)
    action2: front-left leg, top joint
    action3: front-left leg, bottom joint 
    
    LEG 2 (they call this front R)
    action4: back-left leg, top joint
    action5: back-left leg, bottom joint 
    
    LEG 3 (they call this back L)
    action6: back-right leg, top joint
    action7: back-right leg, bottom joint 
    geom_names has 
            ['floor','torso_geom',
            'aux_1_geom','left_leg_geom','left_ankle_geom', --1
            'aux_2_geom','right_leg_geom','right_ankle_geom', --2
            'aux_3_geom','back_leg_geom','third_ankle_geom', --3
            'aux_4_geom','rightback_leg_geom','fourth_ankle_geom'] --4
    '''

    def reset_task(self, value=None):

        if self.task == 'cripple':
            # Pick which leg to remove (0 1 2 are train... 3 is test)
            value = 3
            self.crippled_leg = value if value is not None else np.random.randint(0, 3)

            # Pick which actuators to disable
            self.cripple_mask = np.ones(self.action_space.shape)
            if self.crippled_leg == 0:
                self.cripple_mask[2] = 0
                self.cripple_mask[3] = 0
            elif self.crippled_leg == 1:
                self.cripple_mask[4] = 0
                self.cripple_mask[5] = 0
            elif self.crippled_leg == 2:
                self.cripple_mask[6] = 0
                self.cripple_mask[7] = 0
            elif self.crippled_leg == 3:
                self.cripple_mask[0] = 0
                self.cripple_mask[1] = 0

            # Make the removed leg look red
            geom_rgba = self._init_geom_rgba.copy()
            if self.crippled_leg == 0:
                geom_rgba[3, :3] = np.array([1, 0, 0])
                geom_rgba[4, :3] = np.array([1, 0, 0])
            elif self.crippled_leg == 1:
                geom_rgba[6, :3] = np.array([1, 0, 0])
                geom_rgba[7, :3] = np.array([1, 0, 0])
            elif self.crippled_leg == 2:
                geom_rgba[9, :3] = np.array([1, 0, 0])
                geom_rgba[10, :3] = np.array([1, 0, 0])
            elif self.crippled_leg == 3:
                geom_rgba[12, :3] = np.array([1, 0, 0])
                geom_rgba[13, :3] = np.array([1, 0, 0])
            self.model.geom_rgba = geom_rgba

            # Make the removed leg not affect anything
            temp_size = self._init_geom_size.copy()
            temp_pos = self._init_geom_pos.copy()

            if self.crippled_leg == 0:
                # Top half
                temp_size[3, 0] = temp_size[3, 0]/2
                temp_size[3, 1] = temp_size[3, 1]/2
                # Bottom half
                temp_size[4, 0] = temp_size[4, 0]/2
                temp_size[4, 1] = temp_size[4, 1]/2
                temp_pos[4, :] = temp_pos[3, :]

            elif self.crippled_leg == 1:
                # Top half
                temp_size[6, 0] = temp_size[6, 0]/2
                temp_size[6, 1] = temp_size[6, 1]/2
                # Bottom half
                temp_size[7, 0] = temp_size[7, 0]/2
                temp_size[7, 1] = temp_size[7, 1]/2
                temp_pos[7, :] = temp_pos[6, :]

            elif self.crippled_leg == 2:
                # Top half
                temp_size[9, 0] = temp_size[9, 0]/2
                temp_size[9, 1] = temp_size[9, 1]/2
                # Bottom half
                temp_size[10, 0] = temp_size[10, 0]/2
                temp_size[10, 1] = temp_size[10, 1]/2
                temp_pos[10, :] = temp_pos[9, :]

            elif self.crippled_leg == 3:
                # Top half
                temp_size[12, 0] = temp_size[12, 0]/2
                temp_size[12, 1] = temp_size[12, 1]/2
                # Bottom half
                temp_size[13, 0] = temp_size[13, 0]/2
                temp_size[13, 1] = temp_size[13, 1]/2
                temp_pos[13, :] = temp_pos[12, :]

            self.model.geom_size = temp_size
            self.model.geom_pos = temp_pos

        elif self.task is None:
            pass

        else:
            raise NotImplementedError

        self.model.forward()

"""
if __name__ == '__main__':
    env = AntEnv(task='cripple')
    while True:
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()
"""