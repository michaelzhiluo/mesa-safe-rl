"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise. State
representation is (x, y). Action representation is (dx, dy).
"""

import os
import pickle

import os.path as osp
import numpy as np
# import matplotlib.pyplot as plt
from gym import Env
from gym import utils
from gym.spaces import Box

from obstacle import Obstacle, ComplexObstacle

"""
Constants associated with the PointBot env.
"""

START_POS = [-50, 0]
END_POS = [0, 0]
GOAL_THRESH = 1.
START_STATE = START_POS
GOAL_STATE = END_POS

MAX_FORCE = 1
HORIZON = 100

NOISE_SCALE = 0.05
AIR_RESIST = 0.2

HARD_MODE = False

OBSTACLE = [
        [[-100, 150], [5, 10]],
        [[-100, -80], [-10, 10]],
        [[-100, 150],[-10, -5]]]


CAUTION_ZONE = [
        [[-100, 150], [4, 5]],
        [[-100, 150], [-5, -4]]]


        
OBSTACLE = ComplexObstacle(OBSTACLE)
CAUTION_ZONE = ComplexObstacle(CAUTION_ZONE)



def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)

def teacher_action(state, goal):
    disp = np.subtract(goal, state)
    disp[disp > MAX_FORCE] = MAX_FORCE
    disp[disp < -MAX_FORCE] = -MAX_FORCE
    return disp



class SimplePointBot(Env, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.A = np.eye(2)
        self.B = np.eye(2)
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(2) * MAX_FORCE, np.ones(2) * MAX_FORCE)
        self.observation_space = Box(-np.ones(2) * np.float('inf'), np.ones(2) * np.float('inf'))
        self._max_episode_steps = HORIZON
        self.obstacle = OBSTACLE
        self.caution_zone = CAUTION_ZONE
        self.transition_function = get_random_transitions

    def step(self, a):
        a = process_action(a)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a)
        cur_cost = self.step_cost(self.state, a)
        self.cost.append(cur_cost)
        self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.done = HORIZON <= self.time

        return self.state, cur_cost, self.done, {
                "constraint": self.obstacle(next_state),
                "reward": cur_cost,
                "state": old_state,
                "next_state": next_state,
                "action": a}

    def reset(self):
        self.state = START_STATE + np.random.randn(2)
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        return self.state

    def _next_state(self, s, a, override=False):
        if self.obstacle(s):
            print("obs", s)
            return s
        # if self.caution_zone(s) and not override:
        #     if np.abs(a - safe_action(s, GOAL_STATE)).max() > 0.001:
        #         print(s, a, safe_action(s, GOAL_STATE))
            # print(a, safe_action(s, GOAL_STATE))
            # a = safe_action(s, GOAL_STATE)
        return self.A.dot(s) + self.B.dot(a) + NOISE_SCALE * np.random.randn(len(s))

    def step_cost(self, s, a):
        if HARD_MODE:
            return int(np.linalg.norm(np.subtract(GOAL_STATE, s)) < GOAL_THRESH)
        return -np.linalg.norm(np.subtract(GOAL_STATE, s))

    def values(self):
        return np.cumsum(np.array(self.cost)[::-1])[::-1]

    def sample(self):
        """
        samples a random action from the action space.
        """
        return np.random.random(2) * 2 * MAX_FORCE - MAX_FORCE

    def plot_trajectory(self, states=None):
        if states == None:
            states = self.hist
        states = np.array(states)
        plt.scatter(states[:,0], states[:,2])
        plt.show()

    # Returns whether a state is stable or not
    def is_stable(self, s):
        return np.linalg.norm(np.subtract(GOAL_STATE, s)) <= GOAL_THRESH

    def teacher(self, sess=None):
        return SimplePointBotTeacher()

    def expert_action(self, s):
        return self.teacher._expert_control(s, 0)

    def safe_action(self, s):
        return safe_action(s)


def get_random_transitions(num_transitions):
    env = SimplePointBot()
    transitions = []
    for i in range(num_transitions):
        if np.random.uniform(0, 1) < 0.5:
            state = np.random.uniform(-80, 50), np.random.uniform(-6, -2)
        else:
            state = np.random.uniform(-80, 50), np.random.uniform(2, 6)
        action = np.clip(np.random.randn(2), -1, 1)
        next_state = env._next_state(state, action, override=True)
        constraint = env.obstacle(next_state)
        transitions.append((state, action, constraint, next_state))
    return transitions


def safe_action(state, goal=GOAL_STATE):
    disp = np.subtract(goal, state)
    disp[disp > MAX_FORCE] = MAX_FORCE
    disp[disp < -MAX_FORCE] = -MAX_FORCE
    disp[0] = 0
    return disp * 0.25

def teacher_action(state, goal=GOAL_STATE):
    disp = np.subtract(goal, state)
    disp[disp > MAX_FORCE] = MAX_FORCE
    disp[disp < -MAX_FORCE] = -MAX_FORCE
    return disp

class SimplePointBotTeacher(object):

    def __init__(self):
        self.env = SimplePointBot()
        self.demonstrations = []
        self.outdir = "data/simplepointbot"
        self.goal = GOAL_STATE

    def _generate_trajectory(self):
        """
        The teacher initially tries to go northeast before going to the origin
        """
        transitions = []
        state = self.env.reset()
        for i in range(HORIZON):
            # if i < HORIZON / 2:
            #     action = [0.1, 0.1]
            # else:
            action = self._expert_control(state, i)
            next_state, cost, done, _ = self.env.step(action)
            transitions.append([state, action, cost, next_state, done])
            state = next_state
        assert done, "Did not reach the goal set on task completion."
        V = self.env.values()
        for i, t in enumerate(transitions):
            t.append(V[i])
        # self.env.plot_trajectory()
        return transitions


    def generate_demonstrations(self, num_demos):
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        else:
            raise RuntimeError("Directory %s already exists." % (self.outdir))
        for i in range(num_demos):
            if i % 100 == 0:
                print("Generating Demos: Iteration %d" % i)
            demo = self._generate_trajectory()
            with open(osp.join(self.outdir, "%d.pkl" % (i)), "wb") as f:
                pickle.dump(demo, f)
            self.demonstrations.append(demo)

    def _get_gain(self, t):
        return self.Ks[t]

    def _expert_control(self, s, t):
        return teacher_action(s, self.goal)

if __name__ == '__main__':
    env = SimplePointBot()
    obs = env.reset()
    env.step([1,1])

    for i in range(HORIZON-1):
        env.step([0,0])

    teacher = env.teacher()
    teacher.generate_demonstrations(1000)
    # env.plot_trajectory()

