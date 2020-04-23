from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


TARGET = np.array([0.13345871, 0.21923056, -0.10861196])
THRESH = 0.05
HORIZON = 150

class ReacherSparse3DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.viewer, self.time = None, 0
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.goal = np.copy(TARGET)
        self._max_episode_steps = HORIZON
        # self.obstacle = ReacherObstacle(np.array([0.5, 0.2, 0]), 0.15)
        self.obstacle = ReacherEEObstacle(np.array([0.5, 0.2, 0]), 0.15)
        self.transition_function = get_random_transitions
        mujoco_env.MujocoEnv.__init__(self, os.path.join(dir_path, 'assets/reacher3d.xml'), 2)

    def _step(self, a):
        # a = self.process_action(a)
        old_state = self._get_obs().copy()
        # if not self.obstacle(self.get_EE_pos(old_state[None])):
        self.do_simulation(a, self.frame_skip)
        self.time += 1
        ob = self._get_obs().copy()
        obs_cost = np.sum(np.square(self.get_EE_pos(ob[None]) - self.goal))
        ctrl_cost = 0.01 * np.square(a).sum()
        cost = obs_cost + ctrl_cost

        if obs_cost < 0.05:
            print("goal")
        done = HORIZON <= self.time
        return ob, -cost, done, {
                "constraint": self.obstacle(self.get_EE_pos(ob[None])),
                "reward": -cost,
                "state": old_state,
                "next_state": ob,
                "action": a}

    def process_action(self, action):
        return action

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 2.5
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 270

    def reset_model(self):
        qpos, qvel = np.copy(self.init_qpos), np.copy(self.init_qvel)
        # qpos[-3:] += np.random.normal(loc=0, scale=0.1, size=[3])
        qvel[-3:] = 0
        self.time = 0
        # self.goal = qpos[-3:]
        qpos[-3:] = self.goal = np.copy(TARGET)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat[:-3],
        ])

    def get_EE_pos(self, states):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4], states[:, 4:5], states[:, 5:6], states[:, 6:]

        rot_axis = np.concatenate([np.cos(theta2) * np.cos(theta1), np.cos(theta2) * np.sin(theta1), -np.sin(theta2)],
                                  axis=1)
        rot_perp_axis = np.concatenate([-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1)
        cur_end = np.concatenate([
            0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
            0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
            -0.4 * np.sin(theta2)
        ], axis=1)

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30] = \
                rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis

        return cur_end

    def is_stable(self, ob):
        return (np.sum(np.square(self.get_EE_pos(ob[None]) - self.goal)) < THRESH).astype(bool)

def get_random_transitions(num_transitions, task_demos=False):
    env = ReacherSparse3DEnv()
    transitions = []
    task_transitions = []
    done = False
    for i in range(num_transitions):
        state = env.reset()
        action = np.random.randn(7)
        next_state, reward, done, info = env.step(action)
        constraint = info['constraint']
        transitions.append((state, action, reward, next_state, done))

    if not task_demos:
        return transitions
    else:
        return transitions, task_transitions

class ReacherEEObstacle:
    def __init__(self, center=[0., 0, 0], radius=0.1):
        self.center = np.array(center)
        self.radius = radius

    def __call__(self, x):
        return np.linalg.norm(x - self.center) <= self.radius

class ReacherObstacle:
    def __init__(self, center=[0., 0, 0], radius=0.1, arm_size=0.09, penalty=1):
        # spherical obstacle
        # self.center = tf.convert_to_tensor(center, dtype=tf.dtypes.float32)
        self.center = np.array(center)
        self.radius = radius
        self.collision_radius = radius + arm_size

    def __call__(self, x):
        x = x[:,:,:, :7]
        x_reshaped = tf.reshape(x, (-1, 7))
        bools = tf.zeros(shape[:1], dtype=tf.dtypes.bool)
        points = self.reacher_points(x_reshaped)
        for i in range(1, len(points)):
            v1 = (points[i] - points[i-1])[:, :3]
            v2 = points[i-1][:, :3] - self.center
            v2_other = points[i][:, :3] - self.center
            lambda_num = -tf.reduce_sum(tf.multiply(v1, v2), axis=1)
            lambda_denom = tf.multiply(tf.norm(v1, axis=1), tf.norm(v1, axis=1))
            v3 = tf.cross(v1, v2)
            shortest_dists = tf.norm(v3, axis=1) / tf.norm(v1, axis=1)
            shortest_in_segment = tf.logical_and(lambda_num > 0, lambda_num < lambda_denom)
            actual_dist = tf.multiply(tf.dtypes.cast(shortest_in_segment, tf.dtypes.float32), shortest_dists) + tf.multiply(tf.dtypes.cast(tf.logical_not(shortest_in_segment), tf.dtypes.float32), tf.minimum( tf.norm(v2, axis=1), tf.norm(v2_other, axis=1) ))
            
            #bools = tf.logical_or(bools, curr_bools)
            bools = tf.logical_or(bools, actual_dist < self.collision_radius)
            #print(bools.numpy())
        bools_reshaped = tf.dtypes.cast(tf.reshape(bools, tf.shape(x)[:3]), tf.dtypes.float32)
        return bools_reshaped

    @staticmethod
    def reacher_points(state):
        # state.shape should equal (-1, 7)
        #import ipdb; ipdb.set_trace()
        points = [
                [0, -0.188, 0, 1],
                [0.1, -0.188, 0, 1],
                [0.5, -0.188, 0, 1],
                [0.821, -0.188, 0, 1],
                [1.021, -0.188, 0, 1]
                ]
        points = [
                [0, 0., 0, 1],
                [0, 0., 0, 1],
                [0, 0., 0, 1],
                [0, 0., 0, 1],
                [0, 0., 0, 1]
                ]

        with tf.name_scope('p0'):
            transform = TF_FK.translate(state[:, 0], [0, -0.188, 0]) # (-1, 4, 4)
            points[0] = tf.tensordot(transform, points[0], axes=[[2], [0]]) # (-1, 4, 4) @ (4,)

        with tf.name_scope('p1'):
            transform = transform @ TF_FK.rot_z(state[:, 0]) 
            transform = transform @ TF_FK.translate(state[:, 0], [0.1, 0, 0]) 
            points[1] = tf.tensordot(transform, points[1], axes=[[2], [0]]) # (-1, 4, 4) @ (4,)

        with tf.name_scope('p2'):
            transform = transform @ TF_FK.rot_y(state[:, 1]) 
            transform = transform @ TF_FK.rot_x(state[:, 2]) 
            transform = transform @ TF_FK.translate(state[:, 0], [0.4, 0, 0]) 
            points[2] = tf.tensordot(transform, points[2], axes=[[2], [0]]) # (-1, 4, 4) @ (4,)

        with tf.name_scope('p3'):
            transform = transform @ TF_FK.rot_y(state[:, 3]) 
            transform = transform @ TF_FK.rot_x(state[:, 4]) 
            transform = transform @ TF_FK.translate(state[:, 0], [0.321, 0, 0]) 
            points[3] = tf.tensordot(transform, points[3], axes=[[2], [0]]) # (-1, 4, 4) @ (4,)

        with tf.name_scope('p4'):
            transform = transform @ TF_FK.rot_y(state[:, 5]) 
            transform = transform @ TF_FK.rot_x(state[:, 6]) 
            transform = transform @ TF_FK.translate(state[:, 0], [0.2, 0, 0]) 
            points[4] = tf.tensordot(transform, points[4], axes=[[2], [0]]) # (-1, 4, 4) @ (4,)
        return points

class TF_FK():
    @staticmethod
    def _transpose_correct(mat):
        # turn (4x4x10000) to (10000x4x4)
        return tf.transpose(mat, perm=[2, 0, 1])

    @staticmethod
    def rot_x(theta):
        return TF_FK._transpose_correct(tf.convert_to_tensor([
            [tf.ones_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta)], 
            [tf.zeros_like(theta), tf.cos(theta), -tf.sin(theta), tf.zeros_like(theta)], 
            [tf.zeros_like(theta), tf.sin(theta), tf.cos(theta), tf.zeros_like(theta)], 
            [tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)]]))    

    @staticmethod
    def rot_y(theta):
        return TF_FK._transpose_correct(tf.convert_to_tensor([
            [tf.cos(theta), tf.zeros_like(theta), tf.sin(theta), tf.zeros_like(theta)], 
            [tf.zeros_like(theta), tf.ones_like(theta), tf.zeros_like(theta), tf.zeros_like(theta)], 
            [-tf.sin(theta), tf.zeros_like(theta), tf.cos(theta), tf.zeros_like(theta)], 
            [tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)]]))

    @staticmethod
    def rot_z(theta):
        return TF_FK._transpose_correct(tf.convert_to_tensor([
            [tf.cos(theta), -tf.sin(theta), tf.zeros_like(theta), tf.zeros_like(theta)], 
            [tf.sin(theta), tf.cos(theta), tf.zeros_like(theta), tf.zeros_like(theta)], 
            [tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta), tf.zeros_like(theta)], 
            [tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)]]))

    @staticmethod
    def translate(theta, amount):
        return TF_FK._transpose_correct(tf.convert_to_tensor([
            [tf.ones_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), amount[0] * tf.ones_like(theta)], 
            [tf.zeros_like(theta), tf.ones_like(theta), tf.zeros_like(theta), amount[1] * tf.ones_like(theta)], 
            [tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta), amount[2] * tf.ones_like(theta)], 
            [tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)]]))

    @staticmethod
    def translate_meta(amount):
        return lambda theta: TF_FK.translate(theta, amount)


if __name__ == '__main__':
    import time
    env = ReacherSparse3DEnv()
    env.reset()
    env.render()

    for i in range(10):
        env.step(np.random.randn(7))
        env.render()
        time.sleep(0.2)
