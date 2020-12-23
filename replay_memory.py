'''
Built on on SAC implementation from 
https://github.com/pranz24/pytorch-soft-actor-critic
'''

import random
import numpy as np
from operator import itemgetter


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # print(batch_size, len(self.buffer))
        # import IPython; IPython.embed()
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ConstraintReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.pos_idx = np.zeros(self.capacity)

    def push(self, state, action, reward, next_state, done, mc_reward=None, online_violation=False):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, mc_reward)
        self.pos_idx[self.position] = reward
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, pos_fraction=None):
        if pos_fraction is not None:
            pos_size = int(batch_size * pos_fraction)
            neg_size = batch_size - pos_size
            pos_idx = np.array(
                random.sample(
                    tuple(np.argwhere(self.pos_idx).ravel()), pos_size))
            neg_idx = np.array(
                random.sample(
                    tuple(
                        np.argwhere(
                            (1 - self.pos_idx)[:len(self.buffer)]).ravel()),
                    neg_size))
            idx = np.hstack((pos_idx, neg_idx))
            batch = itemgetter(*idx)(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mc_reward = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, mc_reward

    def __len__(self):
        return len(self.buffer)
