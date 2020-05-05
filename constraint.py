import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import ValueNetwork, QNetworkConstraint, hard_update, soft_update
from replay_memory import ReplayMemory
from utils import soft_update
import os.path as osp

class ValueFunction:
    def __init__(self, params):
        self.gamma_safe = params.gamma_safe
        self.device = params.device
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        self.model = ValueNetwork(params.state_dim, params.hidden_size, params.pred_time).to(self.device)
        self.target = ValueNetwork(params.state_dim, params.hidden_size, params.pred_time).to(self.device)
        self.tau = params.tau_safe
        self.logdir = params.logdir
        self.pred_time = params.pred_time
        self.env_name = params.env_name
        self.opt = params.opt

        if not params.use_target:
            self.tau = 1.
        hard_update(self.target, self.model)

    def train(self, ep, memory, pi=None, lr=0.0003, batch_size=1000, training_iterations=3000, plot=False):
        optim = Adam(self.model.parameters(), lr=lr)

        for j in range(training_iterations):
            state_batch, action_batch, constraint_batch, next_state_batch, _ = memory.sample(batch_size=batch_size)

            with torch.no_grad():
                if self.pred_time:
                    target = (self.gamma_safe * self.target(self.torchify(next_state_batch))[:,0] + 1) * (1 - self.torchify(constraint_batch) )
                else:
                    target = self.torchify(constraint_batch) + self.gamma_safe * self.target(self.torchify(next_state_batch))[:,0] * (1 - self.torchify(constraint_batch) )
            preds = self.model(self.torchify(state_batch))[:,0]
            optim.zero_grad()
            loss = F.mse_loss(preds, target)
            loss.backward()
            optim.step()
            loss = loss.detach().cpu().numpy()
            if j % 100 == 0:
                with torch.no_grad():
                    print("Value Training Iteration %d    Loss: %f"%(j, loss))
            soft_update(self.target, self.model, self.tau)

        if plot:
            if self.env_name == 'maze':
                x_bounds = [-0.3, 0.3]
                y_bounds = [-0.3, 0.3]
            elif self.env_name == 'simplepointbot0':
                x_bounds = [-80, 20]
                y_bounds = [-10, 10]
            elif self.env_name == 'simplepointbot1':
                x_bounds = [-75, 25]
                y_bounds = [-75, 25]
            else:
                raise NotImplementedError("Plotting unsupported for this env")

            states = []
            x_pts = 100
            y_pts = int(x_pts*(x_bounds[1] - x_bounds[0])/(y_bounds[1] - y_bounds[0]) )
            for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
                for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                    states.append([x, y])

            if not self.opt:
                grid = self.model(self.torchify(np.array(states))).detach().cpu().numpy()
                grid = grid.reshape(y_pts, x_pts)
            else:
                raise(NotImplementedError("Need to implement opt"))

            plt.imshow(grid.T)
            plt.savefig(osp.join(self.logdir, "value_" + str(ep)))

    def get_value(self, states, actions=None):
        return self.model(states)


class QFunction:
    def __init__(self, params):
        self.gamma_safe = params.gamma_safe
        self.device = params.device
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        self.ac_dim = params.ac_space.shape[0]
        self.action_space = params.ac_space
        self.model = QNetworkConstraint(params.state_dim, self.ac_dim, params.hidden_size).to(self.device)
        self.model_target = QNetworkConstraint(params.state_dim, self.ac_dim, params.hidden_size).to(self.device)
        self.tau = params.tau
        self.logdir = params.logdir
        self.env_name = params.env_name 
        self.opt = params.opt

    def train(self, ep, memory, pi, lr=0.0003, batch_size=1000, training_iterations=3000, plot=False, num_eval_actions=100):
        optim = Adam(self.model.parameters(), lr=lr)
        for j in range(training_iterations):
            state_batch, action_batch, constraint_batch, next_state_batch, _ = memory.sample(batch_size=batch_size)
            state_batch = self.torchify(state_batch)
            action_batch = self.torchify(action_batch)
            constraint_batch = self.torchify(constraint_batch)
            next_state_batch = self.torchify(next_state_batch)

            with torch.no_grad():
                if not self.opt:
                    next_state_action = pi(next_state_batch)
                    # if ep > 0:
                    #     next_state_action = pi(next_state_batch)
                    # else: # When training on  demo transitions, just learn Q for a random policy
                    #     next_state_action = self.torchify(np.array([self.action_space.sample() for _ in range(batch_size)]))
                    qf1_next_target, qf2_next_target = self.model_target(next_state_batch, next_state_action)
                else:
                    eval_next_states = next_state_batch.repeat(num_eval_actions, 1)
                    eval_next_actions = np.tile(np.array([self.action_space.sample() for _ in range(batch_size)]), (num_eval_actions, 1))
                    eval_next_actions = self.torchify(eval_next_actions)
                    eval_qf1_next_target, eval_qf2_next_target = self.model_target(eval_next_states, eval_next_actions)
                    eval_qf1_next_target = torch.reshape(eval_qf1_next_target, (num_eval_actions, batch_size, -1))
                    eval_qf2_next_target = torch.reshape(eval_qf2_next_target, (num_eval_actions, batch_size, -1))
                    qf1_next_target, _ = torch.max(eval_qf1_next_target, 0, keepdim=False)
                    qf2_next_target, _ = torch.max(eval_qf2_next_target, 0, keepdim=False)

                max_qf_next_target = torch.max(qf1_next_target, qf2_next_target)
                next_qf = constraint_batch.unsqueeze(1) + self.gamma_safe * max_qf_next_target * (1 - constraint_batch.unsqueeze(1) )

            # qf1, qf2 = self.model(state_batch, action_batch)
            qf1, qf2 = self.model(state_batch, pi(state_batch))
            max_qf = torch.max(qf1, qf2)
            qf1_loss = F.mse_loss(qf1, next_qf)
            qf2_loss = F.mse_loss(qf2, next_qf)

            optim.zero_grad()
            qf1_loss.backward(retain_graph=True)
            optim.step()

            optim.zero_grad()
            qf2_loss.backward()
            optim.step()

            if j % 100 == 0:
                with torch.no_grad():
                    print("Q-Value Training Iteration %d    Losses: %f, %f"%(j, qf1_loss, qf2_loss))
            soft_update(self.model_target, self.model, self.tau)


        if plot:
            self.plot(pi, ep, [1, 0], "right")
            self.plot(pi, ep, [-1, 0], "left")
            self.plot(pi, ep, [0, 1], "up")
            self.plot(pi, ep, [0, -1], "down")


    def plot(self, pi, ep, action=None, suffix=""):
        if self.env_name == 'maze':
            x_bounds = [-0.3, 0.3]
            y_bounds = [-0.3, 0.3]
        elif self.env_name == 'simplepointbot0':
            x_bounds = [-80, 20]
            y_bounds = [-10, 10]
        elif self.env_name == 'simplepointbot1':
            x_bounds = [-75, 25]
            y_bounds = [-75, 25]
        else:
            raise NotImplementedError("Plotting unsupported for this env")

        states = []
        x_pts = 100
        y_pts = int(x_pts*(x_bounds[1] - x_bounds[0])/(y_bounds[1] - y_bounds[0]) )
        for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
            for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                states.append([x, y])

        num_states = len(states)
        states = self.torchify(np.array(states))
        if action is None:
            actions = pi(states)
        else:
            actions = self.torchify(np.tile(action, (len(states), 1)))
        # if ep > 0:
        #     actions = pi(states)
        # else:
        #     actions = self.torchify(np.array([self.action_space.sample() for _ in range(num_states)]))

        qf1, qf2 = self.model(states, actions)
        max_qf = torch.max(qf1, qf2)
        grid = max_qf.detach().cpu().numpy()
        grid = grid.reshape(y_pts, x_pts)
        plt.imshow(grid.T)
        plt.savefig(osp.join(self.logdir, "qvalue_" + str(ep) + suffix))



    def get_value(self, states, actions):
        with torch.no_grad():
            q1, q2 = self.model(states, actions)
            return torch.max(q1, q2)


