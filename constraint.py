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
        if not params.use_target:
            self.tau = 1.
        hard_update(self.target, self.model)

    def train(self, ep, memory, epochs=50, lr=1e-3, batch_size=1000, training_iterations=3000, plot=False):
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
            pts = []
            for i in range(100):
                x = -50 + i
                for j in range(100):
                    y = -50 + j
                    pts.append([x, y])
            grid = self.model(self.torchify(np.array(pts))).detach().cpu().numpy().reshape(-1, 100).T

            plt.imshow(grid)
            plt.show()
            assert 0
            plt.imshow(grid)
            plt.savefig(osp.join(self.logdir, "value_" + str(ep)))

    def get_value(self, states):
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

    def train(self, ep, memory, pi, epochs=50, lr=1e-3, batch_size=1000, training_iterations=3000, plot=False):
        optim = Adam(self.model.parameters(), lr=lr)
        for j in range(training_iterations):
            state_batch, action_batch, constraint_batch, next_state_batch, _ = memory.sample(batch_size=batch_size)
            state_batch = self.torchify(state_batch)
            action_batch = self.torchify(action_batch)
            constraint_batch = self.torchify(constraint_batch)
            next_state_batch = self.torchify(next_state_batch)

            with torch.no_grad():
                next_state_action = pi(next_state_batch)
                qf1_next_target, qf2_next_target = self.model_target(next_state_batch, next_state_action)
                max_qf_next_target = torch.max(qf1_next_target, qf2_next_target)
                next_qf = constraint_batch.unsqueeze(1) + self.gamma_safe * max_qf_next_target * (1 - constraint_batch.unsqueeze(1) )

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

        if plot:
            num_eval_actions = 100
            # TODO: plotting code
            states = []
            actions = []
            for i in range(100):
                x = -75 + i
                for j in range(20):
                    y = -10 + j
                    for _ in range(num_eval_actions):
                        states.append([x, y])
                        actions.append(self.action_space.sample())

            states = self.torchify(np.array(states))
            actions = self.torchify(np.array(actions))
            qf1, qf2 = self.model(states, actions)
            max_qf = torch.max(qf1, qf2)
            grid = max_qf.detach().cpu().numpy().reshape(100, 20, -1)
            grid = np.mean(grid, axis=-1).T
            # plt.imshow(grid > 0.8)
            # plt.show()
            plt.imshow(grid)
            plt.show()
            plt.savefig(osp.join(self.logdir, "qvalue_" + str(ep)))

        soft_update(self.model_target, self.model, self.tau)

    def get_qvalue(self, states, actions):
        with torch.no_grad():
            q1, q2 = self.model(states, actions)
            return torch.max(q1, q2)

