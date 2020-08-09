import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from matplotlib.patches import Rectangle
from PIL import Image

from model import ValueNetwork, QNetworkConstraint, hard_update, soft_update
from replay_memory import ReplayMemory
from utils import soft_update
import os.path as osp


class ValueFunction:
    def __init__(self, params):
        self.gamma_safe = params.gamma_safe
        self.device = params.device
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        self.model = ValueNetwork(params.state_dim, params.hidden_size,
                                  params.pred_time).to(self.device)
        self.target = ValueNetwork(params.state_dim, params.hidden_size,
                                   params.pred_time).to(self.device)
        self.tau = params.tau_safe
        self.logdir = params.logdir
        self.pred_time = params.pred_time
        self.env_name = params.env_name
        self.opt = params.opt

        if not params.use_target:
            self.tau = 1.
        hard_update(self.target, self.model)

    def train(self,
              ep,
              memory,
              pi=None,
              lr=0.0003,
              batch_size=1000,
              training_iterations=3000,
              plot=False):
        optim = Adam(self.model.parameters(), lr=lr)

        for j in range(training_iterations):
            state_batch, action_batch, constraint_batch, next_state_batch, _ = memory.sample(
                batch_size=batch_size)

            with torch.no_grad():
                if self.pred_time:
                    target = (self.gamma_safe * self.target(
                        self.torchify(next_state_batch))[:, 0] + 1) * (
                            1 - self.torchify(constraint_batch))
                else:
                    target = self.torchify(
                        constraint_batch) + self.gamma_safe * self.target(
                            self.torchify(next_state_batch))[:, 0] * (
                                1 - self.torchify(constraint_batch))
            preds = self.model(self.torchify(state_batch))[:, 0]
            optim.zero_grad()
            loss = F.mse_loss(preds, target)
            loss.backward()
            optim.step()
            loss = loss.detach().cpu().numpy()
            if j % 100 == 0:
                with torch.no_grad():
                    print(
                        "Value Training Iteration %d    Loss: %f" % (j, loss))
            soft_update(self.target, self.model, self.tau)

        if plot:
            self.plot(ep)

    def plot(self, ep):
        if self.env_name == 'maze' or self.env_name == 'image_maze':
            x_bounds = [-0.3, 0.3]
            y_bounds = [-0.3, 0.3]
        elif self.env_name == 'simplepointbot0':
            x_bounds = [-80, 20]
            y_bounds = [-10, 10]
        elif self.env_name == 'simplepointbot1':
            x_bounds = [-75, 25]
            y_bounds = [-75, 25]
        elif self.env_name == 'car':
            x_bounds = [0, 20]
            y_bounds = [-5, 5]
        else:
            raise NotImplementedError("Plotting unsupported for this env")

        states = []
        x_pts = 100
        y_pts = int(
            x_pts * (x_bounds[1] - x_bounds[0]) / (y_bounds[1] - y_bounds[0]))
        for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
            for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                if self.env_name != 'car':
                    states.append([x, y])
                else:
                    for i in range(100):
                        v = np.random.random(
                        ) * 2 - 1  # random velocities on [-1, 1]
                        states.append([x, y, v])

        if not self.opt:
            if self.env_name != 'car':
                grid = self.model(self.torchify(
                    np.array(states))).detach().cpu().numpy()
                grid = grid.reshape(y_pts, x_pts)
            else:
                grid = []
                for i in range(x_pts * y_pts):
                    grid.append(
                        self.model(self.torchify(np.array(
                            states[i:i + 100]))).detach().cpu().numpy())
                grid = np.array(grid)
                grid = grid.squeeze()
                grid = np.mean(grid, axis=-1)
                grid = grid.reshape((y_pts, x_pts))
        else:
            raise (NotImplementedError("Need to implement opt"))

        if self.env_name == 'simplepointbot0':
            plt.gca().add_patch(
                Rectangle(
                    (0, 25),
                    500,
                    50,
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'))
        elif self.env_name == 'simplepointbot1':
            plt.gca().add_patch(
                Rectangle(
                    (45, 65),
                    10,
                    20,
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'))

        plt.imshow(grid.T)
        plt.savefig(osp.join(self.logdir, "value_" + str(ep)))

    def get_value(self, states, actions=None):
        return self.model(states)


