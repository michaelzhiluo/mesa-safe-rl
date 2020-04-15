import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import ValueNetwork, QNetwork, hard_update, soft_update
from replay_memory import ReplayMemory

class ValueFunction:
    def __init__(self, params):
        self.gamma_safe = params.gamma_safe
        self.device = params.device
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        self.model = ValueNetwork(params.hidden_dim, params.hidden_size).to(self.device)
        self.target = ValueNetwork(params.hidden_dim, params.hidden_size).to(self.device)
        self.tau = params.tau_safe
        if not params.use_target:
            self.tau = 1.
        hard_update(self.target, self.model)

    def train(self, memory, epochs=50, lr=1e-3, batch_size=1000, training_iterations=3000, plot=False):
        optim = Adam(self.model.parameters(), lr=lr)

        for j in range(training_iterations):
            state_batch, action_batch, constraint_batch, next_state_batch, _ = memory.sample(batch_size=batch_size)

            with torch.no_grad():
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
            for i in range(1000):
                x = -75 + i * 0.1
                for j in range(1000):
                    y = -50 + j * 0.1
                    pts.append([x, y])
            grid = self.model(self.torchify(np.array(pts))).detach().cpu().numpy().reshape(-1, 1000).T

            plt.imshow(grid > 0.8)
            plt.show()
            plt.imshow(grid)
            plt.show()

    def get_value(self, states):
        return self.model(states)