import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import ValueNetwork, QNetwork

# TODO: rewrite to make a value function class and to not hardcode things...
def get_value_function(gamma_safe, data_function, recovery_policy, learned_recovery, device='cuda', batch_size=1000, num_transitions=10000, training_iterations=3000, plot=False):

    torchify = lambda x: torch.FloatTensor(x).to(device)

    data = data_function(num_transitions)
    # TODO: may need to fix later, this works now because all the transition data is contiguous in time
    if learned_recovery:
        data_states = np.array([d[0] for d in data])
        data_actions = np.array([d[1] for d in data])
        data_next_states = np.array([d[3] for d in data])
        recovery_policy.train(data_states, data_actions, random=True, next_obs=data_next_states, epochs=50)

    test_size = int(len(data) * 0.1)
    train_size = len(data) - test_size

    train_data = data[:-1000]
    test_data = data[-1000:]

    model = ValueNetwork(data[0][0].shape[0], 200).to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    for j in range(training_iterations):

        train_idx = np.random.randint(0, train_size, batch_size)
        train_states = np.array([data[idx][0] for idx in train_idx])
        train_actions = np.array([data[idx][1] for idx in train_idx])
        train_constraints = np.array([data[idx][2] for idx in train_idx])
        train_next_states = np.array([data[idx][3] for idx in train_idx])


        test_idx = np.arange(0, test_size) + train_size
        test_states = np.array([data[idx][0] for idx in test_idx])
        test_actions = np.array([data[idx][1] for idx in test_idx])
        test_constraints = np.array([data[idx][2] for idx in test_idx])
        test_next_states = np.array([data[idx][3] for idx in test_idx])

        with torch.no_grad():
            target = torchify(train_constraints) + gamma_safe * model(torchify(train_next_states))[:,0] * (1 - torchify(train_constraints) )
        preds = model(torchify(train_states))[:,0]
        optim.zero_grad()
        loss = F.mse_loss(preds, target)
        loss.backward()
        optim.step()
        loss = loss.detach().cpu().numpy()
        if j % 100 == 0:
            with torch.no_grad():
                val_preds = model(torchify(test_states))
                val_targets = torchify(test_constraints) + gamma_safe * model(torchify(test_next_states))[:,0] * (1 - torchify(test_constraints) )
                val_loss = F.mse_loss(val_preds, val_targets).detach().cpu().numpy()
                print("Value Training Iteration %d    Loss: %f"%(j, loss))
                print("Validation Loss %f"%val_loss)

    if plot:
        pts = []
        for i in range(1000):
            x = -75 + i * 0.1
            for j in range(1000):
                y = -50 + j * 0.1
                pts.append([x, y])
        grid = model(torchify(np.array(pts))).detach().cpu().numpy().reshape(-1, 1000).T

        plt.imshow(grid > 0.8)
        plt.show()
        plt.imshow(grid)
        plt.show()

    return model
