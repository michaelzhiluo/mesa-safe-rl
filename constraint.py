from model import ValueNetwork, QNetwork
from simplepointbot import SimplePointBot, get_random_transitions

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt


def get_value_function():

    torchify = lambda x: torch.FloatTensor(x).to('cuda')


    # critic = QNetwork(2, 2, 200)
    # critic_target = QNetwork(2, 2, 200)

    env = SimplePointBot()
    data = get_random_transitions(10000)

    train_data = data[:-1000]
    test_data = data[-1000:]

    model = ValueNetwork(2, 200).to('cuda')
    optim = Adam(model.parameters(), lr=1e-3)
    # optim = Adam(critic.parameters(), lr=1e-3)


    for j in range(3000):

        train_idx = np.random.randint(0, len(train_data), 1000)
        train_states = np.array([data[idx][0] for idx in train_idx])
        train_actions = np.array([data[idx][1] for idx in train_idx])
        train_constraints = np.array([data[idx][2] for idx in train_idx])
        train_next_states = np.array([data[idx][3] for idx in train_idx])


        test_idx = np.arange(0, len(test_data)) + len(train_data)
        test_states = np.array([data[idx][0] for idx in test_idx])
        test_actions = np.array([data[idx][1] for idx in test_idx])
        test_constraints = np.array([data[idx][2] for idx in test_idx])
        test_next_states = np.array([data[idx][3] for idx in test_idx])


        # with torch.no_grad():
        #     qf1_next_target, qf2_next_target = critic_target(torchify(train_next_states), torchify(np.zeros_like(train_actions)))
        #     min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
        #     next_q_value = torchify(train_constraints) + 0.5 * (min_qf_next_target)

        # qf1, qf2 = critic(torchify(train_states), torchify(train_actions))

        # loss = F.mse_loss(qf1, qf1_next_target) + F.mse_loss(qf2, qf2_next_target)


        with torch.no_grad():
            target = torchify(train_constraints) + 0.9 * model(torchify(train_next_states)) * (1 - torchify(train_constraints) )
        # target = torchify(train_constraints)
        preds = model(torchify(train_states))[:,0]
        optim.zero_grad()
        loss = F.mse_loss(preds, target)
        loss.backward()
        optim.step()
        loss = loss.detach().cpu().numpy()
        if j % 100 == 0:
            with torch.no_grad():
                val_preds = model(torchify(test_states))
                val_targets = torchify(test_constraints) + 0.9 * model(torchify(test_next_states)) * (1 - torchify(test_constraints) )
                val_loss = F.mse_loss(val_preds, val_targets).detach().cpu().numpy()
                print("Value Training Iteration %d    Loss: %f"%(j, loss))
                print("Validation Loss %f"%val_loss)

    # pts = []
    # for i in range(1000):
    #     x = -75 + i * 0.1
    #     for j in range(200):
    #         y = -10 + j * 0.1
    #         pts.append([x, y])
    # grid = model(torchify(np.array(pts))).detach().cpu().numpy().reshape(-1, 200)

    # plt.imshow(grid > 0.8)
    # plt.show()
    # plt.imshow(grid)
    # plt.show()

    return model
