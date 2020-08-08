from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import gym

import torch
from torch import nn as nn
from torch.nn import functional as F

from config.utils import swish, get_affine_params
from DotmapUtils import get_required_argument

TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class PtModel(nn.Module):
    def __init__(self, ensemble_size, in_features, out_features):
        super().__init__()

        self.num_nets = ensemble_size

        self.in_features = in_features
        self.out_features = out_features

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size,
                                                     in_features, 200)

        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, 200, 200)

        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, 200, 200)

        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, 200,
                                                     out_features)

        self.inputs_mu = nn.Parameter(
            torch.zeros(in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(
            torch.zeros(in_features), requires_grad=False)

        self.max_logvar = nn.Parameter(
            torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(
            -torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

    def compute_decays(self):

        lin0_decays = 0.00025 * (self.lin0_w**2).sum() / 2.0
        lin1_decays = 0.0005 * (self.lin1_w**2).sum() / 2.0
        lin2_decays = 0.0005 * (self.lin2_w**2).sum() / 2.0
        lin3_decays = 0.00075 * (self.lin3_w**2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def fit_input_stats(self, data):

        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(
            TORCH_DEVICE).float()

    def forward(self, inputs, ret_logvar=False):

        # Transform inputs
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b

        mean = inputs[:, :, :self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)


class ShelfLongConfigModule:
    ENV_NAME = "ShelfLong-v0"
    TASK_HORIZON = 150
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 15
    MODEL_IN, MODEL_OUT = 31, 27
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.ENV.reset()
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }
        self.UPDATE_FNS = [self.update_goal]

        self.goal = None

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def update_goal(self):
        self.goal = self.ENV.goal

    def obs_cost_fn(self, obs):
        pass

    @staticmethod
    def ac_cost_fn(acs):
        return 0.0 * (acs**2).sum(dim=1)  # could do 0.01

    def nn_constructor(self, model_init_cfg):
        ensemble_size = get_required_argument(model_init_cfg, "num_nets",
                                              "Must provide ensemble size")

        load_model = model_init_cfg.get("load_model", False)

        assert load_model is False, 'Has yet to support loading model'

        model = PtModel(ensemble_size, self.MODEL_IN,
                        self.MODEL_OUT * 2).to(TORCH_DEVICE)
        # * 2 because we output both the mean and the variance

        model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

        return model


CONFIG_MODULE = ShelfLongConfigModule
