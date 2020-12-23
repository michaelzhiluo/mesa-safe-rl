import argparse
from copy import deepcopy
from typing import List, Optional
import os
import itertools
import math
import random
import time
import json
import pickle
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os.path as osp
import numpy as np
import higher
import numpy as np
import torch
import torch.autograd as A
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
import torch.distributions as D
import cv2

class FreezeParameters:
    def __init__(self, parameters):
        self.parameters = parameters
        self.param_states = [p.requires_grad for p in self.parameters]

    def __enter__(self):
        for param in self.parameters:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.parameters):
            param.requires_grad = self.param_states[i]

class WLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias_size = None, paaa=None):
        super().__init__()

        self.pa = paaa
        if bias_size is None:
            bias_size = out_features

        dim = 100
        self.z = nn.Parameter(torch.empty(dim).normal_(0, 1. / out_features))
        self.fc = nn.Linear(dim, in_features * out_features + out_features)
        self.seq = self.fc
        self.w_idx = in_features * out_features
        self.weight = self.fc.weight
        self._linear = self.fc
        self.out_f = out_features

    def adaptation_parameters(self):
        return [self.z]

    def forward(self, x: torch.tensor):
        #theta = self.fc(self.z + torch.empty_like(self.z).normal_(0, 1. / self.out_f))
        theta = self.fc(self.z)
        w = theta[:self.w_idx].view(x.shape[-1], -1)
        b = theta[self.w_idx:]
        return x @ w + b

class Linear(nn.Linear):
    def adaptation_parameters(self):
        return list(self.parameters())

class MLP(nn.Module):
    def __init__(self, layer_widths, final_activation = lambda x: x, extra_head_layers = None, w_linear: bool = False, scale=1.0):
        super().__init__()

        if len(layer_widths) < 2:
            raise ValueError('Layer widths needs at least an in-dimension and out-dimension')

        self._final_activation = final_activation
        self.seq = nn.Sequential()
        self._head = extra_head_layers is not None
        self.scale = scale

        if not w_linear:
            linear = Linear
        else:
            linear = WLinear
        self.aparams = []

        for idx in range(len(layer_widths) - 1):
            w = linear(layer_widths[idx], layer_widths[idx + 1])
            self.aparams.extend(w.adaptation_parameters())
            self.seq.add_module(f'fc_{idx}', w)
            if idx < len(layer_widths) - 2:
                self.seq.add_module(f'relu_{idx}', nn.ReLU())

        if extra_head_layers is not None:
            self.pre_seq = self.seq[:-2]
            self.post_seq = self.seq[-2:]

            self.head_seq = nn.Sequential()
            extra_head_layers = [layer_widths[-2] + layer_widths[-1]] + extra_head_layers

            for idx, (infc, outfc) in enumerate(zip(extra_head_layers[:-1], extra_head_layers[1:])):
                self.head_seq.add_module(f'relu_{idx}', nn.ReLU())
                w = linear(extra_head_layers[idx], extra_head_layers[idx + 1])
                self.aparams.extend(w.adaptation_parameters())
                self.head_seq.add_module(f'fc_{idx}', w)

    def bias_parameters(self):
        return [self.seq[0].bias]

    def adaptation_parameters(self):
        return self.parameters()
        #return self.aparams
    
    def forward(self, x: torch.tensor, acts: Optional[torch.tensor] = None):
        if self._head and acts is not None:
            h = self.pre_seq(x)
            head_input = torch.cat((h,acts), -1)
            return self._final_activation(self.post_seq(h))*self.scale, self.head_seq(head_input)
        else:
            return self._final_activation(self.seq(x))*self.scale

class MAMLRAWR(object):
    def __init__(self, obs_space, ac_space, hidden_size, logdir, action_space, args, tmp_env):
        self.env_name = args.env_name
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.logdir = logdir
        self._args = args
        self.tmp_env = tmp_env
        self.gamma_safe = args.gamma_safe
        self.obs_space = obs_space
        self.ac_space = ac_space
        self.pos_fraction = args.pos_fraction if args.pos_fraction >=0 else None
        self.batch_size = 256
        self.inner_batch_size = 256
        self._observation_dim = obs_space.shape[0]
        self._action_dim = ac_space.shape[0]
        self.policy_head = [32, 1]
        self.net_width = 100
        self.net_depth = 3
        self.outer_value_lr = 0.0001#0.00001
        self.outer_policy_lr = 0.00005#0.0001
        self.lrlr = 0.001
        self.inner_policy_lr = 0.0003#0.001
        self.inner_value_lr = 0.0003#0.001
        self.num_tasks = 9
        self.task_batch_size = 5
        self.advantage_head_coef = 0.01
        self._adaptation_temperature = 1.0
        self._gradient_steps_per_iteration = 1
        self._advantage_clamp = np.log(20.0)
        self._action_sigma = 0.01
        self._grad_clip = 1e9
        self._env_seeds = np.random.randint(1e10, size=(int(1e7),))
        self._rollout_counter = 0
        self._maml_steps = 1
        self._max_maml_steps = 1
        self.updates = 0

        self.q_value = True

        import os
        try:
            os.makedirs(logdir + "/1")
            os.makedirs(logdir + "/5")
            os.makedirs(logdir + "/10")
            os.makedirs(logdir + "/20")
            os.makedirs(logdir + "/right")
            os.makedirs(logdir + "/left")
            os.makedirs(logdir + "/up")
            os.makedirs(logdir + "/down")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self._adaptation_policy = MLP([self._observation_dim] +
                                      [256]*2 +#[self.net_width] * self.net_depth +
                                      [self._action_dim],
                                      final_activation=torch.tanh,
                                      extra_head_layers=self.policy_head,
                                      w_linear=False,
                                      scale=ac_space.high[0]).to(self.device)

        if self.q_value:
            self._value_function = MLP([self._observation_dim + self._action_dim] +
                                   [self.net_width] * self.net_depth +
                                   [1],
                                   final_activation=torch.sigmoid,
                                   w_linear=True).to(self.device)
        else:
            self._value_function = MLP([self._observation_dim] +
                                       [self.net_width] * self.net_depth +
                                       [1],
                                       final_activation=torch.sigmoid,
                                       w_linear=True).to(self.device)

        print(self._adaptation_policy, self._value_function)

        # For Meta Update
        self._adaptation_policy_optimizer = O.Adam(self._adaptation_policy.parameters(), lr=self.outer_policy_lr)
        self._value_function_optimizer = O.Adam(self._value_function.parameters(), lr=self.outer_value_lr)

        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        self._policy_lrs = None
        self._value_lrs = None
        self._adv_coef = None

        # Buffer probably declared in main.py
        self._inner_buffers = None
        self._outer_buffers = None

        self._policy_lrs = [torch.nn.Parameter(torch.tensor(float(np.log(self.inner_policy_lr))).to(self.device))
            for p in self._adaptation_policy.adaptation_parameters()]
        self._value_lrs = [torch.nn.Parameter(torch.tensor(float(np.log(self.inner_policy_lr))).to(self.device))
                               for p in self._value_function.adaptation_parameters()]
        self._adv_coef = torch.nn.Parameter(torch.tensor(float(np.log(self.advantage_head_coef))).to(self.device))
                                                                 
        self._policy_lr_optimizer = O.Adam(self._policy_lrs, lr=self.lrlr)
        self._value_lr_optimizer = O.Adam(self._value_lrs, lr=self.lrlr)
        self._adv_coef_optimizer = O.Adam([self._adv_coef], lr=self.lrlr)

        self.online_adapt_policy_opt = None
        self.online_adapt_value_opt = None
        
    def select_action(self, state, eval=False, policy=None):
        if policy is None:
            policy = self._adaptation_policy

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        mu = policy(state)
        if eval is True:
            action = mu
        else:
            action = mu + self._action_sigma * torch.empty_like(mu).normal_()
        
        return action.detach().cpu().numpy()[0]

    def get_value(self, states, actions):
        if self.q_value:
            return self._value_function(torch.cat([states, actions], 1))
        return self._value_function(states)

    def __call__(self, states, actions):
        if self.q_value:
            value = self._value_function(torch.cat([states, actions], 1))
        else:
            value = self._value_function(states)
        return value, value

    def policy_output(self, policy, state_batch):
        mu = policy(state_batch)
        actions = mu + self._action_sigma * torch.empty_like(mu).normal_()
        return actions


    def value_function_loss_on_batch(self, value_function, action_function, task_policy, state_batch, next_state_batch, action_batch, mc_reward_batch, reward_batch, mask_batch, inner: bool = False):
        if self.q_value:
            with torch.no_grad():
                actions_next, _, _ = task_policy.sample(next_state_batch)
                qvalue_next = value_function(torch.cat([next_state_batch, actions_next], 1))
                targets = reward_batch + mask_batch * self.gamma_safe * qvalue_next

            qvalue_estimates = value_function(torch.cat([state_batch, action_batch], 1))

            losses = torch.nn.functional.mse_loss(qvalue_estimates,targets)

            return losses, None, None, None
        else:
            value_estimates = value_function(state_batch)
        
        with torch.no_grad():
            mc_value_estimates = mc_reward_batch

        targets = mc_value_estimates
        if inner:
            pass
        factor = 1
        losses = torch.nn.functional.mse_loss(value_estimates,targets)
 
        return losses, value_estimates.mean(), mc_value_estimates.mean(), mc_value_estimates.std()


    def adaptation_policy_loss_on_batch(self, policy, value_function, state_batch, action_batch, mc_reward_batch, inner: bool = False):
        #self._adaptation_temperature = 0.33333
        if self.q_value:
            actions = self.policy_output(policy, state_batch)
            q_value_estimate = value_function(torch.cat([state_batch, actions], 1))
            losses = q_value_estimate.mean()
            
            return losses, None, None, None
            
        else:
            with torch.no_grad():
                value_estimates = value_function(state_batch)
                action_value_estimates = mc_reward_batch

                advantages = (action_value_estimates - value_estimates).squeeze(-1)

                normalized_advantages = (1 / self._adaptation_temperature) * (advantages - advantages.mean()) / advantages.std()
                normalized_advantages = -normalized_advantages
                weights = normalized_advantages.clamp(max=self._advantage_clamp).exp()
            action_mu, advantage_prediction = policy(state_batch, action_batch)
            action_sigma = torch.empty_like(action_mu).fill_(self._action_sigma)
            action_distribution = D.Normal(action_mu, action_sigma)
            action_log_probs = action_distribution.log_prob(action_batch).sum(-1)
            losses = -(action_log_probs * weights)
        
        adv_prediction_loss = None
        if inner:
            if self.q_value:
                pass
            else:
                adv_prediction_loss = F.softplus(self._adv_coef) *  (advantage_prediction.squeeze() - advantages) ** 2
                losses = losses + adv_prediction_loss
                adv_prediction_loss = adv_prediction_loss.mean()

        return losses.mean(), advantages.mean(), weights, adv_prediction_loss

    def update_model(self, model: nn.Module, optimizer: torch.optim.Optimizer, clip: float = None, extra_grad: list = None):
        if clip is not None:
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        else:
            grad = None

        optimizer.step()
        optimizer.zero_grad()
        
        return grad

    def update_params(self, params: list, optimizer: torch.optim.Optimizer, clip: float = None, extra_grad: list = None):
        optimizer.step()
        optimizer.zero_grad()

    def meta_update_parameters(self, inner_buffers, outer_buffers, writer=None, ep=None, memory=None, policy=None, critic=None, lr=None, batch_size=None, training_iterations=None, plot=None):
        meta_value_grads = []
        meta_policy_grads = []
        train_rewards = []
        rollouts = []
        successes = []
        train_step_index = self.updates
        tasks = random.sample(range(self.num_tasks), self.task_batch_size)

        for i, (train_task_idx, inner_buffer, outer_buffer) in enumerate(zip(range(self.num_tasks), inner_buffers, outer_buffers)):

            # Only train on the randomly selected tasks for this iteration
            if train_task_idx not in tasks:
                continue

            # Data for Inner Adaptation
            state_batch, action_batch, constraint_batch, next_state_batch, mask_batch, mc_reward_batch = inner_buffer.sample(
                batch_size=self.inner_batch_size,
                pos_fraction=self.pos_fraction)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
            constraint_batch = torch.FloatTensor(constraint_batch).to(
                self.device).unsqueeze(1)
            mc_reward_batch = torch.FloatTensor(mc_reward_batch).to(
                self.device).unsqueeze(1)

            # Data for Outer Adaptation
            meta_state_batch, meta_action_batch, meta_constraint_batch, meta_next_state_batch, meta_mask_batch, meta_mc_reward_batch = outer_buffer.sample(
                batch_size=self.batch_size,
                pos_fraction=self.pos_fraction)
            meta_state_batch = torch.FloatTensor(meta_state_batch).to(self.device)
            meta_next_state_batch = torch.FloatTensor(meta_next_state_batch).to(self.device)
            meta_action_batch = torch.FloatTensor(meta_action_batch).to(self.device)
            meta_mask_batch = torch.FloatTensor(meta_mask_batch).to(self.device).unsqueeze(1)
            meta_constraint_batch = torch.FloatTensor(meta_constraint_batch).to(
                self.device).unsqueeze(1)
            meta_mc_reward_batch = torch.FloatTensor(meta_mc_reward_batch).to(
                self.device).unsqueeze(1)

            inner_value_losses = []
            meta_value_losses = []
            inner_policy_losses = []
            adv_policy_losses = []
            meta_policy_losses = []
            value_lr_grads = []
            policy_lr_grads = []
            #inner_mc_means, inner_mc_stds = [], []
            #outer_mc_means, outer_mc_stds = [], []
            #inner_values, outer_values = [], []
            #inner_weights, outer_weights = [], []
            #inner_advantages, outer_advantages = [], []

            ##################################################################################################
            # Adapt value function and collect meta-gradients
            ##################################################################################################
            vf = self._value_function
            vf.train()
            opt = O.SGD([{'params': p, 'lr': None} for p in vf.adaptation_parameters()])
            with higher.innerloop_ctx(vf, opt, override={'lr': [F.softplus(l) for l in self._value_lrs]}, copy_initial_weights=False) as (f_value_function, diff_value_opt):
                for step in range(self._maml_steps):
                    loss, value_inner, mc_inner, mc_std_inner = self.value_function_loss_on_batch(f_value_function, self._adaptation_policy, policy, state_batch, next_state_batch, action_batch, mc_reward_batch, constraint_batch, mask_batch, inner=True)

                    #inner_values.append(value_inner.item())
                    #inner_mc_means.append(mc_inner.item())
                    #inner_mc_stds.append(mc_std_inner.item())
                    diff_value_opt.step(loss)
                    inner_value_losses.append(loss.item())

                # Collect grads for the value function update in the outer loop [L14],
                #  which is not actually performed here
                meta_value_function_loss, value, mc, mc_std = self.value_function_loss_on_batch(f_value_function, self._adaptation_policy, policy, meta_state_batch, meta_next_state_batch, meta_action_batch, meta_mc_reward_batch, meta_constraint_batch, meta_mask_batch)
                total_vf_loss = meta_value_function_loss / self.num_tasks
                total_vf_loss.backward()

                #outer_values.append(value.item())
                #outer_mc_means.append(mc.item())
                #outer_mc_stds.append(mc_std.item())
                '''
                meta_value_losses.append(meta_value_function_loss.item())
                ##################################################################################################
                # Adapt policy and collect meta-gradients
                ##################################################################################################
                adapted_value_function = f_value_function
                opt = O.SGD([{'params': p, 'lr': None} for p in self._adaptation_policy.adaptation_parameters()])
                self._adaptation_policy.train() 
                with higher.innerloop_ctx(self._adaptation_policy, opt, override={'lr': [F.softplus(l) for l in self._policy_lrs]}, copy_initial_weights=False) as (f_adaptation_policy, diff_policy_opt):
                    with FreezeParameters(adapted_value_function.parameters()):
                        for step in range(self._maml_steps):
                            loss, adv, weights, adv_loss = self.adaptation_policy_loss_on_batch(f_adaptation_policy,
                                                                                               adapted_value_function, state_batch, action_batch, mc_reward_batch, inner=True)
                            
                            diff_policy_opt.step(loss)
                            inner_policy_losses.append(loss.item())
                            #adv_policy_losses.append(adv_loss.item())
                            #inner_advantages.append(adv.item())
                            #inner_weights.append(weights.mean().item())

                        meta_policy_loss, outer_adv, outer_weights_, _ = self.adaptation_policy_loss_on_batch(f_adaptation_policy, adapted_value_function, meta_state_batch, meta_action_batch, meta_mc_reward_batch, inner=False)
                        (meta_policy_loss / self.num_tasks).backward()

                        #outer_weights.append(outer_weights_.mean().item())
                        #outer_advantages.append(outer_adv.item())
                        meta_policy_losses.append(meta_policy_loss.item())
                ##################################################################################################
                '''

        # Meta-update value function [L14]
        grad = self.update_model(self._value_function, self._value_function_optimizer, clip=self._grad_clip)

        # Meta-update adaptation policy [L15]
        ap_opt = self._adaptation_policy_optimizer
        state_batch, action_batch, constraint_batch, next_state_batch, mask_batch, mc_reward_batch = memory.sample(
            batch_size=min(batch_size, len(memory)),
            pos_fraction=self.pos_fraction)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch).to(
            self.device).unsqueeze(1)
        mc_reward_batch = torch.FloatTensor(mc_reward_batch).to(
            self.device).unsqueeze(1)
        ap_loss, _, _, _ = self.adaptation_policy_loss_on_batch(self._adaptation_policy, self._value_function, state_batch, action_batch, mc_reward_batch, inner=True)
        ap_opt.zero_grad()
        ap_loss.backward()
        ap_opt.step()

        self._value_function_optimizer.zero_grad()
        #grad = self.update_model(self._adaptation_policy, self._adaptation_policy_optimizer, clip=self._grad_clip)

        if self.lrlr > 0:
            self.update_params(self._value_lrs, self._value_lr_optimizer)
            #self.update_params(self._policy_lrs, self._policy_lr_optimizer)
            self.update_params([self._adv_coef], self._adv_coef_optimizer)

        self.updates+=1
        if self.updates%100==0:
            self.plot(policy, self.updates, [.1, 0], "right", folder_prefix="/right/")
            self.plot(policy, self.updates, [-.1, 0], "left", folder_prefix="/left/")
            self.plot(policy, self.updates, [0, .1], "down", folder_prefix="/down/")
            self.plot(policy, self.updates, [0, -.1], "up", folder_prefix="/up/")
            #self.eval_adaptation(policy, memory)

    def eval_adaptation(self, policy, memory):
        vf = deepcopy(self._value_function)
        ap = deepcopy(self._adaptation_policy)
        opt = O.Adam(vf.parameters(), lr=self.inner_value_lr)
        ap_opt = O.Adam(ap.parameters(), lr=self.inner_policy_lr)

        log_steps = [1,5,10,20]
        for step in range(20):
            state_batch, action_batch, constraint_batch, next_state_batch, mask_batch, mc_reward_batch = memory.sample(
            batch_size=min(self.batch_size, len(memory)),
            pos_fraction=self.pos_fraction)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
            constraint_batch = torch.FloatTensor(constraint_batch).to(
                self.device).unsqueeze(1)
            mc_reward_batch = torch.FloatTensor(mc_reward_batch).to(
                self.device).unsqueeze(1)
            
            vf_loss, _, _, _ = self.value_function_loss_on_batch(vf, ap, policy, state_batch, next_state_batch, action_batch, mc_reward_batch, constraint_batch, mask_batch, inner=True)
            
            opt.zero_grad()
            vf_loss.backward()
            opt.step()

            ap_loss, _, _, _ = self.adaptation_policy_loss_on_batch(ap, vf, state_batch, action_batch, mc_reward_batch, inner=True)
            ap_opt.zero_grad()
            ap_loss.backward()
            ap_opt.step()

            if step+1 in log_steps:
                self.plot(policy, self.updates, [.1, 0], "right", folder_prefix="/" + str(step+1) + "/", critic=vf)

    def update_parameters(self, ep=None, memory=None, policy=None, critic=None, lr=None, batch_size=None, training_iterations=None, plot=None):
        if self.online_adapt_value_opt is None and self.online_adapt_policy_opt is None:
            self.online_adapt_value_opt = O.Adam(self._value_function.parameters(), lr=self.inner_value_lr)
            self.online_adapt_policy_opt = O.Adam(self._adaptation_policy.parameters(), lr=self.inner_policy_lr)
        # Data for Inner Adaptation
        state_batch, action_batch, constraint_batch, next_state_batch, mask_batch, mc_reward_batch = memory.sample(
            batch_size=min(batch_size, len(memory)),
            pos_fraction=self.pos_fraction)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch).to(
            self.device).unsqueeze(1)
        mc_reward_batch = torch.FloatTensor(mc_reward_batch).to(
            self.device).unsqueeze(1)

                
        vf = self._value_function
        vf.train()
        vf_loss, _, _ , _ = self.value_function_loss_on_batch(vf, self._adaptation_policy, policy, state_batch, next_state_batch, action_batch, mc_reward_batch, constraint_batch, mask_batch, inner=True)
        
        self.online_adapt_value_opt.zero_grad()
        vf_loss.backward()
        self.online_adapt_value_opt.step()
        

        self._adaptation_policy.train() 
        actor_loss, _, _, _= self.adaptation_policy_loss_on_batch(self._adaptation_policy,
                                                                           self._value_function, state_batch, action_batch, mc_reward_batch, inner=True)
        # Meta-update value function [L14]
        self.online_adapt_policy_opt.zero_grad()
        actor_loss.backward()
        self.online_adapt_policy_opt.step()

        self.updates+=1
        if self.updates%100==0:
            if self.q_value:
                self.plot(policy, self.updates, [.1, 0], "right", folder_prefix="/right/")
                self.plot(policy, self.updates, [-.1, 0], "left", folder_prefix="/left/")
                self.plot(policy, self.updates, [0, .1], "down", folder_prefix="/down/")
                self.plot(policy, self.updates, [0, -.1], "up", folder_prefix="/up/")
            else:
                self.plot(policy, self.updates)


    def plot(self, pi, ep, action=None, suffix="", folder_prefix = "", critic=None):
        env = self.tmp_env
        if self.env_name in ['maze', 'maze_1', 'maze_2', 'maze_3', 'maze_4', 'maze_5', 'maze_6']:
            x_bounds = [-0.3, 0.3]
            y_bounds = [-0.3, 0.3]
        elif self.env_name == 'simplepointbot0':
            x_bounds = [-80, 20]
            y_bounds = [-10, 10]
        elif self.env_name =='simplepointbot1':
            x_bounds = [-75, 25]
            y_bounds = [-20, 20]

        states = []
        x_pts = 100
        y_pts = int(
            x_pts * (x_bounds[1] - x_bounds[0]) / (y_bounds[1] - y_bounds[0]))
        for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
            for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                if self.env_name == 'image_maze':
                    env.reset(pos=(x, y))
                    obs = process_obs(env._get_obs(images=True))
                    states.append(obs)
                else:
                    states.append([x, y])

        num_states = len(states)
        states = self.torchify(np.array(states))
        if critic is None:
            critic = self._value_function
        critic.eval()
        if self.q_value:
            actions = self.torchify(np.tile(action, (len(states), 1)))
            max_qf = critic(torch.cat([states, actions], 1))
        else:
            max_qf = critic(states)

        grid = max_qf.detach().cpu().numpy()
        grid = grid.reshape(y_pts, x_pts)
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
                    (112.5, 31.25),
                    10*2.5,
                    15*2.5,
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'))


        if self.env_name in ['maze', 'maze_1', 'maze_2', 'maze_3', 'maze_4', 'maze_5', 'maze_6']:
            fig, ax = plt.subplots()
            cmap = plt.get_cmap('jet', 10)
            background = cv2.resize(env._get_obs(images=True), (x_pts, y_pts))
            plt.imshow(background)
            im = ax.imshow(grid.T, alpha=0.6, cmap=cmap, vmin=0.0, vmax=1.0)
            cbar = fig.colorbar(im, ax=ax)
        else:
            plt.imshow(grid.T)
        log_string = self.logdir + "/" + folder_prefix + "value_" + str(ep) + suffix
        plt.savefig(
            log_string,
            bbox_inches='tight')