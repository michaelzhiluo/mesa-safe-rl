'''
Built on on SAC implementation from 
https://github.com/pranz24/pytorch-soft-actor-critic
'''

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy, QNetworkCNN, GaussianPolicyCNN, QNetworkConstraint, QNetworkConstraintCNN, DeterministicPolicyCNN, StochasticPolicy
from dotmap import DotMap
from constraint import ValueFunction
import cv2
from run_multitask import MAMLRAWR

def process_obs(obs):
    im = np.transpose(obs, (2, 0, 1))
    return im


class QSafeWrapper:
    def __init__(self, obs_space, ac_space, hidden_size, logdir, action_space,
                 args, tmp_env):
        self.env_name = args.env_name
        self.goal = args.goal
        self.logdir = logdir
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.ac_space = ac_space
        self.images = args.cnn
        self.encoding = args.vismpc_recovery
        if not self.images:
            self.safety_critic = QNetworkConstraint(
                obs_space.shape[0], ac_space.shape[0],
                hidden_size).to(device=self.device)
            self.safety_critic_target = QNetworkConstraint(
                obs_space.shape[0], ac_space.shape[0],
                args.hidden_size).to(device=self.device)
        else:
            if self.encoding:
                self.safety_critic = QNetworkConstraint(
                    hidden_size, ac_space.shape[0],
                    hidden_size).to(device=self.device)
                self.safety_critic_target = QNetworkConstraint(
                    hidden_size, ac_space.shape[0],
                    args.hidden_size).to(device=self.device)
            else:
                self.safety_critic = QNetworkConstraintCNN(
                    obs_space, ac_space.shape[0], hidden_size,
                    args.env_name).to(self.device)
                self.safety_critic_target = QNetworkConstraintCNN(
                    obs_space, ac_space.shape[0], hidden_size,
                    args.env_name).to(self.device)
        self.awr = False

        import os
        try:
            os.makedirs(logdir + "/right")
            os.makedirs(logdir + "/left")
            os.makedirs(logdir + "/up")
            os.makedirs(logdir + "/down")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.lr = args.lr
        self.safety_critic_optim = Adam(
            self.safety_critic.parameters(), lr=args.lr)
        hard_update(self.safety_critic_target, self.safety_critic)

        self.tau = args.tau_safe
        self.gamma_safe = args.gamma_safe
        self.updates = 0
        self.target_update_interval = args.target_update_interval
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        if not self.images:
            self.policy = StochasticPolicy(obs_space.shape[0],
                                              ac_space.shape[0], hidden_size,
                                              action_space).to(self.device)
        else:
            self.policy = DeterministicPolicyCNN(obs_space, ac_space.shape[0],
                                                 hidden_size, args.env_name,
                                                 action_space).to(self.device)

        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        self.pos_fraction = args.pos_fraction if args.pos_fraction >= 0 else None
        self.ddpg_recovery = args.ddpg_recovery
        self.Q_sampling_recovery = args.Q_sampling_recovery
        self.tmp_env = tmp_env

        self.lagrangian_recovery = args.lagrangian_recovery
        self.recovery_lambda = args.recovery_lambda
        self.eps_safe = args.eps_safe
        self.alpha = args.alpha
        if args.env_name in ['maze', 'maze_1', 'maze_2', 'maze_3', 'maze_4', 'maze_5', 'maze_6']:
            self.tmp_env.reset(pos=(12, 12))

    def update_parameters(self,
                          ep=None,
                          memory=None,
                          policy=None,
                          critic=None,
                          lr=None,
                          batch_size=None,
                          training_iterations=3000,
                          plot=1):
        # TODO: cleanup this is hardcoded for maze
        #state_batch, action_batch, constraint_batch, next_state_batch, mask_batch, mc_reward_batch = memory.sample(
            #batch_size=min(batch_size, len(memory)),
            #pos_fraction=self.pos_fraction)
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

        if self.encoding:
            state_batch_enc = self.encoder(state_batch)
            next_state_batch_enc = self.encoder(next_state_batch)

        if not self.awr:
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = policy.sample(
                    next_state_batch)
                if self.encoding:
                    qf1_next_target, qf2_next_target = self.safety_critic_target(
                        next_state_batch_enc, next_state_action)
                else:
                    qf1_next_target, qf2_next_target = self.safety_critic(
                        next_state_batch, next_state_action)
                min_qf_next_target = torch.max(qf1_next_target, qf2_next_target)
                next_q_value = constraint_batch + mask_batch * self.gamma_safe * (
                    min_qf_next_target)

            # qf1, qf2 = self.safety_critic(state_batch, policy.sample(state_batch)[0])  # Two Q-functions to mitigate positive bias in the policy improvement step
            if self.encoding:
                qf1, qf2 = self.safety_critic(
                    state_batch_enc, action_batch
                )  # Two Q-functions to mitigate positive bias in the policy improvement step
            else:
                qf1, qf2 = self.safety_critic(
                    state_batch, action_batch
                )  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(
                qf1, next_q_value
            )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(
                qf2, next_q_value
            )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

            self.safety_critic_optim.zero_grad()
            (qf1_loss + qf2_loss).backward()
            self.safety_critic_optim.step()
        else:
            qf1, qf2 = self.safety_critic(
                    state_batch, action_batch
            )
            qf_loss = F.mse_loss(qf1, mc_reward_batch) + F.mse_loss(qf2, mc_reward_batch)
            self.safety_critic_optim.zero_grad()
            qf_loss.backward()
            self.safety_critic_optim.step()

        if self.ddpg_recovery:
            pi, log_pi, _ = self.policy.sample(state_batch)

            qf1_pi, qf2_pi = self.safety_critic(state_batch, pi)
            max_sqf_pi = torch.max(qf1_pi, qf2_pi)

            if self.lagrangian_recovery:
                assert critic is not None
                pi, log_pi, _ = policy.sample(state_batch)
                qf1_pi, qf2_pi = critic(state_batch, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                policy_loss = (
                    self.recovery_lambda * (max_sqf_pi - self.eps_safe) -
                    min_qf_pi
                ).mean(
                )  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            else:
                # Ignore AWR doesn't work with Recovery RL
                if self.awr:
                    with torch.no_grad():
                        advantages = (mc_reward_batch - qf1).squeeze(-1)
                        normalized_advantages = (1/0.333333)*(advantages - advantages.mean())/advantages.std()
                        normalized_advantages = - normalized_advantages
                        weights = advantages.clamp(max=np.log(20.0)).exp()
                    
                    cur_dist = self.policy(state_batch)
                    action_log_probs = cur_dist.log_prob(action_batch).sum(-1)  
                    policy_loss = -(action_log_probs * weights).mean()
                else:
                    policy_loss = max_sqf_pi.mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

        if self.updates % self.target_update_interval == 0:
            soft_update(self.safety_critic_target, self.safety_critic,
                        self.tau)
        self.updates += 1

        plot_interval = 100
        if self.env_name == 'image_maze':
            plot_interval = 29000

        if plot and self.updates % plot_interval == 0:
            if self.env_name in ['simplepointbot0', 'simplepointbot1', 'maze', 'maze_1', 'maze_2', 'maze_3', 'maze_4', 'maze_5', 'maze_6']:
                self.plot(policy, self.updates, [.1, 0], "right", folder_prefix="/right/")
                self.plot(policy, self.updates, [-.1, 0], "left", folder_prefix="/left/")
                self.plot(policy, self.updates, [0, .1], "down", folder_prefix="/down/")
                self.plot(policy, self.updates, [0, -.1], "up", folder_prefix="/up/")
            elif self.env_name == 'image_maze':
                self.plot(policy, self.updates, [.3, 0], "right")
                self.plot(policy, self.updates, [-.3, 0], "left")
                self.plot(policy, self.updates, [0, .3], "up")
                self.plot(policy, self.updates, [0, -.3], "down")
            else:
                return
                raise NotImplementedError("Unsupported environment for plotting")          

    def get_value(self, states, actions, encoded=False):
        with torch.no_grad():
            if self.encoding and not encoded:
                q1, q2 = self.safety_critic(self.encoder(states), actions)
            else:
                q1, q2 = self.safety_critic(states, actions)
            return torch.max(q1, q2)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.ddpg_recovery:
            if eval is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]
        elif self.Q_sampling_recovery:
            if not self.images:
                state_batch = state.repeat(1000, 1)
            else:
                state_batch = state.repeat(1000, 1, 1, 1)
            sampled_actions = torch.FloatTensor(
                np.array([self.ac_space.sample() for _ in range(1000)])).to(
                    self.device)
            q_vals = self.get_value(state_batch, sampled_actions)
            min_q_value_idx = torch.argmin(q_vals)
            action = sampled_actions[min_q_value_idx]
            return action.detach().cpu().numpy()
        else:
            assert False

    def plot(self, pi, ep, action=None, suffix="", folder_prefix="", critic=None):
        env = self.tmp_env
        if self.env_name in ['maze', 'maze_1', 'maze_2', 'maze_3', 'maze_4', 'maze_5', 'maze_6']:
            x_bounds = [-0.3, 0.3]
            y_bounds = [-0.3, 0.3]
        elif self.env_name == 'simplepointbot0':
            x_bounds = [-80, 20]
            y_bounds = [-10, 10]
        elif self.env_name == 'simplepointbot1':
            x_bounds = [-75, 25]
            y_bounds = [-20, 20]
        elif self.env_name == 'image_maze':
            x_bounds = [-0.05, 0.25]
            y_bounds = [-0.05, 0.25]
        else:
            raise NotImplementedError("Plotting unsupported for this env")

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
        if not self.encoding and self.env_name=='maze':
            states = np.array(states)
            goal_state = self.tmp_env.get_goal()
            batch_size = states.shape[0]
            goal_states = np.tile(goal_state, (batch_size, 1))
            states = np.concatenate([states, goal_states], axis=1)
            states = self.torchify(states)
        else:
            states = self.torchify(np.array(states))
        actions = self.torchify(np.tile(action, (len(states), 1)))
        # if ep > 0:
        #     actions = pi(states)
        # else:
        #     actions = self.torchify(np.array([self.action_space.sample() for _ in range(num_states)]))

        if critic is None:
            if self.encoding:
                qf1, qf2 = self.safety_critic(self.encoder(states), actions)
            else:
                qf1, qf2 = self.safety_critic(states, actions)
            max_qf = torch.max(qf1, qf2)

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
        log_string = self.logdir + "/" + folder_prefix + "qvalue_" + str(ep) + suffix
        plt.savefig(
            log_string,
            bbox_inches='tight')

    def __call__(self, states, actions):
        if self.encoding:
            return self.safety_critic(self.encoder(states), actions)
        else:
            return self.safety_critic(states, actions)


class SAC(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 args,
                 logdir,
                 im_shape=None,
                 tmp_env=None):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.env_name = args.env_name
        self.logdir = logdir
        self.gamma_safe = args.gamma_safe
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)

        self.device = torch.device("cuda" if args.cuda else "cpu")
        if not args.cnn:
            self.V_safe = ValueFunction(
                DotMap(
                    gamma_safe=self.gamma_safe,
                    device=self.device,
                    state_dim=observation_space.shape[0],
                    hidden_size=200,
                    tau_safe=args.tau_safe,
                    use_target=args.use_target_safe,
                    logdir=logdir,
                    env_name=args.env_name,
                    opt=args.opt_value,
                    pred_time=args.pred_time))

        self.cnn = args.cnn
        # self.Q_safe = QFunction(DotMap(gamma_safe=self.gamma_safe,
        #                                device=self.device,
        #                                state_dim=observation_space.shape[0],
        #                                ac_space=action_space,
        #                                hidden_size=200,
        #                                logdir=logdir,
        #                                env_name=args.env_name,
        #                                opt=args.opt_value,
        #                                tau=args.tau_safe))

        # TODO; cleanup for now this is hard-coded for maze
        if im_shape:
            observation_space = im_shape

        if args.cnn:
            self.critic = QNetworkCNN(observation_space, action_space.shape[0],
                                      args.hidden_size,
                                      args.env_name).to(device=self.device)
        else:
            self.critic = QNetwork(observation_space.shape[0],
                                   action_space.shape[0],
                                   args.hidden_size).to(device=self.device)

        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        if args.cnn:
            self.critic_target = QNetworkCNN(
                observation_space, action_space.shape[0], args.hidden_size,
                args.env_name).to(device=self.device)
        else:
            self.critic_target = QNetwork(
                observation_space.shape[0], action_space.shape[0],
                args.hidden_size).to(device=self.device)

        self.DGD_constraints = args.DGD_constraints
        self.nu = args.nu
        self.update_nu = args.update_nu
        self.cnn = args.cnn
        self.eps_safe = args.eps_safe
        self.use_constraint_sampling = args.use_constraint_sampling
        self.log_nu = torch.tensor(
            np.log(self.nu), requires_grad=True, device=self.device)
        self.nu_optim = Adam([self.log_nu], lr=0.1 * args.lr)

        self.RCPO = args.RCPO
        self.lambda_RCPO = args.lambda_RCPO
        self.log_lambda_RCPO = torch.tensor(
            np.log(self.lambda_RCPO), requires_grad=True, device=self.device)
        self.lambda_RCPO_optim = Adam(
            [self.log_lambda_RCPO],
            lr=0.1 * args.lr)  # Make lambda updated slower than other things

        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(
                    1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            if args.cnn:
                self.policy = GaussianPolicyCNN(
                    observation_space, action_space.shape[0], args.hidden_size,
                    args.env_name, action_space).to(self.device)
            else:
                self.policy = GaussianPolicy(
                    observation_space.shape[0], action_space.shape[0],
                    args.hidden_size, action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            assert not args.cnn
            self.policy = DeterministicPolicy(
                observation_space.shape[0], action_space.shape[0],
                args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        if args.use_value:
            self.safety_critic = self.V_safe
        else:
            if args.meta:
                self.Q_safe = MAMLRAWR(
                observation_space,
                action_space,
                args.hidden_size,
                logdir,
                action_space,
                args,
                tmp_env=tmp_env)
            else:
                self.Q_safe =  QSafeWrapper(
                    observation_space,
                    action_space,
                    args.hidden_size,
                    logdir,
                    action_space,
                    args,
                    tmp_env=tmp_env)
            self.safety_critic = self.Q_safe

    def plot(self, ep, action, suffix):
        if self.env_name == 'reacher':
            x_bounds = np.array([0.03, 0.13]) * 100
            y_bounds = np.array([0.03, 0.13]) * 100

            states = []
            x_pts = 100
            y_pts = int(x_pts * (x_bounds[1] - x_bounds[0]) /
                        (y_bounds[1] - y_bounds[0]))
            for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
                for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                    states.append([x, y, -0.13 * 100])

        num_states = len(states)
        states = self.torchify(np.array(states))
        actions = self.torchify(np.tile(action, (len(states), 1)))
        # if ep > 0:
        #     actions = pi(states)
        # else:
        #     actions = self.torchify(np.array([self.action_space.sample() for _ in range(num_states)]))

        qf1, qf2 = self.critic(states, actions)
        max_qf = torch.min(qf1, qf2)

        grid = max_qf.detach().cpu().numpy()
        grid = grid.reshape(y_pts, x_pts)
        plt.imshow(grid.T)
        plt.savefig(osp.join(self.logdir, "qvalue_" + str(ep) + suffix))

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        self.safe_samples = 100
        if self.use_constraint_sampling:
            if not self.cnn:
                state_batch = state.repeat(self.safe_samples, 1)
            else:
                state_batch = state.repeat(self.safe_samples, 1, 1, 1)
            pi, log_pi, _ = self.policy.sample(state_batch)
            max_qf_constraint_pi = self.safety_critic.get_value(
                state_batch, pi)

            # Threshold with epsilon safe and get idxs and apply to both pi and max_qf_constraint_pi, if empty state
            thresh_idxs = (max_qf_constraint_pi <=
                           self.eps_safe).nonzero()[:, 0]
            # Note: these are auto-normalized
            thresh_probs = torch.exp(log_pi[thresh_idxs])
            thresh_probs = thresh_probs.flatten()

            if list(thresh_probs.size())[0] == 0:
                min_q_value_idx = torch.argmin(max_qf_constraint_pi)
                action = pi[min_q_value_idx, :].unsqueeze(0)
            else:
                prob_dist = torch.distributions.Categorical(thresh_probs)
                sampled_idx = prob_dist.sample()
                action = pi[sampled_idx, :].unsqueeze(0)
        else:
            if eval is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def train_safety_critic(self,
                            ep,
                            memory,
                            pi,
                            lr=0.0003,
                            batch_size=1000,
                            training_iterations=3000,
                            plot=False):
        # TODO: cleanup this is hardcoded for maze

        if self.env_name in ['maze', 'maze_1', 'maze_2', 'maze_3', 'maze_4', 'maze_5', 'maze_6']:
            lr = 1e-3
        self.safety_critic.train(ep, memory, pi, lr, batch_size,
                                 training_iterations, plot)

    def policy_sample(self, states):
        actions, _, _ = self.policy.sample(states)
        return actions

    def get_critic_value(self, states, actions):
        with torch.no_grad():
            q1, q2 = self.critic(states, actions)
            return torch.max(q1, q2).detach().cpu().numpy()

    def update_parameters(self,
                          memory,
                          batch_size,
                          updates,
                          nu=None,
                          safety_critic=None):
        if nu is None:
            nu = self.nu
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(
            self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            min_qf_next_target = torch.min(
                qf1_next_target,
                qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (
                min_qf_next_target)
            if self.RCPO:
                qsafe_batch = torch.max(
                    *safety_critic(state_batch, action_batch))
                assert safety_critic is not None
                next_q_value -= self.lambda_RCPO * qsafe_batch
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        sqf1_pi, sqf2_pi = self.safety_critic(state_batch, pi)
        max_sqf_pi = torch.max(sqf1_pi, sqf2_pi)

        if self.DGD_constraints:
            policy_loss = (
                (self.alpha * log_pi) + nu * (max_sqf_pi - self.eps_safe) -
                1. * min_qf_pi
            ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        else:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean(
            )  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        # Optimize nu
        if self.update_nu:
            nu_loss = (
                self.log_nu * (self.eps_safe - max_sqf_pi).detach()
            ).mean(
            )  # TODO: used log trick here too, just like alpha case, need to understand why this is done.
            self.nu_optim.zero_grad()
            nu_loss.backward()
            self.nu_optim.step()
            self.nu = self.log_nu.exp()

        # Optimize lambda
        if self.RCPO:
            lambda_RCPO_loss = (
                self.log_lambda_RCPO * (self.eps_safe - qsafe_batch).detach()
            ).mean(
            )  # TODO: used log trick here too, just like alpha case, need to understand why this is done.
            self.lambda_RCPO_optim.zero_grad()
            lambda_RCPO_loss.backward()
            self.lambda_RCPO_optim.step()
            self.lambda_RCPO = self.log_lambda_RCPO.exp()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.env_name == 'reacher' and updates % 50 == 0 and not self.cnn:
            self.plot(updates, [0.005, 0, 0], "right")
            self.plot(updates, [-0.005, 0, 0], "left")
            self.plot(updates, [0, 0.005, 0], "up")
            self.plot(updates, [0, -0.005, 0], "down")

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(
        ), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self,
                   env_name,
                   suffix="",
                   actor_path=None,
                   critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
