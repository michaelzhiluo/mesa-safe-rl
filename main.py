'''
Built on on SAC implementation from 
https://github.com/pranz24/pytorch-soft-actor-critic
'''

# -*- coding: utf-8 -*-
import argparse
import datetime
import gym
import os.path as osp
import pickle
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory, ConstraintReplayMemory
from MPC import MPC
from VisualRecovery import VisualRecovery
from dotmap import DotMap
from config import create_config
import os
from env.simplepointbot0 import SimplePointBot
import moviepy.editor as mpy
from video_recorder import VideoRecorder
import cv2
from model import VisualEncoderAttn, TransitionModel, VisualReconModel
from torch import nn, optim
from gen_pointbot0_demos import get_random_transitions_pointbot0
from gen_pointbot1_demos import get_random_transitions_pointbot1
from env.cartpole import transition_function
from env.half_cheetah_disabled import HalfCheetahEnv
from env.ant_disabled import AntEnv

TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
torchify = lambda x: torch.FloatTensor(x).to('cuda')


def linear_schedule(startval, endval, endtime):
    return lambda t: startval + t / endtime * (endval - startval) if t < endtime else endval


def set_seed(seed, env):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)


def dump_logs(test_rollouts, train_rollouts, logdir):
    data = {"test_stats": test_rollouts, "train_stats": train_rollouts}
    with open(osp.join(logdir, "run_stats.pkl"), "wb") as f:
        pickle.dump(data, f)


def print_episode_info(rollout):
    num_violations = 0
    for inf in rollout:
        if 'constraint' in inf:
            num_violations += int(inf['constraint'])
    if "reward" in rollout[-1] and "state" in rollout[-1]:
        print("final reward: %f" % rollout[-1]["reward"])
        if len(rollout[-1]["state"].shape) < 3:
            print(rollout[-1]["state"])
    print("num violations: %d" % num_violations)


def recovery_config_setup(args):
    ctrl_args = DotMap(**{key: val for (key, val) in args.ctrl_arg})
    cfg = create_config(args.env_name, "MPC", ctrl_args, args.override, logdir)
    cfg.ctrl_cfg.pred_time = args.pred_time
    cfg.ctrl_cfg.opt_cfg.reachability_hor = args.reachability_hor
    if args.use_value:
        cfg.ctrl_cfg.use_value = True
    elif args.use_qvalue:
        cfg.ctrl_cfg.use_qvalue = True
    else:
        assert (False)
    cfg.pprint()
    return cfg


def experiment_setup(logdir, args):
    if args.use_recovery and not args.disable_learned_recovery and not (
            args.ddpg_recovery or args.Q_sampling_recovery):
        cfg = recovery_config_setup(args)
        env = cfg.ctrl_cfg.env
        if not args.vismpc_recovery:
            recovery_policy = MPC(cfg.ctrl_cfg)
        else:
            encoder = VisualEncoderAttn(
                args.env_name, args.hidden_size, ch=3).to(device=TORCH_DEVICE)
            transition_model = TransitionModel(
                args.hidden_size,
                env.action_space.shape[0]).to(device=TORCH_DEVICE)
            residual_model = VisualReconModel(
                args.env_name, args.hidden_size).to(device=TORCH_DEVICE)

            dynamics_param_list = list(transition_model.parameters()) + list(
                residual_model.parameters()) + list(encoder.parameters())
            dynamics_optimizer = optim.Adam(
                dynamics_param_list, lr=3e-4, eps=1e-4)
            dynamics_finetune_optimizer = optim.Adam(
                transition_model.parameters(), lr=3e-4, eps=1e-4)

            if args.load_vismpc:
                if 'maze' in args.env_name:
                    model_dicts = torch.load(
                        os.path.join('models', args.model_fname,
                                     'model_19500.pth'))
                else:
                    model_dicts = torch.load(
                        os.path.join('models', args.model_fname,
                                     'model_199900.pth'))

                transition_model.load_state_dict(
                    model_dicts['transition_model'])
                residual_model.load_state_dict(model_dicts['residual_model'])
                encoder.load_state_dict(model_dicts['encoder'])
                dynamics_optimizer.load_state_dict(
                    model_dicts['dynamics_optimizer'])
            else:
                logdir = os.path.join('models', args.model_fname)
                os.makedirs(logdir, exist_ok=True)

            if args.vismpc_recovery:
                cfg.ctrl_cfg.encoder = encoder
                cfg.ctrl_cfg.transition_model = transition_model
                cfg.ctrl_cfg.residual_model = residual_model
                cfg.ctrl_cfg.dynamics_optimizer = dynamics_optimizer
                cfg.ctrl_cfg.dynamics_finetune_optimizer = dynamics_finetune_optimizer
                cfg.ctrl_cfg.hidden_size = args.hidden_size
                cfg.ctrl_cfg.beta = args.beta
                cfg.ctrl_cfg.logdir = logdir
                cfg.ctrl_cfg.batch_size = args.batch_size
                recovery_policy = VisualRecovery(cfg.ctrl_cfg)
    else:
        recovery_policy = None
        if "HalfCheetah" in args.env_name:
            env = HalfCheetahEnv()
        elif "Ant-Disabled" in args.env_name:
            env = AntEnv()
        else:
            env = gym.make(ENV_ID[args.env_name])
    set_seed(args.seed, env)
    agent = agent_setup(env, logdir, args)
    if args.use_recovery and not args.disable_learned_recovery and not (
            args.ddpg_recovery or args.Q_sampling_recovery):
        if args.use_value:
            recovery_policy.update_value_func(agent.V_safe)
        elif args.use_qvalue:
            recovery_policy.update_value_func(agent.Q_safe)
    return agent, recovery_policy, env


def agent_setup(env, logdir, args):
    if "HalfCheetah" in args.env_name:
        tmp_env = HalfCheetahEnv()
    elif "Ant-Disabled" in args.env_name:
        tmp_env = AntEnv()
    elif "reacher" in args.env_name:
        tmp_env = None
    else:
        tmp_env = gym.make(ENV_ID[args.env_name])

    agent = SAC(
        env.observation_space,
        env.action_space,
        args,
        logdir,
        tmp_env=tmp_env)
    return agent


def get_action(state, env, agent, recovery_policy, args, train=True):
    def recovery_thresh(state, action, agent, recovery_policy, args):
        if not args.use_recovery:
            return False

        critic_val = agent.safety_critic.get_value(
            torchify(state).unsqueeze(0),
            torchify(action).unsqueeze(0))

        if args.reachability_test:  # reachability test combined with safety check
            return not recovery_policy.reachability_test(
                state, action, args.eps_safe)
        if args.lookahead_test:
            return not recovery_policy.lookahead_test(state, action,
                                                      args.eps_safe)
        if critic_val > args.eps_safe and not args.pred_time:
            return True
        elif critic_val < args.t_safe and args.pred_time:
            return True
        return False

    policy_state = state
    if args.start_steps > total_numsteps and train:
        action = env.action_space.sample()  # Sample random action
    elif train:
        action = agent.select_action(policy_state)  # Sample action from policy
    else:
        action = agent.select_action(
            policy_state, eval=True)  # Sample action from policy

    # print("test", test)
    if recovery_thresh(state, action, agent, recovery_policy, args):
        recovery = True
        if not args.disable_learned_recovery:
            if args.ddpg_recovery or args.Q_sampling_recovery:
                real_action = agent.safety_critic.select_action(state)
            else:
                real_action = recovery_policy.act(state, 0)
        else:
            real_action = env.safe_action(state)
    else:
        recovery = False
        real_action = np.copy(action)
    return action, real_action, recovery


ENV_ID = {
    'simplepointbot0': 'SimplePointBot-v0',
    'simplepointbot1': 'SimplePointBot-v1',
    'cliffwalker': 'CliffWalker-v0',
    'cliffcheetah': 'CliffCheetah-v0',
    'maze': 'Maze-v0',
    'maze_1': 'Maze1-v0',
    'maze_2': 'Maze2-v0',
    'maze_3': 'Maze3-v0',
    'maze_4': 'Maze4-v0',
    'maze_5': 'Maze5-v0',
    'maze_6': 'Maze6-v0',
    'image_maze': 'ImageMaze-v0',
    'shelf_env': 'Shelf-v0',
    'shelf_dynamic_env': 'ShelfDynamic-v0',
    'shelf_long_env': 'ShelfLong-v0',
    'shelf_dynamic_long_env': 'ShelfDynamicLong-v0',
    'shelf_reach_env': 'ShelfReach-v0',
    'cliffpusher': 'CliffPusher-v0',
    'reacher': 'DVRKReacher-v0',
    'car': 'Car-v0',
    'minitaur': 'Minitaur-v0',
    'cartpole': 'CartPoleLength-v0',
    "HalfCheetah-v2": "HalfCheetah-v2",
    "HalfCheetah-Disabled": "HalfCheetah-Disabled-v0",
    "Ant-Disabled": "Ant-Disabled-v0",
    "Push-v0": "Push-v0",
    "Ant-v2": "Ant-v2",
}


def npy_to_gif(im_list, filename, fps=20):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


def get_constraint_demos(env, args):
    # Get demonstrations
    task_demo_data = None
    obs_seqs = []
    ac_seqs = []
    constraint_seqs = []
    if not args.task_demos:
        if args.env_name == 'reacher':
            constraint_demo_data = pickle.load(
                open(
                    osp.join("demos", "dvrk_reach", "constraint_demos.pkl"),
                    "rb"))
            if args.cnn:
                constraint_demo_data = constraint_demo_data['images']
            else:
                constraint_demo_data = constraint_demo_data['lowdim']
        elif 'maze' in args.env_name:
            if args.env_name == 'maze':
                constraint_demo_data = pickle.load(
                    open(
                        osp.join("demos", args.env_name,
                                 "constraint_demos.pkl"), "rb"))
            else:
                # constraint_demo_data, obs_seqs, ac_seqs, constraint_seqs = env.transition_function(args.num_constraint_transitions)
                demo_data = pickle.load(
                    open(osp.join("demos", args.env_name, "demos.pkl"), "rb"))
                constraint_demo_data = demo_data['constraint_demo_data']
                obs_seqs = demo_data['obs_seqs']
                ac_seqs = demo_data['ac_seqs']
                constraint_seqs = demo_data['constraint_seqs']
        elif args.env_name == 'minitaur':
            constraint_demo_data = pickle.load(
                open(
                    osp.join("demos", args.env_name, "constraint_demos.pkl"),
                    "rb"))
            constraint_demo_data_random = pickle.load(
                open(
                    osp.join("demos", args.env_name,
                             "constraint_demos_random.pkl"), "rb"))
            constraint_demo_data_kinda_random = pickle.load(
                open(
                    osp.join("demos", args.env_name,
                             "constraint_demos_kinda_random.pkl"), "rb"))
            constraint_demo_data_total = constraint_demo_data + constraint_demo_data_random + constraint_demo_data_kinda_random
            constraint_demo_data_list_safe = []
            constraint_demo_data_list_viol = []
            for i in range(len(constraint_demo_data_total)):
                if constraint_demo_data_total[i][2] == 1:
                    constraint_demo_data_list_viol.append(
                        constraint_demo_data_total[i])
            for i in range(len(constraint_demo_data_total)):
                if constraint_demo_data_total[i][2] == 0:
                    constraint_demo_data_list_safe.append(
                        constraint_demo_data_total[i])

            import random
            random.shuffle(constraint_demo_data_list_safe)
            constraint_demo_data = constraint_demo_data_list_viol + constraint_demo_data_list_safe
        elif 'shelf' in args.env_name:
            folder_name = args.env_name.split('_env')[0]
            # if not args.vismpc_recovery:
            if not args.cnn:
                constraint_demo_data = pickle.load(
                    open(
                        osp.join("demos", folder_name, "constraint_demos.pkl"),
                        "rb"))
            else:
                constraint_demo_data = pickle.load(
                    open(
                        osp.join("demos", folder_name,
                                 "constraint_demos_images.pkl"), "rb"))
        else:
            if args.env_name =='simplepointbot0' and args.multitask:
                constraint_demo_data = []
                for i in range(24):
                    data = pickle.load(open("demos/pointbot0_dynamics/constraint_demos_" + str(i) + ".pkl", "rb"))
                    constraint_demo_data.extend(data)
            elif args.env_name =='simplepointbot0' and args.meta:
                constraint_demo_data = get_random_transitions_pointbot0(w1=0.0, w2=0.0, discount=args.gamma_safe, num_transitions=args.num_constraint_transitions)[:200]
            elif args.env_name =='simplepointbot0':
                constraint_demo_data = get_random_transitions_pointbot0(w1=0.0, w2=0.0, discount=args.gamma_safe, num_transitions=args.num_constraint_transitions)
            elif args.env_name =='simplepointbot1' and args.multitask:
                constraint_demo_data = []
                for i in range(25):
                    data = pickle.load(open("demos/pointbot1_dynamics/constraint_demos_" + str(i) + ".pkl", "rb"))
                    constraint_demo_data.extend(data)
            elif args.env_name =='simplepointbot1' and args.meta:
                constraint_demo_data = get_random_transitions_pointbot1(w1=0.0, w2=0.0, discount=args.gamma_safe, num_transitions=args.num_constraint_transitions)[:200]
            elif args.env_name =='simplepointbot1':
                constraint_demo_data = get_random_transitions_pointbot1(w1=0.0, w2=0.0, discount=args.gamma_safe, num_transitions=args.num_constraint_transitions)
            elif args.env_name =='cartpole' and args.multitask:
                constraint_demo_data = []
                for i in range(20):
                    data = pickle.load(open("demos/cartpole_no_task/constraint_demos_" + str(i) + ".pkl", "rb"))
                    constraint_demo_data.extend(data)
            elif args.env_name == 'cartpole':
                constraint_demo_data = []
                data = pickle.load(open("demos/cartpole_no_task/constraint_demos_" + "test" + ".pkl", "rb"))
                import random
                data = random.sample(data, args.test_size)
                constraint_demo_data.extend(data)
            elif args.env_name == "HalfCheetah-Disabled" and args.multitask:
                constraint_demo_data = []
                for i in range(1, 5):
                    data = pickle.load(open("demos/halfcheetah_disabled_no_task/constraint_demos_" + str(i) + ".pkl", "rb"))
                    constraint_demo_data.extend(data)
            elif args.env_name == "HalfCheetah-Disabled":
                # Loading Test Set Data for MESA or for RRL Baseline
                constraint_demo_data = []
                data = pickle.load(open("demos/halfcheetah_disabled_no_task/constraint_demos_" + "5" + ".pkl", "rb"))
                import random
                data = random.sample(data, args.test_size)
                constraint_demo_data.extend(data)
            elif args.env_name == "Ant-Disabled" and args.multitask:
                constraint_demo_data = []
                for i in range(0, 3):
                    data = pickle.load(open("demos/ant_disabled_no_task/constraint_demos_" + str(i) + ".pkl", "rb"))
                    constraint_demo_data.extend(data)
            elif args.env_name == "Ant-Disabled":
                # Loading Test Set Data for MESA or for RRL Baseline
                constraint_demo_data = []
                data = pickle.load(open("demos/ant_disabled_no_task/constraint_demos_" + "3" + ".pkl", "rb"))
                import random
                data = random.sample(data, args.test_size)
                constraint_demo_data.extend(data)
            else:
                constraint_demo_data = env.transition_function(
                    args.num_constraint_transitions)
    else:
        if args.cnn and args.env_name == 'maze':
            constraint_demo_data, task_demo_data_images = env.transition_function(
                args.num_constraint_transitions,
                task_demos=args.task_demos,
                images=True)
            constraint_demo_data = pickle.load(
                open(osp.join("demos", "maze", "constraint_demos.pkl"), "rb"))
        elif 'shelf' in args.env_name:
            folder_name = args.env_name.split('_env')[0]
            if args.cnn:
                task_demo_data = pickle.load(
                    open(
                        osp.join("demos", folder_name,
                                 "task_demos_images.pkl"), "rb"))
            else:
                task_demo_data = pickle.load(
                    open(
                        osp.join("demos", folder_name, "task_demos.pkl"),
                        "rb"))
            if not args.vismpc_recovery:
                if args.cnn:
                    constraint_demo_data = pickle.load(
                        open(
                            osp.join("demos", folder_name,
                                     "constraint_demos_images.pkl"), "rb"))
                else:
                    constraint_demo_data = pickle.load(
                        open(
                            osp.join("demos", folder_name,
                                     "constraint_demos.pkl"), "rb"))

                # Get all violations in front to get as many violations as possible
                constraint_demo_data_list_safe = []
                constraint_demo_data_list_viol = []
                for i in range(len(constraint_demo_data)):
                    if constraint_demo_data[i][2] == 1:
                        constraint_demo_data_list_viol.append(
                            constraint_demo_data[i])
                    else:
                        constraint_demo_data_list_safe.append(
                            constraint_demo_data[i])

                constraint_demo_data = constraint_demo_data_list_viol[:int(
                    0.5 * args.num_constraint_transitions
                )] + constraint_demo_data_list_safe
            else:
                constraint_demo_data = []
                data = pickle.load(
                    open(
                        osp.join("demos", folder_name,
                                 "constraint_demos_images_seqs.pkl"), "rb"))
                obs_seqs = data['obs'][:args.num_constraint_transitions // 25]
                ac_seqs = data['ac'][:args.num_constraint_transitions // 25]
                constraint_seqs = data[
                    'constraint'][:args.num_constraint_transitions // 25]
                for i in range(len(ac_seqs)):
                    ac_seqs[i] = np.array(ac_seqs[i])
                for i in range(len(obs_seqs)):
                    obs_seqs[i] = np.array(obs_seqs[i])
                for i in range(len(constraint_seqs)):
                    constraint_seqs[i] = np.array(constraint_seqs[i])
                ac_seqs = np.array(ac_seqs)
                obs_seqs = np.array(obs_seqs)
                constraint_seqs = np.array(constraint_seqs)
                for i in range(obs_seqs.shape[0]):
                    for j in range(obs_seqs.shape[1] - 1):
                        constraint_demo_data.append(
                            (obs_seqs[i, j], ac_seqs[i, j],
                             constraint_seqs[i, j], obs_seqs[i, j + 1], False))
        else:
            constraint_demo_data, task_demo_data = env.transition_function(
                args.num_constraint_transitions, task_demos=args.task_demos)
    return constraint_demo_data, task_demo_data, obs_seqs, ac_seqs, constraint_seqs


def train_recovery(states, actions, next_states=None, epochs=50):
    if next_states is not None:
        recovery_policy.train(
            states, actions, random=True, next_obs=next_states, epochs=epochs)
    else:
        recovery_policy.train(states, actions)


# TODO: fix this for shelf env...
def process_obs(obs, env_name):
    if 'shelf' in args.env_name:
        obs = cv2.resize(obs, (64, 48), interpolation=cv2.INTER_AREA)
    im = np.transpose(obs, (2, 0, 1))
    return im


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument(
    '--env-name',
    default="HalfCheetah-v2",
    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--logdir', default="runs", help='exterior log directory')
parser.add_argument('--logdir_suffix', default="", help='log directory suffix')
parser.add_argument(
    '--policy',
    default="Gaussian",
    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument(
    '--eval',
    type=bool,
    default=True,
    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for reward (default: 0.99)')
parser.add_argument(
    '--pos_fraction',
    type=float,
    default=-1,
    metavar='G',
    help='fraction of positive examples for critic training')
parser.add_argument(
    '--gamma_safe',
    type=float,
    default=0.5,
    metavar='G',
    help='discount factor for constraints (default: 0.9)')
parser.add_argument(
    '--eps_safe',
    type=float,
    default=0.1,
    metavar='G',
    help='threshold constraints (default: 0.8)')
parser.add_argument(
    '--t_safe',
    type=float,
    default=80,
    metavar='G',
    help='threshold constraints (default: 0.8)')
parser.add_argument(
    '--tau',
    type=float,
    default=0.005,
    metavar='G',  # TODO: idk if this should be 0.005 or 0.0002...
    help='target smoothing coefficient(??) (default: 0.005)')
parser.add_argument(
    '--tau_safe',
    type=float,
    default=0.0002,
    metavar='G',
    help='target smoothing coefficient(??) (default: 0.005)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0003,
    metavar='G',
    help='learning rate (default: 0.0003)')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.2,
    metavar='G',
    help=
    'Temperature parameter ?? determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument(
    '--automatic_entropy_tuning',
    type=bool,
    default=False,
    metavar='G',
    help='Automaically adjust ?? (default: False)')
parser.add_argument(
    '--seed',
    type=int,
    default=123456,
    metavar='N',
    help='random seed (default: 123456)')
parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    metavar='N',
    help='batch size (default: 256)')
parser.add_argument(
    '--num_steps',
    type=int,
    default=1000000,
    metavar='N',
    help='maximum number of steps (default: 1000000)')
parser.add_argument(
    '--num_eps',
    type=int,
    default=1000000,
    metavar='N',
    help='maximum number of episodes (default: 1000000)')
parser.add_argument(
    '--hidden_size',
    type=int,
    default=256,
    metavar='N',
    help='hidden size (default: 256)')
parser.add_argument(
    '--updates_per_step',
    type=int,
    default=1,
    metavar='N',
    help='model updates per simulator step (default: 1)')
parser.add_argument(
    '--start_steps',
    type=int,
    default=100,
    metavar='N',
    help='Steps sampling random actions (default: 10000)')
parser.add_argument(
    '--target_update_interval',
    type=int,
    default=1,
    metavar='N',
    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument(
    '--replay_size',
    type=int,
    default=1000000,
    metavar='N',
    help='size of replay buffer (default: 100000)')
parser.add_argument(
    '--safe_replay_size',
    type=int,
    default=2000000,
    metavar='N',
    help='size of replay buffer for V safe (default: 100000)')
parser.add_argument(
    '--cuda', action="store_true", help='run on CUDA (default: False)')
parser.add_argument(
    '--cnn', action="store_true", help='visual observations (default: False)')
parser.add_argument('--critic_pretraining_steps', type=int, default=3000)
parser.add_argument('--critic_safe_pretraining_steps', type=int, default=10000)

parser.add_argument('--constraint_reward_penalty', type=float, default=0)
parser.add_argument('--safety_critic_penalty', type=float, default=-1)
# For recovery policy
parser.add_argument('--use_target_safe', action="store_true")
parser.add_argument('--disable_learned_recovery', action="store_true")
parser.add_argument('--use_recovery', action="store_true")
parser.add_argument('--ddpg_recovery', action="store_true")
parser.add_argument('--Q_sampling_recovery', action="store_true")
parser.add_argument('--reachability_test', action="store_true")
parser.add_argument('--lookahead_test', action="store_true")
parser.add_argument('--SAC_recovery', action="store_true")
parser.add_argument('--recovery_policy_update_freq', type=int, default=1)
parser.add_argument('--critic_safe_update_freq', type=int, default=1)
parser.add_argument('--task_demos', action="store_true")
parser.add_argument('--filter', action="store_true")
parser.add_argument('--num_filter_samples', type=int, default=100)
parser.add_argument('--max_filter_iters', type=int, default=5)
parser.add_argument('--Q_safe_start_ep', type=int, default=10)
parser.add_argument('--use_value', action="store_true")
parser.add_argument('--use_qvalue', action="store_true")
parser.add_argument('--pred_time', action="store_true")
parser.add_argument('--opt_value', action="store_true")
parser.add_argument('--lagrangian_recovery', action="store_true")
parser.add_argument(
    '--recovery_lambda', type=float, default=0.01, metavar='G',
    help='todo')  # TODO: needs some tuning
parser.add_argument('--num_task_transitions', type=int, default=10000000)
parser.add_argument(
    '--num_constraint_transitions', type=int, default=10000
)  # Make this 20K+ for original shelf env stuff, trying with fewer rn
parser.add_argument('--reachability_hor', type=int, default=2)

# Ablations
parser.add_argument('--disable_offline_updates', action="store_true")
parser.add_argument('--disable_online_updates', action="store_true")
parser.add_argument('--disable_action_relabeling', action="store_true")
parser.add_argument('--add_both_transitions', action="store_true")

# Lagrangian, RSPO
parser.add_argument('--DGD_constraints', action="store_true")
parser.add_argument('--use_constraint_sampling', action="store_true")
parser.add_argument(
    '--nu', type=float, default=0.01, metavar='G',
    help='todo')  # TODO: needs some tuning
parser.add_argument('--update_nu', action="store_true")
parser.add_argument('--nu_schedule', action="store_true")
parser.add_argument(
    '--nu_start',
    type=float,
    default=1e3,
    metavar='G',
    help='start value for nu (high)')
parser.add_argument(
    '--nu_end',
    type=float,
    default=0,
    metavar='G',
    help='end value for nu (low)')

# RCPO
parser.add_argument('--RCPO', action="store_true")
parser.add_argument(
    '--lambda_RCPO', type=float, default=0.01, metavar='G',
    help='todo')  # TODO: needs some tuning

# PLaNet Recoverry
parser.add_argument('--beta', type=float, default=10)
parser.add_argument('--vismpc_recovery', action="store_true")
parser.add_argument('--load_vismpc', action="store_true")
parser.add_argument('--model_fname', default='model1')

# Reward Conditioning
parser.add_argument('--eps_condition', type=float, default=0.3)
parser.add_argument('--conditional', action="store_true")

# Goal-based RL
parser.add_argument('--goal', action="store_true")

parser.add_argument(
    '-ca',
    '--ctrl_arg',
    action='append',
    nargs=2,
    default=[],
    help=
    'Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments'
)
parser.add_argument(
    '-o',
    '--override',
    action='append',
    nargs=2,
    default=[],
    help=
    'Override default parameters, see https://github.com/kchua/handful-of-trials#overrides'
)

# MESA Arguments
parser.add_argument("--meta", action="store_true")

# Multitask Benchmark
parser.add_argument("--multitask", action="store_true")

# Save Replay Buffer (Data Generation for Training and Testing Datasets)
parser.add_argument('--save_replay', action="store_true")

# Iterations to adapt offline-trained agent to test set data (See Phase 2: MESA)
parser.add_argument(
    '--online_iters', type=int, default=500
)

# Size of Test Set (10K for HalfCheetah-Disabled)
parser.add_argument(
    '--test_size', type=int, default=10000
)

args = parser.parse_args()

if args.nu_schedule:
    nu_schedule = linear_schedule(args.nu_start, args.nu_end, args.num_eps)
else:
    nu_schedule = linear_schedule(args.nu, args.nu, 0)

# TODO: clean this up later
if 'shelf' in args.env_name and args.num_constraint_transitions == 10000:
    args.num_constraint_transitions = 20000

if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

logdir = os.path.join(
    args.logdir, '{}_SAC_{}_{}_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
        args.policy, args.logdir_suffix))
print("LOGDIR: ", logdir)
writer = SummaryWriter(logdir=logdir)
pickle.dump(args, open(os.path.join(logdir, "args.pkl"), "wb"))

agent, recovery_policy, env = experiment_setup(logdir, args)

# Memory
memory = ReplayMemory(args.replay_size)
recovery_memory = ConstraintReplayMemory(args.safe_replay_size)

# Training Loop
total_numsteps = 0
updates = 0

conditional_penalty = 0
task_demos = args.task_demos

constraint_demo_data, task_demo_data, obs_seqs, ac_seqs, constraint_seqs = get_constraint_demos(
   env, args)

# Phase 1 MESA: Load in multiple training datasets (N Datasets) into N Replay Buffers
if args.meta:
    if args.env_name == 'maze':
        inner_replay = [ConstraintReplayMemory(args.safe_replay_size) for i in range(100)]
        outer_replay = inner_replay
        for i in range(100):
            data = pickle.load(open("demos/maze_goals/constraint_demos_" + str(i) + ".pkl", "rb"))
            for transition in data:
                inner_replay[i].push(*transition)
    elif args.env_name == 'simplepointbot0':
        inner_replay = [ConstraintReplayMemory(args.safe_replay_size) for i in range(24)]
        outer_replay = inner_replay
        for i in range(24):
            data = pickle.load(open("demos/pointbot0_dynamics/constraint_demos_" + str(i) + ".pkl", "rb"))
            for transition in data:
                inner_replay[i].push(*transition)
    elif args.env_name == 'simplepointbot1':
        inner_replay = [ConstraintReplayMemory(args.safe_replay_size) for i in range(25)]
        outer_replay = inner_replay
        for i in range(25):
            data = pickle.load(open("demos/pointbot1_dynamics/constraint_demos_" + str(i) + ".pkl", "rb"))
            for transition in data:
                inner_replay[i].push(*transition)
    elif args.env_name == 'cartpole':
        inner_replay = [ConstraintReplayMemory(args.safe_replay_size) for i in range(20)]
        outer_replay = inner_replay
        for i in range(20):
            data = pickle.load(open("demos/cartpole_no_task/constraint_demos_" + str(i) + ".pkl", "rb"))
            for transition in data:
                inner_replay[i].push(*transition)
    elif args.env_name == 'HalfCheetah-Disabled':
        inner_replay = [ConstraintReplayMemory(args.safe_replay_size) for i in range(4)]
        outer_replay = inner_replay
        for i in range(4):
            data = pickle.load(open("demos/halfcheetah_disabled_no_task/constraint_demos_" + str(i+1) + ".pkl", "rb"))
            for transition in data:
                inner_replay[i].push(*transition)
    elif args.env_name == 'Ant-Disabled':
        inner_replay = [ConstraintReplayMemory(args.safe_replay_size) for i in range(3)]
        outer_replay = inner_replay
        for i in range(3):
            data = pickle.load(open("demos/ant_disabled_no_task/constraint_demos_" + str(i) + ".pkl", "rb"))
            for transition in data:
                inner_replay[i].push(*transition)


# Phase 1: MESA, Offline Training
num_constraint_violations = 0
# Train recovery policy and associated value function on demos
if not args.disable_offline_updates:
    if (args.use_recovery and not args.disable_learned_recovery
        ) or args.DGD_constraints or args.RCPO:
        if not args.vismpc_recovery:
            demo_data_states = np.array([
                d[0]
                for d in constraint_demo_data[:args.num_constraint_transitions]
            ])
            demo_data_actions = np.array([
                d[1]
                for d in constraint_demo_data[:args.num_constraint_transitions]
            ])
            demo_data_next_states = np.array([
                d[3]
                for d in constraint_demo_data[:args.num_constraint_transitions]
            ])
            num_constraint_transitions = 0
            for transition in constraint_demo_data:
                recovery_memory.push(*transition)
                num_constraint_violations += int(transition[2])
                num_constraint_transitions += 1
                #if num_constraint_transitions == args.num_constraint_transitions:
                    #break
            print("Number of Constraint Transitions: ",
                  num_constraint_transitions)
            print("Number of Constraint Violations: ",
                  num_constraint_violations)
            if args.env_name in [
                    'simplepointbot0', 'simplepointbot1', 'maze', 'image_maze'
            ]:
                plot = True
            else:
                plot = False
            if args.use_qvalue:
                for i in range(args.critic_safe_pretraining_steps):
                    if i % 100 == 0:
                        print("CRITIC SAFE UPDATE STEP: ", i)
                    if args.meta:
                        agent.safety_critic.meta_update_parameters(
                            inner_buffers = inner_replay,
                            outer_buffers = outer_replay,
                            memory=recovery_memory,
                            policy=agent.policy,
                            critic=agent.critic,
                            batch_size=min(args.batch_size,
                                           len(constraint_demo_data)))
                    else:
                        agent.safety_critic.update_parameters(
                            memory=recovery_memory,
                            policy=agent.policy,
                            critic=agent.critic,
                            batch_size=min(args.batch_size,
                                           len(constraint_demo_data)))
                if args.goal:
                    recovery_memory = ConstraintReplayMemory(args.safe_replay_size)
                    constraint_demo_data = pickle.load(open("demos/maze_goals/constraint_demos_-0.2_0.15.pkl", "rb"))[:10000]
                    for transition in constraint_demo_data:
                        recovery_memory.push(*transition)
            else:
                agent.train_safety_critic(
                    0, recovery_memory, agent.policy_sample, plot=0)
            if not (args.ddpg_recovery or args.Q_sampling_recovery
                    or args.DGD_constraints or args.RCPO):
                train_recovery(
                    demo_data_states,
                    demo_data_actions,
                    demo_data_next_states,
                    epochs=50)
        else:
            # Pre-train vis dynamics model if needed
            if not args.load_vismpc:
                recovery_policy.train(
                    obs_seqs,
                    ac_seqs,
                    constraint_seqs,
                    recovery_memory,
                    num_train_steps=20000
                    if "maze" in args.env_name else 200000)
            # Process everything in recovery_memory to be encoded in order to train safety critic
            num_constraint_transitions = 0
            for transition in constraint_demo_data:
                recovery_memory.push(*transition)
                num_constraint_violations += int(transition[2])
                num_constraint_transitions += 1
                if num_constraint_transitions == args.num_constraint_transitions:
                    break
            print("Number of Constraint Transitions: ",
                  num_constraint_transitions)
            print("Number of Constraint Violations: ",
                  num_constraint_violations)
            if args.use_qvalue:
                # Pass encoding function to safety critic:
                agent.safety_critic.encoder = recovery_policy.get_encoding
                # Train safety critic using the encoder
                for i in range(args.critic_safe_pretraining_steps):
                    if i % 100 == 0:
                        print("CRITIC SAFE UPDATE STEP: ", i)
                    agent.safety_critic.update_parameters(
                        memory=recovery_memory,
                        policy=agent.policy,
                        critic=agent.critic,
                        batch_size=min(args.batch_size,
                                       len(constraint_demo_data)))


# If use task demos, add them to memory and train agent
if task_demos:
    num_task_transitions = 0
    for transition in task_demo_data:
        memory.push(*transition)
        num_task_transitions += 1
        if num_task_transitions == args.num_task_transitions:
            break
    print("Number of Task Transitions: ", num_task_transitions)
    for i in range(args.critic_pretraining_steps):
        if i % 100 == 0:
            print("Update: ", i)
        agent.update_parameters(
            memory,
            min(args.batch_size, num_task_transitions),
            updates,
            safety_critic=agent.safety_critic)
        updates += 1

test_rollouts = []
train_rollouts = []
all_ep_data = []

num_viols = 0
num_successes = 0
viol_and_recovery = 0
viol_and_no_recovery = 0
total_viols = 0


# Phase 2: MESA
if args.multitask:
    recovery_memory = ConstraintReplayMemory(args.safe_replay_size)
    if args.env_name =='simplepointbot0':
        constraint_demo_data = get_random_transitions_pointbot0(w1=0.0, w2=0.0, discount=args.gamma_safe, num_transitions=args.num_constraint_transitions)[:200]
    elif args.env_name =='simplepointbot1':
        constraint_demo_data = get_random_transitions_pointbot1(w1=0.0, w2=0.0, discount=args.gamma_safe, num_transitions=args.num_constraint_transitions)[:200]
    elif args.env_name == 'maze': 
        constraint_demo_data = pickle.load(open("demos/maze_goals/constraint_demos_test.pkl", "rb"))[:1000]
    elif args.env_name == "cartpole":
        data = pickle.load(open("demos/cartpole_no_task/constraint_demos_test.pkl", "rb"))
        import random
        data = random.sample(data, args.test_size)
        constraint_demo_data.extend(data)
    elif args.env_name == "HalfCheetah-Disabled":
        data = pickle.load(open("demos/halfcheetah_disabled_no_task/constraint_demos_5.pkl", "rb"))
        import random
        data = random.sample(data, args.test_size)
        constraint_demo_data.extend(data)
    elif args.env_name == "Ant-Disabled":
        data = pickle.load(open("demos/ant_disabled_no_task/constraint_demos_3.pkl", "rb"))
        import random
        data = random.sample(data, args.test_size)
        constraint_demo_data.extend(data)
    for transition in constraint_demo_data:
        recovery_memory.push(*transition)

if args.save_replay:
    recovery_memory = ConstraintReplayMemory(args.safe_replay_size)

if args.meta or args.multitask:
    for i in range(args.online_iters):
        agent.safety_critic.update_parameters(
                        memory=recovery_memory,
                        policy=agent.policy,
                        critic=agent.critic,
                        batch_size=args.batch_size,
                        plot=1)


# Phase 3: MESA (Rest is standard Recovery RL)
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    if args.env_name == 'reacher':
        recorder = VideoRecorder(
            env, osp.join(logdir, 'video_{}.mp4'.format(i_episode)))
    if args.cnn:
        state = process_obs(state, args.env_name)

    train_rollouts.append([])
    ep_states = [state]
    ep_actions = []
    ep_constraints = []

    rollouts = []

    while not done:
        if args.env_name == 'reacher':
            recorder.capture_frame()

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                    memory,
                    min(args.batch_size, len(memory)),
                    updates,
                    safety_critic=agent.safety_critic,
                    nu=nu_schedule(i_episode))
                if args.use_qvalue and not args.disable_online_updates and len(
                        recovery_memory) > args.batch_size and (
                            num_viols + num_constraint_violations
                        ) / args.batch_size > args.pos_fraction:
                    agent.safety_critic.update_parameters(
                        memory=recovery_memory,
                        policy=agent.policy,
                        critic=agent.critic,
                        batch_size=args.batch_size,
                        plot=1)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        action, real_action, recovery_used = get_action(
            state, env, agent, recovery_policy, args)
        next_state, reward, done, info = env.step(real_action)  # Step
        if 'constraint' not in info:
            info['reward'] = reward
            info['state'] = state
            info['next_state'] = next_state 
            info["action"] =  action
            info['constraint'] = 0
        info['recovery'] = recovery_used
        total_viols+= info['constraint']
        #print(reward)
        if args.cnn:
            next_state = process_obs(next_state, args.env_name)

        train_rollouts[-1].append(info)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        if args.constraint_reward_penalty != 0 and info['constraint']:
            reward -= args.constraint_reward_penalty

        if args.safety_critic_penalty > 0:
            critic_val = agent.safety_critic.get_value(
                torchify(state).unsqueeze(0),
                torchify(action).unsqueeze(0)).detach().cpu().numpy()[0, 0]
            reward -= args.safety_critic_penalty * critic_val

        mask = float(not done)
        done = done or episode_steps == env._max_episode_steps

        if args.conditional:
            critic_val = agent.safety_critic.get_value(
                torchify(state).unsqueeze(0),
                torchify(action).unsqueeze(0)).detach().cpu().numpy()[0, 0]
            if not abs(critic_val - args.eps_condition) < 0.07:
                reward -= 0.5 

        if not args.disable_action_relabeling:
            memory.push(state, action, reward, next_state,
                        mask)  # Append transition to memory
        else:
            memory.push(state, real_action, reward, next_state,
                        mask)  # Append transition to memory

        rollouts.append([state, real_action, info['constraint'], next_state, mask])
        if args.use_recovery or args.DGD_constraints or args.RCPO:
            #recovery_memory.push(state, real_action, info['constraint'],
                                 #next_state, mask)
            if recovery_used and args.add_both_transitions:
                memory.push(state, real_action, reward, next_state,
                            mask)  # Append transition to memory

        state = next_state
        ep_states.append(state)
        ep_actions.append(real_action)
        ep_constraints.append([info['constraint']])

    if args.use_recovery or args.save_replay:
        mc_reward =0
        discount=args.gamma_safe
        for transition in rollouts[::-1]:
            mc_reward = transition[2] + discount * mc_reward
            transition.append(mc_reward)
            recovery_memory.push(*transition)

    if args.env_name == 'reacher':
        recorder.capture_frame()
        recorder.close()

    if info['constraint']:
        num_viols += 1
        if info['recovery']:
            viol_and_recovery += 1
        else:
            viol_and_no_recovery += 1

    if "shelf" in args.env_name and info['reward'] > -0.5:
        num_successes += 1
    elif "point" in args.env_name and info['reward'] > -4:
        num_successes += 1
    elif "maze" in args.env_name and -info['reward'] < 0.03:
        num_successes += 1
    elif "cartpole" in args.env_name and episode_reward > 160:
        num_successes += 1


    if (args.use_recovery and not args.disable_learned_recovery
        ) and not args.disable_online_updates:
        all_ep_data.append({
            'obs': np.array(ep_states), 
            'ac': np.array(ep_actions),
            'constraint': np.array(ep_constraints)
        })
        if i_episode % args.recovery_policy_update_freq == 0 and not (
                args.ddpg_recovery or args.Q_sampling_recovery
                or args.DGD_constraints):
            if not args.vismpc_recovery:
                train_recovery([ep_data['obs'] for ep_data in all_ep_data],
                               [ep_data['ac'] for ep_data in all_ep_data])
                all_ep_data = []
            else:
                recovery_policy.train_dynamics(
                    i_episode, recovery_memory
                )  # Tbh we could train this on everything collected, but are not right now
        if i_episode % args.critic_safe_update_freq == 0 and args.use_recovery:
            if args.env_name in [
                    'simplepointbot0', 'simplepointbot1', 'maze', 'image_maze'
            ]:
                plot = 0
            else:
                plot = False
            if args.use_value:
                agent.train_safety_critic(
                    i_episode,
                    recovery_memory,
                    agent.policy_sample,
                    training_iterations=50,
                    batch_size=100,
                    plot=plot)

    writer.add_scalar('reward/train', episode_reward, i_episode)
    writer.add_scalar('total_violations', total_viols, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".
          format(i_episode, total_numsteps, episode_steps,
                 round(episode_reward, 2)))
    print_episode_info(train_rollouts[-1])
    print("Num Violations So Far: %d" % num_viols)
    print("Violations with Recovery: %d" % viol_and_recovery)
    print("Violations with No Recovery: %d" % viol_and_no_recovery)
    print("Num Successes So Far: %d" % num_successes)

    if total_numsteps > args.num_steps or i_episode > args.num_eps:
        break

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 1
        for j in range(episodes):
            test_rollouts.append([])
            state = env.reset()

            # TODO; clean up the following code
            if 'maze' in args.env_name:
                im_list = [env._get_obs(images=True)]
            elif 'shelf' in args.env_name:
                im_list = [env.render().squeeze()]
            elif 'cartpole' in args.env_name:
                im_list = [env.get_image()]

            if args.cnn:
                state = process_obs(state, args.env_name)

            episode_reward = 0
            episode_steps = 0
            done = False
            while not done:
                action, real_action, recovery_used = get_action(
                    state, env, agent, recovery_policy, args, train=False)
                next_state, reward, done, info = env.step(real_action)  # Step
                info['recovery'] = recovery_used
                done = done or episode_steps == env._max_episode_steps

                # TODO: clean up the following code
                if 'maze' in args.env_name:
                    im_list.append(env._get_obs(images=True))
                elif 'shelf' in args.env_name:
                    im_list.append(env.render().squeeze())
                elif 'cartpole' in args.env_name:
                    im_list.append(env.get_image())

                if args.cnn:
                    next_state = process_obs(next_state, args.env_name)

                test_rollouts[-1].append(info)
                episode_reward += reward
                episode_steps += 1
                state = next_state

            print_episode_info(test_rollouts[-1])
            avg_reward += episode_reward

            if 'maze' in args.env_name or 'shelf' in args.env_name or 'cartpole' in args.env_name:
                npy_to_gif(
                    im_list,
                    osp.join(logdir, "test_" + str(i_episode) + "_" + str(j)))
            

            # Save Replay Buffer
            if "HalfCheetah" in args.env_name and args.save_replay:
                with open("demos/halfcheetah_disabled_no_task/constraint_demos_5"  + ".pkl", 'wb') as handle:
                    pickle.dump(recovery_memory.buffer, handle)
            elif "cartpole" in args.env_name and args.save_replay:
                with open("demos/cartpole_no_task/constraint_demos_test"  + ".pkl", 'wb') as handle:
                    pickle.dump(recovery_memory.buffer, handle)
            elif "Ant-Disabled" in args.env_name and args.save_replay:
                with open("demos/ant_disabled_no_task/constraint_demos_3"  + ".pkl", 'wb') as handle:
                    pickle.dump(recovery_memory.buffer, handle)

        avg_reward /= episodes
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(
            episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    dump_logs(test_rollouts, train_rollouts, logdir)

env.close()
