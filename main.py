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
from replay_memory import ReplayMemory
from MPC import MPC
from dotmap import DotMap
from config import create_config
import os
from env.simplepointbot0 import SimplePointBot
from env.simplepointbot1 import SimplePointBot
# from env.maze import MazeNavigation

ENV_ID = {'simplepointbot0': 'SimplePointBot-v0', 'simplepointbot1': 'SimplePointBot-v1'}

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--gamma_safe', type=float, default=0.5, metavar='G',
                    help='discount factor for constraints (default: 0.9)')
parser.add_argument('--eps_safe', type=float, default=0.1, metavar='G',
                    help='threshold constraints (default: 0.8)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=10000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=100, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--cnn', action="store_true", 
                    help='visual observations (default: False)')

# For PETS
parser.add_argument('--learned_recovery', action="store_true")
parser.add_argument('--recovery_policy_update_freq', type=int, default=1)
parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                    help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                    help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
args = parser.parse_args()

#TesnorboardX
logdir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "")
writer = SummaryWriter(logdir=logdir)
pickle.dump(args, open(os.path.join(logdir, "args.pkl"), "wb") )

if args.learned_recovery:
    ctrl_args = DotMap(**{key: val for (key, val) in args.ctrl_arg})
    cfg = create_config(args.env_name, "MPC", ctrl_args, args.override, logdir)
    cfg.pprint()
    cfg.ctrl_cfg.use_value = True
    recovery_policy = MPC(cfg.ctrl_cfg)
else:
    recovery_policy = None
# Environment
# env = NormalizedActions(gym.make(args.env_name))

if args.learned_recovery:
    env = cfg.ctrl_cfg.env
else:
    env = gym.make(ENV_ID[args.env_name])

torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = SAC(env.observation_space, env.action_space, env.transition_function, recovery_policy, args)

if args.learned_recovery:
    recovery_policy.update_value_func(agent.value)

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

test_rollouts = []
train_rollouts = []

all_ep_data = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    train_rollouts.append([])
    ep_states = [state]
    ep_actions = []

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
            if agent.value(torch.FloatTensor(state).to('cuda').unsqueeze(0)) > args.eps_safe:
                if args.learned_recovery:
                    print("RECOVERY", agent.value(torch.FloatTensor(state).to('cuda').unsqueeze(0)))
                    real_action = recovery_policy.act(state, 0)
                else:
                    real_action = env.safe_action(state)
            else:
                print("NOT RECOVERY", agent.value(torch.FloatTensor(state).to('cuda').unsqueeze(0)))
                real_action = action
        else:
            action = agent.select_action(state)  # Sample action from policy
            if agent.value(torch.FloatTensor(state).to('cuda').unsqueeze(0)) > args.eps_safe:
                if args.learned_recovery:
                    print("RECOVERY", agent.value(torch.FloatTensor(state).to('cuda').unsqueeze(0)))
                    real_action = recovery_policy.act(state, 0)
                else:
                    real_action = env.safe_action(state)
            else:
                print("NOT RECOVERY", agent.value(torch.FloatTensor(state).to('cuda').unsqueeze(0)))
                real_action = action
        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, info = env.step(real_action) # Step
        train_rollouts[-1].append(info)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

        ep_states.append(state)
        ep_actions.append(real_action)

    ep_states = np.array(ep_states)
    ep_actions = np.array(ep_actions)

    if args.learned_recovery:
        if i_episode % args.recovery_policy_update_freq == 0:
            recovery_policy.train(
                [ep_data['obs'] for ep_data in all_ep_data],
                [ep_data['ac'] for ep_data in all_ep_data]
            )
            all_ep_data = []
        else:
            all_ep_data.append({'obs': np.array(ep_states), 'ac': np.array(ep_actions)})

    num_violations = 0
    for inf in train_rollouts[-1]:
        num_violations += int(inf['constraint'])
    print("final reward: %f"%reward)
    print("num violations: %d"%num_violations)

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            test_rollouts.append([])
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)
                if agent.value(torch.FloatTensor(state).to('cuda').unsqueeze(0)) > args.eps_safe:
                    if args.learned_recovery:
                        real_action = recovery_policy.act(state, 0)
                    else:
                        real_action = env.safe_action(state)
                else:
                    real_action = action
                next_state, reward, done, info = env.step(real_action)
                test_rollouts[-1].append(info)
                episode_reward += reward


                state = next_state
            num_violations = 0
            for inf in test_rollouts[-1]:
                num_violations += int(inf['constraint'])
            print("final reward: %f"%reward)
            print("num violations: %d"%num_violations)
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

        data = {
            "test_stats": test_rollouts,
            "train_stats": train_rollouts
        }
        with open(osp.join(logdir, "run_stats.pkl"), "wb") as f:
            pickle.dump(data, f)

env.close()

