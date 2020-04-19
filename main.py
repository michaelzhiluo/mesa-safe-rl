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
import moviepy.editor as mpy

torchify = lambda x: torch.FloatTensor(x).to('cuda')

ENV_ID = {'simplepointbot0': 'SimplePointBot-v0', 
          'simplepointbot1': 'SimplePointBot-v1',
          'cliffwalker': 'CliffWalker-v0',
          'cliffcheetah': 'CliffCheetah-v0',
          'maze': 'Maze-v0',
          'shelf_env': 'Shelf-v0',
          'cliffpusher': 'CliffPusher-v0'
          }

def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')

def process_obs(obs):
    im = np.transpose(obs, (2, 0, 1))
    return im

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
parser.add_argument('--tau', type=float, default=0.0002, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--tau_safe', type=float, default=0.005, metavar='G',
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
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=100, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
                    help='size of replay buffer (default: 100000)')
parser.add_argument('--safe_replay_size', type=int, default=100000, metavar='N',
                    help='size of replay buffer for V safe (default: 100000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--cnn', action="store_true", 
                    help='visual observations (default: False)')
parser.add_argument('--critic_pretraining_steps', type=int, default=200)

parser.add_argument('--constraint_reward_penalty', type=float, default=-1)
# For recovery policy
parser.add_argument('--use_target_safe', action="store_true")
parser.add_argument('--disable_learned_recovery', action="store_true")
parser.add_argument('--use_recovery', action="store_true")
parser.add_argument('--recovery_policy_update_freq', type=int, default=1)
parser.add_argument('--critic_safe_update_freq', type=int, default=1e10) # TODO: by default, not updating on-policy, but will need to for non-pointbot envs
parser.add_argument('--task_demos', action="store_true")
parser.add_argument('--filter', action="store_true")
parser.add_argument('--num_filter_samples', type=int, default=100)
parser.add_argument('--max_filter_iters', type=int, default=5)
parser.add_argument('--Q_safe_start_ep', type=int, default=10)

parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                    help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                    help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
parser.add_argument('--num_demo_transitions', type=int, default=10000)
args = parser.parse_args()

#TesnorboardX
logdir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "")
writer = SummaryWriter(logdir=logdir)
pickle.dump(args, open(os.path.join(logdir, "args.pkl"), "wb") )

if args.use_recovery and not args.disable_learned_recovery:
    ctrl_args = DotMap(**{key: val for (key, val) in args.ctrl_arg})
    cfg = create_config(args.env_name, "MPC", ctrl_args, args.override, logdir)
    cfg.pprint()
    cfg.ctrl_cfg.use_value = True
    recovery_policy = MPC(cfg.ctrl_cfg)
else:
    recovery_policy = None
# Environment
# env = NormalizedActions(gym.make(args.env_name))

if args.use_recovery and not args.disable_learned_recovery:
    env = cfg.ctrl_cfg.env
else:
    env = gym.make(ENV_ID[args.env_name])

torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
# TODO; cleanup for now this is hard-coded for maze
if args.cnn and args.env_name == 'maze':
    agent = SAC(env.observation_space, env.action_space, args, im_shape=(64, 64, 3))
else:
    agent = SAC(env.observation_space, env.action_space, args)

if args.use_recovery and not args.disable_learned_recovery:
    recovery_policy.update_value_func(agent.V_safe)

# Memory
memory = ReplayMemory(args.replay_size)
V_safe_memory = ReplayMemory(args.safe_replay_size)
Q_safe_memory = ReplayMemory(args.safe_replay_size)

# Training Loop
total_numsteps = 0
updates = 0

task_demos = args.task_demos or (not args.filter) or (args.eps_safe == 1)

# Get demonstrations
if not task_demos:
    constraint_demo_data = env.transition_function(args.num_demo_transitions)
else:
    # TODO: cleanup, for now this is hard-coded for maze
    if args.cnn and args.env_name == 'maze':
        constraint_demo_data, task_demo_data_images = env.transition_function(args.num_demo_transitions, task_demos=task_demos, images=True)
    else:
        constraint_demo_data, task_demo_data = env.transition_function(args.num_demo_transitions, task_demos=task_demos)

# Train recovery policy and associated value function on demos
if args.use_recovery and not args.disable_learned_recovery:
    demo_data_states = np.array([d[0] for d in constraint_demo_data])
    demo_data_actions = np.array([d[1] for d in constraint_demo_data])
    demo_data_next_states = np.array([d[3] for d in constraint_demo_data])
    recovery_policy.train(demo_data_states, demo_data_actions, random=True, next_obs=demo_data_next_states, epochs=50)

for transition in constraint_demo_data:
    V_safe_memory.push(*transition)
agent.V_safe.train(V_safe_memory)

# Train Qsafe on demos for filtering
if args.filter:
    for i, transition in enumerate(constraint_demo_data):
        if i < 100:
            Q_safe_memory.push(*transition)
    agent.Q_safe.train(Q_safe_memory, agent.policy_sample, epochs=10, training_iterations=50, batch_size=50)

# TODO: cleanup, for now this is hard-coded for maze
if args.cnn and args.env_name == 'maze':
    task_demo_data = task_demo_data_images

# If use task demos, add them to memory and train agent
if task_demos:
    for transition in task_demo_data:
        memory.push(*transition)
    for _ in range(args.critic_pretraining_steps):
        agent.update_parameters(memory, args.batch_size, updates)
        updates += 1


test_rollouts = []
train_rollouts = []

all_ep_data = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    # TODO; cleanup for now this is hard-coded for maze
    if args.cnn:
        if args.env_name == 'maze':
            im_state = env.sim.render(64, 64, camera_name= "cam0")
            im_state = process_obs(im_state)
        else:
            state = process_obs(state)

    train_rollouts.append([])
    ep_states = [state]
    ep_actions = []
    while not done:
        # print("EP STEP", episode_steps)
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
            if args.use_recovery and agent.V_safe.get_value(torch.FloatTensor(state).to('cuda').unsqueeze(0)) > args.eps_safe:
                if not args.disable_learned_recovery:
                    print("RECOVERY", agent.V_safe.get_value(torch.FloatTensor(state).to('cuda').unsqueeze(0)))
                    real_action = recovery_policy.act(state, 0)
                else:
                    real_action = env.safe_action(state)
            else:
                print("NOT RECOVERY", agent.V_safe.get_value(torch.FloatTensor(state).to('cuda').unsqueeze(0)))
                if not args.filter or i_episode < args.Q_safe_start_ep:
                    real_action = action
                else:
                    state_batch = np.tile(state, [args.num_filter_samples, 1])
                    found_safe_action = False
                    for _ in range(args.max_filter_iters):
                        action_batch = np.array([env.action_space.sample() for _ in range(args.num_filter_samples)])
                        Q_safe_values = agent.Q_safe.get_qvalue(torch.FloatTensor(state_batch).to('cuda'), torch.FloatTensor(action_batch).to('cuda')).flatten()
                        thresh_idxs = np.argwhere(Q_safe_values <= args.eps_safe).flatten() # Get indices where you are sufficiently safe
                        if len(thresh_idxs): # If there is something safe we done
                            found_safe_action = True
                            break
                    if found_safe_action:
                        filtered_state_batch = state_batch[thresh_idxs]
                        filtered_action_batch = action_batch[thresh_idxs]
                        Q_values = agent.get_critic_value(torch.FloatTensor(filtered_state_batch).to('cuda'), torch.FloatTensor(filtered_action_batch).to('cuda')).flatten() # Get Q values for filtered actions
                        real_action = filtered_action_batch[np.argmax(Q_values)]
                    else: # Backup action
                        real_action = action_batch[np.argmin(Q_safe_values)]
        else:
            # TODO; cleanup for now this is hard-coded for maze
            if args.cnn and args.env_name == 'maze':
                action = agent.select_action(im_state) 
            else:
                action = agent.select_action(state)  # Sample action from policy

            if args.use_recovery and agent.V_safe.get_value(torch.FloatTensor(state).to('cuda').unsqueeze(0)) > args.eps_safe:
                if not args.disable_learned_recovery:
                    print("RECOVERY", agent.V_safe.get_value(torch.FloatTensor(state).to('cuda')))
                    real_action = recovery_policy.act(state, 0)
                else:
                    real_action = env.safe_action(state)
            else:
                print("NOT RECOVERY HERE", agent.V_safe.get_value(torch.FloatTensor(state).to('cuda').unsqueeze(0)))
                if not args.filter or i_episode < args.Q_safe_start_ep:
                    real_action = action
                else:
                    state_batch = np.tile(state, [args.num_filter_samples, 1])
                    found_safe_action = False
                    for _ in range(args.max_filter_iters):
                         # TODO; cleanup for now this is hard-coded for maze
                        if args.cnn and args.env_name == 'maze':
                            action_batch = np.array([agent.select_action(im_state) for _ in range(args.num_filter_samples)])
                        else:
                            action_batch = np.array([agent.select_action(state) for _ in range(args.num_filter_samples)])

                        Q_safe_values = agent.Q_safe.get_qvalue(torch.FloatTensor(state_batch).to('cuda'), torch.FloatTensor(action_batch).to('cuda')).flatten()
                        thresh_idxs = np.argwhere(Q_safe_values <= args.eps_safe).flatten() # Get indices where you are sufficiently safe

                        if len(thresh_idxs): # If there is something safe we done
                            found_safe_action = True
                            break
                    if found_safe_action:
                        filtered_state_batch = state_batch[thresh_idxs]
                        filtered_action_batch = action_batch[thresh_idxs]
                        Q_values = agent.get_critic_value(torch.FloatTensor(filtered_state_batch).to('cuda'), torch.FloatTensor(filtered_action_batch).to('cuda')).flatten() # Get Q values for filtered actions
                        real_action = filtered_action_batch[np.argmax(Q_values)]
                    else: # Backup action
                        real_action = action_batch[np.argmin(Q_safe_values)]

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
        # TODO; cleanup for now this is hard-coded for maze
        if args.cnn:
            if args.env_name == 'maze':
                im_next_state = env.sim.render(64, 64, camera_name= "cam0")
                im_next_state = process_obs(im_next_state)
            else:
                next_state = process_obs(next_state)

        train_rollouts[-1].append(info)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        if args.constraint_reward_penalty > 0 and info['constraint']:
            reward -= args.constraint_reward_penalty

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        if episode_steps == env._max_episode_steps:
            done = True

        mask = float(not done)
        # done = done or episode_steps == env._max_episode_steps

        # TODO; cleanup for now this is hard-coded for maze
        if args.cnn and args.env_name == 'maze':
            memory.push(im_state, action, reward, im_next_state, mask) # Append transition to memory
        else:
            memory.push(state, action, reward, next_state, mask) # Append transition to memory


        V_safe_memory.push(state, action, info['constraint'], next_state, mask)

        state = next_state

        # TODO; cleanup for now this is hard-coded for maze
        if args.cnn and args.env_name == 'maze':
            im_state = im_next_state

        ep_states.append(state)
        ep_actions.append(real_action)

    ep_states = np.array(ep_states)
    ep_actions = np.array(ep_actions)

    if args.env_name == 'cliffwalker' or args.env_name == 'cliffcheetah':
        print("FINAL X POSITION", ep_states[-1][0])

    if args.use_recovery and not args.disable_learned_recovery:
        if i_episode % args.recovery_policy_update_freq == 0:
            recovery_policy.train(
                [ep_data['obs'] for ep_data in all_ep_data],
                [ep_data['ac'] for ep_data in all_ep_data]
            )
            all_ep_data = []
        else:
            all_ep_data.append({'obs': np.array(ep_states), 'ac': np.array(ep_actions)})

        if i_episode % args.critic_safe_update_freq == 0:
            agent.V_safe.train(V_safe_memory, epochs=10, training_iterations=50)
    if i_episode % args.critic_safe_update_freq == 0:
        if args.filter:
            agent.Q_safe.train(Q_safe_memory, agent.policy_sample, epochs=10, training_iterations=50, batch_size=50)

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
        episodes = 5
        for j in range(episodes):
            test_rollouts.append([])
            state = env.reset()
            # TODO; cleanup for now this is hard-coded for maze
            if args.env_name == 'maze':
                im_list = [env.sim.render(64, 64, camera_name= "cam0")]
            if args.cnn:
                if args.env_name == 'maze':
                    im_state = env.sim.render(64, 64, camera_name= "cam0")
                    im_state = process_obs(im_state)
                else:
                    state = process_obs(state)

            episode_reward = 0
            episode_steps = 0
            done = False
            while not done:
                # TODO; cleanup for now this is hard-coded for maze
                if args.cnn and args.env_name == 'maze':
                    action = agent.select_action(im_state, eval=True) 
                else:
                    action = agent.select_action(state, eval=True)  # Sample action from policy

                if args.use_recovery and agent.V_safe.get_value(torch.FloatTensor(state).to('cuda').unsqueeze(0)) > args.eps_safe:
                    if not args.disable_learned_recovery:
                        print("RECOVERY", agent.V_safe.get_value(torch.FloatTensor(state).to('cuda')))
                        real_action = recovery_policy.act(state, 0)
                    else:
                        real_action = env.safe_action(state)
                else:
                    print("NOT RECOVERY TEST", agent.V_safe.get_value(torch.FloatTensor(state).to('cuda').unsqueeze(0)))
                    if not args.filter or i_episode < args.Q_safe_start_ep:
                        real_action = action
                    else:
                        state_batch = np.tile(state, [args.num_filter_samples, 1])
                        found_safe_action = False
                        for _ in range(args.max_filter_iters):
                            # TODO; cleanup for now this is hard-coded for maze
                            if args.cnn and args.env_name == 'maze':
                                action_batch = np.array([agent.select_action(im_state) for _ in range(args.num_filter_samples)])
                            else:
                                action_batch = np.array([agent.select_action(state) for _ in range(args.num_filter_samples)])

                            Q_safe_values = agent.Q_safe.get_qvalue(torch.FloatTensor(state_batch).to('cuda'), torch.FloatTensor(action_batch).to('cuda')).flatten()
                            thresh_idxs = np.argwhere(Q_safe_values <= args.eps_safe).flatten() # Get indices where you are sufficiently safe
                            if len(thresh_idxs): # If there is something safe we done
                                found_safe_action = True
                                break
                        if found_safe_action:
                            filtered_state_batch = state_batch[thresh_idxs]
                            filtered_action_batch = action_batch[thresh_idxs]
                            Q_values = agent.get_critic_value(torch.FloatTensor(filtered_state_batch).to('cuda'), torch.FloatTensor(filtered_action_batch).to('cuda')).flatten() # Get Q values for filtered actions
                            real_action = filtered_action_batch[np.argmax(Q_values)]
                        else: # Backup action
                            real_action = action_batch[np.argmin(Q_safe_values)]

                next_state, reward, done, info = env.step(real_action) # Step

                if args.env_name == 'maze':
                    im_list.append(env.sim.render(64, 64, camera_name= "cam0"))
                # TODO; cleanup for now this is hard-coded for maze
                if args.cnn:
                    if args.env_name == 'maze':
                        im_next_state = env.sim.render(64, 64, camera_name= "cam0")
                        im_next_state = process_obs(im_next_state)
                    else:
                        next_state = process_obs(next_state)

                test_rollouts[-1].append(info)
                episode_reward += reward
                episode_steps += 1

                if episode_steps == env._max_episode_steps:
                    done = True

                state = next_state
                # TODO; cleanup for now this is hard-coded for maze
                if args.cnn and args.env_name == 'maze':
                    im_state = im_next_state

            num_violations = 0
            for inf in test_rollouts[-1]:
                num_violations += int(inf['constraint'])
            print("final reward: %f"%reward)
            print("num violations: %d"%num_violations)
            avg_reward += episode_reward

            if args.env_name == 'maze':
                npy_to_gif(im_list, osp.join(logdir, "test_" + str(i_episode) + "_" + str(j)))

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

