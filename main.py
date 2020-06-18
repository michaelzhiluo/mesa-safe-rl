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
from replay_memory import ReplayMemory
from MPC import MPC
from dotmap import DotMap
from config import create_config
import os
from env.simplepointbot0 import SimplePointBot
import moviepy.editor as mpy
from video_recorder import VideoRecorder
import cv2

torchify = lambda x: torch.FloatTensor(x).to('cuda')

def set_seed(seed, env):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

def dump_logs(test_rollouts, train_rollouts, logdir):
    data = {
        "test_stats": test_rollouts,
        "train_stats": train_rollouts
    }
    with open(osp.join(logdir, "run_stats.pkl"), "wb") as f:
        pickle.dump(data, f)

def print_episode_info(rollout):
    num_violations = 0
    for inf in rollout:
        num_violations += int(inf['constraint'])
    print("final reward: %f"%rollout[-1]["reward"])
    print(rollout[-1]["state"])
    print("num violations: %d"%num_violations)


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
        assert(False)
    cfg.pprint()
    return cfg

def experiment_setup(logdir, args):
    if args.use_recovery and not args.disable_learned_recovery:
        cfg = recovery_config_setup(args)
        recovery_policy = MPC(cfg.ctrl_cfg)
        env = cfg.ctrl_cfg.env
    else:
        recovery_policy = None
        env = gym.make(ENV_ID[args.env_name])
    set_seed(args.seed, env)
    agent = agent_setup(env, logdir, args)
    if args.use_recovery and not args.disable_learned_recovery:
        if args.use_value:
            recovery_policy.update_value_func(agent.V_safe)
        elif args.use_qvalue: 
            recovery_policy.update_value_func(agent.Q_safe)
    return agent, recovery_policy, env

def agent_setup(env, logdir, args):
    if args.cnn and 'maze' in args.env_name:
        agent = SAC(env.observation_space, env.action_space, args, logdir, im_shape=(64, 64, 3))
    elif args.cnn and 'shelf' in args.env_name:
        agent = SAC(env.observation_space, env.action_space, args, logdir, im_shape=(48, 64, 3))
    else:
        agent = SAC(env.observation_space, env.action_space, args, logdir)
    return agent


def get_action(state, env, agent, recovery_policy, args, train=True, im_state=None):
    def recovery_thresh(state, action, agent, recovery_policy, args):
        if not args.use_recovery:
            return False
        critic_val = agent.safety_critic.get_value(torchify(state).unsqueeze(0), torchify(action).unsqueeze(0)) # TODO: make sure this is exactly equal to reachability_hor=1
        if args.reachability_test: # reachability test combined with safety check
            return not recovery_policy.reachability_test(state, action, args.eps_safe)
        if args.lookahead_test:
            return not recovery_policy.lookahead_test(state, action, args.eps_safe)
        if critic_val > args.eps_safe and not args.pred_time:
            return True
        elif critic_val < args.t_safe and args.pred_time:
            return True
        return False



    policy_state = im_state if im_state is not None else state
    if args.start_steps > total_numsteps and train:
        action = env.action_space.sample()  # Sample random action
    elif train:
        action = agent.select_action(policy_state)  # Sample action from policy
    else:
        action = agent.select_action(policy_state, eval=True)  # Sample action from policy

    # print("test", test)
    if recovery_thresh(state, action, agent, recovery_policy, args):
        recovery = True
        if not args.disable_learned_recovery:
            real_action = recovery_policy.act(state, 0)
        else:
            real_action = env.safe_action(state)
    else:
        recovery = False
        real_action = np.copy(action)
    return action, real_action, recovery


ENV_ID = {'simplepointbot0': 'SimplePointBot-v0', 
          'simplepointbot1': 'SimplePointBot-v1',
          'cliffwalker': 'CliffWalker-v0',
          'cliffcheetah': 'CliffCheetah-v0',
          'maze': 'Maze-v0',
          'image_maze': 'ImageMaze-v0',
          'shelf_env': 'Shelf-v0',
          'shelf_dynamic_env': 'ShelfDynamic-v0',
          'shelf_reach_env': 'ShelfReach-v0',
          'cliffpusher': 'CliffPusher-v0',
          'reacher': 'Reacher-v0',
          'car': 'Car-v0'
          }

def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


def get_constraint_demos(env, args):
    # Get demonstrations
    task_demo_data = None
    if not args.task_demos:
        if args.env_name == 'reacher':
            constraint_demo_data = pickle.load(open(osp.join("demos", "reacher", "data.pkl"), "rb"))
        elif args.env_name == 'maze':
            constraint_demo_data = pickle.load(open(osp.join("demos", "maze", "constraint_demos.pkl"), "rb"))
        elif 'shelf' in args.env_name:
            folder_name = args.env_name.split('_env')[0]
            constraint_demo_data = pickle.load(open(osp.join("demos", folder_name, "constraint_demos.pkl"), "rb"))
        else:
            constraint_demo_data = env.transition_function(args.num_constraint_transitions)
    else:
        # TODO: cleanup, for now this is hard-coded for maze
        if args.cnn and args.env_name == 'maze':
            constraint_demo_data, task_demo_data_images = env.transition_function(args.num_constraint_transitions, task_demos=args.task_demos, images=True)
            constraint_demo_data = pickle.load(open(osp.join("demos", "maze", "constraint_demos.pkl"), "rb"))
        elif 'shelf' in args.env_name:
            folder_name = args.env_name.split('_env')[0]
            if args.cnn:
                task_demo_data = pickle.load(open(osp.join("demos", folder_name, "task_demos_images.pkl"), "rb"))
            else:
                task_demo_data = pickle.load(open(osp.join("demos", folder_name, "task_demos.pkl"), "rb"))
            constraint_demo_data = pickle.load(open(osp.join("demos", folder_name, "constraint_demos.pkl"), "rb"))
            # constraint_demo_data = None # TODO: temp fix later
        else:
            constraint_demo_data, task_demo_data = env.transition_function(args.num_constraint_transitions, task_demos=args.task_demos)
    return constraint_demo_data, task_demo_data


def train_recovery(states, actions, next_states=None, epochs=50):
    if next_states is not None:
        recovery_policy.train(states, actions, random=True, next_obs=next_states, epochs=epochs)
    else:
        recovery_policy.train(states, actions)


# TODO: fix this for shelf env...
def process_obs(obs, env_name):
    if 'shelf' in args.env_name:
        obs = cv2.resize(obs, (64, 48), interpolation=cv2.INTER_AREA)
    im = np.transpose(obs, (2, 0, 1))
    return im

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--logdir', default="runs",
                    help='exterior log directory')
parser.add_argument('--logdir_suffix', default="",
                    help='log directory suffix')
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
parser.add_argument('--t_safe', type=float, default=80, metavar='G',
                    help='threshold constraints (default: 0.8)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G', # TODO: idk if this should be 0.005 or 0.0002...
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
parser.add_argument('--num_eps', type=int, default=1000000, metavar='N',
                    help='maximum number of episodes (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=100, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 100000)')
parser.add_argument('--safe_replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer for V safe (default: 100000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--cnn', action="store_true", 
                    help='visual observations (default: False)')
parser.add_argument('--critic_pretraining_steps', type=int, default=3000)

parser.add_argument('--constraint_reward_penalty', type=float, default=-1)
parser.add_argument('--safety_critic_penalty', type=float, default=-1)
# For recovery policy
parser.add_argument('--use_target_safe', action="store_true")
parser.add_argument('--disable_learned_recovery', action="store_true")
parser.add_argument('--use_recovery', action="store_true")
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
parser.add_argument('--num_task_transitions', type=int, default=10000000)
parser.add_argument('--num_constraint_transitions', type=int, default=10000) # Make this 20K+ for original shelf env stuff, trying with fewer rn
parser.add_argument('--reachability_hor', type=int, default=2)

parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                    help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                    help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
args = parser.parse_args()

# TODO: clean this up later
if 'shelf' in args.env_name:
    args.num_constraint_transitions = 20000

if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

logdir = os.path.join(args.logdir, '{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, args.logdir_suffix))
print("LOGDIR: ", logdir)
writer = SummaryWriter(logdir=logdir)
pickle.dump(args, open(os.path.join(logdir, "args.pkl"), "wb") )

agent, recovery_policy, env = experiment_setup(logdir, args)

# Memory
memory = ReplayMemory(args.replay_size)
recovery_memory = ReplayMemory(args.safe_replay_size)

# Training Loop
total_numsteps = 0
updates = 0

task_demos = args.task_demos

constraint_demo_data, task_demo_data = get_constraint_demos(env, args)

# Train recovery policy and associated value function on demos
if args.use_recovery and not args.disable_learned_recovery:
    demo_data_states = np.array([d[0] for d in constraint_demo_data])
    demo_data_actions = np.array([d[1] for d in constraint_demo_data])
    demo_data_next_states = np.array([d[3] for d in constraint_demo_data])
    train_recovery(demo_data_states, demo_data_actions, demo_data_next_states, epochs=50)
    num_constraint_transitions = 0
    num_viols = 0
    for transition in constraint_demo_data:
        recovery_memory.push(*transition)
        num_viols += int(transition[2])
        num_constraint_transitions += 1
        if num_constraint_transitions == args.num_constraint_transitions:
            break
    print("Number of Constraint Transitions: ", num_constraint_transitions)
    print("Number of Constraint Violations: ", num_viols)
    if args.env_name in ['simplepointbot0', 'simplepointbot1', 'maze', 'image_maze']:
        plot = True
    else:
        plot = False
    agent.train_safety_critic(0, recovery_memory, agent.policy_sample, plot=plot)

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
        agent.update_parameters(memory, min(args.batch_size, num_task_transitions), updates)
        updates += 1


test_rollouts = []
train_rollouts = []
all_ep_data = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    if args.env_name == 'reacher':
        recorder = VideoRecorder(env, osp.join(logdir, 'video_{}.mp4'.format(i_episode)))
    # TODO; cleanup for now this is hard-coded for maze
    if args.cnn and 'maze' in args.env_name:
        im_state = process_obs(env._get_obs(images=True), args.env_name)
    elif args.cnn and 'shelf' in args.env_name:
        im_state = process_obs(env.render(), args.env_name)
    else:
        im_state = None

    train_rollouts.append([])
    ep_states = [state]
    ep_actions = []
    ep_constraints = []

    while not done:
        if args.env_name == 'reacher':
            recorder.capture_frame()

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

        action, real_action, recovery_used = get_action(state, env, agent, recovery_policy, args, im_state=im_state)
        next_state, reward, done, info = env.step(real_action) # Step
        info['recovery'] = recovery_used


        # TODO; cleanup for now this is hard-coded for maze
        if args.cnn and 'maze' in args.env_name:
            im_next_state = process_obs(env._get_obs(images=True), args.env_name)
        elif args.cnn and 'shelf' in args.env_name:
            im_next_state = process_obs(env.render(), args.env_name)

        train_rollouts[-1].append(info)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        if args.constraint_reward_penalty > 0 and info['constraint']:
            reward -= args.constraint_reward_penalty

        if args.safety_critic_penalty > 0:
            critic_val = agent.safety_critic.get_value(torchify(state).unsqueeze(0), torchify(action).unsqueeze(0)).detach().cpu().numpy()[0, 0]
            reward -= args.safety_critic_penalty * critic_val

        mask = float(not done)
        done = done or episode_steps == env._max_episode_steps
        # TODO; cleanup for now this is hard-coded for maze
        if args.cnn and ('maze' in args.env_name or 'shelf' in args.env_name):
            memory.push(im_state, action, reward, im_next_state, mask)
        else:
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

        if args.use_recovery:
            recovery_memory.push(state, action, info['constraint'], next_state, mask)
        state = next_state
        if args.cnn and ('maze' in args.env_name or 'shelf' in args.env_name):
            im_state = im_next_state

        ep_states.append(state)
        ep_actions.append(real_action)
        ep_constraints.append([info['constraint']])

    if args.env_name == 'reacher':
        recorder.capture_frame()
        recorder.close()

    if args.use_recovery and not args.disable_learned_recovery:
        all_ep_data.append({'obs': np.array(ep_states), 'ac': np.array(ep_actions), 'constraint': np.array(ep_constraints)})
        if i_episode % args.recovery_policy_update_freq == 0:
            train_recovery([ep_data['obs'] for ep_data in all_ep_data], [ep_data['ac'] for ep_data in all_ep_data])
            all_ep_data = []
        if i_episode % args.critic_safe_update_freq == 0 and args.use_recovery:
            if args.env_name in ['simplepointbot0', 'simplepointbot1', 'maze', 'image_maze']:
                plot = True
            else:
                plot = False
            agent.train_safety_critic(i_episode, recovery_memory, agent.policy_sample, training_iterations=50, batch_size=100, plot=plot)

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    print_episode_info(train_rollouts[-1])

    if total_numsteps > args.num_steps or i_episode > args.num_eps:
        break

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 5
        for j in range(episodes):
            test_rollouts.append([])
            state = env.reset()

            # TODO; clean up the following code
            if 'maze' in args.env_name:
                im_list = [env._get_obs(images=True)]
            elif 'shelf' in args.env_name:
                im_list = [env.render().squeeze()]
            if args.cnn and 'maze' in args.env_name:
                im_state = process_obs(env._get_obs(images=True), args.env_name)
            elif args.cnn and 'shelf' in args.env_name:
                im_state = process_obs(env.render(), args.env_name)

            episode_reward = 0
            episode_steps = 0
            done = False
            while not done:
                action, real_action, recovery_used = get_action(state, env, agent, recovery_policy, args, train=False, im_state=im_state)
                next_state, reward, done, info = env.step(real_action) # Step
                info['recovery'] = recovery_used
                done = done or episode_steps == env._max_episode_steps

                # TODO: clean up the following code
                if 'maze' in args.env_name:
                    im_list.append(env._get_obs(images=True))
                elif 'shelf' in args.env_name:
                    im_list.append(env.render().squeeze())
                if args.cnn and 'maze' in args.env_name:
                    im_next_state = process_obs(env._get_obs(images=True), args.env_name)
                elif args.cnn and 'shelf' in args.env_name:
                    im_next_state = process_obs(env.render(), args.env_name)

                test_rollouts[-1].append(info)
                episode_reward += reward
                episode_steps += 1
                state = next_state

                if args.cnn and ('maze' in args.env_name or 'shelf' in args.env_name):
                    im_state = im_next_state

            print_episode_info(test_rollouts[-1])
            avg_reward += episode_reward

            if 'maze' in args.env_name or 'shelf' in args.env_name:
                npy_to_gif(im_list, osp.join(logdir, "test_" + str(i_episode) + "_" + str(j)))

        avg_reward /= episodes
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    dump_logs(test_rollouts, train_rollouts, logdir)
        
env.close()
