import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from tensorboardX import SummaryWriter
import cv2
import os
import moviepy.editor as mpy
from env.shelf_dynamic_env import ShelfDynamicEnv
import pickle

HYPERPARAMS = {
    'T': 25,  # length of each episode
    'image_height': 48,
    'image_width': 64,
}


def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


def process_obs(obs):
    agent_img_height = HYPERPARAMS['image_height']
    agent_img_width = HYPERPARAMS['image_width']
    im = obs
    im = cv2.resize(
        im, (agent_img_width, agent_img_height), interpolation=cv2.INTER_AREA)
    im = np.transpose(im, (2, 0, 1))
    return im


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument(
    '--env-name',
    default="ShelfEnv",
    help='Mujoco Gym environment (default: ShelfEnv')
parser.add_argument(
    '--start_steps',
    type=int,
    default=5000,
    metavar=
    'N',  # TODO: think about what this is approperiate to be...maybe even lower, or make it higher
    # so can explore sufficiently after the demos are over??
    help='Steps sampling random actions (default: 10000)')
parser.add_argument(
    '--num_demos',
    type=int,
    default=250,
    metavar='N',
    help='num demos (default: 250)')
parser.add_argument(
    '--seed',
    type=int,
    default=123456,
    metavar='N',
    help='random seed (default: 123456)')
parser.add_argument(
    '--cnn', action="store_true", help='visual observations (default: False)')
parser.add_argument(
    '--cuda', action="store_true", help='run on CUDA (default: False)')
parser.add_argument(
    '--demo_filter_constraints',
    action="store_true",
    help='make sure all demos satisfy constraints (default: False)')
parser.add_argument('--demo_quality', default='high')
parser.add_argument('--dense_reward', action="store_true")
parser.add_argument('--fixed_env', action="store_true")
parser.add_argument('--gt_state', action="store_true")
parser.add_argument('--early_termination', action="store_true")
parser.add_argument('--early_termination_success', action="store_true")
parser.add_argument(
    '--use_constraint_penalty',
    action="store_true",
    help='use constraints penalty (default: False)')
parser.add_argument(
    '--constraint_penalty',
    type=int,
    default=1,
    metavar='N',
    help='constraint penalty (default: 10)')
parser.add_argument('--constraint_demos', action="store_true")
parser.add_argument('--save_rollouts', action="store_true")

args = parser.parse_args()

# Environment
env = gym.make('ShelfReach-v0')

torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

print("ENV STUFF")
print("OBSERVATION SPACE", env.observation_space)
print("ACTION SPACE", env.action_space.low)
print("ACTION SPACE", env.action_space.high)

# Training Loop
total_numsteps = 0
updates = 0

demo_transitions = []
demo_rollouts = []
i_demos = 0
while i_demos < args.num_demos:
    if i_demos % 50 == 0:
        print("Demo #: ", i_demos)
    state = env.reset()
    demo_rollouts.append([])
    if not args.gt_state:
        state = process_obs(state)
    episode_steps = 0
    episode_reward = 0
    episode_constraints = 0
    done = False

    t = 0
    while not done:
        if args.constraint_demos:
            action = env.expert_action(t, noise_std=0.05)
        else:
            action = env.expert_action(t, noise_std=0.005)

        next_state, reward, done, info = env.step(action)  # Step

        if episode_steps == env._max_episode_steps:
            done = True

        if done and reward > 0:
            reward = 5
            info['reward'] = 5

        constraint = info['constraint']
        if args.use_constraint_penalty and constraint:
            reward += args.constraint_penalty * (-int(constraint))

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        episode_constraints += constraint

        mask = float(not done)

        if not args.gt_state:
            next_state = process_obs(next_state)

        if args.constraint_demos:
            if constraint:
                demo_transitions.append((state, action, constraint, next_state,
                                         mask))
                demo_rollouts[-1].append((state, action, constraint,
                                          next_state, mask))
            else:
                if np.random.random() < 0.1:
                    demo_transitions.append((state, action, constraint,
                                             next_state, mask))
                    demo_rollouts[-1].append((state, action, constraint,
                                              next_state, mask))
        else:
            demo_transitions.append((state, action, reward, next_state, mask))
            demo_rollouts[-1].append((state, action, constraint, next_state,
                                      mask))

        state = next_state
        t += 1

    print("DEMO EPISODE REWARD", episode_reward)
    print("DEMO EPISODE CONSTRAINTS", episode_constraints)
    print("DEMO EPISODE STEPS", episode_steps)

    if not args.constraint_demos:
        if episode_reward > 0 and episode_constraints == 0:
            i_demos += 1
        else:
            # Remove last rollout if it doesn't do the task...
            demo_transitions = demo_transitions[:-t]
            demo_rollouts.pop()
    else:
        i_demos += 1

if args.constraint_demos:
    f_name = "constraint_demos"
    if args.save_rollouts:
        f_name += "_rollouts"
    if not args.gt_state:
        f_name += "_images"
    f_name += ".pkl"

    if not args.save_rollouts:
        pickle.dump(demo_transitions,
                    open(os.path.join("demos/shelf_reach", f_name), "wb"))
    else:
        pickle.dump(demo_rollouts,
                    open(os.path.join("demos/shelf_reach", f_name), "wb"))
else:
    f_name = "task_demos"
    if args.save_rollouts:
        f_name += "_rollouts"
    if not args.gt_state:
        f_name += "_images"
    f_name += ".pkl"

    if not args.save_rollouts:
        pickle.dump(demo_transitions,
                    open(os.path.join("demos/shelf_reach", f_name), "wb"))
    else:
        pickle.dump(demo_rollouts,
                    open(os.path.join("demos/shelf_reach", f_name), "wb"))
