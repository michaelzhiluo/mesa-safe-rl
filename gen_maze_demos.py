from env.maze import *
from env.mazes import Maze1Navigation, Maze2Navigation, Maze3Navigation, Maze4Navigation, Maze5Navigation, Maze6Navigation
import numpy as np
import pickle



def get_random_episodes(num_transitions,
                           images=False,
                           save_rollouts=False,
                           task_demos=False,
                           env_cls = None):
    env = env_cls()
    transitions = []
    num_constraints = 0
    total = 0
    rollouts = []
    done = True
    discount = 0.5
    # Rollout episodes instead
    while True:
        if done:
            if len(rollouts):
                mc_reward =0
                for transition in rollouts[::-1]:
                    if transition[2] >= 1.0:
                        mc_reward = 1.0
                    else:
                        mc_reward = transition[2] + discount * mc_reward
                    transition.append(mc_reward)
                transitions.extend(rollouts)
                if total > num_transitions //2:
                    break

            sample = np.random.uniform(0, 1, 1)[0]
            if sample < 0.3:  # maybe make 0.2 to 0.3
                mode = 'e'
            elif sample < 0.6:
                mode = 'm'
            else:
                mode = 'h'
            state = env.reset(mode, check_constraint=False, demos=True)
            rollouts = []
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        done = len(rollouts)==19
        constraint = info['constraint']
        rollouts.append([state, action, constraint, next_state, not done]) 
        total += 1
        num_constraints += int(constraint)
        state = next_state

    rollouts = []
    while True:
        if done:
            if len(rollouts):
                mc_reward =0
                for transition in rollouts[::-1]:
                    if transition[2] >= 1.0:
                        mc_reward = 1.0
                    else:
                        mc_reward = transition[2] + discount * mc_reward
                    transition.append(mc_reward)
                transitions.extend(rollouts)
                if total > num_transitions:
                    break

            sample = np.random.uniform(0, 1, 1)[0]
            if sample < 0.3:  # maybe make 0.2 to 0.3
                mode = 'e'
            elif sample < 0.6:
                mode = 'm'
            else:
                mode = 'h'
            state = env.reset(mode, check_constraint=False, demos=True)
            rollouts = []
        action = env.expert_action()
        next_state, reward, done, info = env.step(action)
        constraint = info['constraint']
        done = len(rollouts)==19
        rollouts.append([state, action, constraint, next_state, not done]) 
        total += 1
        num_constraints += int(constraint)
        state = next_state

    print("data dist", total, num_constraints)
    return transitions


constraint_demo_data = get_random_episodes(30000, env_cls= MazeNavigation)

num_constraint_transitions = 0
num_constraint_violations = 0
for transition in constraint_demo_data:
    num_constraint_violations += int(transition[2])
    num_constraint_transitions += 1
print("Number of Constraint Transitions: ",
      num_constraint_transitions)
print("Number of Constraint Violations: ",
      num_constraint_violations)

with open("demos/maze/constraint_demos.pkl", 'wb') as handle:
    pickle.dump(constraint_demo_data, handle)