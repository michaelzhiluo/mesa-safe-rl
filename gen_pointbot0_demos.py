from env.simplepointbot0 import SimplePointBot, SimplePointBotTeacher
import numpy as np
import pickle

def get_random_transitions_pointbot0(w1,
                           w2,
                           discount,
                           num_transitions,
                           task_demos=False,
                           save_rollouts=False):
    env = SimplePointBot(w1 = w1, w2 = w2)
    transitions = []
    rollouts = []
    done = True
    while True:
        if done:
            if len(rollouts):
                mc_reward =0
                for transition in rollouts[::-1]:
                    mc_reward = transition[2] + discount * mc_reward
                    transition.append(mc_reward)
                transitions.extend(rollouts)
                if len(transitions) > num_transitions:
                    break

            # Reset
            if np.random.uniform(0, 1) < 0.5:
                state = np.array(
                    [np.random.uniform(-80, 50),
                     np.random.uniform(-5, -2)])
            else:
                state = np.array(
                    [np.random.uniform(-80, 50),
                     np.random.uniform(2, 5)])
            rollouts = []

        action = np.clip(np.random.randn(2), -1, 1)
        next_state = env._next_state(state, action, override=True)
        constraint = env.obstacle(next_state)
        done = constraint
        reward = env.step_cost(state, action)
        rollouts.append([state, action, constraint, next_state,
                             not constraint])
        state = next_state

    return transitions

if __name__ == '__main__':
    counter =0
    num_constraint_transitions = 30000
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i==0 and j==0:
                continue
            constraint_demo_data = get_random_transitions_pointbot0(w1=i, w2=j, discount=0.8, num_transitions = num_constraint_transitions)

            num_constraint_transitions = 0
            num_constraint_violations = 0
            for transition in constraint_demo_data:
                num_constraint_violations += int(transition[2])
                num_constraint_transitions += 1
            print("Number of Constraint Transitions: ",
                  num_constraint_transitions)
            print("Number of Constraint Violations: ",
                  num_constraint_violations)

            with open("demos/pointbot_0/constraint_demos_" + str(counter) + ".pkl", 'wb') as handle:
                pickle.dump(constraint_demo_data, handle)
            print(counter)
            counter+=1