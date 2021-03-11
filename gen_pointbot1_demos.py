from env.simplepointbot1 import SimplePointBot, SimplePointBotTeacher
import numpy as np
import pickle

def get_random_transitions_pointbot1(w1,
                           w2,
                           discount,
                           num_transitions,
                           task_demos=False,
                           save_rollouts=False):
    env = SimplePointBot(w1 = w1, w2 = w2)
    transitions = []
    rollouts = []
    done = True
    total =0
    if w1 is None or w2 is None:
        w1 = 0.0
        w2 = 0.0
    print(w1, w2, discount, len(transitions))
    while True:
        if done:
            if len(rollouts):
                mc_reward =0
                for transition in rollouts[::-1]:
                    mc_reward = transition[2] + discount * mc_reward
                    transition.append(mc_reward)
                if total > num_transitions / 3:
                    print(total, num_transitions)
                    break

            state = np.array(
                [np.random.uniform(-40, 10),
                np.random.uniform(-25, 25)])
            while env.obstacle(state):
                state = np.array(
                    [np.random.uniform(-40, 10),
                     np.random.uniform(-25, 25)])
            rollouts = []

        action = np.clip(np.random.randn(2), -1, 1)
        next_state = env._next_state(state, action, override=True)
        constraint = env.obstacle(next_state)
        done = constraint or len(rollouts)==10
        reward = env.step_cost(state, action)
        rollouts.append([state, action, constraint, next_state,
                             not constraint])
        state = next_state
        total+=1

    rollouts = []
    done = True
    total = 0
    print(w1, w2, discount, len(transitions))
    while True:
        if done:
            if len(rollouts):
                mc_reward =0
                for transition in rollouts[::-1]:
                    mc_reward = transition[2] + discount * mc_reward
                    transition.append(mc_reward)
                transitions.extend(rollouts)
                if total > num_transitions /4:
                    break

            state = np.array(
            [np.random.uniform(-35-w1, -30-w1),
             np.random.uniform(-12, 12)])
            rollouts = []

        action = np.clip(
                np.array([np.random.uniform(0.5, 1, 1),
                          np.random.randn(1)]), -1, 1).ravel()
        next_state = env._next_state(state, action, override=True)
        constraint = env.obstacle(next_state)
        done = constraint or len(rollouts)==10
        reward = env.step_cost(state, action)
        rollouts.append([state, action, constraint, next_state,
                             not constraint])
        state = next_state
        total+=1

    rollouts = []
    done = True
    total = 0
    print(w1, w2, discount, len(transitions))
    while True:
        if done:
            if len(rollouts):
                mc_reward =0
                for transition in rollouts[::-1]:
                    mc_reward = transition[2] + discount * mc_reward
                    transition.append(mc_reward)
                transitions.extend(rollouts)
                if total > num_transitions /4:
                    break

            state = np.array(
            [np.random.uniform(-20+w1, -15+w1),
             np.random.uniform(-12, 12)])
            rollouts = []

        action = np.clip(
                np.array([np.random.uniform(-1, -0.5, 1),
                          np.random.randn(1)]), -1, 1).ravel()
        next_state = env._next_state(state, action, override=True)
        constraint = env.obstacle(next_state)
        done = constraint or len(rollouts)==10
        reward = env.step_cost(state, action)
        rollouts.append([state, action, constraint, next_state,
                             not constraint])
        state = next_state
        total+=1

    rollouts = []
    done = True
    total = 0
    print(w1, w2, discount, len(transitions))
    while True:
        if done:
            if len(rollouts):
                mc_reward =0
                for transition in rollouts[::-1]:
                    mc_reward = transition[2] + discount * mc_reward
                    transition.append(mc_reward)
                transitions.extend(rollouts)
                if total > num_transitions /4:
                    break

            state = np.array(
            [np.random.uniform(-30-w1, -20-w1),
             np.random.uniform(10+w2, 15+w2)])
            rollouts = []

        action = np.clip(
                np.array([np.random.randn(1),
                          np.random.uniform(-1, -0.5, 1)]), -1, 1).ravel()
        next_state = env._next_state(state, action, override=True)
        constraint = env.obstacle(next_state)
        done = constraint or len(rollouts)==9
        reward = env.step_cost(state, action)
        rollouts.append([state, action, constraint, next_state,
                             not constraint])
        state = next_state
        total+=1

    rollouts = []
    done = True
    total = 0
    print(w1, w2, discount, len(transitions))
    while True:
        if done:
            if len(rollouts):
                mc_reward =0
                for transition in rollouts[::-1]:
                    mc_reward = transition[2] + discount * mc_reward
                    transition.append(mc_reward)
                transitions.extend(rollouts)
                if total > num_transitions /4:
                    break

            state = np.array(
            [np.random.uniform(-30-w1, -20-w1),
             np.random.uniform(-15-w2, -10-w2)])
            rollouts = []

        action = np.clip(
                np.array([np.random.randn(1),
                          np.random.uniform(0.5, 1, 1)]), -1, 1).ravel()
        next_state = env._next_state(state, action, override=True)
        constraint = env.obstacle(next_state)
        done = constraint or len(rollouts)==9
        reward = env.step_cost(state, action)
        rollouts.append([state, action, constraint, next_state,
                             not constraint])
        state = next_state
        total+=1
    print(len(transitions))
    return transitions



if __name__ == '__main__':
    counter =0
    num_transitions = 30000
    for i in range(0, 25):
        print(counter)
        w_1 = np.random.uniform(low=-5.0, high=5.0)
        w_2 = np.random.uniform(low=-5.0, high=5.0)
        constraint_demo_data = get_random_transitions_pointbot1(w1=None, w2=None, discount=0.65, num_transitions = num_transitions)

        num_constraint_transitions = 0
        num_constraint_violations = 0
        for transition in constraint_demo_data:
            num_constraint_violations += int(transition[2])
            num_constraint_transitions += 1
        print("Number of Constraint Transitions: ",
              num_constraint_transitions)
        print("Number of Constraint Violations: ",
              num_constraint_violations)

        with open("demos/pointbot1_dynamics/constraint_demos_" + str(counter) + ".pkl", 'wb') as handle:
            pickle.dump(constraint_demo_data, handle)
        counter+=1