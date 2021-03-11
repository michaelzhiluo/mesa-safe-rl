from env.cartpole import CartPoleEnv, transition_function
import numpy as np
import pickle



if __name__ == '__main__':
    counter =0
    num_transitions = 10000
    for i in range(0, 20):
        print(counter)
        w_1 = np.random.uniform(0.4, 0.8)
        constraint_demo_data = transition_function(num_transitions, w_1, 0.8)
        num_constraint_transitions = 0
        num_constraint_violations = 0
        for transition in constraint_demo_data:
            num_constraint_violations += int(transition[2])
            num_constraint_transitions += 1
        print("Number of Constraint Transitions: ",
              num_constraint_transitions)
        print("Number of Constraint Violations: ",
              num_constraint_violations)

        with open("demos/cartpole/constraint_demos_" + str(counter) + ".pkl", 'wb') as handle:
            pickle.dump(constraint_demo_data, handle)
        counter+=1