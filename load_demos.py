import numpy as np
import pickle
import os.path as osp
import os

def load_demos(demo_dir):
	print("Demo Directory", osp.join("demos", demo_dir, "run_stats.pkl"))
	with open(osp.join("demos", demo_dir, "run_stats.pkl"), "rb") as f:
		data = pickle.load(f)

	safe_transitions = []
	unsafe_transitions = []
	all_transitions = []
	t, s = 0, 0
	for traj in data['train_stats']:
		for step_stats in traj:
			t += 1
			state = step_stats['state']
			action = step_stats['action']
			constraint = step_stats['constraint']
			next_state = step_stats['next_state']
			done = 0
			if constraint:
				unsafe_transitions.append((state, action, constraint, next_state, done))
				s += 1
			else:
				safe_transitions.append((state, action, constraint, next_state, done))
			all_transitions.append((state, action, constraint, next_state, done))
	print("Total transitions", t, "Violating Transitions", s) # to modify proportions later
	return all_transitions

if __name__ == '__main__':
	load_demos("reacher")
