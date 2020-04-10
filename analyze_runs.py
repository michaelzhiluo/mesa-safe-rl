import os.path as osp
import numpy as np
import pickle
import matplotlib.pyplot as plt

normal = "2020-04-09_14-39-35_SAC_SimplePointBot-v0_Gaussian_"
normal2 = "2020-04-09_14-42-46_SAC_SimplePointBot-v0_Gaussian_"
recovery2 = "2020-04-09_14-50-39_SAC_SimplePointBot-v0_Gaussian_"

recoveryfull = "2020-04-09_15-01-45_SAC_SimplePointBot-v0_Gaussian_"
baselinefull = "2020-04-09_15-05-57_SAC_SimplePointBot-v0_Gaussian_"

recoveryhardcode = "2020-04-09_15-17-43_SAC_SimplePointBot-v0_Gaussian_"

if __name__ == '__main__':
	fname = "run_stats.pkl"
	exp_dir = osp.join("runs", recoveryhardcode)
	# exp_dir = osp.join("runs", baselinefull)

	with open(osp.join(exp_dir, fname), "rb") as f:
		data = pickle.load(f)

	print("Train Stats")

	train_stats = data['train_stats']

	train_violations = []
	for traj_stats in train_stats:
		train_violations.append([])
		for step_stats in traj_stats:
			train_violations[-1].append(step_stats['constraint'])

	train_means = np.array(train_violations).sum(1)



	test_stats = data['test_stats']

	test_violations = []
	for traj_stats in test_stats:
		test_violations.append([])
		for step_stats in traj_stats:
			test_violations[-1].append(step_stats['constraint'])

	test_means = np.array(test_violations).sum(1)


	plt.plot(train_means, c='r')
	plt.plot(test_means)
	plt.title("Constraint Violations vs. Episode: Fixed Recovery Policy")
	plt.xlabel("Episode")
	plt.ylabel("Num Constraint Violations")
	plt.savefig("recoveryhardcodefull.png")
	plt.show()



	train_stats = data['train_stats']

	train_violations = []
	for traj_stats in train_stats:
		train_violations.append([])
		for step_stats in traj_stats:
			train_violations[-1].append(step_stats['reward'])

	train_means = np.array(train_violations).sum(1)



	test_stats = data['test_stats']

	test_violations = []
	for traj_stats in test_stats:
		test_violations.append([])
		for step_stats in traj_stats:
			test_violations[-1].append(step_stats['reward'])

	test_means = np.array(test_violations).sum(1)


	plt.plot(train_means, c='r')
	plt.plot(test_means)
	plt.title("Reward vs. Episode: Fixed Recovery Policy")
	plt.xlabel("Episode")
	plt.ylabel("Reward")
	plt.savefig("recoveryhardcoderewardsfull.png")
	plt.show()

