import os.path as osp
import numpy as np
import pickle
import matplotlib.pyplot as plt

experiment_map = {
	"pointbot0": {
		"algs": {
			"recovery": "2020-04-15_18-07-44_SAC_simplepointbot0_Gaussian_",
			"sac_norecovery": "2020-04-15_17-18-59_SAC_simplepointbot0_Gaussian_",
			"sac_penalty1": "2020-04-15_17-21-40_SAC_simplepointbot0_Gaussian_",
			"sac_penalty10": "2020-04-15_18-01-42_SAC_simplepointbot0_Gaussian_",
			"sac_penalty100": "2020-04-15_18-24-50_SAC_simplepointbot0_Gaussian_",
			"q-filter": "2020-04-16_12-46-37_SAC_simplepointbot0_Gaussian_"
		},
		"outfile": "pointbot0.png"
	},
	"pointbot1": {
		"algs": {
			"recovery": "2020-04-15_20-35-58_SAC_simplepointbot1_Gaussian_",
			"sac_norecovery": "2020-04-15_21-42-14_SAC_simplepointbot1_Gaussian_",
			"sac_penalty1": "2020-04-15_21-42-32_SAC_simplepointbot1_Gaussian_",
			"sac_penalty10": "2020-04-15_21-43-02_SAC_simplepointbot1_Gaussian_",
			"sac_penalty100": "2020-04-15_21-43-28_SAC_simplepointbot1_Gaussian_"
		},
		"outfile": "pointbot1.png"
	}
}


names = {
	"sac_norecovery": "SAC",
	"sac_penalty1": "SAC (penalty 1)",
	"sac_penalty10": "SAC (penalty 10)",
	"sac_penalty100": "SAC (penalty 100)",
	"recovery": "SAC + Recovery",
	"q-filter": "Q-Filter"
}


colors = {
	"sac_norecovery": "g",
	"sac_penalty1": "orange",
	"sac_penalty10": "black",
	"sac_penalty100": "purple",
	"recovery": "red",
	"q-filter": "blue"
}


def plot_experiment(experiment):
	fig, axs = plt.subplots(2, figsize=(16, 9))
	axs[0].title.set_text("Constraint Violations vs. Episode")
	# axs[0].set_ylim(-0.1, 1.1)
	axs[0].set_xlabel("Episode")
	axs[0].set_ylabel("Num Constraint Violations")

	axs[1].title.set_text("Reward vs. Episode")
	axs[1].set_ylim(-10000, 0)
	axs[1].set_xlabel("Episode")
	axs[1].set_ylabel("Reward")

	for alg in experiment_map[experiment]["algs"]:
		exp_dir = experiment_map[experiment]["algs"][alg]
		fname = osp.join("runs", exp_dir, "run_stats.pkl")
		with open(fname, "rb") as f:
			data = pickle.load(f)
		train_stats = data['train_stats']

		train_violations = []
		train_rewards = []
		for traj_stats in train_stats:
			train_violations.append([])
			train_rewards.append(0)
			for step_stats in traj_stats:
				train_violations[-1].append(step_stats['constraint'])
				train_rewards[-1] += step_stats['reward']


		train_violations = np.array(train_violations).sum(1) > 0
		train_violations = np.cumsum(train_violations)
		train_rewards = np.array(train_rewards)

		axs[0].plot(train_violations, c=colors[alg], label=names[alg])
		axs[1].plot(train_rewards, c=colors[alg], label=names[alg])

	axs[0].legend(loc="lower right")
	axs[1].legend(loc="lower right")
	plt.savefig(experiment_map[experiment]["outfile"])
	plt.show()


experiment = "pointbot0"

if __name__ == '__main__':
	plot_experiment(experiment)
