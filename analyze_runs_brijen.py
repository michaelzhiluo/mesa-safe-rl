import os.path as osp
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

from plotting_utils import get_color, get_legend_name


def get_directory(dirname, suffix, parent="data"):
    dirs = [osp.join(parent, dirname, d) for d in os.listdir(osp.join(parent, dirname)) if d.endswith(suffix)]
    return dirs

experiment_map = {
    "maze": {
        "algs": {
        "sac_norecovery": get_directory("maze", "vanilla"),
            # "sac_penalty1": get_directory("maze", "reward_1"),
            # "sac_penalty10": get_directory("maze", "reward_10"),
            "sac_penalty100": get_directory("maze", "reward_100"),
            # "sac_lagrangian_1": get_directory("maze", "nu_1_update"),
            # "sac_lagrangian_10": get_directory("maze", "nu_10_update"),
            "sac_lagrangian_100": get_directory("maze", "nu_100_update"),
            # "lookahead": get_directory("maze", "lookahead"),
            "recovery": get_directory("maze", "recovery"),
            "test": get_directory("temp", "test"),
        },
        "outfile": "maze_plot.png"
    },
    "pointbot0": {
        "algs": {
            "sac_norecovery": get_directory("pointbot0", "vanilla"),
            # "sac_penalty1": get_directory("pointbot0", "reward_1"),
            # "sac_penalty10": get_directory("pointbot0", "reward_10"),
            # "sac_penalty100": get_directory("pointbot0", "reward_100"),
            "sac_penalty": get_directory("pointbot0", "reward_1000"),
            # "sac_penalty3000": get_directory("pointbot0", "reward_3000"),
            # "sac_lagrangian_1": get_directory("pointbot0", "nu_1"),
            # "sac_lagrangian_10": get_directory("pointbot0", "nu_10"),
            # "sac_lagrangian_100": get_directory("pointbot0", "nu_100"),
            # "sac_lagrangian_1000": get_directory("pointbot0", "nu_1000"),
            "sac_lagrangian": get_directory("pointbot0", "nu_5000"),
            # "sac_lagrangian_3000": get_directory("pointbot0", "nu_3000"),
            # "rcpo_1": get_directory("pointbot0", "rcpo_1"),
            # "rcpo_10": get_directory("pointbot0", "rcpo_10"),
            # "rcpo_100": get_directory("pointbot0", "rcpo_100"),
            # "rcpo_1000": get_directory("pointbot0", "rcpo_1000"),
            # "rcpo_5000": get_directory("pointbot0", "rcpo_5000"),
            "sac_rcpo": get_directory("pointbot0", "rcpo_1000"),
            # "lookahead": get_directory("pointbot0", "lookahead"),
            "sac_recovery_pets": get_directory("pointbot0", "pets"),
            "sac_recovery_ddpg": get_directory("pointbot0", "ddpg"),
        },
        "outfile": "pointbot0.png"
    },
    "pointbot1": {
        "algs": {
            "sac_vanilla": get_directory("pointbot1", "vanilla"),
            # "sac_penalty1": get_directory("pointbot1", "reward_1"),
            # "sac_penalty10": get_directory("pointbot1", "reward_10"),
            # "sac_penalty100": get_directory("pointbot1", "reward_100"),
            # "sac_penalty1000": get_directory("pointbot1", "reward_1000"),
            "sac_penalty": get_directory("pointbot1", "reward_3000"),
            # "sac_lagrangian_1": get_directory("pointbot1", "nu_1"),
            # "sac_lagrangian_10": get_directory("pointbot1", "nu_10"),
            # "sac_lagrangian_100": get_directory("pointbot1", "nu_100_update"),
            # "sac_lagrangian_500": get_directory("pointbot1", "nu_500_update"), # rerun this
            "sac_lagrangian": get_directory("pointbot1", "nu_1000"),
            # "sac_lagrangian_5000": get_directory("pointbot1", "nu_5000"),
            # "rcpo_1": get_directory("pointbot1", "rcpo_1"),
            # "rcpo_10": get_directory("pointbot1", "rcpo_10"),
            # "rcpo_100": get_directory("pointbot1", "rcpo_100"),
            # "rcpo_1000": get_directory("pointbot1", "rcpo_1000"),
            "sac_rcpo": get_directory("pointbot1", "rcpo_5000"),
            "sac_recovery_pets": get_directory("pointbot1", "pets"),
            "sac_recovery_ddpg": get_directory("pointbot1", "ddpg"),
        },
        "outfile": "pointbot1.png"
    },
    "shelf": { # Sparse reward instead... (all up to 2800)
        "algs": {
            # "sac_norecovery": get_directory("shelf", "vanilla"),
            # "sac_penalty3": get_directory("shelf", "reward_3"),
            # "sac_penalty10": get_directory("shelf", "reward_10"),
            # "sac_lagrangian_3": get_directory("shelf", "nu_3_update"),
            "sac_lagrangian_10_alpha_05": get_directory("shelf", "nu_10_update_alpha_05"),
            # "sac_lagrangian_10_alpha_2": get_directory("shelf", "nu_10_update_alpha_2"),
            # "recovery_pets_alpha_05_eps_04": get_directory("shelf", "recovery_pets_alpha_05_eps_04"),
            # "recovery_pets_alpha_2_eps_5": get_directory("shelf", "recovery_pets_alpha_2_eps_5"),
            # "recovery_pets_alpha_2_eps_6": get_directory("shelf", "recovery_pets_alpha_2_eps_6"),
            # "recovery_ddpg_alpha_2_eps_6": get_directory("shelf", "recovery_ddpg_alpha_2_eps_6"),
            # "recovery_ddpg_alpha_2_eps_4": get_directory("shelf", "recovery_ddpg_alpha_2_eps_4"),
            "recovery_pets_alpha_05_eps_6": get_directory("shelf", "recovery_pets_alpha_05_eps_6"),
            # "recovery_pets_alpha_05_eps_5": get_directory("shelf", "recovery_pets_alpha_05_eps_5"),
            # "recovery_pets_alpha_2_eps_4": get_directory("shelf", "recovery_pets_alpha_2_eps_4"),
            "recovery_ddpg_alpha_2_eps_5": get_directory("shelf", "recovery_ddpg_alpha_2_eps_5"),
            # "recovery_ddpg_alpha_05_eps_5": get_directory("shelf", "recovery_ddpg_alpha_05_eps_5"),
            # "recovery_ddpg_alpha_05_eps_4": get_directory("shelf", "recovery_ddpg_alpha_05_eps_4"),
            # "recovery_ddpg_alpha_05_eps_6": get_directory("shelf", "recovery_ddpg_alpha_05_eps_6"),
            # "ddpg": get_directory("shelf", "ddpg"),
        },
        "outfile": "shelf.png"
    },
    "shelf_dynamic": { # Sparse reward instead... (all up to 2800)
        "algs": {
            "sac_norecovery": get_directory("shelf_dynamic", "vanilla"),
            # "sac_penalty3": get_directory("shelf_dynamic", "reward_3"),
            "sac_penalty10": get_directory("shelf_dynamic", "reward_10"),
            # "sac_lagrangian_1": get_directory("shelf_dynamic", "nu_1_update"),
            "sac_lagrangian_3": get_directory("shelf_dynamic", "nu_3_update"),
            # "sac_lagrangian_10": get_directory("shelf_dynamic", "nu_10_update"),
            "ddpg": get_directory("shelf_dynamic", "ddpg"),
        },
        "outfile": "shelf_dynamic.png"
    },
}


names = {
    "sac_norecovery": "SAC",
    "sac_penalty1": "SAC (penalty 1)",
    "sac_penalty20": "SAC (penalty 20)",
    "sac_penalty50": "SAC (penalty 50)",
    "sac_penalty10": "SAC (penalty 10)",
    "sac_penalty75": "SAC (penalty 75)",
    "sac_penalty100": "SAC (penalty 100)",
    "sac_penalty1000": "SAC (penalty 1000)",
    "sac_penalty3000": "SAC (penalty 3000)",
    "recovery": "SAC + Recovery (PETS)",
    "sac_lagrangian" : "SAC + Lagrangian",
    "sac_lagrangian_1" : "SAC + Lagrangian (1)",
    "sac_lagrangian_3" : "SAC + Lagrangian (3)",
    "sac_lagrangian_10" : "SAC + Lagrangian (10)",
    "sac_lagrangian_10" : "SAC + Lagrangian (10)",
    "sac_lagrangian_100" : "SAC + Lagrangian (100)",
    "sac_lagrangian_500" : "SAC + Lagrangian (500)",
    "sac_lagrangian_5000" : "SAC + Lagrangian (5000)",
    "sac_lagrangian_1000" : "SAC + Lagrangian (1000)",
    "sac_lagrangian_10000" : "SAC + Lagrangian (10000)",
    "sac_lagrangian_3000" : "SAC + Lagrangian (3000)",
    "rcpo_1" : "SAC + RCPO (1)",
    "rcpo_10" : "SAC + RCPO (10)",
    "rcpo_100" : "SAC + RCPO (100)",
    "rcpo_1000" : "SAC + RCPO (1000)",
    "rcpo_3000" : "SAC + RCPO (3000)",
    "rcpo_5000" : "SAC + RCPO (5000)",
    "recovery_0.1": "SAC + Recovery (eps=0.1)",
    "recovery_0.2": "SAC + Recovery (eps=0.2)",
    "recovery_0.25": "SAC + Recovery (eps=0.25)",
    "recovery_0.3": "SAC + Recovery (eps=0.3)",
    "recovery_0.4": "SAC + Recovery (eps=0.4)",
    "recovery_0.4_20k": "SAC + Recovery",
    "recovery_0.5": "SAC + Recovery (eps=0.5)",
    "recovery_0.6": "SAC + Recovery (eps=0.6)",
    "recovery_0.8": "SAC + Recovery (eps=0.8)",
    "recovery_0.9": "SAC + Recovery (eps=0.9)",
    "sac_penalty1": "SAC (penalty 1)",
    "sac_penalty3": "SAC (penalty 3)",
    "sac_penalty5": "SAC (penalty 5)",
    "sac_penalty10": "SAC (penalty 10)",
    "sac_penalty25": "SAC (penalty 25)",
    "worst_action": "SAC + Recovery (worst action)",
    "lookahead": "SAC + Recovery (lookahead)",
    "test": "test",
    "ddpg": "SAC + Recovery (DDPG)",
    "sac_lagrangian_10_alpha_05": "nu_10_update_alpha_05",
    "sac_lagrangian_10_alpha_2": "nu_10_update_alpha_2",
    "recovery_pets_alpha_05_eps_04": "recovery_pets_alpha_05_eps_04",
    "recovery_pets_alpha_2_eps_5": "recovery_pets_alpha_2_eps_5",
    "recovery_pets_alpha_2_eps_6": "recovery_pets_alpha_2_eps_6",
    "recovery_ddpg_alpha_2_eps_6": "recovery_ddpg_alpha_2_eps_6",
    "recovery_ddpg_alpha_2_eps_4": "recovery_ddpg_alpha_2_eps_4",
    "recovery_pets_alpha_05_eps_6": "recovery_pets_alpha_05_eps_6",
    "recovery_pets_alpha_05_eps_5": "recovery_pets_alpha_05_eps_5",
    "recovery_pets_alpha_2_eps_4": "recovery_pets_alpha_2_eps_4",
    "recovery_ddpg_alpha_2_eps_5": "recovery_ddpg_alpha_2_eps_5",
    "recovery_ddpg_alpha_05_eps_5": "recovery_ddpg_alpha_05_eps_5",
    "recovery_ddpg_alpha_05_eps_4": "recovery_ddpg_alpha_05_eps_4",
    "recovery_ddpg_alpha_05_eps_6": "recovery_ddpg_alpha_05_eps_6",
}


colors = {
    "sac_norecovery": "g",
    "sac_penalty1": "blue",
    "sac_penalty20": "orange",
    "sac_penalty50": "black",
    "sac_penalty10": "orange",
    "sac_penalty75": "blue",
    "sac_penalty100": "purple",
    "sac_penalty1000": "blue",
    "sac_penalty3000": "blue",
    "recovery": "red",
    "sac_lagrangian": "pink",
    "sac_lagrangian_1": "red",
    "sac_lagrangian_3": "red",
    "sac_lagrangian_10": "orange",
    "sac_lagrangian_10": "cyan",
    "sac_lagrangian_100": "cyan",
    "sac_lagrangian_500": "orange",
    "sac_lagrangian_5000": "orange",
    "sac_lagrangian_1000": "magenta",
    "sac_lagrangian_10000": "magenta",
    "sac_lagrangian_3000": "magenta",
    "rcpo_1": "orange",
    "rcpo_10": "cyan",
    "rcpo_100": "magenta",
    "rcpo_1000": "magenta",
    "rcpo_3000": "purple",
    "rcpo_5000": "purple",
    "recovery_0.2": "blue",
    "recovery_0.25": "cyan",
    "recovery_0.3": "black",
    "recovery_0.4_20k": "blue",
    "recovery_0.4_5k": "red",
    "recovery_0.4": "blue",
    "recovery_0.6": "cyan",
    "recovery_0.8": "purple",
    "sac_penalty3": "orange",
    "sac_penalty5": "yellow",
    "sac_penalty25": "magenta",
    "worst_action": "gold",
    "lookahead": "green",
    "test": "green",
    "ddpg": "black",
    "sac_lagrangian_10_alpha_05": "blue",
    "sac_lagrangian_10_alpha_2": "orange",
    "recovery_pets_alpha_05_eps_04": "red",
    "recovery_pets_alpha_2_eps_5": "purple",
    "recovery_pets_alpha_2_eps_6": "black",
    "recovery_ddpg_alpha_2_eps_6": "grey",
    "recovery_ddpg_alpha_2_eps_4": "green",
    "recovery_pets_alpha_05_eps_6": "cyan",
    "recovery_pets_alpha_05_eps_5": "magenta",
    "recovery_pets_alpha_2_eps_4": "yellow",
    "recovery_ddpg_alpha_2_eps_5": "pink",
    "recovery_ddpg_alpha_05_eps_5": "gold",
    "recovery_ddpg_alpha_05_eps_4": "silver",
    "recovery_ddpg_alpha_05_eps_6": "gold",
}

def get_stats(data):
    minlen = min([len(d) for d in data])
    data = [d[:minlen] for d in data]
    mu = np.mean(data, axis=0)
    lb = mu - np.std(data, axis=0) / np.sqrt(len(data))
    ub = mu + np.std(data, axis=0) / np.sqrt(len(data))
    return mu, lb, ub


eps = {
    "maze": 1500,
    "pointbot0": 300,
    "pointbot1": 300,
    "shelf": 4000,
    "shelf_dynamic": 3000
}


envname = {
    "maze": "Maze",
    "pointbot0": "Navigation 1",
    "pointbot1": "Navigation 2",
    "shelf": "Shelf",
    "shelf_dynamic": "Dynamic Shelf"
}


yscaling = {
    "maze": 0.25,
    "pointbot0": 0.9,
    "pointbot1": 0.4,
    "shelf": 0.1,
    "shelf_dynamic": 0.3
}


def plot_experiment(experiment): # 3000 for normal shelf...

    max_eps = eps[experiment]

    fig, axs = plt.subplots(2, figsize=(16, 16))

    axs[0].set_title("%s: Cumulative Constraint Violations vs. Episode"%envname[experiment], fontsize=30)
    axs[0].set_ylim(-0.1, int(yscaling[experiment] * max_eps) + 1)
    axs[0].set_xlabel("Episode", fontsize=24)
    axs[0].set_ylabel("Cumulative Constraint Violations", fontsize=24)
    axs[0].tick_params(axis='both', which='major', labelsize=21)

    axs[1].set_title("%s: Cumulative Task Successes vs. Episode"%envname[experiment], fontsize=30)
    axs[1].set_ylim(0, int(max_eps)+1)
    axs[1].set_xlabel("Episode", fontsize=24)
    axs[1].set_ylabel("Cumulative Task Successes", fontsize=24)
    axs[1].tick_params(axis='both', which='major', labelsize=21)

    plt.subplots_adjust(hspace=0.3)

    for alg in experiment_map[experiment]["algs"]:
        print(alg)
        exp_dirs = experiment_map[experiment]["algs"][alg]
        fnames = [osp.join(exp_dir, "run_stats.pkl") for exp_dir in exp_dirs]

        task_successes_list = []
        train_rewards_list = []
        train_violations_list = []
        recovery_called_list = []
        recovery_called_constraint_list = []

        for fname in fnames:
            with open(fname, "rb") as f:
                data = pickle.load(f)
            train_stats = data['train_stats']

            train_violations = []
            train_rewards = []
            last_rewards = []
            recovery_called = []
            print(fname)
            for traj_stats in train_stats:
                train_violations.append([])
                recovery_called.append([])
                train_rewards.append(0)
                last_reward = 0
                for step_stats in traj_stats:
                    train_violations[-1].append(step_stats['constraint'])
                    recovery_called[-1].append(0)
                    train_rewards[-1] += step_stats['reward']
                    last_reward = step_stats['reward']
                last_rewards.append(last_reward)
                # print(last_reward)

            recovery_called = np.array([np.sum(t) > 0 for t in recovery_called])[:max_eps].astype(int) # For now just look at whether a recovery was called at any point
            ep_lengths = np.array([len(t) for t in train_violations])[:max_eps]
            train_violations = np.array([np.sum(t) > 0 for t in train_violations])[:max_eps]
            recovery_called_constraint = np.bitwise_and(recovery_called, train_violations)

            recovery_called = np.cumsum(recovery_called)
            train_violations = np.cumsum(train_violations)
            print(train_violations, len(train_violations))
            recovery_called_constraint = np.cumsum(recovery_called_constraint)

            train_rewards = np.array(train_rewards)[:max_eps]
            last_rewards = np.array(last_rewards)[:max_eps]

            if experiment == 'maze':
                task_successes = (-last_rewards < 0.03).astype(int)
            elif 'shelf' in experiment:
                task_successes = (last_rewards == 0).astype(int)
            elif "pointbot0" in experiment:
                task_successes = (last_rewards > -4).astype(int)
            else:
                task_successes = (last_rewards > -4).astype(int)

            task_successes = np.cumsum(task_successes)
            task_successes_list.append(task_successes)

            train_violations_list.append(train_violations)
            recovery_called_list.append(recovery_called)
            recovery_called_constraint_list.append(recovery_called_constraint)

        task_successes_list = np.array(task_successes_list)
        train_violations_list = np.array(train_violations_list)
        recovery_called_list = np.array(recovery_called_list)
        recovery_called_constraint_list = np.array(recovery_called_constraint_list)

        ts_mean, ts_lb, ts_ub = get_stats(task_successes_list)
        tv_mean, tv_lb, tv_ub = get_stats(train_violations_list)
        trec_mean, trec_lb, trec_ub = get_stats(recovery_called_list)
        trec_constraint_mean, trec_constraint_lb, trec_constraint_ub = get_stats(recovery_called_constraint_list)

        axs[0].fill_between(range(tv_mean.shape[0]), tv_ub, tv_lb,
                     color=get_color(alg), alpha=.25, label=get_legend_name(alg))
        axs[0].plot(tv_mean, get_color(alg))
        axs[1].fill_between(range(ts_mean.shape[0]), ts_ub, ts_lb,
                     color=get_color(alg), alpha=.25)
        axs[1].plot(ts_mean, get_color(alg), label=get_legend_name(alg))
        

    axs[0].legend(loc="upper left", fontsize=20)
    axs[1].legend(loc="upper left", fontsize=20)
    plt.savefig(experiment_map[experiment]["outfile"], bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    experiment = "pointbot1"
    plot_experiment(experiment)

