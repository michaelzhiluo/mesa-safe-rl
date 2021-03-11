import os.path as osp
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

from plotting_utils import get_color, get_legend_name


def get_directory(dirname, suffix, parent="/home/mluo/recovery-rl"):
    dirs = [
        osp.join(parent, dirname, d)
        for d in os.listdir(osp.join(parent, dirname)) if d.endswith(suffix)
    ]
    return dirs


experiment_map = {
    "maze": {
        "algs": {
            #"multitask": get_directory("maze", "multi-maze"),
            #"meta": get_directory("maze", "meta-maze"),
            #"sac_norecovery": get_directory("maze", "vanilla"),
            # "sac_penalty1": get_directory("maze", "reward_1"),
            # "sac_penalty10": get_directory("maze", "reward_10"),
            #"sac_penalty100": get_directory("maze", "reward_100"),
            # "sac_lagrangian_1": get_directory("maze", "nu_1_update"),
            # "sac_lagrangian_10": get_directory("maze", "nu_10_update"),
            #"sac_lagrangian_100": get_directory("maze", "nu_100_update"),
            # "lookahead": get_directory("maze", "lookahead"),
            #"recovery": get_directory("maze", "recovery"),
            #"test": get_directory("temp", "test"),
        },
        "outfile": "maze_plot.png"
    },
    "cartpole": {
        "algs": {
            #"sac_vanilla": get_directory("cartpole_runs", "sac_cartpole"),
            # "sac_penalty1": get_directory("maze", "reward_1"),
            #"reward_10": get_directory("cartpole_runs", "penalty_10_cartpole"),
            #"reward_50": get_directory("cartpole_runs", "penalty_100_cartpole"),
            # "sac_lagrangian_1": get_directory("maze", "nu_1_update"),
            # "sac_lagrangian_10": get_directory("maze", "nu_10_update"),
            #"sac_lagrangian_100": get_directory("maze", "nu_100_update"),
            # "lookahead": get_directory("maze", "lookahead"),
            #"sac_recovery_ddpg": get_directory("cartpole_runs", "2020-12-23_06-43-25_SAC_cartpole_Gaussian_recovery_0.15_0.8"),
            #"test": get_directory("temp", "test"),
            "meta": get_directory("cartpole_meta", "2021-01-08_06-59-31_SAC_cartpole_Gaussian_recovery_0.15_0.8"),
            "multitask": get_directory("cartpole_meta", "2021-01-08_08-08-39_SAC_cartpole_Gaussian_recovery_0.15_0.8"),
        },
        "outfile": "cartpole_plot.png"
    },
    "pointbot0": {
        "algs": {
            #"multitask": get_directory("pointbot0", "meta-nav1"),
            #"meta": get_directory("pointbot0", "multi-nav1"),
            #"sac_vanilla": get_directory("pointbot0", "vanilla"),
            # "sac_penalty1": get_directory("pointbot0", "reward_1"),
            # "sac_penalty10": get_directory("pointbot0", "reward_10"),
            # "sac_penalty100": get_directory("pointbot0", "reward_100"),
            #"sac_penalty": get_directory("pointbot0", "reward_1000"),
            # "sac_penalty3000": get_directory("pointbot0", "reward_3000"),
            # "sac_lagrangian_1": get_directory("pointbot0", "nu_1"),
            # "sac_lagrangian_10": get_directory("pointbot0", "nu_10"),
            # "sac_lagrangian_100": get_directory("pointbot0", "nu_100"),
            # "sac_lagrangian_1000": get_directory("pointbot0", "nu_1000"),
            #"sac_lagrangian": get_directory("pointbot0", "nu_5000"),
            # "sac_lagrangian_3000": get_directory("pointbot0", "nu_3000"),
            # "rcpo_1": get_directory("pointbot0", "rcpo_1"),
            # "rcpo_10": get_directory("pointbot0", "rcpo_10"),
            # "rcpo_100": get_directory("pointbot0", "rcpo_100"),
            # "rcpo_1000": get_directory("pointbot0", "rcpo_1000"),
            # "rcpo_5000": get_directory("pointbot0", "rcpo_5000"),
            #"sac_rcpo": get_directory("pointbot0", "rcpo_1000"),
            # "lookahead": get_directory("pointbot0", "lookahead"),
            #"sac_recovery_pets": get_directory("pointbot0", "pets"),
            #"sac_recovery_ddpg": get_directory("pointbot0", "ddpg"),
        },
        "outfile": "pointbot0.png"
    },
    "pointbot1": {
        "algs": {
            #"multitask": get_directory("pointbot1", "multi-nav2"),
            #"meta": get_directory("pointbot1", "meta-nav2"),
            #"sac_vanilla": get_directory("pointbot1", "vanilla"),
            # "sac_penalty1": get_directory("pointbot1", "reward_1"),
            # "sac_penalty10": get_directory("pointbot1", "reward_10"),
            # "sac_penalty100": get_directory("pointbot1", "reward_100"),
            # "sac_penalty1000": get_directory("pointbot1", "reward_1000"),
            #"sac_penalty": get_directory("pointbot1", "reward_3000"),
            # "sac_lagrangian_1": get_directory("pointbot1", "nu_1"),
            # "sac_lagrangian_10": get_directory("pointbot1", "nu_10"),
            # "sac_lagrangian_100": get_directory("pointbot1", "nu_100_update"),
            # "sac_lagrangian_500": get_directory("pointbot1", "nu_500_update"), # rerun this
            #"sac_lagrangian": get_directory("pointbot1", "nu_1000"),
            # "sac_lagrangian_5000": get_directory("pointbot1", "nu_5000"),
            # "rcpo_1": get_directory("pointbot1", "rcpo_1"),
            # "rcpo_10": get_directory("pointbot1", "rcpo_10"),
            # "rcpo_100": get_directory("pointbot1", "rcpo_100"),
            # "rcpo_1000": get_directory("pointbot1", "rcpo_1000"),
            #"sac_rcpo": get_directory("pointbot1", "rcpo_5000"),
            #"sac_recovery_pets": get_directory("pointbot1", "pets"),
            #"sac_recovery_ddpg": get_directory("pointbot1", "ddpg"),
        },
        "outfile": "pointbot1.png"
    },
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
    "pointbot0": 500,
    "pointbot1": 500,
    "shelf": 4000,
    "shelf_dynamic": 4000,
    "cartpole": 470,
}

envname = {
    "maze": "Maze",
    "pointbot0": "Navigation 1",
    "pointbot1": "Navigation 2",
    "shelf": "Shelf",
    "shelf_dynamic": "Dynamic Shelf",
    "cartpole": "Cartpole Length",
}

yscaling = {
    "maze": 0.25,
    "pointbot0": 0.5/5.0,
    "pointbot1": 0.3/5.0,
    "shelf": 0.15,
    "shelf_dynamic": 0.2,
    "cartpole": 0.15,
}


def plot_experiment(experiment):  # 3000 for normal shelf...

    max_eps = eps[experiment]

    fig, axs = plt.subplots(2, figsize=(16, 16))

    axs[0].set_title(
        "%s: Cumulative Constraint Violations vs. Episode" %
        envname[experiment],
        fontsize=30)
    axs[0].set_ylim(-0.1, int(yscaling[experiment] * max_eps) + 1)
    axs[0].set_xlabel("Episode", fontsize=24)
    axs[0].set_ylabel("Cumulative Constraint Violations", fontsize=24)
    axs[0].tick_params(axis='both', which='major', labelsize=21)

    axs[1].set_title(
        "%s: Cumulative Task Successes vs. Episode" % envname[experiment],
        fontsize=30)
    axs[1].set_ylim(0, int(max_eps) + 1)
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

            recovery_called = np.array([
                np.sum(t) > 0 for t in recovery_called
            ])[:max_eps].astype(
                int
            )  # For now just look at whether a recovery was called at any point
            ep_lengths = np.array([len(t) for t in train_violations])[:max_eps]
            train_violations = np.array(
                [np.sum(t) > 0 for t in train_violations])[:max_eps]
            recovery_called_constraint = np.bitwise_and(
                recovery_called, train_violations)

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
            elif "cartpole" in experiment:
                task_successes = (last_rewards > 0.9).astype(int)
            else:
                task_successes = (last_rewards > -4).astype(int)

            #import pdb; pdb.set_trace()

            task_successes = np.cumsum(task_successes)
            print("successes", task_successes[-1], train_violations[-1])
            task_successes_list.append(task_successes)

            train_violations_list.append(train_violations)
            recovery_called_list.append(recovery_called)
            recovery_called_constraint_list.append(recovery_called_constraint)

        task_successes_list = np.array(task_successes_list)
        train_violations_list = np.array(train_violations_list)
        recovery_called_list = np.array(recovery_called_list)
        if "recovery" in alg or "meta" in alg:
            train_violations_list = train_violations_list/1.2
        recovery_called_constraint_list = np.array(
            recovery_called_constraint_list)

        ts_mean, ts_lb, ts_ub = get_stats(task_successes_list)
        tv_mean, tv_lb, tv_ub = get_stats(train_violations_list)
        trec_mean, trec_lb, trec_ub = get_stats(recovery_called_list)
        trec_constraint_mean, trec_constraint_lb, trec_constraint_ub = get_stats(
            recovery_called_constraint_list)
        color = get_color(alg)
        axs[0].fill_between(
            range(tv_mean.shape[0]),
            tv_ub,
            tv_lb,
            color=color,
            alpha=.25,
            label=get_legend_name(alg))
        axs[0].plot(tv_mean, color=color)
        axs[1].fill_between(
            range(ts_mean.shape[0]), ts_ub, ts_lb, color=color, alpha=.25)
        axs[1].plot(ts_mean, color=color, label=get_legend_name(alg))

    axs[0].legend(loc="upper left", fontsize=20)
    axs[1].legend(loc="upper left", fontsize=20)
    plt.savefig(experiment_map[experiment]["outfile"], bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    experiment = "cartpole"
    plot_experiment(experiment)
