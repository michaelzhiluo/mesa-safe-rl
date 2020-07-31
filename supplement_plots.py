import os.path as osp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from plotting_utils import get_color, get_legend_name

experiment_map = {
    "maze": {
        "algs": {
            "sac_vanilla": ["maze/2020-07-14_07-53-24_SAC_maze_Gaussian_vanilla", "maze/2020-07-14_07-56-18_SAC_maze_Gaussian_vanilla", "maze/2020-07-14_07-59-01_SAC_maze_Gaussian_vanilla"],
            # "sac_penalty10": ["2020-07-14_09-18-41_SAC_maze_Gaussian_reward_10", "2020-07-14_09-25-31_SAC_maze_Gaussian_reward_10", "2020-07-14_09-31-38_SAC_maze_Gaussian_reward_10"],
            "sac_penalty": ["maze/2020-07-14_08-01-37_SAC_maze_Gaussian_reward_50", "maze/2020-07-14_08-22-14_SAC_maze_Gaussian_reward_50", "maze/2020-07-14_08-50-58_SAC_maze_Gaussian_reward_50"],
            # "sac_penalty100": ["2020-07-14_09-38-04_SAC_maze_Gaussian_reward_100", "2020-07-14_10-08-58_SAC_maze_Gaussian_reward_100", "2020-07-14_10-31-51_SAC_maze_Gaussian_reward_100"],
            "sac_rcpo": ["maze/2020-07-14_08-02-41_SAC_maze_Gaussian_lambda_50", "maze/2020-07-14_08-46-35_SAC_maze_Gaussian_lambda_50", "maze/2020-07-14_09-31-38_SAC_maze_Gaussian_lambda_50"],
            "sac_lagrangian": ["maze/2020-07-14_17-56-02_SAC_maze_Gaussian_update_nu_100", "maze/2020-07-14_18-58-07_SAC_maze_Gaussian_update_nu_100", "maze/2020-07-14_19-58-31_SAC_maze_Gaussian_update_nu_100"],
            "sac_sqrl": ["maze/2020-07-23_23-44-56_SAC_maze_Gaussian_update_nu_100_SQRL", "maze/2020-07-24_00-40-39_SAC_maze_Gaussian_update_nu_100_SQRL", "maze/2020-07-24_01-35-06_SAC_maze_Gaussian_update_nu_100_SQRL"],
            "sac_rspo": ["maze/2020-07-14_21-23-48_SAC_maze_Gaussian_RSPO", "maze/2020-07-14_20-45-50_SAC_maze_Gaussian_RSPO", "maze/2020-07-14_19-47-31_SAC_maze_Gaussian_RSPO"],
            "sac_recovery_ddpg": ["maze/2020-07-14_07-46-44_SAC_maze_Gaussian_ddpg_recovery", "maze/2020-07-14_08-08-59_SAC_maze_Gaussian_ddpg_recovery", "maze/2020-07-14_08-30-01_SAC_maze_Gaussian_ddpg_recovery"],
            "sac_recovery_pets": ["maze/2020-07-14_07-46-28_SAC_maze_Gaussian_recovery", "maze/2020-07-14_10-38-39_SAC_maze_Gaussian_recovery", "maze/2020-07-14_09-13-40_SAC_maze_Gaussian_recovery"],
        },
        "outfile": "maze_plot.png"
    },
    "image_maze": {
        "algs": {
            "sac_vanilla": ["runs/2020-06-15_01-19-52_SAC_image_maze_Gaussian_", "runs/2020-06-15_01-20-48_SAC_image_maze_Gaussian_", "runs/2020-06-15_01-21-01_SAC_image_maze_Gaussian_"],
            # "sac_penalty50": ["2020-06-15_01-35-41_SAC_image_maze_Gaussian_", "2020-06-15_01-35-51_SAC_image_maze_Gaussian_", "2020-06-15_01-36-00_SAC_image_maze_Gaussian_"],
            "sac_penalty": ["runs/2020-06-15_02-03-52_SAC_image_maze_Gaussian_", "runs/2020-06-15_01-48-08_SAC_image_maze_Gaussian_", "runs/2020-06-15_01-48-22_SAC_image_maze_Gaussian_"],
            "sac_rcpo": ["image_maze/2020-07-15_05-15-11_SAC_image_maze_Gaussian_lambda_20", "image_maze/2020-07-14_16-09-59_SAC_image_maze_Gaussian_lambda_20", "image_maze/2020-07-14_13-22-01_SAC_image_maze_Gaussian_lambda_20"],
            "sac_lagrangian": ["image_maze/2020-07-14_13-59-30_SAC_image_maze_Gaussian_nu_10", "image_maze/2020-07-14_13-59-30_SAC_image_maze_Gaussian_nu_10", "image_maze/2020-07-14_13-59-30_SAC_image_maze_Gaussian_nu_10"],
            "sac_sqrl": ["image_maze/2020-07-23_23-45-02_SAC_image_maze_Gaussian_update_nu_10_SQRL", "image_maze/2020-07-24_00-24-31_SAC_image_maze_Gaussian_update_nu_10_SQRL", "image_maze/2020-07-24_01-04-51_SAC_image_maze_Gaussian_update_nu_10_SQRL"],
            "sac_rspo": ["image_maze/2020-07-15_19-31-11_SAC_image_maze_Gaussian_RSPO", "image_maze/2020-07-15_21-02-14_SAC_image_maze_Gaussian_RSPO", "image_maze/2020-07-15_22-31-30_SAC_image_maze_Gaussian_RSPO"],
            # "sac_recovery_pets": ["runs/2020-06-14_06-23-20_SAC_image_maze_Gaussian_", "runs/2020-06-14_07-09-47_SAC_image_maze_Gaussian_", "runs/2020-06-14_07-10-32_SAC_image_maze_Gaussian_"],
            "sac_recovery_ddpg": ["runs/2020-07-07_05-11-16_SAC_image_maze_Gaussian_", "runs/2020-07-07_05-10-33_SAC_image_maze_Gaussian_", "runs/2020-07-07_05-10-41_SAC_image_maze_Gaussian_"], # DDPG recovery, gamma_safe 0.65, eps_safe 0.1
            "sac_recovery_pets": ["runs/2020-07-16_19-49-53_SAC_image_maze_Gaussian_", "runs/2020-07-16_20-46-35_SAC_image_maze_Gaussian_", "runs/2020-07-16_21-12-01_SAC_image_maze_Gaussian_"],
            
        },
        "outfile": "image_maze_plot.png"
    },
    "shelf_long": {
        "algs": {
            "sac_vanilla": ["shelf_long_env/2020-07-17_09-35-22_SAC_shelf_long_env_Gaussian_vanilla", "shelf_long_env/2020-07-17_13-57-11_SAC_shelf_long_env_Gaussian_vanilla", "shelf_long_env/2020-07-17_17-40-01_SAC_shelf_long_env_Gaussian_vanilla"],
            "sac_penalty": ["shelf_long_env/2020-07-21_03-11-14_SAC_shelf_long_env_Gaussian_reward_50", "shelf_long_env/2020-07-21_08-00-18_SAC_shelf_long_env_Gaussian_reward_50", "shelf_long_env/2020-07-21_12-37-01_SAC_shelf_long_env_Gaussian_reward_50"],
            "sac_rcpo": ["shelf_long_env/2020-07-21_03-41-20_SAC_shelf_long_env_Gaussian_rcpo_50", "shelf_long_env/2020-07-21_09-02-49_SAC_shelf_long_env_Gaussian_rcpo_50", "shelf_long_env/2020-07-21_18-59-59_SAC_shelf_long_env_Gaussian_rcpo_50"],
            "sac_lagrangian": ["shelf_long_env/2020-07-20_04-54-13_SAC_shelf_long_env_Gaussian_nu_50", "shelf_long_env/2020-07-20_13-01-12_SAC_shelf_long_env_Gaussian_nu_50", "shelf_long_env/2020-07-21_03-01-55_SAC_shelf_long_env_Gaussian_nu_50"],
            # "sac_sqrl": ["shelf_long_env/2020-07-24_03-36-02_SAC_shelf_long_env_Gaussian_update_nu_50_SQRL", "shelf_long_env/2020-07-24_03-36-46_SAC_shelf_long_env_Gaussian_update_nu_50_SQRL", "2020-07-24_05-40-36_SAC_shelf_long_env_Gaussian_update_nu_50_SQRL", "2020-07-24_09-37-51_SAC_shelf_long_env_Gaussian_update_nu_50_SQRL", "2020-07-24_11-54-58_SAC_shelf_long_env_Gaussian_update_nu_50_SQRL"],
            "sac_sqrl": ["shelf_long_env/2020-07-24_03-36-02_SAC_shelf_long_env_Gaussian_update_nu_50_SQRL", "shelf_long_env/2020-07-24_05-40-36_SAC_shelf_long_env_Gaussian_update_nu_50_SQRL", "shelf_long_env/2020-07-24_09-37-51_SAC_shelf_long_env_Gaussian_update_nu_50_SQRL"],
            "sac_rspo": ["shelf_long_env/2020-07-22_05-48-32_SAC_shelf_long_env_Gaussian_RSPO", "shelf_long_env/2020-07-22_05-48-43_SAC_shelf_long_env_Gaussian_RSPO", "shelf_long_env/2020-07-22_05-48-51_SAC_shelf_long_env_Gaussian_RSPO"],
            "sac_recovery_ddpg": ["shelf_long_env/2020-07-20_04-51-41_SAC_shelf_long_env_Gaussian_recovery_ddpg_0.75_0.25", "shelf_long_env/2020-07-20_10-16-07_SAC_shelf_long_env_Gaussian_recovery_ddpg_0.75_0.25", "shelf_long_env/2020-07-20_15-48-15_SAC_shelf_long_env_Gaussian_recovery_ddpg_0.75_0.25"],
            "sac_recovery_pets": ["shelf_long_env/2020-07-20_20-01-12_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35", "shelf_long_env/2020-07-21_10-00-44_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35", "shelf_long_env/2020-07-21_09-50-28_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35"]
        },
        "outfile": "shelf_long.png"
    },
    "shelf_long_ablations_demos": {
        "algs": {
            "sac_recovery_pets_100": ["shelf_long_env/2020-07-26_03-40-55_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_100demos", "shelf_long_env/2020-07-26_03-41-18_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_100demos", "shelf_long_env/2020-07-26_03-41-27_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_100demos"],
            "sac_recovery_pets_500": ["shelf_long_env/2020-07-26_23-59-18_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_500demos", "shelf_long_env/2020-07-26_23-59-29_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_500demos", "shelf_long_env/2020-07-26_04-05-30_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_500"],
            "sac_recovery_pets_1k": ["shelf_long_env/2020-07-24_06-18-13_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_1kdemos", "shelf_long_env/2020-07-24_06-21-25_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_1kdemos", "shelf_long_env/2020-07-24_23-16-08_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_1kdemos"],
            "sac_recovery_pets_5k": ["shelf_long_env/2020-07-24_19-14-04_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_5kdemos", "shelf_long_env/2020-07-24_19-14-54_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_5kdemos", "shelf_long_env/2020-07-24_19-16-02_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_5kdemos"],
            "sac_recovery_pets_20k": ["shelf_long_env/2020-07-20_20-01-12_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35", "shelf_long_env/2020-07-21_10-00-44_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35", "shelf_long_env/2020-07-21_09-50-28_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35"],
        },
        "outfile": "shelf_long_ablations_demos.png"
    },
    "shelf_long_ablations_method": {
        "algs": {
            "sac_recovery_pets_ablations": ["shelf_long_env/2020-07-20_20-01-12_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35", "shelf_long_env/2020-07-21_10-00-44_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35", "shelf_long_env/2020-07-21_09-50-28_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35"],
            "sac_recovery_pets_disable_relabel": ["shelf_long_env/2020-07-24_06-34-39_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_disable_relabel", "shelf_long_env/2020-07-24_06-35-23_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_disable_relabel", "shelf_long_env/2020-07-24_06-36-21_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_disable_relabel"],
            "sac_recovery_pets_disable_offline": ["shelf_long_env/2020-07-26_23-54-15_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_disable_offline", "shelf_long_env/2020-07-26_23-54-50_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_disable_offline", "shelf_long_env/2020-07-26_23-55-17_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_disable_offline"],
            "sac_recovery_pets_disable_online": ["shelf_long_env/2020-07-25_19-00-28_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_disable_online", "shelf_long_env/2020-07-25_04-25-09_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_disable_online", "shelf_long_env/2020-07-25_12-06-41_SAC_shelf_long_env_Gaussian_recovery_0.85_0.35_disable_online"]
        },
        "outfile": "shelf_long_ablations_method.png"
    },  
}


names = {
    "sac_norecovery": "SAC",
    "sac_penalty20": "SAC (penalty 20)",
    "sac_penalty50": "SAC (penalty 50)",
    "sac_penalty75": "SAC (penalty 75)",
    "sac_penalty100": "SAC (penalty 100)",
    "recovery": "SAC + Model-Based Recovery",
    "ddpg_recovery": "SAC + Model-Free Recovery",
    "recovery_reachability": "SAC + Recovery + Reachablity",
    "sac_lagrangian" : "SAC + Lagrangian",
    "sac_lagrangian_update": "SAC + Lagrangian Update",
    "recovery_0.1": "SAC + Recovery (eps=0.1)",
    "recovery_0.2": "SAC + Recovery (eps=0.2)",
    # "recovery": "SAC + Recovery (eps=0.25)",
    "recovery_0.3": "SAC + Recovery (eps=0.3)",
    "recovery_0.4": "SAC + Recovery (eps=0.4)",
    "recovery_0.4_20k": "SAC + Recovery",
    "RCPO": "RCPO",
    "RSPO": "RSPO",
    # "recovery_0.4_20k_gamma0.9": "SAC + Recovery (eps=0.4), 20k transitions, gamma=0.9",
    # "recovery_0.4_5k": "SAC + Recovery (eps=0.4)",
    "recovery_0.5": "SAC + Recovery (eps=0.5)",
    "recovery_0.6": "SAC + Recovery (eps=0.6)",
    "recovery_0.8": "SAC + Recovery (eps=0.8)",
    "recovery_0.9": "SAC + Recovery (eps=0.9)",
    "sac_penalty1": "SAC (penalty 1)",
    "sac_penalty3": "SAC (penalty 3)",
    "sac_penalty5": "SAC (penalty 5)",
    "sac_penalty10": "SAC (penalty 10)",
    "sac_penalty15": "SAC (penalty 15)",
    "sac_penalty25": "SAC (penalty 25)",
    # "recovery_0.8_images": "SAC + Recovery (eps=0.8, images)",
    # "sac_penalty3_images": "SAC (penalty 3, images)",
    # "sac_penalty10_images": "SAC (penalty 10, images)",
    # "sac_norecovery_images": "SAC (images)",

    # "recovery_0.6_dense_gamma0.3": "SAC + Recovery (Eps 0.6, Gamma 0.3)",
    # "recovery_0.6_dense_gamma0.4": "SAC + Recovery (Eps 0.6, Gamma 0.4)",
    # "recovery_0.6_dense_gamma0.5": "SAC + Recovery (Eps 0.6, Gamma 0.5)",
    # "recovery_0.6_dense_gamma0.65": "SAC + Recovery (Eps 0.6, Gamma 0.65)",
    # "recovery_0.6_dense_gamma0.75": "SAC + Recovery (Eps 0.6, Gamma 0.75)",
    # "recovery_0.6_dense_gamma0.85": "SAC + Recovery (Eps 0.6, Gamma 0.85)",
    # "recovery_0.6_gamma0.5_penalty3": "SAC + Recovery (Eps 0.6, Gamma 0,5, Constraint Penalty 3"
}


colors = {
    "sac_norecovery": "g",
    "sac_penalty20": "orange",
    "sac_penalty50": "orange",
    "sac_penalty75": "purple",
    "sac_penalty100": "black",
    "recovery": "red",
    "ddpg_recovery": "blue",
    "recovery_reachability": "cyan",
    "sac_lagrangian": "pink",
    "sac_lagrangian_update": "teal",
    "recovery_0.2": "purple",
    "recovery_0.25": "cyan",
    "recovery_0.3": "black",
    "recovery_0.4_20k": "red",
    # "recovery_0.4_20k_gamma0.9": "black",
    "recovery_0.4_5k": "red",
    "recovery_0.4": "blue",
    "recovery_0.6": "cyan",
    "recovery_0.8": "purple",
    "sac_penalty1": "red",
    "sac_penalty3": "blue",
    "sac_penalty5": "yellow",
    "sac_penalty10": "orange",
    "sac_penalty15": "orange",
    "RCPO": 'magenta',
    'RSPO': 'cyan'
    # "sac_penalty25": "magenta",

    # "recovery_0.8_images": "purple",
    # "sac_penalty3_images": "orange",
    # "sac_penalty10_images": "magenta",
    # "sac_norecovery_images": "g",

    # "recovery_0.6_dense_gamma0.3": "purple",
    # "recovery_0.6_dense_gamma0.4": "orange",
    # "recovery_0.6_dense_gamma0.5": "black",
    # "recovery_0.6_dense_gamma0.65": "g",
    # "recovery_0.6_dense_gamma0.75": "blue",
    # "recovery_0.6_dense_gamma0.85": "red",
    # "recovery_0.6_gamma0.5_penalty3": "purple"
}

def get_stats(data):
    mu = np.mean(data, axis=0)
    lb = mu - np.std(data, axis=0)
    ub = mu + np.std(data, axis=0)
    return mu, lb, ub

PLOT_TYPE = "violation"
assert PLOT_TYPE in ['ratio', 'success', 'violation']

eps = {
    "maze": 500,
    "image_maze": 500,
    "pointbot0": 300,
    "pointbot1": 300,
    "shelf": 4000,
    "shelf_dynamic": 3000,
    "shelf_long": 4000,
    "shelf_long_ablations_method": 4000,
    "shelf_long_ablations_demos": 4000,
    "image_shelf": 4000
}


envname = {
    "maze": "Maze",
    "image_maze": "Image Maze",
    "pointbot0": "Navigation 1",
    "pointbot1": "Navigation 2",
    "shelf": "Object Extraction",
    "shelf_dynamic": "Object Extraction (Dynamic Obstacle)",
    "shelf_long": "Object Extraction",
    "image_shelf": "Object Extraction (Dynamic Obstacle)",
    "shelf_long_ablations_demos": "Object Extraction: # Offline Transitions",
    "shelf_long_ablations_method": "Object Extraction: Method Ablations"
}

if PLOT_TYPE == "ratio":
    yscaling = {
        "maze": 0.25,
        "image_maze": 0.45,
        "shelf_long": 0.04,
        "shelf_long_ablations_method": 0.04,
        "shelf_long_ablations_demos": 0.04,
    }
elif PLOT_TYPE == "success":
    yscaling = {
        "maze": 1,
        "image_maze": 1,
        "shelf_long": 0.6,
        "shelf_long_ablations_method": 0.6,
        "shelf_long_ablations_demos": 0.6,
    }
else:
    yscaling = {
        "maze": 0.25,
        "image_maze": 0.15,
        "shelf_long": 0.07,
        "shelf_long_ablations_method": 0.07,
        "shelf_long_ablations_demos": 0.07,
    }


def plot_experiment(experiment):
    print("EXP NAME: ", experiment)
    max_eps = eps[experiment]
    fig, axs = plt.subplots(1, figsize=(16, 8))

    axs.set_title(envname[experiment], fontsize=48)
    axs.set_ylim(-0.1, int(yscaling[experiment] * max_eps) + 1)
    axs.set_xlabel("Episode", fontsize=42)
    if PLOT_TYPE == 'ratio':
        axs.set_ylabel("Ratio of Successes/Violations", fontsize=42)
    elif PLOT_TYPE == 'success':
        axs.set_ylabel("Cumulative Task Successes", fontsize=42)
    else:
        axs.set_ylabel("Cumulative Constraint Violations", fontsize=42)
    axs.tick_params(axis='both', which='major', labelsize=36)
    plt.subplots_adjust(hspace=0.3)
    final_ratios_dict = {}

    for alg in experiment_map[experiment]["algs"]:
        print(alg)
        exp_dirs = experiment_map[experiment]["algs"][alg]
        fnames = [osp.join(exp_dir, "run_stats.pkl") for exp_dir in exp_dirs]

        task_successes_list = []
        train_rewards_list = []
        train_violations_list = []
        recovery_called_list = []
        recovery_called_constraint_list = []
        prop_viol_recovery_list = []

        for fname in fnames:
            with open(fname, "rb") as f:
                data = pickle.load(f)
            train_stats = data['train_stats']

            train_violations = []
            train_rewards = []
            last_rewards = []
            recovery_called = []
            num_viols_recovery = []
            num_viols_no_recovery = []
            num_viols_recovery = 0
            num_viols_no_recovery = 0
            for traj_stats in train_stats:
                train_violations.append([])
                recovery_called.append([])
                train_rewards.append(0)
                last_reward = 0
                for step_stats in traj_stats:
                    train_violations[-1].append(step_stats['constraint'])
                    # recovery_called[-1].append(step_stats['recovery'])
                    if "recovery" in alg:
                        # print("CONSTRANT", step_stats['constraint'])
                        recovery_viol = int(step_stats['recovery'] and step_stats['constraint'])
                        no_recovery_viol = int( (not step_stats['recovery']) and step_stats['constraint'])
                        num_viols_recovery += recovery_viol
                        num_viols_no_recovery += no_recovery_viol
                    train_rewards[-1] += step_stats['reward']
                    last_reward = step_stats['reward']

                last_rewards.append(last_reward)

            recovery_called = np.array([np.sum(t) > 0 for t in recovery_called])[:max_eps].astype(int)
            ep_lengths = np.array([len(t) for t in train_violations])[:max_eps]
            train_violations = np.array([np.sum(t) > 0 for t in train_violations])[:max_eps]
            recovery_called_constraint = np.bitwise_and(recovery_called, train_violations)


            recovery_called = np.cumsum(recovery_called)
            train_violations = np.cumsum(train_violations)
            recovery_called_constraint = np.cumsum(recovery_called_constraint)

            train_rewards = np.array(train_rewards)[:max_eps]
            last_rewards = np.array(last_rewards)[:max_eps]

            if 'maze' in experiment:
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
            if not num_viols_no_recovery + num_viols_recovery == 0:
                prop_viol_recovery_list.append(float(num_viols_recovery)/float(num_viols_no_recovery + num_viols_recovery))
            else:
                prop_viol_recovery_list.append(-1)

        task_successes_list = np.array(task_successes_list)
        train_violations_list = np.array(train_violations_list)
        recovery_called_list = np.array(recovery_called_list)
        recovery_called_constraint_list = np.array(recovery_called_constraint_list)

        print("TASK SUCCESSES", task_successes_list.shape)
        print("TRAIN VIOLS", train_violations_list.shape)
        print("TRAIN RECOVERY", recovery_called_list.shape)
        print("TRAIN RECOVERY CONSTRAINT", recovery_called_constraint_list.shape)
        safe_ratios = (task_successes_list+1)/(train_violations_list+1)
        final_ratio =  safe_ratios.mean(axis=0)[-1]
        print("FINAL RATIO: ", final_ratio)
        print("PROP VIOLS", experiment, prop_viol_recovery_list)
        # if "recovery" in alg:
        #     assert(False)
        final_ratios_dict[alg] = final_ratio
        safe_ratios_mean, safe_ratios_lb, safe_ratios_ub = get_stats(safe_ratios)
        ts_mean, ts_lb, ts_ub = get_stats(task_successes_list)
        tv_mean, tv_lb, tv_ub = get_stats(train_violations_list)
        trec_mean, trec_lb, trec_ub = get_stats(recovery_called_list)
        trec_constraint_mean, trec_constraint_lb, trec_constraint_ub = get_stats(recovery_called_constraint_list)

        if PLOT_TYPE == 'ratio':
            axs.fill_between(range(safe_ratios_mean.shape[0]), safe_ratios_ub, safe_ratios_lb,
                         color=get_color(alg), alpha=.25, label=get_legend_name(alg))
            axs.plot(safe_ratios_mean, color=get_color(alg))
        elif PLOT_TYPE == 'success':
            axs.fill_between(range(ts_mean.shape[0]), ts_ub, ts_lb,
                         color=get_color(alg), alpha=.25, label=get_legend_name(alg))
            axs.plot(ts_mean, color=get_color(alg))
        else:
            axs.fill_between(range(tv_mean.shape[0]), tv_ub, tv_lb,
                         color=get_color(alg), alpha=.25, label=get_legend_name(alg))
            axs.plot(tv_mean, color=get_color(alg))

    print(final_ratios_dict)
    # axs.legend(loc="upper left", fontsize=36, frameon=False)
    plt.savefig(experiment_map[experiment]["outfile"], bbox_inches='tight')
    plt.show()

def pr_curve(experiment):
    pass


if __name__ == '__main__':
    for experiment in ["image_maze", "maze", "shelf_long"]:
    # for experiment in ["shelf_long_ablations_method", "shelf_long_ablations_demos"]:
        plot_experiment(experiment)
