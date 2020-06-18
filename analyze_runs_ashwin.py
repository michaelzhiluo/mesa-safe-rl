import os.path as osp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

experiment_map = {
    "maze": {
        "algs": {
            "recovery": ["2020-04-22_00-28-46_SAC_maze_Gaussian_", "2020-04-22_03-19-56_SAC_maze_Gaussian_", "2020-04-22_07-59-58_SAC_maze_Gaussian_"],
            # "recovery": ["2020-06-01_06-12-07_SAC_maze_Gaussian_", "2020-06-01_06-12-31_SAC_maze_Gaussian_", "2020-06-01_06-12-41_SAC_maze_Gaussian_"], # Latest run on master...not *quite* as good, maybe need to look into it still?
            # "recovery_reachability" : ["2020-06-01_10-17-26_SAC_maze_Gaussian_", "2020-06-01_10-17-37_SAC_maze_Gaussian_", "2020-06-01_10-17-50_SAC_maze_Gaussian_"], # reachability stuff barely gives a win so not using it for now
            "sac_norecovery": ["2020-04-22_00-50-15_SAC_maze_Gaussian_", "2020-04-22_09-54-41_SAC_maze_Gaussian_", "2020-04-22_09-56-59_SAC_maze_Gaussian_"],
            # "sac_penalty5": ["2020-04-22_05-55-55_SAC_maze_Gaussian_", "2020-04-22_09-54-56_SAC_maze_Gaussian_", "2020-04-22_09-58-36_SAC_maze_Gaussian_"],
            # "sac_penalty10": ["2020-04-22_05-56-18_SAC_maze_Gaussian_", "2020-04-22_09-55-06_SAC_maze_Gaussian_", "2020-04-22_10-09-11_SAC_maze_Gaussian_"],
            "sac_penalty20": ["2020-04-22_05-56-33_SAC_maze_Gaussian_", "2020-04-22_09-55-13_SAC_maze_Gaussian_", "2020-04-22_10-43-14_SAC_maze_Gaussian_"],
            "sac_penalty50": ["2020-04-27_09-20-34_SAC_maze_Gaussian_", "2020-04-27_09-20-54_SAC_maze_Gaussian_", "2020-04-27_09-21-14_SAC_maze_Gaussian_"],
            "sac_penalty75": ["2020-04-27_09-25-19_SAC_maze_Gaussian_", "2020-04-27_09-25-35_SAC_maze_Gaussian_", "2020-04-27_09-25-48_SAC_maze_Gaussian_"],
            "sac_penalty100": ["2020-04-27_09-23-49_SAC_maze_Gaussian_", "2020-04-27_09-24-03_SAC_maze_Gaussian_", "2020-04-27_09-24-15_SAC_maze_Gaussian_"],
            "sac_lagrangian": ["2020-04-30_09-42-50_SAC_maze_Gaussian_", "2020-04-30_09-42-55_SAC_maze_Gaussian_", "2020-04-30_10-16-13_SAC_maze_Gaussian_"]
        },
        "outfile": "maze_plot.png"
    },
    "image_maze": {
        "algs": {
            "recovery": ["2020-06-14_06-23-20_SAC_image_maze_Gaussian_", "2020-06-14_07-09-47_SAC_image_maze_Gaussian_", "2020-06-14_07-10-32_SAC_image_maze_Gaussian_"],
            "sac_norecovery": ["2020-06-15_01-19-52_SAC_image_maze_Gaussian_", "2020-06-15_01-20-48_SAC_image_maze_Gaussian_", "2020-06-15_01-21-01_SAC_image_maze_Gaussian_"],
            # "sac_penalty50": ["2020-06-15_01-35-41_SAC_image_maze_Gaussian_", "2020-06-15_01-35-51_SAC_image_maze_Gaussian_", "2020-06-15_01-36-00_SAC_image_maze_Gaussian_"],
            "sac_penalty20": ["2020-06-15_02-03-52_SAC_image_maze_Gaussian_", "2020-06-15_01-48-08_SAC_image_maze_Gaussian_", "2020-06-15_01-48-22_SAC_image_maze_Gaussian_"],
            # "sac_penalty5": ["2020-06-15_02-22-21_SAC_image_maze_Gaussian_", "2020-06-15_02-22-29_SAC_image_maze_Gaussian_", "2020-06-15_02-22-36_SAC_image_maze_Gaussian_"],
            "RCPO": ["2020-06-15_08-34-02_SAC_image_maze_Gaussian_", "2020-06-15_08-36-06_SAC_image_maze_Gaussian_", "2020-06-15_08-47-25_SAC_image_maze_Gaussian_"],
            "sac_lagrangian": ["2020-06-15_19-53-56_SAC_image_maze_Gaussian_", "2020-06-15_20-48-22_SAC_image_maze_Gaussian_", "2020-06-15_19-54-24_SAC_image_maze_Gaussian_"], # nu = 1
            "sac_lagrangian_update": ["2020-06-16_00-11-54_SAC_image_maze_Gaussian_", "2020-06-16_00-12-06_SAC_image_maze_Gaussian_", "2020-06-16_00-12-27_SAC_image_maze_Gaussian_"] 
            # "sac_lagrangian": ["2020-06-15_07-58-23_SAC_image_maze_Gaussian_", "2020-06-15_07-58-28_SAC_image_maze_Gaussian_", "2020-06-15_07-58-34_SAC_image_maze_Gaussian_"], # nu = 10
            # "sac_lagrangian": ["2020-06-15_02-58-01_SAC_image_maze_Gaussian_", "2020-06-15_02-58-09_SAC_image_maze_Gaussian_", "2020-06-15_02-58-18_SAC_image_maze_Gaussian_"], # nu = 50
            # "sac_lagrangian": ["2020-06-15_04-21-43_SAC_image_maze_Gaussian_", "2020-06-15_04-21-52_SAC_image_maze_Gaussian_", "2020-06-15_04-21-59_SAC_image_maze_Gaussian_"], # nu = 200
        },
        "outfile": "image_maze_plot.png"
    },
    "pointbot0": {
        "algs": {
            "sac_norecovery": ["2020-04-30_03-27-49_SAC_simplepointbot0_Gaussian_", "2020-04-30_04-29-05_SAC_simplepointbot0_Gaussian_", "2020-04-30_04-35-14_SAC_simplepointbot0_Gaussian_"],
            "sac_lagrangian": ["2020-04-30_04-02-57_SAC_simplepointbot0_Gaussian_", "2020-04-30_04-12-58_SAC_simplepointbot0_Gaussian_", "2020-04-30_04-21-06_SAC_simplepointbot0_Gaussian_"]
        },
        "outfile": "pointbot0.png"
    },
    "pointbot1": {
        "algs": {
            "sac_norecovery": ["2020-04-30_07-55-32_SAC_simplepointbot1_Gaussian_", "2020-04-30_07-55-17_SAC_simplepointbot1_Gaussian_", "2020-04-30_07-55-05_SAC_simplepointbot1_Gaussian_"],
            "sac_lagrangian": ["2020-04-30_05-20-57_SAC_simplepointbot1_Gaussian_", "2020-04-30_04-45-47_SAC_simplepointbot1_Gaussian_", "2020-04-30_08-13-34_SAC_simplepointbot1_Gaussian_"]
        },
        "outfile": "pointbot1.png"
    },
    # "shelf": {  # Up to 2800
    #     "algs": {
    #         "sac_norecovery": ["2020-05-02_10-02-27_SAC_shelf_env_Gaussian_", "2020-05-02_23-46-58_SAC_shelf_env_Gaussian_", "2020-05-02_23-47-31_SAC_shelf_env_Gaussian_"],
    #         # "recovery_0.1": ["2020-05-02_10-25-48_SAC_shelf_env_Gaussian_"],
    #         # "recovery_0.2": ["2020-05-02_10-34-18_SAC_shelf_env_Gaussian_"],
    #         "recovery_0.3": ["2020-05-02_23-50-24_SAC_shelf_env_Gaussian_", "2020-05-02_23-51-35_SAC_shelf_env_Gaussian_", "2020-05-02_23-52-27_SAC_shelf_env_Gaussian_"],
    #         "recovery_0.4": ["2020-05-02_10-35-04_SAC_shelf_env_Gaussian_", "2020-05-02_23-48-11_SAC_shelf_env_Gaussian_", "2020-05-02_23-49-08_SAC_shelf_env_Gaussian_"],
    #         "recovery_0.6": ["2020-05-02_23-53-33_SAC_shelf_env_Gaussian_", "2020-05-02_23-54-31_SAC_shelf_env_Gaussian_", "2020-05-02_23-55-08_SAC_shelf_env_Gaussian_"],
    #         "recovery_0.8": ["2020-05-02_10-37-10_SAC_shelf_env_Gaussian_", "2020-05-02_23-58-07_SAC_shelf_env_Gaussian_", "2020-05-02_23-58-33_SAC_shelf_env_Gaussian_"],
    #         # "recovery_0.9": ["2020-05-02_10-49-34_SAC_shelf_env_Gaussian_"]
    #         "sac_penalty1": ["2020-05-03_22-17-28_SAC_shelf_env_Gaussian_", "2020-05-03_22-17-50_SAC_shelf_env_Gaussian_", "2020-05-03_22-18-43_SAC_shelf_env_Gaussian_"],
    #         "sac_penalty3": ["2020-05-03_22-20-30_SAC_shelf_env_Gaussian_", "2020-05-03_22-21-35_SAC_shelf_env_Gaussian_", "2020-05-03_22-22-46_SAC_shelf_env_Gaussian_"],
    #         "sac_penalty5": ["2020-05-03_22-31-48_SAC_shelf_env_Gaussian_", "2020-05-03_22-32-17_SAC_shelf_env_Gaussian_", "2020-05-03_22-32-44_SAC_shelf_env_Gaussian_"],
    #         # "sac_penalty10": ["2020-05-03_22-34-18_SAC_shelf_env_Gaussian_", "2020-05-03_22-34-38_SAC_shelf_env_Gaussian_", "2020-05-03_22-34-54_SAC_shelf_env_Gaussian_"],
    #         "sac_penalty25": ["2020-05-04_02-24-15_SAC_shelf_env_Gaussian_", "2020-05-04_02-24-21_SAC_shelf_env_Gaussian_", "2020-05-04_02-24-26_SAC_shelf_env_Gaussian_"],
    #         "sac_lagrangian": ["2020-05-04_04-34-38_SAC_shelf_env_Gaussian_", "2020-05-04_04-34-54_SAC_shelf_env_Gaussian_", "2020-05-04_04-35-08_SAC_shelf_env_Gaussian_"]
    #     },
    #     "outfile": "shelf.png"
    # } 
    "shelf": { # Sparse reward instead... (all up to 2800)
        "algs": {
            "sac_norecovery": ["2020-05-07_20-54-58_SAC_shelf_env_Gaussian_", "2020-05-07_20-55-14_SAC_shelf_env_Gaussian_", "2020-05-07_20-55-33_SAC_shelf_env_Gaussian_"],
            "recovery": ["2020-05-07_21-03-10_SAC_shelf_env_Gaussian_", "2020-05-07_21-03-22_SAC_shelf_env_Gaussian_", "2020-05-07_21-03-33_SAC_shelf_env_Gaussian_"], # eps_safe 0.4, 20k
            # "recovery_0.4_20k_gamma0.9": ["2020-05-18_01-27-18_SAC_shelf_env_Gaussian_", "2020-05-18_01-27-28_SAC_shelf_env_Gaussian_", "2020-05-18_01-27-38_SAC_shelf_env_Gaussian_"],
            # "recovery_0.4_5k": ["2020-05-09_04-36-14_SAC_shelf_env_Gaussian_", "2020-05-09_04-36-20_SAC_shelf_env_Gaussian_", "2020-05-09_04-36-27_SAC_shelf_env_Gaussian_"],
            # "recovery_0.8_images": ["2020-05-10_05-37-07_SAC_shelf_env_Gaussian_", "2020-05-10_05-36-50_SAC_shelf_env_Gaussian_", "2020-05-10_05-36-34_SAC_shelf_env_Gaussian_"],
            # "sac_penalty3_images": ["2020-05-11_04-30-15_SAC_shelf_env_Gaussian_", "2020-05-11_04-30-28_SAC_shelf_env_Gaussian_", "2020-05-11_04-30-36_SAC_shelf_env_Gaussian_"],
            # "sac_penalty10_images": ["2020-05-11_04-31-11_SAC_shelf_env_Gaussian_", "2020-05-11_04-31-22_SAC_shelf_env_Gaussian_", "2020-05-11_04-31-32_SAC_shelf_env_Gaussian_"],
            # "sac_norecovery_images": ["2020-05-17_05-25-13_SAC_shelf_env_Gaussian_", "2020-05-17_05-25-29_SAC_shelf_env_Gaussian_", "2020-05-17_05-25-43_SAC_shelf_env_Gaussian_"],
            # "recovery_0.6_gamma0.5_penalty3": ["2020-05-17_05-57-36_SAC_shelf_env_Gaussian_", "2020-05-17_05-57-43_SAC_shelf_env_Gaussian_", "2020-05-17_05-57-54_SAC_shelf_env_Gaussian_"]
            "sac_penalty3": ["2020-05-07_21-36-57_SAC_shelf_env_Gaussian_", "2020-05-07_21-37-04_SAC_shelf_env_Gaussian_", "2020-05-07_21-37-11_SAC_shelf_env_Gaussian_"],
            "sac_penalty10": ["2020-05-07_21-37-17_SAC_shelf_env_Gaussian_", "2020-05-07_21-37-25_SAC_shelf_env_Gaussian_", "2020-05-07_21-37-32_SAC_shelf_env_Gaussian_"],
            "RCPO": ["2020-06-15_19-49-04_SAC_shelf_env_Gaussian_", "2020-06-15_19-49-23_SAC_shelf_env_Gaussian_", "2020-06-15_19-49-39_SAC_shelf_env_Gaussian_"],
            "sac_lagrangian": ["2020-06-15_09-50-15_SAC_shelf_env_Gaussian_", "2020-06-15_09-50-45_SAC_shelf_env_Gaussian_", "2020-06-15_09-51-01_SAC_shelf_env_Gaussian_"], # nu = 1
            "sac_lagrangian_update": ["2020-06-16_02-30-16_SAC_shelf_env_Gaussian_", "2020-06-16_00-10-45_SAC_shelf_env_Gaussian_", "2020-06-16_00-10-57_SAC_shelf_env_Gaussian_"] 
            # "sac_lagrangian": ["2020-06-15_09-52-16_SAC_shelf_env_Gaussian_", "2020-06-15_09-52-43_SAC_shelf_env_Gaussian_"] # nu = 10
            # "sac_lagrangian": ["2020-06-15_09-53-42_SAC_shelf_env_Gaussian_", "2020-06-15_09-53-53_SAC_shelf_env_Gaussian_", "2020-06-15_09-54-06_SAC_shelf_env_Gaussian_"] # nu = 100
            # "recovery_0.4_20k_gamma0.5": ["2020-05-17_05-48-29_SAC_shelf_env_Gaussian_", "2020-05-17_05-50-23_SAC_shelf_env_Gaussian_", "2020-05-17_05-50-14_SAC_shelf_env_Gaussian_"]
            # "recovery_0.6_dense_gamma0.3": ["2020-05-15_23-16-25_SAC_shelf_env_Gaussian_", "2020-05-15_23-16-30_SAC_shelf_env_Gaussian_", "2020-05-15_23-16-34_SAC_shelf_env_Gaussian_"],
            # "recovery_0.6_dense_gamma0.4": ["2020-05-15_23-15-53_SAC_shelf_env_Gaussian_", "2020-05-15_23-15-57_SAC_shelf_env_Gaussian_", "2020-05-15_23-16-02_SAC_shelf_env_Gaussian_"],
            # "recovery_0.6_dense_gamma0.5": ["2020-05-15_23-16-07_SAC_shelf_env_Gaussian_", "2020-05-15_23-16-12_SAC_shelf_env_Gaussian_", "2020-05-15_23-16-20_SAC_shelf_env_Gaussian_"],
            # "recovery_0.6_dense_gamma0.65": ["2020-05-14_00-00-17_SAC_shelf_env_Gaussian_", "2020-05-14_00-00-26_SAC_shelf_env_Gaussian_", "2020-05-14_00-00-36_SAC_shelf_env_Gaussian_"],
            # "recovery_0.6_dense_gamma0.75": ["2020-05-13_23-59-02_SAC_shelf_env_Gaussian_", "2020-05-13_23-59-25_SAC_shelf_env_Gaussian_", "2020-05-13_23-59-43_SAC_shelf_env_Gaussian_"],
            # "recovery_0.6_dense_gamma0.85": ["2020-05-13_23-56-30_SAC_shelf_env_Gaussian_", "2020-05-13_21-30-48_SAC_shelf_env_Gaussian_", "2020-05-13_21-30-54_SAC_shelf_env_Gaussian_"]
            # "unconstrained_images": [],
            # "recovery_0.6_dense_gamma0.5_constraint_penalty0.3": []
        },
        "outfile": "shelf.png"
    },
    "shelf_dynamic": {
        "algs": {
            "sac_norecovery": ["2020-05-24_09-07-14_SAC_shelf_dynamic_env_Gaussian_", "2020-05-24_09-07-23_SAC_shelf_dynamic_env_Gaussian_", "2020-05-24_09-08-59_SAC_shelf_dynamic_env_Gaussian_"],
            "sac_penalty3": ["2020-05-24_09-05-16_SAC_shelf_dynamic_env_Gaussian_", "2020-05-24_10-26-43_SAC_shelf_dynamic_env_Gaussian_", "2020-05-24_09-08-28_SAC_shelf_dynamic_env_Gaussian_"],
            "sac_penalty10": ["2020-05-25_09-48-06_SAC_shelf_dynamic_env_Gaussian_", "2020-05-25_09-47-23_SAC_shelf_dynamic_env_Gaussian_", "2020-05-25_09-48-15_SAC_shelf_dynamic_env_Gaussian_"],
            # "recovery_0.2": ["2020-05-25_09-41-29_SAC_shelf_dynamic_env_Gaussian_", "2020-05-25_09-42-02_SAC_shelf_dynamic_env_Gaussian_", "2020-05-25_20-21-35_SAC_shelf_dynamic_env_Gaussian_"],
            "recovery": ["2020-05-26_03-32-21_SAC_shelf_dynamic_env_Gaussian_", "2020-05-26_03-32-30_SAC_shelf_dynamic_env_Gaussian_", "2020-05-26_03-32-37_SAC_shelf_dynamic_env_Gaussian_"], # eps_safe=0.25
            "RCPO": ["2020-06-15_19-47-45_SAC_shelf_dynamic_env_Gaussian_", "2020-06-15_19-48-05_SAC_shelf_dynamic_env_Gaussian_", "2020-06-15_19-48-19_SAC_shelf_dynamic_env_Gaussian_"],
            "sac_lagrangian": ["2020-06-04_04-31-13_SAC_shelf_dynamic_env_Gaussian_", "2020-06-03_23-56-14_SAC_shelf_dynamic_env_Gaussian_", "2020-06-15_19-41-22_SAC_shelf_dynamic_env_Gaussian_"], # nu = 1 (tried nu=1, 10, 100 this was best)
            "sac_lagrangian_update": ["2020-06-16_00-02-34_SAC_shelf_dynamic_env_Gaussian_", "2020-06-16_00-03-21_SAC_shelf_dynamic_env_Gaussian_", "2020-06-16_00-03-30_SAC_shelf_dynamic_env_Gaussian_"] 
            # "recovery_0.3": ["2020-05-24_21-59-33_SAC_shelf_dynamic_env_Gaussian_", "2020-05-25_09-38-26_SAC_shelf_dynamic_env_Gaussian_", "2020-05-25_09-38-51_SAC_shelf_dynamic_env_Gaussian_"],
            # "recovery_0.4": ["2020-05-24_21-57-43_SAC_shelf_dynamic_env_Gaussian_", "2020-05-24_21-59-23_SAC_shelf_dynamic_env_Gaussian_", "2020-05-24_21-59-28_SAC_shelf_dynamic_env_Gaussian_"]
        },
        "outfile": "shelf_dynamic.png"
    },
    "image_shelf": {
        "algs": {
            "sac_norecovery": ["2020-06-17_07-48-06_SAC_shelf_env_Gaussian_", "2020-06-17_07-50-05_SAC_shelf_env_Gaussian_"],
            # "recovery": ["2020-06-17_03-58-30_SAC_shelf_env_Gaussian_", "2020-06-17_03-58-44_SAC_shelf_env_Gaussian_", "2020-06-17_07-40-38_SAC_shelf_env_Gaussian_"], # eps_safe = 0.25
            # "recovery": ["2020-06-17_07-46-08_SAC_shelf_env_Gaussian_", "2020-06-17_07-46-16_SAC_shelf_env_Gaussian_"], # eps_safe = 0.3
            "recovery": ["2020-06-17_03-57-09_SAC_shelf_env_Gaussian_", "2020-06-17_03-57-51_SAC_shelf_env_Gaussian_"],
            "sac_penalty10": ["2020-06-18_04-43-05_SAC_shelf_env_Gaussian_", "2020-06-18_04-43-29_SAC_shelf_env_Gaussian_"]
        },
        "outfile": "image_shelf.png"
    }  
}


names = {
    "sac_norecovery": "SAC",
    "sac_penalty20": "SAC (penalty 20)",
    "sac_penalty50": "SAC (penalty 50)",
    "sac_penalty75": "SAC (penalty 75)",
    "sac_penalty100": "SAC (penalty 100)",
    "recovery": "SAC + Recovery",
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
    "sac_penalty20": "blue",
    "sac_penalty50": "orange",
    "sac_penalty75": "purple",
    "sac_penalty100": "black",
    "recovery": "red",
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
    "RCPO": 'magenta'
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

<<<<<<< Updated upstream
def plot_experiment(experiment, max_eps=3000): # 3000 for normal shelf...
=======
def plot_experiment(experiment, max_eps=2100): # 3000 for normal shelf...
>>>>>>> Stashed changes

    fig, axs = plt.subplots(4, figsize=(16, 27))

    axs[0].set_title("Cumulative Constraint Violations vs. Episode", fontsize=20)
    axs[0].set_ylim(-0.1, int(0.8*max_eps) + 1)
    axs[0].set_xlabel("Episode", fontsize=16)
    axs[0].set_ylabel("Cumulative Constraint Violations", fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=14)

    axs[1].set_title("Cumulative Task Successes vs. Episode", fontsize=20)
    axs[1].set_ylim(0, int(max_eps)+1)
    axs[1].set_xlabel("Episode", fontsize=16)
    axs[1].set_ylabel("Cumulative Task Successes", fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=14)

    axs[2].set_title("Cumulative Recovery Calls vs. Episode", fontsize=20)
    axs[2].set_ylim(-0.1, max_eps + 1)
    axs[2].set_xlabel("Episode", fontsize=16)
    axs[2].set_ylabel("Cumulative Recovery Calls", fontsize=16)
    axs[2].tick_params(axis='both', which='major', labelsize=14)

    axs[3].set_title("Cumulative Recovery Calls + Constraint Violated vs. Episode", fontsize=20)
    axs[3].set_ylim(-0.1, max_eps + 1)
    axs[3].set_xlabel("Episode", fontsize=16)
    axs[3].set_ylabel("Cumulative Recovery Calls + Constraint Violated", fontsize=16)
    axs[3].tick_params(axis='both', which='major', labelsize=14)

    alg_names = experiment_map[experiment]["algs"].keys()
    penalty_names = [name for name in alg_names if "penalty" in name]
    for i in range(len(penalty_names)):
        for j in range(len(penalty_names)):
            if i > j and int(penalty_names[i].split("penalty")[1]) < int(penalty_names[j].split("penalty")[1]):
                tmp = penalty_names[i]
                penalty_names[i] = penalty_names[j]
                penalty_names[j] = tmp

    if 'sac_norecovery' in alg_names:
        alg_names_new = ['sac_norecovery']
    else:
        alg_names_new = []
    alg_names_new += penalty_names
    if 'sac_lagrangian' in alg_names:
        alg_names_new += ['sac_lagrangian']
    if 'sac_lagrangian_update' in alg_names:
        alg_names_new += ['sac_lagrangian_update']
    if 'RCPO' in alg_names:
        alg_names_new += ['RCPO']
    if 'recovery' in alg_names:
        alg_names_new += ['recovery']

    print("ALG NAMES NEW: ", alg_names_new)
    for alg in alg_names_new:
        exp_dirs = experiment_map[experiment]["algs"][alg]
        fnames = [osp.join("runs", exp_dir, "run_stats.pkl") for exp_dir in exp_dirs]

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
            for traj_stats in train_stats:
                train_violations.append([])
                recovery_called.append([])
                train_rewards.append(0)
                last_reward = 0
                for step_stats in traj_stats:
                    train_violations[-1].append(step_stats['constraint'])
                    # recovery_called[-1].append(step_stats['recovery'])
                    if alg == "recovery" or alg == "recovery_reachability":
                        # recovery_called[-1].append(step_stats['recovery'])
                        recovery_called[-1].append(0)
                    else:
                        recovery_called[-1].append(0)
                    train_rewards[-1] += step_stats['reward']
                    last_reward = step_stats['reward']
                last_rewards.append(last_reward)

            recovery_called = np.array([np.sum(t) > 0 for t in recovery_called])[:max_eps].astype(int) # For now just look at whether a recovery was called at any point
            ep_lengths = np.array([len(t) for t in train_violations])[:max_eps]
            train_violations = np.array([np.sum(t) > 0 for t in train_violations])[:max_eps]
            recovery_called_constraint = np.bitwise_and(recovery_called, train_violations)

            recovery_called = np.cumsum(recovery_called)
            train_violations = np.cumsum(train_violations)
            recovery_called_constraint = np.cumsum(recovery_called_constraint)

            train_rewards = np.array(train_rewards)[:max_eps]
            last_rewards = np.array(last_rewards)[:max_eps]

            for i in range(len(train_rewards)):
                if ep_lengths[i] != 50:
                    diff = 50 - ep_lengths[i]
                    train_rewards[i] += diff * last_rewards[i]

            if 'maze' in experiment:
                task_successes = (-last_rewards < 0.03).astype(int)
            elif 'shelf' in experiment:
                task_successes = (last_rewards > 4.5).astype(int)
            else:
                assert False

            task_successes = np.cumsum(task_successes)
            task_successes_list.append(task_successes)

            # x = np.arange(len(last_rewards))
            # xnew = np.linspace(x.min(), x.max(), 100)
            # spl = make_interp_spline(x,last_rewards, k=3)
            # last_rewards_smooth = spl(xnew)

            # x = np.arange(len(train_rewards))
            # xnew = np.linspace(x.min(), x.max(), 100)
            # spl = make_interp_spline(x,train_rewards, k=3)
            # train_rewards_smooth = spl(xnew)

            # if experiment == 'maze' or 'shelf' in experiment:
            #     train_rewards_list.append(last_rewards_smooth)
            # else:
            #     train_rewards_list.append(train_rewards_smooth)

            train_violations_list.append(train_violations)
            recovery_called_list.append(recovery_called)
            recovery_called_constraint_list.append(recovery_called_constraint)

        task_successes_list = np.array(task_successes_list)
        # train_rewards_list = np.array(train_rewards_list)
        train_violations_list = np.array(train_violations_list)
        recovery_called_list = np.array(recovery_called_list)
        recovery_called_constraint_list = np.array(recovery_called_constraint_list)

        print("TASK SUCCESSES", task_successes_list.shape)
        # print("TRAIN REWARDS", train_rewards_list.shape)
        print("TRAIN VIOLS", train_violations_list.shape)
        print("TRAIN RECOVERY", recovery_called_list.shape)
        print("TRAIN RECOVERY CONSTRAINT", recovery_called_constraint_list.shape)

        ts_mean, ts_lb, ts_ub = get_stats(task_successes_list)
        # tr_mean, tr_lb, tr_ub = get_stats(train_rewards_list)
        tv_mean, tv_lb, tv_ub = get_stats(train_violations_list)
        trec_mean, trec_lb, trec_ub = get_stats(recovery_called_list)
        trec_constraint_mean, trec_constraint_lb, trec_constraint_ub = get_stats(recovery_called_constraint_list)

        axs[0].fill_between(range(tv_mean.shape[0]), tv_ub, tv_lb,
                     color=colors[alg], alpha=.5, label=names[alg])
        axs[0].plot(tv_mean, colors[alg])
        # axs[1].fill_between(xnew, tr_ub, tr_lb,
        #              color=colors[alg], alpha=.5, label=names[alg])
        # axs[1].plot(xnew, tr_mean, colors[alg])
        axs[1].fill_between(range(ts_mean.shape[0]), ts_ub, ts_lb,
                     color=colors[alg], alpha=.5)
        axs[1].plot(ts_mean, colors[alg], label=names[alg])
        
        axs[2].fill_between(range(trec_mean.shape[0]), trec_ub, trec_lb,
                     color=colors[alg], alpha=.5, label=names[alg])
        axs[2].plot(range(trec_mean.shape[0]), trec_mean, colors[alg])

        axs[3].fill_between(range(trec_constraint_mean.shape[0]), trec_constraint_ub, trec_constraint_lb,
                     color=colors[alg], alpha=.5)
        axs[3].plot(trec_constraint_mean, colors[alg], label=names[alg])

    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper left")
    axs[2].legend(loc="upper left")
    axs[3].legend(loc="upper left")
    plt.savefig(experiment_map[experiment]["outfile"])
    plt.show()


if __name__ == '__main__':
    # experiment = "shelf_dynamic"
    # experiment = "image_maze"
<<<<<<< Updated upstream
    experiment = "shelf"
=======
    # experiment = "shelf"
    experiment = "image_shelf"
>>>>>>> Stashed changes
    plot_experiment(experiment)

# "recovery_0.4": ["2020-05-04_03-41-46_SAC_shelf_env_Gaussian_", "2020-05-04_03-49-11_SAC_shelf_env_Gaussian_", "2020-05-04_03-42-53_SAC_shelf_env_Gaussian_"], # Bad results: planhor=5
# "recovery_0.4": ["2020-05-04_03-00-46_SAC_shelf_env_Gaussian_", "2020-05-04_03-01-05_SAC_shelf_env_Gaussian_", "2020-05-04_03-01-17_SAC_shelf_env_Gaussian_"], # Bad results: include constraint penalty
