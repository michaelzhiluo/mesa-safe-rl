import numpy as np

# colors = {
#     "sac_recovery_pets": "red",
#     "sac_recovery_ddpg": "blue",
#     "sac_penalty": "green",
#     "sac_lagrangian": "black",
#     "sac_rcpo": "purple",
#     "sac_rspo": "orange",
#     "sac_vanilla": "olive",
#     "sac_sqrl": "magenta",
#     "sac_recovery_disable_relabel": "blue",
#     "sac_recovery_pets_1k": "green",
#     "sac_recovery_pets_5k": "purple"
# }

colors = {
    "sac_vanilla": "#AA5D1F",
    "sac_lagrangian": "#BA2DC1",
    "sac_rspo": "#6C2896",
    "sac_sqrl": "#D43827",
    "sac_penalty": "#4899C5",
    "sac_rcpo": "#34539C",
    "sac_recovery_ddpg": "red",
    "sac_recovery_pets": "#349C26",
    "sac_recovery_pets_100": "#AA5D1F",
    "sac_recovery_pets_500": "#34539C",
    "sac_recovery_pets_1k": "#4899C5",
    "sac_recovery_pets_5k": "#D43827",
    "sac_recovery_pets_20k": "#349C26",
    "sac_recovery_pets_100": "#AA5D1F",
    "sac_recovery_pets_500": "#34539C",
    "sac_recovery_pets_1k": "#4899C5",
    "sac_recovery_pets_5k": "#D43827",
    "sac_recovery_pets_20k": "#349C26",
    "reward_5": "#AA5D1F",
    "reward_10": "#34539C",
    "reward_15": "#4899C5",
    "reward_25": "#D43827",
    "reward_50": "#349C26",
    "nu_5": "#AA5D1F",
    "nu_10": "#34539C",
    "nu_15": "#4899C5",
    "nu_25": "#D43827",
    "nu_50": "#349C26",
    "lambda_5": "#AA5D1F",
    "lambda_10": "#34539C",
    "lambda_15": "#4899C5",
    "lambda_25": "#D43827",
    "lambda_50": "#349C26",
    "eps_0.15": "#AA5D1F",
    "eps_0.25": "#34539C",
    "eps_0.35": "#4899C5",
    "eps_0.45": "#D43827",
    "eps_0.55": "#349C26",
    "sac_recovery_pets_ablations": "#349C26",
    "sac_recovery_pets_disable_relabel": "#34539C",
    "sac_recovery_pets_disable_offline": "#AA5D1F",
    "sac_recovery_pets_disable_online": "#D43827",
    "multitask": "#AA5D1F",
    "meta": "#BA2DC1",
}

# colors = {
#     "sac_recovery_pets": (0, 0.45, 0.7),
#     "sac_recovery_ddpg": (0.8, 0.6, 0.7),
#     "sac_penalty": (0, 0.6, 0.5),
#     "sac_lagrangian": "black",
#     "sac_rcpo": (0.8, 0.4, 0),
#     "sac_rspo": (0.9, 0.6, 0),
#     "sac_vanilla": (0.35, 0.7, 0.9),
#     "sac_sqrl": (0.2, 0.7, 0.3)
# }

names = {
    "sac_recovery_pets": "SAC + Model-Based Recovery",
    "sac_recovery_pets_ablations": "Ours: Recovery RL (MB Recovery)",
    "sac_recovery_ddpg": "SAC + Model-Free Recovery",
    "sac_penalty": "SAC + Reward Penalty (RCPO)",
    "sac_lagrangian": "SAC + Lagrangian",
    "sac_rcpo": "SAC + Critic Penalty (RCPO)",
    "sac_rspo": "SAC + RSPO",
    "sac_vanilla": "SAC",
    "sac_sqrl": "SQRL",
    "sac_recovery_pets_100": "100",
    "sac_recovery_pets_500": "500",
    "sac_recovery_pets_1k": "1K",
    "sac_recovery_pets_5k": "5K",
    "sac_recovery_pets_20k": "20K",
    "reward_5": "$\lambda = 5$",
    "reward_10": "$\lambda = 10$",
    "reward_15": "$\lambda = 15$",
    "reward_25": "$\lambda = 25$",
    "reward_50": "$\lambda = 50$",
    "nu_5": "$\lambda = 5$",
    "nu_10": "$\lambda = 10$",
    "nu_15": "$\lambda = 15$",
    "nu_25": "$\lambda = 25$",
    "nu_50": "$\lambda = 50$",
    "lambda_5": "$\lambda = 5$",
    "lambda_10": "$\lambda = 10$",
    "lambda_15": "$\lambda = 15$",
    "lambda_25": "$\lambda = 25$",
    "lambda_50": "$\lambda = 50$",
    "eps_0.15": "$\epsilon_{risk} = 0.15$",
    "eps_0.25": "$\epsilon_{risk} = 0.25$",
    "eps_0.35": "$\epsilon_{risk} = 0.35$",
    "eps_0.45": "$\epsilon_{risk} = 0.45$",
    "eps_0.55": "$\epsilon_{risk} = 0.55$",
    "sac_recovery_pets_disable_relabel": "Ours - Action Relabeling",
    "sac_recovery_pets_disable_offline": "Ours - Offline Training",
    "sac_recovery_pets_disable_online": "Ours - Online Training",
    "multitask": "Multitask",
    "meta": "Metalearning",
}


def get_color(algname, alt_color_map={}):
    if algname in colors:
        return colors[algname]
    elif algname in alt_color_map:
        return alt_color_map[algname]
    else:
        return np.random.rand(3, )


def get_legend_name(algname, alt_name_map={}):
    if algname in names:
        return names[algname]
    elif algname in alt_name_map:
        return alt_name_map[algname]
    else:
        return algname
