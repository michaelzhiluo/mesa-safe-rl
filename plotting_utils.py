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
    "sac_recovery_ddpg": "#60CC38",
    "sac_recovery_pets": "#349C26",
    "sac_recovery_pets_100": "#AA5D1F",
    "sac_recovery_pets_500": "#34539C",
    "sac_recovery_pets_1k": "#4899C5",
    "sac_recovery_pets_5k": "#D43827",
    "sac_recovery_pets_20k": "#349C26",
    "sac_recovery_pets_ablations": "#349C26",
    "sac_recovery_pets_disable_relabel": "#34539C",
    "sac_recovery_pets_disable_offline": "#AA5D1F",
    "sac_recovery_pets_disable_online": "#D43827"
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
    "sac_recovery_pets_disable_relabel": "Ours - Action Relabeling",
    "sac_recovery_pets_disable_offline": "Ours - Offline Training",
    "sac_recovery_pets_disable_online": "Ours - Online Training"
}


def get_color(algname, alt_color_map={}):
	if algname in colors:
		return colors[algname]
	elif algname in alt_color_map:
		return alt_color_map[algname]
	else:
		return np.random.rand(3,)


def get_legend_name(algname, alt_name_map={}):
	if algname in names:
		return names[algname]
	elif algname in alt_name_map:
		return alt_name_map[algname]
	else:
		return algname
