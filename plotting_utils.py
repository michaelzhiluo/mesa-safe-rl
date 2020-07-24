import numpy as np

colors = {
    "sac_recovery_pets": "red",
    "sac_recovery_ddpg": "blue",
    "sac_penalty": "green",
    "sac_lagrangian": "black",
    "sac_rcpo": "purple",
    "sac_rspo": "orange",
    "sac_vanilla": "olive",
    "sac_sqrl": "magenta"
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
    "sac_recovery_ddpg": "SAC + Model-Free Recovery",
    "sac_penalty": "SAC + Reward Penalty (RCPO)",
    "sac_lagrangian": "SAC + Lagrangian",
    "sac_rcpo": "SAC + Critic Penalty (RCPO)",
    "sac_rspo": "SAC + RSPO",
    "sac_vanilla": "SAC",
    "sac_sqrl": "SQRL"
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
