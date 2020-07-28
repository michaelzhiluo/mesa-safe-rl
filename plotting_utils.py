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
    "sac_vanilla": "#7776bc",
    "sac_lagrangian": "#aef78e",
    "sac_rspo": "#8ff499",
    "sac_sqrl": "#66a182",
    "sac_penalty": "#b7c335",
    "sac_rcpo": "#be8d39",
    "sac_recovery_ddpg": "#f88585",
    "sac_recovery_pets": "#830404",
    "sac_recovery_pets_100": "#7776bc",
    "sac_recovery_pets_500": "#be8d39",
    "sac_recovery_pets_1k": "#b7c335",
    "sac_recovery_pets_5k": "#66a182",
    "sac_recovery_pets_20k": "#830404",
    "sac_recovery_pets_ablations": "#830404",
    "sac_recovery_pets_disable_relabel": "#be8d39",
    "sac_recovery_pets_disable_offline": "#f88585",
    "sac_recovery_pets_disable_online": "#66a182"
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
    "sac_recovery_pets_ablations": "Ours",
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
