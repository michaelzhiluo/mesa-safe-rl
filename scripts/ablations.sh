#!/bin/bash

# ----- # Demos Ablations -----

# 100 constraint demos
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_constraint_transitions 100 --pos_fraction 0.3

# 500 constraint demos
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_constraint_transitions 500 --pos_fraction 0.3

# 1000 constraint demos
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_constraint_transitions 1000 --pos_fraction 0.3

# 5000 constraint demos
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_constraint_transitions 5000 --pos_fraction 0.3

# 20000 constraint demos
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_constraint_transitions 20000 --pos_fraction 0.3

# ----- # Method Ablations -----

# Disable Action Relabeling
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --disable_action_relabeling --pos_fraction 0.3

# Disable Online Updates
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --disable_online_updates --pos_fraction 0.3

# Disable Offline Updates
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --disable_offline_updates --pos_fraction 0.3

