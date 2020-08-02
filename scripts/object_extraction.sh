#!/bin/bash

# Recovery RL (model-free recovery)
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --ddpg_recovery --pos_fraction 0.3

# Recovery RL (model-based recovery)
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --pos_fraction 0.3

# Unconstrained
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --pos_fraction 0.3

# Lagrangian Relaxation
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --DGD_constraints --update_nu --nu 50 --pos_fraction 0.3

# RSPO
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --DGD_constraints --use_qvalue --nu_schedule --nu_start 100 --pos_fraction 0.3 

# SQRL
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --DGD_constraints --update_nu --use_constraint_sampling --nu 50 --pos_fraction 0.3

# Reward Penalty
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --constraint_reward_penalty 50 --pos_fraction 0.3

# RCPO
python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --RCPO --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --lambda 50 --pos_fraction 0.3
