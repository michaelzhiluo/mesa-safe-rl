#!/bin/bash

# Recovery RL (model-free recovery)
python -m main --cuda --env-name shelf_dynamic_long_env --task_demos --num_task_transitions 1000 --tau 0.0002 --replay_size 100000 --num_eps 4000 --use_qvalue --use_recovery --gamma_safe 0.85 --eps_safe 0.35 --recovery_policy_update_freq 20  --pos_fraction 0.3 --ddpg_recovery

# Recovery RL (model-based recovery)
python -m main --cuda --env-name shelf_dynamic_long_env --task_demos --num_task_transitions 1000 --tau 0.0002 --replay_size 100000 --num_eps 4000 --use_qvalue --use_recovery --gamma_safe 0.85 --eps_safe 0.25 --recovery_policy_update_freq 20  --pos_fraction 0.3

# Unconstrained
python -m main --cuda --env-name shelf_dynamic_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --pos_fraction 0.3

# Lagrangian Relaxation
python -m main --cuda --env-name shelf_dynamic_long_env --task_demos --num_task_transitions 1000 --tau 0.0002 --replay_size 100000 --num_eps 4000 --use_qvalue --DGD_constraints --nu 20 --gamma_safe 0.85 --eps_safe 0.25 --update_nu --pos_fraction 0.3

# RSPO
python -m main --cuda --env-name shelf_dynamic_long_env --task_demos --num_task_transitions 1000 --tau 0.0002 --replay_size 100000 --num_eps 4000 --use_qvalue --DGD_constraints --nu_start 40 --gamma_safe 0.85 --eps_safe 0.25 --nu_schedule --pos_fraction 0.3

# SQRL
python -m main --cuda --env-name shelf_dynamic_long_env --task_demos --num_task_transitions 1000 --tau 0.0002 --replay_size 100000 --num_eps 4000 --use_qvalue --DGD_constraints --nu 20 --gamma_safe 0.85 --eps_safe 0.25 --update_nu --use_constraint_sampling --pos_fraction 0.3

# Reward Penalty
python -m main --cuda --env-name shelf_dynamic_long_env --task_demos --num_task_transitions 1000 --tau 0.0002 --replay_size 100000 --num_eps 4000 --constraint_reward_penalty 25 --pos_fraction 0.3

# RCPO
python -m main --cuda --env-name shelf_dynamic_long_env --task_demos --num_task_transitions 1000 --tau 0.0002 --replay_size 100000 --num_eps 4000 --RCPO --lambda_RCPO 10 --use_qvalue --gamma_safe 0.85 --eps_safe 0.25 --pos_fraction 0.3
