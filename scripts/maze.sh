#!/bin/bash

# Recovery RL (model-free recovery)
python -m main --cuda --env-name maze --use_recovery --ddpg_recovery --use_qvalue --critic_safe_update_freq 5 --recovery_policy_update_freq 5 --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3

# Recovery RL (model-based recovery)
python -m main --cuda --env-name maze --use_recovery --use_qvalue --recovery_policy_update_freq 5 --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3

# Unconstrained
python main.py --env-name maze --cuda

# Lagrangian Relaxation
python -m main --cuda --env-name maze --use_qvalue --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --DGD_constraints --nu 100 --update_nu

# RSPO
python -m main --cuda --env-name maze --use_qvalue --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --DGD_constraints --nu_schedule --nu_start 200

# SQRL
python -m main --cuda --env-name maze --use_qvalue --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --DGD_constraints --use_constraint_sampling --nu 100 --update_nu

# Reward Penalty
python main.py --env-name maze --cuda --constraint_reward_penalty 50

# RCPO
python -m main --cuda --env-name maze --use_qvalue --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --RCPO --lambda 50
