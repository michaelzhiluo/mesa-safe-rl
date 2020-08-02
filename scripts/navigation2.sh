#!/bin/bash

# Recovery RL (model-free recovery)
python main.py --env-name simplepointbot1 --cuda --use_recovery --gamma_safe 0.65 --eps_safe 0.1 --use_qvalue --ddpg_recovery

# Recovery RL (model-based recovery)
python main.py --env-name simplepointbot1 --cuda --use_recovery --gamma_safe 0.65 --eps_safe 0.1 --use_qvalue

# Unconstrained
python main.py --env-name simplepointbot1 --cuda

# Lagrangian Relaxation
python main.py --env-name simplepointbot1 --cuda --use_qvalue --DGD_constraints --nu 1000 --gamma_safe 0.65 --eps_safe 0.1 --update_nu

# RSPO
python main.py --env-name simplepointbot1 --cuda --use_qvalue --DGD_constraints --nu_start 2000 --gamma_safe 0.65 --eps_safe 0.1 --nu_schedule

# SQRL
python main.py --env-name simplepointbot1 --cuda --use_qvalue --DGD_constraints --nu 1000 --gamma_safe 0.65 --eps_safe 0.1 --update_nu --use_constraint_sampling

# Reward Penalty
python main.py --env-name simplepointbot1 --cuda --constraint_reward_penalty 3000

# RCPO
python main.py --env-name simplepointbot1 --cuda --use_qvalue --DGD_constraints --lambda_RCPO 5000 --gamma_safe 0.65 --eps_safe 0.1 --RCPO
