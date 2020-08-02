#!/bin/bash

# Recovery RL (model-free recovery)
python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.3 --use_qvalue --ddpg_recovery

# Recovery RL (model-based recovery)
python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.3 --use_qvalue

# Unconstrained
python main.py --env-name simplepointbot0 --cuda

# Lagrangian Relaxation
python main.py --env-name simplepointbot0 --cuda --use_qvalue --DGD_constraints --nu 5000 --gamma_safe 0.8 --eps_safe 0.3 --update_nu

# RSPO
python main.py --env-name simplepointbot0 --cuda --use_qvalue --DGD_constraints --nu_start 10000 --gamma_safe 0.8 --eps_safe 0.3 --nu_schedule

# SQRL
python main.py --env-name simplepointbot0 --cuda --use_qvalue --DGD_constraints --nu 5000 --gamma_safe 0.8 --eps_safe 0.3 --update_nu --use_constraint_sampling

# Reward Penalty
python main.py --env-name simplepointbot0 --cuda --constraint_reward_penalty 1000

# RCPO
python main.py --env-name simplepointbot0 --cuda --use_qvalue --DGD_constraints --lambda_RCPO 1000 --gamma_safe 0.8 --eps_safe 0.3 --RCPO
