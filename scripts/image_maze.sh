#!/bin/bash

# Recovery RL (model-free recovery)
python -m main --cuda --env-name image_maze --use_recovery --use_qvalue --ddpg_recovery --recovery_policy_update_freq 200000 --gamma_safe 0.65 --eps_safe 0.1 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 20000

# Recovery RL (model-based recovery)
python -m main --cuda --env-name image_maze --use_recovery --use_qvalue --recovery_policy_update_freq 200000 --gamma_safe 0.6 --eps_safe 0.05 --cnn --critic_safe_pretraining_steps 30000 --num_constraint_transitions 20000 --model_fname image_maze_dynamics --beta 10 --vismpc_recovery

# Unconstrained
python main.py --env-name image_maze --cuda

# Lagrangian Relaxation
python -m main --cuda --env-name image_maze --use_qvalue --gamma_safe 0.65 --eps_safe 0.1 --cnn --DGD_constraints --nu 10  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 20000 --update_nu

# RSPO
python -m main --cuda --env-name image_maze --use_qvalue --gamma_safe 0.65 --eps_safe 0.1 --cnn --DGD_constraints --nu_schedule --nu_start 20  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 20000 

# SQRL
python -m main --cuda --env-name image_maze --use_qvalue --gamma_safe 0.65 --eps_safe 0.1 --cnn --DGD_constraints --use_constraint_sampling --nu 10 --update_nu  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 20000

# Reward Penalty
python main.py --env-name image_maze --cuda --constraint_reward_penalty 20

# RCPO
python -m main --cuda --env-name image_maze --use_qvalue --gamma_safe 0.65 --eps_safe 0.1 --cnn --RCPO --lambda 20  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 20000
