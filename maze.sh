#!/bin/bash
# Recovery RL PETS Recovery
# for i in {1..3}
# do
# 	echo "Recovery Run $i"
# 	python -m main --cuda --env-name maze --use_recovery --use_qvalue --critic_safe_update_freq 5 --recovery_policy_update_freq 5 --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --num_eps 500 --seed $i --logdir maze --logdir_suffix recovery
# done

# # Recovery RL DDPG Recovey
# for i in {1..3}
# do
# 	echo "DDPG Recovery Run $i"
# 	python -m main --cuda --env-name maze --use_recovery --ddpg_recovery --use_qvalue --critic_safe_update_freq 5 --recovery_policy_update_freq 5 --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --num_eps 500 --seed $i --logdir maze --logdir_suffix ddpg_recovery
# done

# # Unconstrained
# for i in {1..3}
# do
# 	echo "SAC Run $i"
# 	python main.py --env-name maze --cuda --logdir maze --logdir_suffix vanilla --num_eps 500 --seed $i
# done

# # Reward Penalty 50
# for i in {1..3}
# do
# 	echo "Reward Penalty 50 Run $i"
# 	python main.py --env-name maze --cuda --constraint_reward_penalty 50 --logdir maze --logdir_suffix reward_50 --num_eps 500 --seed $i

# done

# # Reward Penalty 10
# for i in {1..3}
# do
# 	echo "Reward Penalty 10 Run $i"
# 	python main.py --env-name maze --cuda --constraint_reward_penalty 10 --logdir maze --logdir_suffix reward_10 --num_eps 500 --seed $i

# done

# # Reward Penalty 100
# for i in {1..3}
# do
# 	echo "Reward Penalty 100 Run $i"
# 	python main.py --env-name maze --cuda --constraint_reward_penalty 100 --logdir maze --logdir_suffix reward_100 --num_eps 500 --seed $i

# done

# # SAC Lagrangian Nu=10
# for i in {1..3}
# do
# 	echo "Lagrangian Nu 1 Run $i"
# 	python -m main --cuda --env-name maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --DGD_constraints --nu 1 --logdir maze --logdir_suffix nu_1 --num_eps 500 --seed $i
# done

# # SAC Lagrangian Nu=50
# for i in {1..3}
# do
# 	echo "Lagrangian Nu 10 Run $i"
# 	python -m main --cuda --env-name maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --DGD_constraints --nu 10 --logdir maze --logdir_suffix nu_10 --num_eps 500 --seed $i
# done

# # SAC Lagrangian Nu=100
# for i in {1..3}
# do
# 	echo "Lagrangian Nu 100 Run $i"
# 	python -m main --cuda --env-name maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --DGD_constraints --nu 100 --logdir maze --logdir_suffix nu_100 --num_eps 500 --seed $i
# done

# RCPO Lambda=50
# for i in {1..3}
# do
# 	echo "RCPO Lambda 50 Run $i"
# 	python -m main --cuda --env-name maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --RCPO --lambda 50 --logdir maze --logdir_suffix lambda_50 --num_eps 500 --seed $i
# done

# # RCPO Lambda=10
# for i in {1..3}
# do
# 	echo "RCPO Lambda 10 Run $i"
# 	python -m main --cuda --env-name maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --RCPO --lambda 10 --logdir maze --logdir_suffix lambda_10 --num_eps 500 --seed $i
# done

# # RCPO Lambda=100
# for i in {1..3}
# do
# 	echo "RCPO Lambda 100 Run $i"
# 	python -m main --cuda --env-name maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --RCPO --lambda 100 --logdir maze --logdir_suffix lambda_100 --num_eps 500 --seed $i
# done


