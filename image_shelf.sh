#!/bin/bash

# Recovery RL PLaNet Recovery
# for i in {1..3}
# do
# 	echo "PlaNet Recovery Run $i"
# 	python -m main --cuda --env-name shelf_env --use_recovery --use_qvalue --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.85 --eps_safe 0.35 --cnn --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --model_fname model_shelf3 --beta 10 --vismpc_recovery --load_vismpc --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix recovery_0.85_0.35 --pos_fraction 0.3 --load_vismpc
# done

# # Recovery RL PLaNet Recovery
# for i in {1..3}
# do
# 	echo "PlaNet Recovery Run $i"
# 	python -m main --cuda --env-name shelf_env --use_recovery --use_qvalue --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.75 --eps_safe 0.35 --cnn --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --model_fname model_shelf3 --beta 10 --vismpc_recovery --load_vismpc --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix recovery_0.75_0.35 --pos_fraction 0.3 --load_vismpc
# done

# # Recovery RL PLaNet Recovery
# for i in {1..3}
# do
# 	echo "PlaNet Recovery Run $i"
# 	python -m main --cuda --env-name shelf_env --use_recovery --use_qvalue --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.85 --eps_safe 0.25 --cnn --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --model_fname model_shelf3 --beta 10 --vismpc_recovery --load_vismpc --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix recovery_0.85_0.25 --pos_fraction 0.3 --load_vismpc
# done

# # Recovery RL PLaNet Recovery
# for i in {1..3}
# do
# 	echo "PlaNet Recovery Run $i"
# 	python -m main --cuda --env-name shelf_env --use_recovery --use_qvalue --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.75 --eps_safe 0.25 --cnn --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --model_fname model_shelf3 --beta 10 --vismpc_recovery --load_vismpc --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix recovery_0.75_0.25 --pos_fraction 0.3 --load_vismpc
# done

# # Recovery RL DDPG Recovery (DONE)
# for i in {1..3}
# do
# 	echo "DDPG Recovery Run $i"
# 	python -m main --cuda --env-name shelf_env --use_recovery --use_qvalue --ddpg_recovery --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.6 --eps_safe 0.35 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 5000000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix ddpg_recovery_0.6_0.35 --pos_fraction 0.3
# done

# SAC Lagrangian Nu=50
# for i in {1..3}
# do
# 	echo "SAC Lagrangian Nu=50 Run $i"
# 	python -m main --cuda --env-name shelf_env --use_qvalue --critic_safe_update_freq 200000 --gamma_safe 0.6 --eps_safe 0.35 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 5000000 --DGD_constraints --nu 50 --update_nu --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix nu_50 --pos_fraction 0.3
# done

# SAC Lagrangian Nu=100
# for i in {1..3}
# do
# 	echo "SAC Lagrangian Nu=100 Run $i"
# 	python -m main --cuda --env-name shelf_env --use_qvalue --critic_safe_update_freq 200000 --gamma_safe 0.6 --eps_safe 0.35 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 30000 --DGD_constraints --nu 100 --update_nu --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix nu_100 --pos_fraction 0.3
# done

# # SAC Lagrangian Nu=10
# for i in {1..3}
# do
# 	echo "SAC Lagrangian Nu=10 $i"
# 	python -m main --cuda --env-name shelf_env --use_qvalue --critic_safe_update_freq 200000 --gamma_safe 0.6 --eps_safe 0.35 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 500000 --DGD_constraints --nu 10 --update_nu --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix nu_10 --pos_fraction 0.3
# done

# RCPO, RSPO, SQRL

# for i in {1..3}
# do
# 	echo "SAC RCPO Lambda=10 Run $i"
# 	python -m main --cuda --env-name shelf_env --use_qvalue --critic_safe_update_freq 200000 --gamma_safe 0.6 --eps_safe 0.35 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 500000 --RCPO --lambda 10 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix lambda_10 --pos_fraction 0.3
# done

# for i in {1..3}
# do
# 	echo "SAC RSPO Run $i"
# 	python -m main --cuda --env-name shelf_env --use_qvalue --critic_safe_update_freq 200000 --gamma_safe 0.6 --eps_safe 0.35 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 500000 --DGD_constraints --nu_schedule --nu_start 100 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix RSPO --pos_fraction 0.3
# done

# for i in {1..3}
# do
# 	echo "SAC SQRL Run $i"
# 	python -m main --cuda --env-name shelf_env --use_qvalue --critic_safe_update_freq 200000 --gamma_safe 0.6 --eps_safe 0.35 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 500000 --DGD_constraints --nu 50 --update_nu --use_constraint_sampling --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix SQRL --pos_fraction 0.3
# done

# Vanilla
# for i in {1..3}
# do
# 	echo "SAC Run $i"
# 	python -m main --cuda --env-name shelf_env --cnn  --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix vanilla --pos_fraction 0.3
# done

# # Reward Penalty
# for i in {1..3}
# do
# 	echo "SAC Reward 10"
# 	python -m main --cuda --env-name shelf_env --cnn  --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix reward_10 --pos_fraction 0.3 --constraint_reward_penalty 10
# done
