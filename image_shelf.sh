#!/bin/bash

# Recovery RL PLaNet Recovery
# for i in {1..3}
# do
# 	echo "PlaNet Recovery Run $i"
# 	python -m main --cuda --env-name shelf_env --use_recovery --use_qvalue --critic_safe_update_freq 200000 --recovery_policy_update_freq 100 --gamma_safe 0.85 --eps_safe 0.35 --cnn --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --model_fname model_shelf_qvalue --beta 10 --vismpc_recovery --load_vismpc --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix recovery_0.85_0.35 --pos_fraction 0.3 --load_vismpc
# done

# # Recovery RL PLaNet Recovery
# for i in {1..3}
# do
# 	echo "PlaNet Recovery Run $i"
# 	python -m main --cuda --env-name shelf_env --use_recovery --use_qvalue --critic_safe_update_freq 200000 --recovery_policy_update_freq 100 --gamma_safe 0.7 --eps_safe 0.35 --cnn --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --model_fname model_shelf_qvalue --beta 10 --vismpc_recovery --load_vismpc --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix recovery_0.7_0.35 --pos_fraction 0.3 --load_vismpc
# done

# # Recovery RL PLaNet Recovery
# for i in {1..3}
# do
# 	echo "PlaNet Recovery Run $i"
# 	python -m main --cuda --env-name shelf_env --use_recovery --use_qvalue --critic_safe_update_freq 200000 --recovery_policy_update_freq 100 --gamma_safe 0.55 --eps_safe 0.35 --cnn --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --model_fname model_shelf_qvalue --beta 10 --vismpc_recovery --load_vismpc --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix recovery_0.55_0.35 --pos_fraction 0.3 --load_vismpc
# done

# # Recovery RL DDPG Recovery
# for i in {1..3}
# do
# 	echo "DDPG Recovery Run $i"
# 	python -m main --cuda --env-name shelf_env --use_recovery --use_qvalue --ddpg_recovery --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.7 --eps_safe 0.35 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 30000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix ddpg_recovery_0.7_0.35 --pos_fraction 0.3
# done

# # Recovery RL DDPG Recovery
# for i in {1..3}
# do
# 	echo "DDPG Recovery Run $i"
# 	python -m main --cuda --env-name shelf_env --use_recovery --use_qvalue --ddpg_recovery --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.55 --eps_safe 0.35 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 30000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix ddpg_recovery_0.55_0.35 --pos_fraction 0.3
# done

# Vanilla
# for i in {1..3}
# do
# 	echo "SAC Run $i"
# 	python -m main --cuda --env-name shelf_env --cnn  --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix vanilla --pos_fraction 0.3
# done

# Reward Penalty
# for i in {1..3}
# do
# 	echo "SAC Reward 3"
# 	python -m main --cuda --env-name shelf_env --cnn  --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix reward_3 --pos_fraction 0.3 --constraint_reward_penalty 3
# done

# # Reward Penalty
# for i in {1..3}
# do
# 	echo "SAC Reward 10"
# 	python -m main --cuda --env-name shelf_env --cnn  --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix reward_10 --pos_fraction 0.3 --constraint_reward_penalty 10
# done

# # Reward Penalty
# for i in {1..3}
# do
# 	echo "SAC Reward 15"
# 	python -m main --cuda --env-name shelf_env --cnn  --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix reward_15 --pos_fraction 0.3 --constraint_reward_penalty 15
# done

