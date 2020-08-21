#!/bin/bash

# Recovery RL PETS Recovery
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_qvalue --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_0.85_0.25 --pos_fraction 0.3 --seed $i
# done

# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_0.75_0.25 --pos_fraction 0.3 --seed $i
# done

# Recovery RL PETS Recovery
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.15 --use_qvalue --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_0.85_0.15 --pos_fraction 0.3 --seed $i
# done

# # Recovery RL PETS Recovery
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.35 --use_qvalue --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_0.75_0.35 --pos_fraction 0.3 --seed $i
# done

# # Recovery RL PETS Recovery
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.35 --use_qvalue --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_0.75_0.25 --pos_fraction 0.3 --seed $i
# done

# Recovery RL PETS Recovery
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.15 --use_qvalue --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_0.75_0.15 --pos_fraction 0.3 --seed $i
# done

# # Recovery RL DDPG Recovery
# Recovery RL DDPG
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --gamma_safe 0.85 --eps_safe 0.25 --use_qvalue --ddpg_recovery --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_ddpg_0.85_0.25 --pos_fraction 0.3 --seed $i
# done

# # Recovery RL DDPG Recovery Images
# Recovery RL DDPG
for i in {3..3}
do
	python -m main --cuda --env-name shelf_long_env --cnn --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --task_demos --tau 0.0002 --alpha 0.05 --replay_size 100000 --num_task_transitions 5000 --use_recovery --gamma_safe 0.75 --eps_safe 0.15 --use_qvalue --ddpg_recovery --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_ddpg_0.85_0.25_images --pos_fraction 0.3 --seed $i
done

# Unconstrained
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --num_eps 4000 --logdir shelf_long_env --logdir_suffix vanilla --pos_fraction 0.3 --seed $i
# done

# Reward Penalty 10
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --num_eps 4000 --constraint_reward_penalty 10 --logdir shelf_long_env --logdir_suffix reward_10 --pos_fraction 0.3 --seed $i
# done

# SAC Lagrangian Nu = 10 (Update nu)
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --DGD_constraints --update_nu --nu 10 --num_eps 4000 --logdir shelf_long_env --logdir_suffix nu_10 --pos_fraction 0.3 --seed $i
# done

# SQRL Nu = 50 (Update nu)
# for i in {1..3}
# do
# 	echo "SQRL Nu 50 Update Nu Run $i"
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --DGD_constraints --update_nu --use_constraint_sampling --nu 50 --num_eps 4000 --logdir shelf_long_env --logdir_suffix update_nu_50_SQRL --pos_fraction 0.3 --seed $i
# done

# SAC Lagrangian RSPO (Update nu) # Start at 2x best nu and decay
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --num_eps 4000 --DGD_constraints --use_qvalue  --nu_schedule --nu_start 200 -logdir shelf_long_env --logdir_suffix RSPO --pos_fraction 0.3 --seed $i
# done

# RCPO Lambda 10
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --num_eps 4000 --RCPO --use_qvalue  --lambda 10 -logdir shelf_long_env --logdir_suffix lambda_10 --pos_fraction 0.3 --seed $i
# done