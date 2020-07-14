#!/bin/bash
# Recovery RL DDPG Recovery
for i in {1..3}
do
	echo "DDPG Recovery Run $i"
	python -m main --cuda --env-name image_maze --use_recovery --use_qvalue --ddpg_recovery --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.65 --eps_safe 0.1 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 20000 --num_eps 500 --seed $i --logdir image_maze --logdir_suffix ddpg_recovery
done

# SAC Lagrangian Nu=1
for i in {1..3}
do
	echo "Lagrangian Nu 1 Run $i"
	python -m main --cuda --env-name image_maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.65 --eps_safe 0.1 --cnn --DGD_constraints --nu 1 --logdir image_maze --logdir_suffix nu_1 --num_eps 500 --seed $i
done

# SAC Lagrangian Nu=10
for i in {1..3}
do
	echo "Lagrangian Nu 10 Run $i"
	python -m main --cuda --env-name image_maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.65 --eps_safe 0.1 --cnn --DGD_constraints --nu 10 --logdir image_maze --logdir_suffix nu_10 --num_eps 500 --seed $i
done

# SAC Lagrangian Nu=100
for i in {1..3}
do
	echo "Lagrangian Nu 100 Run $i"
	python -m main --cuda --env-name image_maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.65 --eps_safe 0.1 --cnn --DGD_constraints --nu 100 --logdir image_maze --logdir_suffix nu_100 --num_eps 500 --seed $i
done

# RCPO Lambda=5
for i in {1..3}
do
	echo "RCPO Lambda 5 Run $i"
	python -m main --cuda --env-name image_maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.65 --eps_safe 0.1 --cnn --RCPO --lambda 5 --logdir image_maze --logdir_suffix lambda_5 --num_eps 500 --seed $i
done

# RCPO Lambda=20
for i in {1..3}
do
	echo "RCPO Lambda 10 Run $i"
	python -m main --cuda --env-name image_maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.65 --eps_safe 0.1 --cnn --RCPO --lambda 20 --logdir image_maze --logdir_suffix lambda_20 --num_eps 500 --seed $i
done

# RCPO Lambda=100
for i in {1..3}
do
	echo "RCPO Lambda 100 Run $i"
	python -m main --cuda --env-name image_maze --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.65 --eps_safe 0.1 --cnn --RCPO --lambda 100 --logdir image_maze --logdir_suffix lambda_100 --num_eps 500 --seed $i
done

# Note: Reward Penalty and Model Based Runs are already done... can include for now, but need to calibrate model based run later... (right now it uses value function)