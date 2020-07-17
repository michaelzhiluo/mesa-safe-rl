#!/bin/bash

# Recovery RL PLaNet Recovery
for i in {1..3}
do
	echo "PlaNet Recovery Run $i"
	python -m main --cuda --env-name shelf --use_recovery --use_qvalue --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.6 --eps_safe 0.25 --cnn --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --model_fname model_shelf_lowdata --beta 10 --vismpc_recovery --load_vismpc --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix recovery_0.6_0.25
done

# # Recovery RL DDPG Recovery
# for i in {1..3}
# do
# 	echo "DDPG Recovery Run $i"
# 	python -m main --cuda --env-name shelf --use_recovery --use_qvalue --ddpg_recovery --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.6 --eps_safe 0.25 --cnn  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --num_eps 4000 --seed $i --logdir image_shelf --logdir_suffix ddpg_recovery_0.6_0.25
# done

# # SAC Lagrangian Nu=1
# for i in {1..3}
# do
# 	echo "Lagrangian Nu 1 Run $i"
# 	python -m main --cuda --env-name shelf --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.6 --eps_safe 0.25 --cnn --DGD_constraints --nu 1  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --logdir image_shelf --logdir_suffix nu_1 --num_eps 4000 --seed $i
# done

# # SAC Lagrangian Nu=10
# for i in {1..3}
# do
# 	echo "Lagrangian Nu 10 Run $i"
# 	python -m main --cuda --env-name shelf --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.6 --eps_safe 0.25 --cnn --DGD_constraints --nu 10  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --logdir image_shelf --logdir_suffix nu_10 --num_eps 4000 --seed $i
# done

# # SAC Lagrangian Nu=100      
# for i in {1..3}
# do
# 	echo "Lagrangian Nu 100 Run $i"
# 	python -m main --cuda --env-name shelf --use_qvalue --critic_safe_update_freq 5 --gamma_safe 0.6 --eps_safe 0.25 --cnn --DGD_constraints --nu 100  --critic_safe_pretraining_steps 30000 --num_constraint_transitions 50000 --logdir image_shelf --logdir_suffix nu_100 --num_eps 4000 --seed $i
# done
