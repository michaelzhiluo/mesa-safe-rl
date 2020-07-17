#!/bin/bash

# DDPG Recovery
for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_env --task_demos --alpha 0.05 --num_task_transitions 500 --tau 0.0002 --replay_size 100000 --num_eps 3000 --use_qvalue --use_recovery --ddpg_recovery --gamma_safe 0.85 --eps_safe 0.25
done

# PETS Recovery
for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_env --task_demos --alpha 0.05 --num_task_transitions 500 --tau 0.0002 --replay_size 100000 --num_eps 3000 --use_qvalue --use_recovery --gamma_safe 0.85 --eps_safe 0.25
done

# Lagrangian
for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_env --task_demos --alpha 0.05 --num_task_transitions 500 --tau 0.0002 --replay_size 100000 --num_eps 3000 --use_qvalue --DGD_constraints --nu 10 --gamma_safe 0.85 --eps_safe 0.25
done

# RCPO
for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_env --task_demos --alpha 0.05 --num_task_transitions 500 --tau 0.0002 --replay_size 100000 --num_eps 3000 --use_qvalue --RCPO --lambda_RCPO 10 --gamma_safe 0.85 --eps_safe 0.25
done

# Vanilla
for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_env --task_demos --alpha 0.05 --num_task_transitions 500 --tau 0.0002 --replay_size 100000 --num_eps 3000
done

# Reward Penalty
for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_env --task_demos --alpha 0.05 --num_task_transitions 500 --tau 0.0002 --replay_size 100000 --num_eps 3000 --constraint_reward_penalty 10
done
