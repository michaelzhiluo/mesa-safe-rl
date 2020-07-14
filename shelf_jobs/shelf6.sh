#!/bin/bash

for i in {1..3}
do
	python -m main --cuda --env-name shelf_env --task_demos --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.5 --use_qvalue --num_task_transitions 500 --logdir shelf --logdir_suffix recovery_ddpg_alpha_2_eps_5 --num_eps 4000 --ddpg_recovery
done

for i in {1..3}
do
	python -m main --cuda --env-name shelf_env --task_demos --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --alpha 0.05 --eps_safe 0.5 --use_qvalue --num_task_transitions 500 --logdir shelf --logdir_suffix recovery_ddpg_alpha_05_eps_5 --num_eps 4000 --ddpg_recovery
done


