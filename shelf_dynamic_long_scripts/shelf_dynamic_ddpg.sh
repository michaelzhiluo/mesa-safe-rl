#!/bin/bash

# Recovery RL DDPG
for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_long_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --ddpg_recovery --num_eps 3000 --logdir shelf_dynamic_long_env --logdir_suffix recovery_ddpg_0.85_0.35 --pos_fraction 0.3
done

for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_long_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_qvalue --ddpg_recovery --num_eps 3000 --logdir shelf_dynamic_long_env --logdir_suffix recovery_ddpg_0.85_0.25 --pos_fraction 0.3
done

for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_long_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.45 --use_qvalue --ddpg_recovery --num_eps 3000 --logdir shelf_dynamic_long_env --logdir_suffix recovery_ddpg_0.85_0.45 --pos_fraction 0.3
done
