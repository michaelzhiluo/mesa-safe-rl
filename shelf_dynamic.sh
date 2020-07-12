#!/bin/bash
for i in {1..3}
do
	python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_qvalue --ddpg_recovery --num_eps 3000 --logdir shelf_dynamic --logdir_suffix recovery_ddpg
done


# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.15 --use_qvalue --ddpg_recovery --num_eps 3000 --logdir shelf_dynamic --logdir_suffix eps15
# done
