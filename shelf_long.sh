#!/bin/bash

# Unconstrained
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 4000 --logdir shelf_long_env --logdir_suffix vanilla --pos_fraction 0.3 --seed $i
# done

# Recovery RL DDPG
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_qvalue --ddpg_recovery --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_ddpg_0.85_0.25 --pos_fraction 0.3 --seed $i
# done

# Recovery RL DDPG
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_qvalue --ddpg_recovery --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_ddpg_0.85_0.35 --pos_fraction 0.3 --seed $i
# done

# Recovery RL DDPG
# for i in {1..3}
# do
# 	python -m main --cuda --env-name shelf_long_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --ddpg_recovery --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_ddpg_0.75_0.25 --pos_fraction 0.3 --seed $i
# done

