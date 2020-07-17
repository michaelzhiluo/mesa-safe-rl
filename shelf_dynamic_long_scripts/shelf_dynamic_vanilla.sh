#!/bin/bash

for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_long_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 3000 --logdir shelf_dynamic_long_env --logdir_suffix vanilla
done

for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_long_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 3000 --logdir shelf_dynamic_long_env --logdir_suffix penalty_10 --constraint_reward_penalty 10
done

for i in {1..3}
do
	python -m main --cuda --seed $i --env-name shelf_dynamic_long_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --num_eps 3000 --logdir shelf_dynamic_long_env --logdir_suffix penalty_5 --constraint_reward_penalty 5
done
