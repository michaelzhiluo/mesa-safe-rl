#!/bin/bash
for i in {1..3}
do
	echo "Recovery Run $i"
	python main.py --env-name simplepointbot1 --cuda --use_recovery --gamma_safe 0.75 --eps_safe 0.2 --use_qvalue --logdir pointbot1 --logdir_suffix recovery --num_steps 30000
done

for i in {1..3}
do
	echo "Lookahead Recovery Run $i"
	python main.py --env-name simplepointbot1 --cuda --use_recovery --gamma_safe 0.75 --eps_safe 0.2 --use_qvalue --lookahead_test --logdir pointbot1 --logdir_suffix lookahead --num_steps 30000
done


for i in {1..3}
do
	echo "SAC Run $i"
	python main.py --env-name simplepointbot1 --cuda --logdir pointbot1 --logdir_suffix vanilla --num_steps 30000
done


for i in {1..3}
do
	echo "Reward Penalty 1 Run $i"
	python main.py --env-name simplepointbot1 --cuda --constraint_reward_penalty 1 --logdir pointbot1 --logdir_suffix reward_1 --num_steps 30000

done


for i in {1..3}
do
	echo "Reward Penalty 10 Run $i"
	python main.py --env-name simplepointbot1 --cuda --constraint_reward_penalty 10 --logdir pointbot1 --logdir_suffix reward_10 --num_steps 30000
done


for i in {1..3}
do
	echo "Reward Penalty 100 Run $i"
	python main.py --env-name simplepointbot1 --cuda --constraint_reward_penalty 100 --logdir pointbot1 --logdir_suffix reward_100 --num_steps 30000
done


for i in {1..3}
do
	echo "Lagrangian Nu 1 Run $i"
	python main.py --env-name simplepointbot1 --cuda --use_qvalue --DGD_constraints --nu 1 --gamma_safe 0.75 --logdir pointbot1 --logdir_suffix nu_1 --num_steps 30000
done

for i in {1..3}
do
	echo "Lagrangian Nu 10 Run $i"
	python main.py --env-name simplepointbot1 --cuda --use_qvalue --DGD_constraints --nu 10 --gamma_safe 0.75 --logdir pointbot1 --logdir_suffix nu_10 --num_steps 30000
done

for i in {1..3}
do
	echo "Lagrangian Nu 100 Run $i"
	python main.py --env-name simplepointbot1 --cuda --use_qvalue --DGD_constraints --nu 100 --gamma_safe 0.75 --logdir pointbot1 --logdir_suffix nu_100 --num_steps 30000
done

