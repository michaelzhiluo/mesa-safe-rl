#!/bin/bash
# for i in {1..3}
# do
# 	echo "Recovery Run $i"
# 	python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.3 --use_qvalue --logdir pointbot0 --logdir_suffix recovery_ddpg --num_eps 300 --seed $i --ddpg_recovery
# done


# for i in {1..3}
# do
# 	echo "Recovery Run $i"
# 	python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.3 --use_qvalue --logdir pointbot0 --logdir_suffix recovery_pets --num_eps 300 --seed $i
# done


# for i in {1..3}
# do
# 	echo "Lookahead Recovery Run $i"
# 	python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.2 --use_qvalue --lookahead_test --logdir pointbot0 --logdir_suffix lookahead --num_eps 100 --seed $i
# done


# for i in {1..3}
# do
# 	echo "SAC Run $i"
# 	python main.py --env-name simplepointbot0 --cuda --logdir pointbot0 --logdir_suffix vanilla --num_eps 100 --seed $i
# done


# for i in {1..3}
# do
# 	echo "Reward Penalty 1 Run $i"
# 	python main.py --env-name simplepointbot0 --cuda --constraint_reward_penalty 1 --logdir pointbot0 --logdir_suffix reward_1 --num_eps 100 --seed $i

# done


# for i in {1..3}
# do
# 	echo "Reward Penalty 10 Run $i"
# 	python main.py --env-name simplepointbot0 --cuda --constraint_reward_penalty 10 --logdir pointbot0 --logdir_suffix reward_10 --num_eps 100 --seed $i
# done


# for i in {1..3}
# do
# 	echo "Reward Penalty 100 Run $i"
# 	python main.py --env-name simplepointbot0 --cuda --constraint_reward_penalty 100 --logdir pointbot0 --logdir_suffix reward_100 --num_eps 100 --seed $i
# done

# for i in {1..3}
# do
# 	echo "Reward Penalty 1000 Run $i"
# 	python main.py --env-name simplepointbot0 --cuda --constraint_reward_penalty 1000 --logdir pointbot0 --logdir_suffix reward_1000 --num_eps 300 --seed $i
# done


# for i in {1..3}
# do
# 	echo "Reward Penalty 3000 Run $i"
# 	python main.py --env-name simplepointbot0 --cuda --constraint_reward_penalty 3000 --logdir pointbot0 --logdir_suffix reward_3000 --num_eps 300 --seed $i
# done



for i in {1..3}
do
	echo "Lagrangian Nu 1000 Run $i"
	python main.py --env-name simplepointbot0 --cuda --use_qvalue --DGD_constraints --nu 1000 --gamma_safe 0.8 --eps_safe 0.3 --logdir pointbot0 --logdir_suffix nu_1000 --num_eps 300 --seed $i --update_nu
done

for i in {1..3}
do
	echo "Lagrangian Nu 5000 Run $i"
	python main.py --env-name simplepointbot0 --cuda --use_qvalue --DGD_constraints --nu 5000 --gamma_safe 0.8 --eps_safe 0.3 --logdir pointbot0 --logdir_suffix nu_5000 --num_eps 300 --seed $i --update_nu
done

# for i in {1..3}
# do
# 	echo "Lagrangian Nu 10000 Run $i"
# 	python main.py --env-name simplepointbot0 --cuda --use_qvalue --DGD_constraints --nu 10000 --gamma_safe 0.8 --logdir pointbot0 --logdir_suffix nu_10000 --num_eps 300 --seed $i --update_nu
# done

