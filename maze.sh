#!/bin/bash
for i in {1..3}
do
	echo "Recovery Run $i"
	python -m main --cuda --env-name maze --use_recovery --use_value --critic_safe_update_freq 5 --recovery_policy_update_freq 5 --gamma_safe 0.85 --eps_safe 0.05  --logdir maze --logdir_suffix recovery --num_eps 2000
done

for i in {1..3}
do
	echo "Lookahead Recovery Run $i"
	python -m main --cuda --env-name maze --use_recovery --lookahead_test --use_value --critic_safe_update_freq 5 --recovery_policy_update_freq 5 --gamma_safe 0.85 --eps_safe 0.05  --logdir maze --logdir_suffix lookahead --num_eps 2000
done


for i in {1..3}
do
	echo "SAC Run $i"
	python main.py --env-name maze --cuda --logdir maze --logdir_suffix vanilla --num_eps 2000
done


for i in {1..3}
do
	echo "Reward Penalty 1 Run $i"
	python main.py --env-name maze --cuda --constraint_reward_penalty 1 --logdir maze --logdir_suffix reward_1 --num_eps 2000

done


for i in {1..3}
do
	echo "Reward Penalty 10 Run $i"
	python main.py --env-name maze --cuda --constraint_reward_penalty 10 --logdir maze --logdir_suffix reward_10 --num_eps 2000
done


for i in {1..3}
do
	echo "Reward Penalty 100 Run $i"
	python main.py --env-name maze --cuda --constraint_reward_penalty 100 --logdir maze --logdir_suffix reward_100 --num_eps 2000
done


# for i in {1..3}
# do
# 	echo "Lagrangian Nu 1 Run $i"
# 	python main.py --env-name maze --cuda --use_qvalue --DGD_constraints --nu 1 --gamma_safe 0.8 --logdir maze --logdir_suffix nu_1 --num_eps 2000
# done

# for i in {1..3}
# do
# 	echo "Lagrangian Nu 10 Run $i"
# 	python main.py --env-name maze --cuda --use_qvalue --DGD_constraints --nu 10 --gamma_safe 0.8 --logdir maze --logdir_suffix nu_10 --num_eps 2000
# done

# for i in {1..3}
# do
# 	echo "Lagrangian Nu 100 Run $i"
# 	python main.py --env-name maze --cuda --use_qvalue --DGD_constraints --nu 100 --gamma_safe 0.8 --logdir maze --logdir_suffix nu_100 --num_eps 2000
# done
