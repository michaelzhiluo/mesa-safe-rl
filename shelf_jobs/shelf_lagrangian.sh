#!/bin/bash

for i in {1..3}
do
	python -m main --cuda --env-name shelf_env --task_demos --tau 0.0002 --replay_size 100000 --DGD_constraints --update_nu --nu 10 --gamma_safe 0.85 --eps_safe 0.4 --use_qvalue --num_task_transitions 500 --alpha 0.05 --logdir shelf --logdir_suffix nu_10_update_alpha_05 --num_eps 4000
done

for i in {1..3}
do
	python -m main --cuda --env-name shelf_env --task_demos --tau 0.0002 --replay_size 100000 --DGD_constraints --update_nu --nu 10 --gamma_safe 0.85 --eps_safe 0.4 --use_qvalue --num_task_transitions 500 --logdir shelf --logdir_suffix nu_10_update_alpha_2 --num_eps 4000
done
