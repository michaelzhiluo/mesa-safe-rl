# Ablate:: # Task Demos, # Constraint Demos (main paper), offline training, online training, add both transitions, action relabeling (main paper), 3x5 grid of gamma_safe, eps_safe for which env? (main paper)
# Do these for just MB-recovery?

# -------------------------:-------------------------:-------------------------MUST RUN FOR PAPER--------------------------------------------------:-------------------------

# Recovery RL PETS Recovery (no need to rerun)
# for i in {1..3}
# do
# 	echo "Recovery RL PETS Recovery Run $i"
# 	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_eps 4000 --logdir shelf_long_env --logdir_suffix recovery_0.85_0.35 --pos_fraction 0.3 --seed $i
# done

# Recovery RL PETS Recovery 1000 constraint demos
for i in {1..3}
do
	echo "Recovery RL PETS Recovery 1000 demos Run $i"
	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_eps 4000 --num_constraint_transitions 1000 --logdir shelf_long_env --logdir_suffix recovery_0.85_0.35_1kdemos --pos_fraction 0.3 --seed $i
done

# Recovery RL PETS Recovery 5000 constraint demos
for i in {1..3}
do
	echo "Recovery RL PETS Recovery 5000 demos Run $i"
	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_eps 4000 --num_constraint_transitions 5000 --logdir shelf_long_env --logdir_suffix recovery_0.85_0.35_5kdemos --pos_fraction 0.3 --seed $i
done

# Recovery RL PETS Recovery disable action relabeling
for i in {1..3}
do
	echo "Recovery RL PETS Recovery No Ac Relabeling Run $i"
	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_eps 4000 --disable_action_relabeling --logdir shelf_long_env --logdir_suffix recovery_0.85_0.35_disable_relabel --pos_fraction 0.3 --seed $i
done

# -------------------------:-------------------------:------------------------- END MUST RUN FOR PAPER--------------------------------------------------:-------------------------

# FOR supplementary material (in order of priority, most important first, this can be done later since only due a week after initial submission):

# Disable offline updates
for i in {1..3}
do
	echo "Recovery RL PETS Recovery Disable Offline Run $i"
	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_eps 4000 --disable_offline_updates --logdir shelf_long_env --logdir_suffix recovery_0.85_0.35_disable_offline --pos_fraction 0.3 --seed $i
done

# Disable online updates
for i in {1..3}
do
	echo "Recovery RL PETS Recovery Disable Online Run $i"
	python -m main --cuda --env-name shelf_long_env --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_eps 4000 --disable_online_updates --logdir shelf_long_env --logdir_suffix recovery_0.85_0.35_disable_online --pos_fraction 0.3 --seed $i
done

# GRID (TODO: create 3 x 5 grid of different gamma_safe, eps_safe and create a ROC curve)


