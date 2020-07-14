# --- POINTBOT 0 ENV ---
# Recovery RL (Master)
python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.1 --use_value
python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.2 --use_qvalue

# Unconstrained (Master)
python main.py --env-name simplepointbot0 --cuda 

# Reward Penalty (Master)
python main.py --env-name simplepointbot0 --cuda --constraint_reward_penalty 1

# Lagrangian (saclagrangian-new)
python main.py --env-name simplepointbot0 --cuda --use_qvalue --DGD_constraints --nu {} --eps_safe 0.2

# RCPO (RCPO)
python main.py --env-name simplepointbot0 --cuda --eps_safe 0.1 --gamma_safe 0.8 --RCPO --lamda {} --use_value


# RCPO (fast_update)
python main.py --env-name simplepointbot0 --cuda --gamma_safe 0.5 --eps_safe 0.2 --use_qvalue --RCPO --lambda_RCPO 1000

# --- POINTBOT 1 ENV ---
# Recovery RL (Master)
python main.py --env-name simplepointbot1 --cuda --use_recovery --gamma_safe 0.9 --eps_safe 0.1 --use_value
python main.py --env-name simplepointbot1 --cuda --use_recovery --gamma_safe 0.75 --eps_safe 0.2 --use_qvalue

# Unconstrained (Master)
python main.py --env-name simplepointbot1 --cuda 

# Reward Penalty (Master)
python main.py --env-name simplepointbot1 --cuda --constraint_reward_penalty 10

# Lagrangian (saclagrangian-new)
python main.py --env-name simplepointbot1 --cuda --use_qvalue --DGD_constraints --nu {} --eps_safe 0.2

# RCPO (RCPO)
python main.py --env-name simplepointbot1 --cuda --eps_safe 0.1 --gamma_safe 0.9 --RCPO --lamda {} --use_value

# --- MAZE ENV --- 
# Recovery RL (Master)
python -m main --cuda --env-name maze --use_recovery --use_value --critic_safe_update_freq 5 --recovery_policy_update_freq 5 --gamma_safe 0.85 --eps_safe 0.05

# Unconstrained (Master)
python -m main --cuda --env-name maze

# Reward Penalty (Master)
python -m main --cuda --env-name maze --constraint_reward_penalty 50

# Lagrangian (fast_update)
python main.py --env-name maze --cuda --use_qvalue --DGD_constraints --update_nu --nu {} --gamma_safe 0.85 --eps_safe 0.05

# Lagrangian (saclagrangian-new)
python main.py --env-name maze --cuda --use_qvalue --DGD_constraints --nu {} --eps_safe 0.05

# RCPO (RCPO)
python -m main --cuda --env-name maze --eps_safe 0.05 --gamma_safe 0.85 --RCPO --lambda {} --use_value

# --- SHELF ENV ---
# Data Gen: 
# Task demos: python -m gen_shelf_demos --cuda --gt_state --num_demos 250 
# Constraint demos: python -m gen_shelf_demos --cuda --gt_state --num_demos 10000 --constraint_demos
# Task demos for RCPO: python -m gen_shelf_demos --cuda --gt_state --num_demos 250 --RCPO_demos

# Recovery RL (Master)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.4 --use_value

# Unconstrained (Master)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000

# Reward Penalty (Master)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --constraont_reward_penalty 3

# Lagrangian (saclagrangian-new)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --gamma_safe 0.85 --eps_safe 0.4 --critic_safe_update_freq 20 --use_qvalue --DGD_constraints --nu 1

# Lagrangian (fast_update)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --DGD_constraints --update_nu --nu 10 --gamma_safe 0.85 --eps_safe 0.4 --use_qvalue --num_task_transitions 500

# RCPO (RCPO)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --RCPO --lamda {} --eps_safe 0.4 --gamma_safe 0.85 --critic_safe_update_freq 20 --use_value


# --- DYNAMIC SHELF ENV ---
# Data Gen:
# Task demos: python -m gen_dynamic_shelf_demos --cuda --gt_state --num_demos 250 
# Constraint demos: python -m gen_dynamic_shelf_demos --cuda --gt_state --num_demos 10000 --constraint_demos
# Task demos for RCPO: python -m gen_dynamic_shelf_demos --cuda --gt_state --num_demos 250 --RCPO_demos

# Recovery RL (Master)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_value

# Unconstrained (Master)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000

# Reward Penalty (Master)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --constraint_reward_penalty 3

# Lagrangian (saclagrangian-new)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --critic_safe_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_qvalue --DGD_constraints --nu 1

# RCPO (RCPO)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --RCPO --lamda {} --critic_safe_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_value


# --- IMAGE MAZE ENV ---

# Recovery RL (vismpc-recovery)
python -m main --cuda --env-name image_maze --use_recovery --use_value --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.8 --eps_safe 0.05 --cnn --vismpc_recovery --num_constraint_transitions 10000 --model_fname model2_lowdata --beta 10 --kappa 10000 --load_vismpc

# Unconstrained (Master)
python -m main --cuda --env-name image_maze --cnn

# Reward Penalty (Master)
python -m main --cuda --env-name image_maze --cnn --constraint_reward_penalty 20

# Lagrangian (saclagrangian-new)
python -m main --cuda --env-name image_maze --cnn --use_qvalue --eps_safe 0.05 --gamma_safe 0.8 --critic_safe_update_freq 20 --DGD_constraints --nu {}

# RCPO (RCPO)
python -m main --cuda --env-name image_maze --cnn --RCPO --lamda {} --eps_safe 0.05 --gamma_safe 0.8 --critic_safe_update_freq 200000 --use_value

# --- IMAGE SHELF ENV ---
# Data Gen:
# Task demos: python -m gen_shelf_demos --cuda --num_demos 250 (vismpc-recovery)
# Constraint demos: python -m gen_shelf_demos --cuda --num_demos 10000 --constraint_demos --vismpc_train_data (vismpc-recovery)
# Task demos for RCPO: python -m gen_shelf_demos --cuda --num_demos 250 --RCPO_demos (RCPO)

# Recovery RL (vismpc-recovery) (model_shelf3 for high data)
python -m main --cuda --env-name shelf_env --use_recovery --use_value --critic_safe_update_freq 20000 --recovery_policy_update_freq 20000 --gamma_safe 0.85 --eps_safe 0.25 --cnn --vismpc_recovery --num_constraint_transitions 250000 --model_fname model_shelf_lowdata --beta 10 --kappa 10000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --load_vismpc

# Unconstrained (Master)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --cnn

# Reward Penalty (Master)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --cnn --constraint_reward_penalty 10

# Lagrangian (saclagrangian-new) --> Note for this need to change images to true in env

# RCPO (RCPO) --> Note for this need to change images to true in env

# --- IMAGE DYNAMIC SHELF ENV ---
# Data Gen:
# Task demos: python -m gen_dynamic_shelf_demos --cuda --num_demos 250 (vismpc-recovery)
# Constraint demos: python -m gen_dynamic_shelf_demos --cuda --num_demos 10000 --constraint_demos --vismpc_train_data (vismpc-recovery)
# Task demos for RCPO: python -m gen_dynamic_shelf_demos --cuda --num_demos 250 --RCPO_demos (RCPO)

# Recovery RL (vismpc-recovery)
python -m main --cuda --env-name shelf_dynamic_env --use_recovery --use_value --critic_safe_update_freq 20000 --recovery_policy_update_freq 20000 --gamma_safe 0.85 --eps_safe 0.1 --cnn --vismpc_recovery --num_constraint_transitions 250000 --model_fname model_shelf_dynamic --beta 10 --kappa 10000 --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --load_vismpc

# Unconstrained (Master)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --cnn

# Reward Penalty (Master)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --cnn --constraint_reward_penalty 3

# Lagrangian (saclagrangian-new) --> Note for this need to change images to true in env

# RCPO (RCPO) --> Note for this need to change images to true in env
