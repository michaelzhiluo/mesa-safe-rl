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
python main.py --env-name simplepointbot0 --cuda --eps_safe 0.1 --RCPO --lamda 1

# Safety Critic Penalty (Master)
python main.py --env-name simplepointbot0 --cuda --safety_critic_penalty 1

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
python main.py --env-name simplepointbot1 --cuda --eps_safe 0.1 --RCPO --lamda 10

# Safety Critic Penalty (Master)
python main.py --env-name simplepointbot1 --cuda --safety_critic_penalty {}

# --- MAZE ENV --- 
# Recovery RL (Master)
python -m main --cuda --env-name maze --use_recovery --use_value --critic_safe_update_freq 5 --recovery_policy_update_freq 5 --gamma_safe 0.85 --eps_safe 0.05

# Unconstrained (Master)
python -m main --cuda --env-name maze

# Reward Penalty (Master)
python -m main --cuda --env-name maze --constraint_reward_penalty 50

# Lagrangian (saclagrangian-new)
python main.py --env-name maze --cuda --use_qvalue --DGD_constraints --nu {} --eps_safe 0.05

# RCPO (RCPO)
python -m main --cuda --env-name maze --eps_safe 0.05 --RCPO

# Safety Critic Penalty (Master)
python main.py --env-name maze --cuda --safety_critic_penalty {}

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

# RCPO (RCPO)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --RCPO --lamda 3

# Safety Critic Penalty (Master)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --safety_critic_penalty {}

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
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --constraont_reward_penalty 3

# Lagrangian (saclagrangian-new)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --critic_safe_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_qvalue --DGD_constraints --nu 1

# RCPO (RCPO)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --RCPO --lamda 3

# Safety Critic Penalty (Master)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --safety_critic_penalty {}

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
python -m main --cuda --env-name image_maze --cnn --RCPO --lamda 20

# Safety Critic Penalty (vismpc-recovery)
python -m main --cuda --env-name image_maze --cnn --safety_critic_penalty 20 --model_fname model2_lowdata --beta 10 --kappa 10000 --load_vismpc  --use_value

# --- IMAGE SHELF ENV ---
# Data Gen:
# Task demos: python -m gen_shelf_demos --cuda --num_demos 250 
# Constraint demos: python -m gen_shelf_demos --cuda --num_demos 10000 --constraint_demos
# Task demos for RCPO: python -m gen_shelf_demos --cuda --num_demos 250 --RCPO_demos

# Recovery RL
python -m main --cuda --env-name shelf_env --use_recovery --use_value --critic_safe_update_freq 20000 --recovery_policy_update_freq 20000 --gamma_safe 0.85 --eps_safe 0.4 --cnn --vismpc_recovery --num_constraint_transitions 250000 --model_fname model_shelf3 --beta 10 --kappa 10000 --load_vismpc

# Unconstrained

# Reward Penalty

# Lagrangian

# RCPO

# --- IMAGE DYNAMIC SHELF ENV ---
# Data Gen:
# Task demos: python -m gen_dynamic_shelf_demos --cuda --num_demos 250 
# Constraint demos: python -m gen_dynamic_shelf_demos --cuda --num_demos 10000 --constraint_demos
# Task demos for RCPO: python -m gen_dynamic_shelf_demos --cuda --num_demos 250 --RCPO_demos

# Recovery RL

# Unconstrained

# Reward Penalty

# Lagrangian

# RCPO
