# --- POINTBOT 0 ENV ---
# Recovery RL (Master)
python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.1 --use_value
python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.2 --use_qvalue

# Unconstrained (Master)

# Reward Penalty (Master)

# Lagrangian (saclagrangian-new)

# RCPO (RCPO)

# --- POINTBOT 1 ENV ---
# Recovery RL (Master)
python main.py --env-name simplepointbot1 --cuda --use_recovery --gamma_safe 0.9 --eps_safe 0.1 --use_value
python main.py --env-name simplepointbot1 --cuda --use_recovery --gamma_safe 0.75 --eps_safe 0.2 --use_qvalue

# Unconstrained (Master)

# Reward Penalty (Master)

# Lagrangian (saclagrangian-new)

# RCPO (RCPO)

# --- MAZE ENV --- 
# Recovery RL (Master)
python -m main --cuda --env-name maze --use_recovery --use_value --critic_safe_update_freq 5 --recovery_policy_update_freq 5 --gamma_safe 0.85 --eps_safe 0.05

# Unconstrained (Master)
python -m main --cuda --env-name maze

# Reward Penalty (Master)
python -m main --cuda --env-name maze --constraont_reward_penalty 50

# Lagrangian (saclagrangian-new)

# RCPO (RCPO)
python -m main --cuda --env-name maze --eps_safe 0.05 --RCPO

# --- SHELF ENV ---
# Recovery RL (Master)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.4 --use_value

# Unconstrained (Master)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000

# Reward Penalty (Master)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --constraont_reward_penalty 3

# Lagrangian (saclagrangian-new)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --gamma_safe 0.85 --eps_safe 0.4 --critic_safe_update_freq 20 --use_qvalue --DGD_constraints --nu 1

# RCPO (RCPO)
python -m main --cuda --env-name shelf_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --RCPO --lamda {}

# --- DYNAMIC SHELF ENV ---

# Recovery RL (Master)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --use_recovery --critic_safe_update_freq 20 --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_value

# Unconstrained (Master)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000

# Reward Penalty (Master)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --constraont_reward_penalty 3

# Lagrangian (saclagrangian-new)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --critic_safe_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_qvalue --DGD_constraints --nu 1

# RCPO (RCPO)
python -m main --cuda --env-name shelf_dynamic_env --task_demos --alpha 0.05 --tau 0.0002 --replay_size 100000 --RCPO --lambda {}

# --- IMAGE MAZE ENV ---

# Recovery RL (vismpc-recovery)
python -m main --cuda --env-name image_maze --use_recovery --use_value --critic_safe_update_freq 200000 --recovery_policy_update_freq 200000 --gamma_safe 0.8 --eps_safe 0.05 --cnn --vismpc_recovery --num_constraint_transitions 10000 --model_fname model2_lowdata --beta 10 --kappa 10000 --load_vismpc

# Unconstrained (Master)
python -m main --cuda --env-name image_maze --cnn

# Reward Penalty (Master)
python -m main --cuda --env-name image_maze --cnn --constraint_reward_penalty {}

# Lagrangian (saclagrangian-new)
python -m main --cuda --env-name image_maze --cnn --use_qvalue --eps_safe 0.05 --gamma_safe 0.8 --critic_safe_update_freq {} --DGD_constraints --nu {}

# RCPO (RCPO)
python -m main --cuda --env-name image_maze --cnn --RCPO --lambda {}

# --- IMAGE SHELF ENV ---

# Recovery RL

# Unconstrained

# Reward Penalty

# Lagrangian

# RCPO
