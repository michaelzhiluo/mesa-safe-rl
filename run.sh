# --- POINTBOT 0 ENV ---
# Recovery RL
python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.1 --use_value
python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.2 --use_qvalue
# Unconstrained

# Constrained

# Lagrangian

# RCPO

# --- POINTBOT 1 ENV ---
# Recovery RL
python main.py --env-name simplepointbot1 --cuda --use_recovery --gamma_safe 0.9 --eps_safe 0.1 --use_value
python main.py --env-name simplepointbot1 --cuda --use_recovery --gamma_safe 0.75 --eps_safe 0.2 --use_qvalue
# Unconstrained

# Constrained

# Lagrangian

# RCPO

# --- MAZE ENV --- 
# Recovery RL
python -m main --cuda --env-name maze --use_recovery --use_value --critic_safe_update_freq 5 --recovery_policy_update_freq 5 --gamma_safe 0.85 --eps_safe 0.05
# Unconstrained

# Constrained

# Lagrangian

# RCPO

# --- SHELF ENV ---
# Recovery RL

# Unconstrained

# Constrained

# Lagrangian

# RCPO

# --- DYNAMIC SHELF ENV ---

# Recovery RL

# Unconstrained

# Constrained

# Lagrangian

# RCPO

# --- IMAGE MAZE ENV ---

# Recovery RL

# Unconstrained

# Constrained

# Lagrangian

# RCPO

# --- SHELF ENV ---

# Recovery RL

# Unconstrained

# Constrained

# Lagrangian

# RCPO
