# To run channel/tightrope env
python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.1 --use_value
python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 1 --use_value --pred_time --t_safe 70

# To run obstacle env
python main.py --cuda --env-name SimplePointBot-v1 --gamma_safe 0.9 --num_steps 40000

# To run maze env
python -m main --cuda --env-name maze --use_recovery --use_value --critic_safe_update_freq 5 --recovery_policy_update_freq 5 --gamma_safe 0.85 --eps_safe 0.05