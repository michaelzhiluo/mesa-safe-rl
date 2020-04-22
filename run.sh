# To run channel/tightrope env
python main.py --env-name simplepointbot0 --cuda --use_recovery --gamma_safe 0.8 --eps_safe 0.1

# To run obstacle env
python main.py --cuda --env-name SimplePointBot-v1 --gamma_safe 0.9 --num_steps 40000
