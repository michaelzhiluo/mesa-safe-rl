# To run channel/tightrope env
python main.py --cuda --env-name SimplePointBot-v0

# To run obstacle env
python main.py --cuda --env-name SimplePointBot-v1 --gamma_safe 0.9 --num_steps 40000
