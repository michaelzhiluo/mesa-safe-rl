### Description
------------
Implementation of MESA: Offline Meta-RL for Safe Adaptation and Fault Tolerance. The repo builds on top of Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones. The main file is main.py in the root directory. MESA's meta-learning takes place in run_multitask.py. The core SAC implementation can be found in sac.py and is built on the implementation from https://github.com/pranz24/pytorch-soft-actor-critic.
This file also implements constraint critic training for Recovery RL.

### Reproducing Experiments
------------

To reproduce experiments (1) download data from \todo{insert} link and place it in a folder called data/ in the root directory. The commands to run the experiments are in 