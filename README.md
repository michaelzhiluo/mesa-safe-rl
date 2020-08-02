### Description
------------
Implementation of Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones. The main file is main.py in the root
directory. The core sac implementation can be found in sac.py and is built on the implementation from https://github.com/pranz24/pytorch-soft-actor-critic.
This file also implements constraint critic training for Recovery RL. All environments can be found in the env folder. For the model-based recovery
policy, for low dimensional experiments we build on the PETS implementation provided in https://github.com/quanvuong/handful-of-trials-pytorch while
for image-based experiments we build on the latent dynamics model from Goal-Aware Prediction: Learning to Model What Matters (ICML 2020). The relevant
code can be found in MPC.py and VisualRecovery.py respectively and the relevant configs can be found in the config folder. The file model.py contains
the core architectures for the SAC implementations and for the latent dynamics model. Recovery RL and all comparisons share the same SAC and safety
critic implementation and differ only in how the safety critic is utilized. All environments are implemented in the env folder. The object extraction
environments are built on top of the cartrgipper environment from https://github.com/SudeepDasari/visual_foresight.

The files most directly related to the scientific contribution of the paper are main.py, which implements the main logic for Recovery RL and all comparisons,
and sac.py, which implements SAC and safety critic training. MPC.py and VisualRecovery.py are relevant for model-based recovery. 

### Reproducing Experiments
------------

To reproduce experiments (1) download data from \todo{insert} link and place it in a folder called data in the root directory. Then, run the following
bash scripts to reproduce the results from each of the reported experiments:

# Navigation 1
. navigation1.sh

# Navigation 2
. navigation2.sh

# Maze
. maze.sh

# Object Extraction
. object_extraction.sh

# Object Extraction (Dynamic Obstacle)
. object_extraction_dynamic_obs.sh

# Ablations
. ablations.sh