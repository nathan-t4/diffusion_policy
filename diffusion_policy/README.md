# Diffusion Policy
- `configs/`: yaml configuration files for training (used in `train.py`)
- `models/`: torch neural network models for diffusion policy (1D UNet for diffusion backbone + ResNet loader)
- `dataset.py`: load and process pushT data from `data/` into torch datasets (normalization, sample sequences)
- `envs.py`: Environment wrapper for huggingface/gym-pusht Gymnasium environment + helper functions for initializing environments
- `train.py`: Trainer class to train and evaluate DP models on the PushT environment. Tested for training and evaluating environment with rgb image (96, 96, 3) + agent end (2) gripper pos. Includes wandb logging, continue training functionality, batch evaluations, and more.