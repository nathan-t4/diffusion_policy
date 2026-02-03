# Data Shapes

v1 has T=12468 data points
v2 and v7 have T=25650 data points 

Image data shape (T, 96, 96, 3)
Action data shape (T, 2)
State data shape (T, 5)
T represents time indices, corresponding to len(episode_ends) trajectories

# Environment

Can either train using state + action or iamge + action data




# TODO: 
- look at colab notebooks to start training
- how to evaluate (95% coverage = success in environment)
- final evaluation should use multiple seeds (500? https://huggingface.co/lerobot/diffusion_pusht)
- write continue_training code
- do data visualization to see difference between v1 and v2 (set initial condition and actions on pushT environment)
- comprehensive evaluation (10+ initial conditions)
- run v1 training and compare btw v1 and v2

State is 9 waypoints on the T, Image uses RGB (96x96)
