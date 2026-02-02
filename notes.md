# Data Shapes

v1 has T=12468 data points
v2 and v7 have T=25650 data points 

Image data shape (T, 96, 96, 3)
Action data shape (T, 2)
State data shape (T, 5)
T represents time indices, corresponding to len(episode_ends) trajectories

# Environment

Can either train using state + action or iamge + action data