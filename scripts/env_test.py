import gymnasium as gym
import gym_pusht

from diffusion_policy.envs import PushTImageEnv

env = PushTImageEnv()
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()