import gymnasium as gym
import gym_pusht
import numpy as np

from diffusion_policy.envs import PushTImageEnv, PushTStateEnv

env = PushTImageEnv()
# env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="human")
obs, info = env.reset()

print(f"obs: {obs}")
# img_obs = env.unwrapped._get_img(env.unwrapped._draw(), env.unwrapped.observation_width, env.unwrapped.observation_height)

# print(f"img_obs: {np.array(img_obs).shape}, max: {np.max(img_obs)}, min: {np.min(img_obs)}")

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"obs: {np.array(obs)}")
    if terminated or truncated:
        obs, info = env.reset()
    env.render()

env.render()
env.close()