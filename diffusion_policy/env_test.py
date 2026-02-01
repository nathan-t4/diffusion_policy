import gymnasium as gym
import gym_pusht

# render modes "human" 

env = gym.make("gym_pusht/PushT-v0", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()