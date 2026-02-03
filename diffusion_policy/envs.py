import gymnasium as gym
import gym_pusht
import numpy as np

from typing import Literal

class PushTEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = {}
        raw_obs, info = self.env.reset(**kwargs)
        obs['image'] = self.get_img_obs()
        obs['agent_pos'] = raw_obs['agent_pos'][:2]
        return obs, info

    def step(self, action):
        obs = {}
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        obs['image'] = self.get_img_obs()
        obs['agent_pos'] = raw_obs['agent_pos'][:2]
        return obs, reward, terminated, truncated, info
    
    def get_img_obs(self):
        img = self.env.unwrapped._get_img(
            self.env.unwrapped._draw(), 
            self.env.unwrapped.observation_width, 
            self.env.unwrapped.observation_height
        )
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)

        return img_obs

def PushTImageEnv(render_mode: Literal["human", "rgb_array"] = "human"):
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode=render_mode)
    return PushTEnvWrapper(env)

def PushTStateEnv(render_mode: Literal["human", "rgb_array"] = "human"):
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=render_mode)
    return env