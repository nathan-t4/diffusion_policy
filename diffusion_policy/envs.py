import gymnasium as gym
import gym_pusht

from typing import Literal

def PushTImageEnv(render_mode: Literal["human", "rgb_array"] = "human"):
    return gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode=render_mode)

def PushTStateEnv(render_mode: Literal["human", "rgb_array"] = "human"):
    return gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=render_mode)