import zarr
import numpy as np
import gymnasium as gym
import gym_pusht
from skvideo.io import vwrite

from diffusion_policy.envs import PushTImageEnv, PushTStateEnv

def main():
    data_version = "v2"
    dataset_path = f"data/pusht_cchi_{data_version}.zarr"
    dataset = zarr.open(dataset_path, mode="r")
    train_image_data = dataset['data']['img']
    train_keypoint_data = dataset['data']['keypoint']
    train_action_data = dataset['data']['action']
    train_state_data = dataset['data']['state']
    train_contact_data = dataset['data']['n_contacts']
    episode_ends = dataset['meta']['episode_ends'][:]

    # print("train_state_data:", train_state_data[:2]) # (T, 5) - [agent_x, agent_y, block_x, block_y, block_angle]
    # print("train_image_data:", train_image_data[:2]) # (T, 96, 96, 3)
    # print("train_action_data:", train_action_data[:2]) # (T, 2) - [target_x, target_y]
    print("episode_ends:", len(episode_ends))
    # print("train_keypoint_data:", train_keypoint_data[:2]) # (T, 9, 2) - [x, y]
    # print("train_contact_data:", train_contact_data) # (T, 1) - [n_contacts]
    # vwrite("train_image_data.mp4", train_image_data[:episode_ends[0]])

    # env = PushTImageEnv()
    # state = train_state_data[0] # The IC is not consistent with image! (same angle but wrong block_x, block_y)
    # obs, info = env.reset(options={"reset_to_state": state})
    # done = False
    # i = 0
    # rewards = []
    # imgs = []
    # while i < episode_ends[0]: 
    #     action = train_action_data[i]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated | truncated
    #     if done:
    #         obs, info = env.reset()
    #         break
    #     i += 1
    #     imgs.append(obs['image'])
    #     # imgs.append(env.unwrapped.get_img_obs())
    #     rewards.append(reward)
    #     env.render()
    
    # print("Done", i)
    # print("Max reward:", max(rewards))
    # vwrite("env_simulated_data.mp4", np.array(imgs).transpose(0, 2, 3, 1) * 255)
    # env.close()
    
if __name__ == "__main__":
    main()