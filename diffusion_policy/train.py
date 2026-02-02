#@markdown ### **Imports**
# diffusion policy import
import pathlib
import collections
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from datetime import datetime
import wandb
from skvideo.io import vwrite
import re
import glob
import os
import yaml

from diffusion_policy.envs import PushTImageEnv
from diffusion_policy.dataset import PushTImageDataset, PushTStateDataset, normalize_data, unnormalize_data
from diffusion_policy.models.config import init_ema_model, load_pretrained_model

class Trainer:
    def __init__(self, cfg: OmegaConf, existing_model_path: str | None = None):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.continue_training = existing_model_path is not None

        if self.continue_training:
            existing_path = pathlib.Path(existing_model_path)
            if existing_path.suffix == '.pt':
                self.output_dir = existing_path.parent.parent
            else:
                self.output_dir = existing_path
        else:
            self.output_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/"

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.data_version = self.cfg.task.dataset.zarr_path.split('/')[-1].split('.')[0]

        # training parameters
        self.num_epochs = 100
        self.eval_every_epochs = 25
        self.eval_n_episodes = cfg.task.env_runner.n_envs
        self.max_eval_steps = cfg.task.env_runner.max_steps
        # state-space parameters
        self.pred_horizon = cfg.policy.horizon
        self.obs_horizon = cfg.policy.n_obs_steps
        self.action_horizon = cfg.policy.n_action_steps
        self.action_dim = cfg.policy.shape_meta.action.shape[0]
        self.num_inference_steps = cfg.policy.num_inference_steps


        self.load_dataset()
        self.create_noise_scheduler()
        self.create_nets()
        self.create_optimizer()
        self.create_eval_env()
        
        # Initialize log file
        self.log_file_path = pathlib.Path(self.output_dir) / "training_log.txt"
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Append if continuing training, otherwise write new
        if self.continue_training and self.log_file_path.exists():
            # Don't overwrite, just append
            pass
        else:
            with open(self.log_file_path, 'w') as f:
                f.write("Epoch\tTrain_Loss\tValidation_Reward\n")
        
        # Initialize wandb
        wandb_kwargs = {
            "project": "diffusion-policy",
            "name": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "config": {
                "num_epochs": self.num_epochs,
                "pred_horizon": self.pred_horizon,
                "obs_horizon": self.obs_horizon,
                "action_horizon": self.action_horizon,
                "action_dim": self.action_dim,
                "num_inference_steps": self.num_inference_steps,
                "data_version": self.data_version,
                "output_dir": str(self.output_dir),
            }
        }
        
        # Resume wandb run if continuing training
        if self.continue_training:
            run_id = self.find_wandb_run_id()
            if run_id:
                wandb_kwargs["id"] = run_id
                wandb_kwargs["resume"] = "must"
                # Use the original name from the existing run
                wandb_kwargs["name"] = None  # Let wandb use the existing name
                print(f"Resuming wandb run with ID: {run_id}")
            else:
                print("Warning: Could not find wandb run ID. Creating a new wandb run.")
        
        wandb.init(**wandb_kwargs)

    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir) / "checkpoints" / f"checkpoint_{tag}.pt"
    
    def find_latest_checkpoint_epoch(self):
        """Find the latest checkpoint epoch number from training log."""
        log_file = pathlib.Path(self.output_dir) / "training_log.txt"
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Skip header
                    # Get the last epoch from the log
                    last_line = lines[-1].strip()
                    if last_line:
                        parts = last_line.split('\t')
                        if parts[0].isdigit():
                            return int(parts[0])
        
        return 0
    
    def find_wandb_run_id(self):
        wandb_dir = pathlib.Path("wandb")
        if not wandb_dir.exists():
            return None
        
        output_dir_str = str(self.output_dir)
        output_dir_normalized = str(pathlib.Path(output_dir_str)).rstrip('/')
        
        # Search through all wandb run directories
        # Format: run-YYYYMMDD_HHMMSS-<run_id>
        for run_dir in wandb_dir.glob("run-*-*"):
            if not run_dir.is_dir():
                continue
            
            # Read the config file to check output_dir
            config_file = run_dir / "files" / "config.yaml"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Check if output_dir matches
                    if 'output_dir' in config and 'value' in config['output_dir']:
                        config_output_dir = str(pathlib.Path(config['output_dir']['value'])).rstrip('/')
                        if config_output_dir == output_dir_normalized:
                            # Extract run ID from directory name
                            # Format: run-YYYYMMDD_HHMMSS-<run_id>
                            match = re.search(r'run-\d{8}_\d{6}-(.+)$', run_dir.name)
                            if match:
                                return match.group(1)
                except Exception as e:
                    # Skip if we can't read the config file
                    continue
        
        return None
    
    def save_checkpoint(self, tag='latest'):
        checkpoint_path = self.get_checkpoint_path(tag)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.nets.state_dict(), checkpoint_path)
    
    def log(self, epoch, train_loss=None, validation_reward=None):
        # Log to text file
        with open(self.log_file_path, 'a') as f:
            if validation_reward is not None:
                mean_reward = validation_reward.get("mean_reward", 0.0)
                train_loss_str = f"{train_loss:.6f}" if train_loss is not None else "-"
                f.write(f"{epoch}\t{train_loss_str}\t{mean_reward:.6f}\n")
            elif train_loss is not None:
                f.write(f"{epoch}\t{train_loss:.6f}\t-\n")
            else:
                f.write(f"{epoch}\t-\t-\n")
        
        # Log to wandb
        log_dict = {"epoch": epoch}
        if train_loss is not None:
            log_dict["train_loss"] = train_loss
        if validation_reward is not None:
            # Handle both dict (from batch_evaluate) and single value
            if isinstance(validation_reward, dict):
                log_dict["validation_reward"] = validation_reward.get("mean_reward", 0.0)
                log_dict["task_success_rate"] = validation_reward.get("task_success_rate", 0.0)
        wandb.log(log_dict)

    def load_dataset(self):
        dataset_path = self.cfg.task.dataset.zarr_path

        self.dataset = PushTImageDataset(
            dataset_path=dataset_path,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon
        )
        # save training data statistics (min, max) for each dim
        self.stats = self.dataset.stats

        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.cfg.dataloader.batch_size,
            num_workers=self.cfg.dataloader.num_workers,
            shuffle=self.cfg.dataloader.shuffle,
            # accelerate cpu-gpu transfer
            pin_memory=self.cfg.dataloader.pin_memory,
            # don't kill worker process afte each epoch
            persistent_workers=self.cfg.dataloader.persistent_workers
        )
    
    def create_noise_scheduler(self):
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.cfg.policy.noise_scheduler.num_train_timesteps,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule=self.cfg.policy.noise_scheduler.beta_schedule,
            # clip output to [-1,1] to improve stability
            clip_sample=self.cfg.policy.noise_scheduler.clip_sample,
            # our network predicts noise (instead of denoised action)
            prediction_type=self.cfg.policy.noise_scheduler.prediction_type
        )
        
    
    def create_nets(self):
        self.nets = init_ema_model(self.cfg)

        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        self.ema = EMAModel(
            parameters=self.nets.parameters(),
            power=self.cfg.ema.power,
            inv_gamma=self.cfg.ema.inv_gamma,
            max_value=self.cfg.ema.max_value,
            min_value=self.cfg.ema.min_value,
            update_after_step=self.cfg.ema.update_after_step
        )

    
    def create_optimizer(self):
        # Standard ADAM optimizer
        # Note that EMA parametesr are not optimized
        self.optimizer = torch.optim.AdamW(
            params=self.nets.parameters(),
            lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.weight_decay)

        # Cosine LR schedule with linear warmup
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.dataloader) * self.num_epochs
        )
    
    def create_eval_env(self):
        self.eval_env = PushTImageEnv()

    def run_training(self):
        # Determine starting epoch
        start_epoch = 0
        if self.continue_training:
            # Load from checkpoint_final.pt
            checkpoint_path = self.get_checkpoint_path(tag='final')
            if checkpoint_path.exists():
                print(f"Loading checkpoint from: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path)
                self.nets.load_state_dict(state_dict)
            
            # Find the latest epoch we trained to from training log
            start_epoch = self.find_latest_checkpoint_epoch()
            print(f"Resuming training from epoch {start_epoch + 1}")

        # if start_epoch == 0:
        #     self.batch_evaluate(self.nets, tag='initial') 

        self.nets.train()
        mean_epoch_loss = None

        with tqdm(range(start_epoch, self.num_epochs), desc='Epoch', initial=start_epoch, total=self.num_epochs) as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(self.dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # device transfer
                        nimage = nbatch['image'][:,:self.obs_horizon].to(self.device)
                        nagent_pos = nbatch['agent_pos'][:,:self.obs_horizon].to(self.device)
                        naction = nbatch['action'].to(self.device)
                        B = nagent_pos.shape[0]

                        # encoder vision features
                        image_features = self.nets['vision_encoder'](
                            nimage.flatten(end_dim=1))
                        image_features = image_features.reshape(
                            *nimage.shape[:2],-1)
                        # (B,obs_horizon,D)

                        # concatenate vision feature and low-dim obs
                        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                        obs_cond = obs_features.flatten(start_dim=1)
                        # (B, obs_horizon * obs_dim)

                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=self.device)

                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (B,), device=self.device
                        ).long()

                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = self.noise_scheduler.add_noise(
                            naction, noise, timesteps)

                        # predict the noise residual
                        noise_pred = self.nets['noise_pred_net'](
                            noisy_actions, timesteps, global_cond=obs_cond)

                        # L2 loss
                        loss = nn.functional.mse_loss(noise_pred, noise)

                        # optimize
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        self.lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        self.ema.step(self.nets.parameters())

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                
                mean_epoch_loss = np.mean(epoch_loss)
                tglobal.set_postfix(loss=mean_epoch_loss)
                
                # Save checkpoint and validate every self.eval_every_epochs
                if (epoch_idx + 1) % self.eval_every_epochs == 0:
                    reward = self.batch_evaluate(self.nets, tag=f'epoch_{epoch_idx + 1}')
                    self.save_checkpoint(tag=f'epoch_{epoch_idx + 1}')
                    self.log(epoch_idx + 1, train_loss=mean_epoch_loss, validation_reward=reward)
                else:
                    self.log(epoch_idx + 1, train_loss=mean_epoch_loss)

        # Save checkpoint at the end of training
        self.save_checkpoint(tag='final')
        
        # Final evaluation and logging
        final_reward = self.batch_evaluate(self.nets, tag='final')
        # Only log train_loss if we actually trained (mean_epoch_loss was set)
        if mean_epoch_loss is not None:
            self.log(self.num_epochs, train_loss=mean_epoch_loss, validation_reward=final_reward)
        else:
            self.log(self.num_epochs, validation_reward=final_reward)
        
        wandb.finish()

        return self.ema

    def batch_evaluate(self, ema_nets: nn.Module | None = None, checkpoint_path: str | None = None, tag: str | None = None, seed: int | None = None):
        rewards = []
        successes = 0
        eval_seed = self.cfg.task.env_runner.test_start_seed
        for i in range(self.eval_n_episodes):
            result = self.evaluate(ema_nets, checkpoint_path, tag, seed=eval_seed + i, save = i == self.eval_n_episodes - 1)
            # evaluate returns a dict with 'task_success' and 'reward'
            success = result.get('task_success')
            reward = result.get('reward')
            successes += int(success)
            rewards.append(reward)
        
        rewards = np.array(rewards)
        
        return {
            "task_success_rate": successes / self.eval_n_episodes,
            "mean_reward": np.mean(rewards),
            "raw_rewards": rewards,
        }
    
    def evaluate(self, ema_nets: nn.Module | None = None, checkpoint_path: str | None = None, tag: str | None = None, seed: int | None = None, save: bool = False):
        # eval parameters
        # render = False
        # max_steps = 10

        if ema_nets is None:
            if checkpoint_path is None:
                checkpoint_path = self.get_checkpoint_path()
            ema_nets = load_pretrained_model(self.cfg, checkpoint_path)
        
        ema_nets.eval()
        
        if seed is None:
            seed = self.cfg.task.env_runner.test_start_seed

        obs, info = self.eval_env.reset(seed=seed)

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * self.obs_horizon, maxlen=self.obs_horizon)

        self.eval_env.render()

        # save visualization and rewards
        imgs = [self.eval_env.get_img_obs()]
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=self.max_eval_steps, desc="Eval PushTImageEnv", leave=False) as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon number of observations
                images = np.stack([x['image'] for x in obs_deque])
                agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

                # normalize observation
                nagent_poses = normalize_data(agent_poses, stats=self.stats['agent_pos'])
                # images are already normalized to [0,1]
                nimages = images

                # device transfer
                nimages = torch.from_numpy(nimages).to(self.device, dtype=torch.float32)
                # (2,3,96,96)
                nagent_poses = torch.from_numpy(nagent_poses).to(self.device, dtype=torch.float32)
                # (2,2)

                # infer action
                with torch.no_grad():
                    # get image features
                    image_features = ema_nets['vision_encoder'](nimages)
                    # (2,512)

                    # concat with low-dim observations
                    obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, self.pred_horizon, self.action_dim), device=self.device)
                    naction = noisy_action

                    # init scheduler
                    self.noise_scheduler.set_timesteps(self.num_inference_steps)

                    for k in self.noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = ema_nets['noise_pred_net'](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=self.stats['action'])

                # only take action_horizon number of actions
                start = self.obs_horizon - 1
                end = start + self.action_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, terminated, truncated, info = self.eval_env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    imgs.append(self.eval_env.get_img_obs())

                    self.eval_env.render()

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if terminated or truncated or step_idx > self.max_eval_steps:
                        done = True
                    if done:
                        break
        
        if save:
            movie_filename = f"evaluation_{tag}.mp4" if tag else "evaluation.mp4"
            movie_path = pathlib.Path(self.output_dir).joinpath("rollouts", movie_filename)
            movie_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert list of (3, 96, 96) arrays to (N, 96, 96, 3) and uint8
            # Images are CHW format normalized to [0, 1], need HWC format [0, 255]
            movie_imgs = np.stack(imgs)  # (N, 3, 96, 96)
            movie_imgs = np.transpose(movie_imgs, (0, 2, 3, 1))  # (N, 96, 96, 3)
            movie_imgs = (movie_imgs * 255).astype(np.uint8)  # Convert to [0, 255]
            vwrite(str(movie_path), movie_imgs)

        return {
            'task_success': max(rewards) > 0.95,
            'reward': max(rewards),
        }

if __name__ == '__main__':
    cfg = OmegaConf.load("diffusion_policy/config.yaml")
    OmegaConf.resolve(cfg)

    existing_model_path = None
    trainer = Trainer(cfg, existing_model_path=existing_model_path)
    trainer.run_training()