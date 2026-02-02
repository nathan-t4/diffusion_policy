import torch
import torch.nn as nn
from omegaconf import OmegaConf

from diffusion_policy.models.resnet import get_resnet, replace_bn_with_gn
from diffusion_policy.models.unet import ConditionalUnet1D

def init_ema_model(config: OmegaConf):
    obs_horizon = config.policy.n_obs_steps

    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet('resnet18')

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    # agent_pos is 2 dimensional
    lowdim_obs_dim = 2
    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    # device transfer
    device = torch.device('cuda')
    _ = nets.to(device) 

    return nets

def load_pretrained_model(config: OmegaConf, model_path: str):
    nets = init_ema_model(config)
    state_dict = torch.load(model_path)
    nets.load_state_dict(state_dict)
    return nets