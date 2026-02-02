import hydra
from omegaconf import OmegaConf
from pprint import pprint

def test(config: OmegaConf):
    OmegaConf.resolve(config)
    pprint(config)
    return config

if __name__ == "__main__":
    config = OmegaConf.load("diffusion_policy/config.yaml")
    test(config)