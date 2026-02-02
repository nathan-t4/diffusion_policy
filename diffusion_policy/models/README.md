PyTorch neural network module definitions

- `resnet.py`: helper functions to load pretrained ResNet vision models from torchvision (including replacing batchnorms with groupnorms)
- `unet.py`: UNet backbone for diffusion model, conditioned on observation history 
- `config.py`: helper function to initialize resnet + 1D conditional UNet together