# Diffusion Policy for PushT Environment

Task Environment: pushT — https://github.com/huggingface/gym-pusht

# TODO: 
- look at colab notebooks to start training
- how to evaluate (95% coverage = success in environment)
- final evaluation should use multiple seeds (500? https://huggingface.co/lerobot/diffusion_pusht)
- write continue_training code

State is 9 waypoints on the T, Image uses RGB (96x96)

# Install
Setup environment with `uv sync`

Then install the environment
``` 
    uv pip install -e .
```

## Special Installation Instructions

### ROCm 7.1
The following command uses the ROCm 7.1 PyTorch wheel. If you want a different version of ROCm, modify the command accordingly.
```
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.1

```

### CUDA
Follow torch installation instructions on the official website [[Link](https://pytorch.org/get-started/locally/)]
```
    uv pip install torch torchvision
```

