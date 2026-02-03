# Diffusion Policy for PushT Environment

Task Environment: pushT — https://github.com/huggingface/gym-pusht

# Install
Setup environment with `uv sync`

Then install the environment
``` 
source .venv/bin/activate
uv pip install -e .
```

## Special Installation Instructions

### ROCm 7.1
The following command uses the ROCm 7.1 PyTorch wheel. If you want a different version of ROCm, modify the command accordingly.
```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.1

```

### CUDA
Follow torch installation instructions on the official website [[Link](https://pytorch.org/get-started/locally/)], e.g.:
```
uv pip install torch torchvision
```

# Train DP on PushT environment 
Change config flag to train on another version of data (v1 or v2)
```
python3 diffusion_policy/train.py --config=v1
```

## Example: success with v2 policy (trained with 2x data) while failure with v1 policy

### v1
<video width="320" height="320" controls>
  <source src="evaluation_v1_100ep_100001.mp4" type="video/mp4">
</video>

### v2
<video width="320" height="320" controls>
  <source src="evaluation_v2_100ep_100001.mp4" type="video/mp4">
</video>

## Training Runs Info
- runs/20260202_213008 (v1, 100 epochs)
- runs/20260202_223257 (v2, 100 epochs)

Wandb training report: [[Link](https://api.wandb.ai/links/nathan-t4-n-a/jytx7c9j)]

# Credits
- Official DP Repository: https://github.com/real-stanford/diffusion_policy
- HuggingFace PushT Environment: https://github.com/huggingface/gym-pusht 
- Data: https://drive.google.com/drive/folders/1tL7WRNSsIjPAGuD5yAJuRMJKIe-DAz0J 