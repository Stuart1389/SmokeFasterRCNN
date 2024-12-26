import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("torchvision version:", torchvision.__version__)

"""
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
"""

#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121