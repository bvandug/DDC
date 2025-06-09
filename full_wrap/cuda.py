import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Built with CUDA:", torch.version.cuda)
print("Running on device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
