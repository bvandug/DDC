import torch

print("🔍 Checking PyTorch CUDA availability...")

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Using Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("⚠️ No CUDA devices detected. Make sure:")
    print("  • You have a supported NVIDIA GPU")
    print("  • CUDA drivers are installed correctly")
    print("  • PyTorch was installed with CUDA support")


import torch

print("🚀 torch.__version__:", torch.__version__)
print("🧠 CUDA available:", torch.cuda.is_available())
print("🛠 CUDA version (PyTorch was built with):", torch.version.cuda)
print("📦 Torch compiled with CUDA support:", torch.backends.cuda.is_built())

if torch.cuda.is_available():
    print(f"✅ Detected GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No CUDA GPU detected or torch not built with CUDA support.")
