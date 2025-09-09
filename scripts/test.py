import torch

print("=== GPU Verification ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x @ y
    
    print(f"✅ Success! Operation performed on: {x.device}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
else:
    print("❌ GPU not available")