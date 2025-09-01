print("=== Starting Debug Test ===")

# Test 1: Basic imports
print("1. Testing basic imports...")
try:
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"   ❌ PyTorch failed: {e}")

try:
    import numpy as np
    print(f"   ✅ NumPy: {np.__version__}")
except Exception as e:
    print(f"   ❌ NumPy failed: {e}")

try:
    import diffusers
    print(f"   ✅ Diffusers: {diffusers.__version__}")
except Exception as e:
    print(f"   ❌ Diffusers failed: {e}")

# Test 2: Import Stable Diffusion
print("2. Testing Stable Diffusion imports...")
try:
    from diffusers import StableDiffusionPipeline
    print("   ✅ StableDiffusionPipeline imported")
except Exception as e:
    print(f"   ❌ StableDiffusionPipeline failed: {e}")

print("=== Debug Test Complete ===")