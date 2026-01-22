"""
GPU Environment Setup for Aggressive LSTM Training

This script configures the environment to use the CUDA 12.8 toolkit
found in tools/tools/cuda-12.8/ for maximum GPU performance.
"""

import os
import sys
from pathlib import Path

def setup_cuda_environment():
    """Setup CUDA environment variables for optimal performance"""
    
    # CUDA toolkit path
    cuda_path = Path("../Tools/cuda-12.8").resolve()
    
    if not cuda_path.exists():
        print("❌ CUDA toolkit not found at expected location")
        return False
    
    print(f"✅ Found CUDA 12.8 toolkit at: {cuda_path}")
    
    # Set environment variables
    os.environ['CUDA_HOME'] = str(cuda_path)
    os.environ['CUDA_PATH'] = str(cuda_path)
    os.environ['CUDA_ROOT'] = str(cuda_path)
    
    # Add CUDA bin to PATH
    cuda_bin = cuda_path / "bin"
    current_path = os.environ.get('PATH', '')
    if str(cuda_bin) not in current_path:
        os.environ['PATH'] = f"{cuda_bin};{current_path}"
    
    # Add CUDA lib to PATH
    cuda_lib = cuda_path / "lib" / "x64"
    if str(cuda_lib) not in current_path:
        os.environ['PATH'] = f"{cuda_lib};{os.environ['PATH']}"
    
    # cuDNN settings
    os.environ['CUDNN_PATH'] = str(cuda_path)
    
    # Performance optimizations
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
    os.environ['CUDA_CACHE_DISABLE'] = '0'    # Enable caching
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # Memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("✅ CUDA environment variables configured")
    return True

def check_gpu_availability():
    """Check if GPU is available and working"""
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
            
            return True
        else:
            print("❌ CUDA not available in PyTorch")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def install_cuda_pytorch():
    """Install CUDA-enabled PyTorch"""
    print("Installing CUDA-enabled PyTorch...")
    
    # Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.8)
    install_cmd = [
        "pip", "install", "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    import subprocess
    try:
        result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
        print("✅ CUDA PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install CUDA PyTorch: {e}")
        return False

def main():
    """Main setup process"""
    print("GPU Environment Setup for Aggressive LSTM Training")
    print("=" * 60)
    
    # Step 1: Setup CUDA environment
    cuda_ok = setup_cuda_environment()
    
    # Step 2: Check current GPU availability
    gpu_available = check_gpu_availability()
    
    # Step 3: Install CUDA PyTorch if needed
    if cuda_ok and not gpu_available:
        print("\nInstalling CUDA-enabled PyTorch...")
        pytorch_ok = install_cuda_pytorch()
        
        if pytorch_ok:
            print("\nRechecking GPU availability...")
            gpu_available = check_gpu_availability()
    
    # Summary
    print("\n" + "=" * 60)
    print("GPU Setup Summary:")
    print(f"CUDA Toolkit: {'✅ CONFIGURED' if cuda_ok else '❌ FAILED'}")
    print(f"GPU Available: {'✅ READY' if gpu_available else '❌ NOT AVAILABLE'}")
    
    if cuda_ok and gpu_available:
        print("\n🚀 GPU environment ready for aggressive LSTM training!")
        print("\nOptimizations enabled:")
        print("- CUDA 12.8 with cuDNN 9")
        print("- Mixed precision training (AMP)")
        print("- Optimized memory allocation")
        print("- Async GPU execution")
        
        return True
    else:
        print("\n⚠️  GPU setup incomplete. Training will use CPU.")
        return False

if __name__ == "__main__":
    main()