"""
Test GPU Memory Manager

Quick test to verify GPUMemoryManager functionality.
"""

import sys
sys.path.append('src')

from src.ai.models.gpu_memory_manager import GPUMemoryManager
import torch

def test_gpu_memory_manager():
    """Test GPUMemoryManager basic functionality"""
    print("Testing GPU Memory Manager")
    print("=" * 50)
    
    # Initialize memory manager
    memory_manager = GPUMemoryManager(reserve_memory_gb=1.0)
    
    # Test 1: Basic memory monitoring
    print("\n1. Memory Monitoring Test")
    print("-" * 30)
    
    metrics = memory_manager.monitor_memory_usage()
    print(f"Total Memory: {metrics.total_memory_gb:.1f} GB")
    print(f"Allocated: {metrics.allocated_memory_gb:.1f} GB")
    print(f"Cached: {metrics.cached_memory_gb:.1f} GB")
    print(f"Free: {metrics.free_memory_gb:.1f} GB")
    print(f"Utilization: {metrics.memory_utilization*100:.1f}%")
    
    # Test 2: Available memory check
    print("\n2. Available Memory Test")
    print("-" * 30)
    
    available = memory_manager.get_available_memory()
    print(f"Available Memory: {available:.1f} GB")
    
    # Test 3: Memory summary
    print("\n3. Memory Summary Test")
    print("-" * 30)
    
    summary = memory_manager.get_memory_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Test 4: Batch size recommendation
    print("\n4. Batch Size Recommendation Test")
    print("-" * 30)
    
    # Test with LSTM parameters (252 sequence length, 55 features)
    recommended_batch = memory_manager.get_recommended_batch_size(
        base_batch_size=128,
        sequence_length=252,
        feature_count=55
    )
    print(f"Recommended batch size for LSTM: {recommended_batch} (requested: 128)")
    
    # Test 5: Memory allocation test
    print("\n5. Memory Allocation Test")
    print("-" * 30)
    
    if memory_manager.gpu_available:
        # Try to allocate a small tensor
        try:
            test_tensor = torch.randn(1000, 1000, device=memory_manager.device)
            print("✅ Small tensor allocation successful")
            
            # Monitor memory after allocation
            metrics_after = memory_manager.monitor_memory_usage()
            print(f"Memory after allocation: {metrics_after.allocated_memory_gb:.1f} GB")
            
            # Cleanup
            del test_tensor
            memory_manager.cleanup_memory()
            print("✅ Memory cleanup completed")
            
        except Exception as e:
            print(f"❌ Memory allocation test failed: {e}")
    else:
        print("⚠️  GPU not available - skipping allocation test")
    
    # Test 6: Memory availability check
    print("\n6. Memory Availability Check")
    print("-" * 30)
    
    # Check if we have enough memory for different scenarios
    scenarios = [
        ("Small batch (32)", 32 * 252 * 55 * 4 / (1024**3)),
        ("Medium batch (64)", 64 * 252 * 55 * 4 / (1024**3)),
        ("Large batch (128)", 128 * 252 * 55 * 4 / (1024**3)),
        ("Very large batch (256)", 256 * 252 * 55 * 4 / (1024**3))
    ]
    
    for scenario_name, required_gb in scenarios:
        available = memory_manager.is_memory_available(required_gb)
        status = "✅ Available" if available else "❌ Not available"
        print(f"{scenario_name} ({required_gb:.1f} GB): {status}")
    
    print("\n" + "=" * 50)
    print("GPU Memory Manager Test Complete")
    
    return memory_manager

if __name__ == "__main__":
    test_gpu_memory_manager()