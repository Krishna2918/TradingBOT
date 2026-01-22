"""
Test GPU Memory Manager Error Handling

Test the enhanced error handling and recovery mechanisms.
"""

import sys
sys.path.append('src')

from src.ai.models.gpu_memory_manager import GPUMemoryManager
import torch

def test_memory_error_handling():
    """Test memory error handling and recovery"""
    print("Testing GPU Memory Manager Error Handling")
    print("=" * 60)
    
    # Initialize memory manager
    memory_manager = GPUMemoryManager(reserve_memory_gb=1.0)
    
    if not memory_manager.gpu_available:
        print("⚠️  GPU not available - limited testing")
        return
    
    # Test 1: Memory health check
    print("\n1. Memory Health Check")
    print("-" * 40)
    
    health = memory_manager.check_memory_health()
    print(f"Status: {health['status']}")
    print(f"Warnings: {len(health['warnings'])}")
    for warning in health['warnings']:
        print(f"  ⚠️  {warning}")
    print(f"Recommendations: {len(health['recommendations'])}")
    for rec in health['recommendations']:
        print(f"  💡 {rec}")
    
    # Test 2: Fallback recommendations
    print("\n2. Fallback Recommendation Test")
    print("-" * 40)
    
    # Test different memory requirements
    test_scenarios = [
        ("Small batch training", 1.0),
        ("Medium batch training", 4.0),
        ("Large batch training", 8.0),
        ("Very large batch training", 15.0)  # More than available
    ]
    
    for scenario, required_gb in test_scenarios:
        rec = memory_manager.get_fallback_recommendation(scenario, required_gb)
        print(f"\nScenario: {scenario} ({required_gb:.1f} GB)")
        print(f"  Should fallback to CPU: {rec['should_fallback_to_cpu']}")
        print(f"  Reason: {rec['reason']}")
        if rec['alternatives']:
            print("  Alternatives:")
            for alt in rec['alternatives']:
                print(f"    - {alt}")
    
    # Test 3: Emergency cleanup
    print("\n3. Emergency Cleanup Test")
    print("-" * 40)
    
    # Allocate some memory first
    try:
        test_tensors = []
        for i in range(3):
            tensor = torch.randn(1000, 1000, device=memory_manager.device)
            test_tensors.append(tensor)
        
        print("✅ Allocated test tensors")
        
        # Check memory before cleanup
        metrics_before = memory_manager.monitor_memory_usage()
        print(f"Memory before cleanup: {metrics_before.allocated_memory_gb:.1f} GB")
        
        # Perform emergency cleanup
        success = memory_manager.emergency_memory_cleanup()
        print(f"Emergency cleanup success: {success}")
        
        # Check memory after cleanup
        metrics_after = memory_manager.monitor_memory_usage()
        print(f"Memory after cleanup: {metrics_after.allocated_memory_gb:.1f} GB")
        
        # Cleanup test tensors
        del test_tensors
        memory_manager.cleanup_memory()
        
    except Exception as e:
        print(f"❌ Emergency cleanup test failed: {e}")
    
    # Test 4: Batch size recommendations under memory pressure
    print("\n4. Batch Size Under Memory Pressure")
    print("-" * 40)
    
    # Simulate memory pressure by allocating a large tensor
    try:
        # Allocate about 6GB to simulate memory pressure
        pressure_tensor = torch.randn(int(6 * 1024**3 / 4), device=memory_manager.device)
        print("✅ Simulated memory pressure (6GB allocated)")
        
        # Test batch size recommendations
        original_batch = 128
        recommended = memory_manager.get_recommended_batch_size(
            base_batch_size=original_batch,
            sequence_length=252,
            feature_count=55
        )
        
        print(f"Original batch size: {original_batch}")
        print(f"Recommended under pressure: {recommended}")
        
        # Check memory availability
        available = memory_manager.get_available_memory()
        print(f"Available memory under pressure: {available:.1f} GB")
        
        # Cleanup
        del pressure_tensor
        memory_manager.cleanup_memory()
        print("✅ Memory pressure simulation cleaned up")
        
    except Exception as e:
        print(f"❌ Memory pressure test failed: {e}")
        # Try to cleanup anyway
        try:
            memory_manager.emergency_memory_cleanup()
        except:
            pass
    
    # Test 5: Simulated OOM handling
    print("\n5. Simulated OOM Error Handling")
    print("-" * 40)
    
    # Create a fake OOM error
    fake_oom_error = RuntimeError("CUDA error: out of memory")
    
    recovery_info = memory_manager.handle_oom_error(fake_oom_error)
    
    print(f"Error type: {recovery_info['error_type']}")
    print(f"Cleanup successful: {recovery_info['cleanup_successful']}")
    print(f"Available after cleanup: {recovery_info['available_after_cleanup_gb']:.1f} GB")
    print("Recovery recommendations:")
    for rec in recovery_info['recommendations']:
        print(f"  - {rec}")
    
    print("\n" + "=" * 60)
    print("Memory Error Handling Test Complete")

if __name__ == "__main__":
    test_memory_error_handling()