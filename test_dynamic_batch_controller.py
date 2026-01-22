"""
Test Dynamic Batch Controller

Test the dynamic batch size adjustment functionality.
"""

import sys
sys.path.append('src')

from src.ai.models.gpu_memory_manager import GPUMemoryManager
from src.ai.models.dynamic_batch_controller import DynamicBatchController
import torch

def test_dynamic_batch_controller():
    """Test dynamic batch controller functionality"""
    print("Testing Dynamic Batch Controller")
    print("=" * 60)
    
    # Initialize components
    memory_manager = GPUMemoryManager(reserve_memory_gb=1.0)
    batch_controller = DynamicBatchController(
        memory_manager=memory_manager,
        initial_batch_size=128,
        min_batch_size=8,
        max_batch_size=512
    )
    
    if not memory_manager.gpu_available:
        print("⚠️  GPU not available - limited testing")
        return
    
    # Test 1: Configuration
    print("\n1. Batch Controller Configuration")
    print("-" * 40)
    
    # Configure with LSTM parameters
    batch_controller.configure(
        sequence_length=252,
        feature_count=55,
        bytes_per_element=4
    )
    
    print("✅ Batch controller configured for LSTM training")
    print(f"Current batch size: {batch_controller.current_batch_size}")
    print(f"Effective batch size: {batch_controller.get_effective_batch_size()}")
    print(f"Gradient accumulation steps: {batch_controller.gradient_accumulation_steps}")
    
    # Test 2: Optimal batch size calculation
    print("\n2. Optimal Batch Size Calculation")
    print("-" * 40)
    
    available_memory = memory_manager.get_available_memory()
    optimal_batch = batch_controller.calculate_optimal_batch_size(available_memory)
    
    print(f"Available memory: {available_memory:.1f} GB")
    print(f"Optimal batch size: {optimal_batch}")
    
    # Test different memory scenarios
    test_scenarios = [
        ("High memory", 8.0),
        ("Medium memory", 4.0),
        ("Low memory", 2.0),
        ("Very low memory", 0.5)
    ]
    
    for scenario, memory_gb in test_scenarios:
        optimal = batch_controller.calculate_optimal_batch_size(memory_gb)
        print(f"{scenario} ({memory_gb:.1f} GB): Optimal batch size = {optimal}")
    
    # Test 3: Batch size adjustment
    print("\n3. Batch Size Adjustment Testing")
    print("-" * 40)
    
    # Test adjustment under different memory pressures
    memory_scenarios = [
        ("Low usage", 30.0),
        ("Moderate usage", 60.0),
        ("High usage", 85.0),
        ("Critical usage", 95.0)
    ]
    
    for scenario, memory_pct in memory_scenarios:
        print(f"\nTesting {scenario} ({memory_pct:.0f}% memory usage):")
        
        old_batch = batch_controller.current_batch_size
        adjusted = batch_controller.adjust_batch_size(memory_pct)
        new_batch = batch_controller.current_batch_size
        
        if adjusted:
            print(f"  ✅ Adjusted: {old_batch} → {new_batch}")
            print(f"  Gradient accumulation: {batch_controller.gradient_accumulation_steps}")
            print(f"  Effective batch size: {batch_controller.get_effective_batch_size()}")
        else:
            print(f"  ➡️  No adjustment needed (batch size: {new_batch})")
    
    # Test 4: Gradient accumulation
    print("\n4. Gradient Accumulation Testing")
    print("-" * 40)
    
    # Enable gradient accumulation for target batch size
    target_batch = 256
    accumulation_steps = batch_controller.enable_gradient_accumulation(target_batch)
    
    print(f"Target effective batch size: {target_batch}")
    print(f"Current batch size: {batch_controller.current_batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Actual effective batch size: {batch_controller.get_effective_batch_size()}")
    
    # Test 5: Aggressive mode
    print("\n5. Aggressive Mode Testing")
    print("-" * 40)
    
    print("Testing conservative mode:")
    batch_controller.set_aggressive_mode(False)
    conservative_optimal = batch_controller.calculate_optimal_batch_size(available_memory)
    print(f"  Conservative optimal batch: {conservative_optimal}")
    
    print("Testing aggressive mode:")
    batch_controller.set_aggressive_mode(True)
    aggressive_optimal = batch_controller.calculate_optimal_batch_size(available_memory)
    print(f"  Aggressive optimal batch: {aggressive_optimal}")
    
    # Test 6: Batch statistics
    print("\n6. Batch Statistics")
    print("-" * 40)
    
    stats = batch_controller.get_batch_statistics()
    
    print(f"Current batch size: {stats['current_batch_size']}")
    print(f"Effective batch size: {stats['effective_batch_size']}")
    print(f"Total adjustments: {stats['total_adjustments']}")
    
    if stats['total_adjustments'] > 0:
        print(f"Batch size range: {stats['batch_size_range']['min']}-{stats['batch_size_range']['max']} (avg: {stats['batch_size_range']['avg']:.1f})")
        print(f"Memory utilization range: {stats['memory_utilization_range']['min']:.1f}%-{stats['memory_utilization_range']['max']:.1f}% (avg: {stats['memory_utilization_range']['avg']:.1f}%)")
        
        print("Adjustment reasons:")
        for reason, count in stats['adjustment_reasons'].items():
            print(f"  {reason}: {count} times")
        
        print("Recent adjustments:")
        for event in stats['recent_adjustments'][-3:]:  # Show last 3
            print(f"  {event.timestamp.strftime('%H:%M:%S')}: {event.old_batch_size} → {event.new_batch_size} ({event.reason})")
    
    # Test 7: Initial batch size recommendation
    print("\n7. Initial Batch Size Recommendation")
    print("-" * 40)
    
    recommended = batch_controller.recommend_initial_batch_size()
    print(f"Recommended initial batch size: {recommended}")
    
    # Test 8: Memory pressure simulation
    print("\n8. Memory Pressure Simulation")
    print("-" * 40)
    
    try:
        # Allocate memory to simulate pressure
        print("Simulating memory pressure...")
        
        # Allocate about 4GB to simulate memory usage
        pressure_tensors = []
        for i in range(4):
            tensor = torch.randn(int(1024**3 / 4), device=memory_manager.device)  # ~1GB each
            pressure_tensors.append(tensor)
        
        print("✅ Memory pressure applied")
        
        # Check memory state
        metrics = memory_manager.monitor_memory_usage()
        print(f"Memory utilization under pressure: {metrics.memory_utilization*100:.1f}%")
        
        # Test batch size adjustment under pressure
        old_batch = batch_controller.current_batch_size
        adjusted = batch_controller.adjust_batch_size(metrics.memory_utilization * 100, force_adjustment=True)
        
        if adjusted:
            print(f"✅ Batch size adjusted under pressure: {old_batch} → {batch_controller.current_batch_size}")
        else:
            print("➡️  No adjustment made under pressure")
        
        # Cleanup pressure tensors
        del pressure_tensors
        memory_manager.cleanup_memory()
        print("✅ Memory pressure simulation cleaned up")
        
    except Exception as e:
        print(f"❌ Memory pressure simulation failed: {e}")
        # Try to cleanup anyway
        try:
            memory_manager.emergency_memory_cleanup()
        except:
            pass
    
    # Test 9: Reset functionality
    print("\n9. Reset Functionality")
    print("-" * 40)
    
    print(f"Before reset - Batch size: {batch_controller.current_batch_size}")
    batch_controller.reset_to_initial()
    print(f"After reset - Batch size: {batch_controller.current_batch_size}")
    print(f"Effective batch size: {batch_controller.get_effective_batch_size()}")
    
    print("\n" + "=" * 60)
    print("Dynamic Batch Controller Test Complete")
    
    return batch_controller

if __name__ == "__main__":
    test_dynamic_batch_controller()