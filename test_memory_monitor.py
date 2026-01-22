"""
Test Memory Monitor

Test the memory monitoring and logging functionality.
"""

import sys
sys.path.append('src')

from src.ai.models.gpu_memory_manager import GPUMemoryManager
from src.ai.models.memory_monitor import MemoryMonitor
import torch
import time

def test_memory_monitor():
    """Test memory monitoring functionality"""
    print("Testing Memory Monitor")
    print("=" * 50)
    
    # Initialize components
    memory_manager = GPUMemoryManager(reserve_memory_gb=1.0)
    monitor = MemoryMonitor(memory_manager, log_dir="logs")
    
    if not memory_manager.gpu_available:
        print("⚠️  GPU not available - limited testing")
        return
    
    # Test 1: Basic logging
    print("\n1. Basic Memory Logging")
    print("-" * 30)
    
    # Log initial state
    monitor.log_memory_usage(epoch=0, step=0, operation="initialization", force_log=True)
    print("✅ Initial memory state logged")
    
    # Test 2: Simulated training with memory logging
    print("\n2. Simulated Training Memory Logging")
    print("-" * 30)
    
    try:
        # Simulate training epochs with varying memory usage
        test_tensors = []
        
        for epoch in range(3):
            print(f"Simulating epoch {epoch+1}")
            
            # Allocate memory to simulate batch processing
            for step in range(5):
                # Simulate different batch sizes
                batch_size = 32 + (step * 16)  # 32, 48, 64, 80, 96
                
                # Allocate tensor to simulate batch
                tensor_size = batch_size * 100  # Smaller tensors for testing
                tensor = torch.randn(tensor_size, 100, device=memory_manager.device)
                test_tensors.append(tensor)
                
                # Log memory usage
                monitor.log_memory_usage(
                    epoch=epoch+1, 
                    step=step+1, 
                    batch_size=batch_size,
                    operation="forward_pass",
                    notes=f"Processing batch {step+1}"
                )
                
                # Small delay to simulate processing
                time.sleep(0.1)
            
            # Simulate epoch end cleanup
            if epoch == 1:  # Clean up after epoch 2
                del test_tensors[5:10]  # Remove some tensors
                memory_manager.cleanup_memory()
                monitor.log_memory_usage(
                    epoch=epoch+1, 
                    step=0, 
                    operation="epoch_cleanup",
                    notes="Cleaned up intermediate tensors",
                    force_log=True
                )
        
        print("✅ Simulated training completed")
        
    except Exception as e:
        print(f"❌ Simulated training failed: {e}")
    
    # Test 3: Memory summary
    print("\n3. Memory Usage Summary")
    print("-" * 30)
    
    summary = monitor.get_memory_summary()
    print(f"Total log entries: {summary.get('total_log_entries', 0)}")
    print(f"Monitoring duration: {summary.get('monitoring_duration_minutes', 0):.1f} minutes")
    
    if 'memory_utilization' in summary:
        util = summary['memory_utilization']
        print(f"Memory utilization - Min: {util['min_pct']:.1f}%, Max: {util['max_pct']:.1f}%, Avg: {util['avg_pct']:.1f}%")
        print(f"Peak memory usage: {util['peak_memory_gb']:.1f} GB")
    
    if 'alerts' in summary:
        alerts = summary['alerts']
        print(f"Total alerts: {alerts['total_alerts']}")
        print(f"Warning alerts: {alerts['warning_alerts']}")
        print(f"Critical alerts: {alerts['critical_alerts']}")
    
    # Test 4: Efficiency report
    print("\n4. Memory Efficiency Report")
    print("-" * 30)
    
    efficiency_report = monitor.get_memory_efficiency_report()
    print(f"Efficiency Score: {efficiency_report.get('efficiency_score', 'N/A')}")
    print(f"Efficiency Note: {efficiency_report.get('efficiency_note', 'N/A')}")
    print(f"Average Utilization: {efficiency_report.get('average_utilization_pct', 0):.1f}%")
    print(f"Peak Utilization: {efficiency_report.get('peak_utilization_pct', 0):.1f}%")
    
    if efficiency_report.get('recommendations'):
        print("Recommendations:")
        for rec in efficiency_report['recommendations']:
            print(f"  - {rec}")
    
    # Test 5: Save logs
    print("\n5. Save Detailed Logs")
    print("-" * 30)
    
    try:
        monitor.save_detailed_log()
        print("✅ Detailed logs saved")
        print(f"CSV log: {monitor.log_file}")
        print(f"JSON log: {monitor.json_log_file}")
    except Exception as e:
        print(f"❌ Error saving logs: {e}")
    
    # Test 6: Create visualization (optional - requires matplotlib)
    print("\n6. Memory Usage Visualization")
    print("-" * 30)
    
    try:
        plot_path = monitor.create_memory_usage_plot()
        if plot_path:
            print(f"✅ Memory usage plot created: {plot_path}")
        else:
            print("⚠️  Plot creation skipped (matplotlib may not be available)")
    except Exception as e:
        print(f"⚠️  Plot creation failed: {e}")
    
    # Cleanup
    try:
        if 'test_tensors' in locals():
            del test_tensors
        memory_manager.cleanup_memory()
        print("\n✅ Test cleanup completed")
    except:
        pass
    
    print("\n" + "=" * 50)
    print("Memory Monitor Test Complete")
    
    return monitor

if __name__ == "__main__":
    test_memory_monitor()