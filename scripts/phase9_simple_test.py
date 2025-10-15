#!/usr/bin/env python3
"""
Phase 9 Simple Test - GPU & Ollama Lifecycle

Simplified test for core functionality without complex mocking.
"""

import sys
import os
import logging
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ollama_lifecycle_basic():
    """Test basic Ollama lifecycle functionality."""
    print("\n[TEST] Basic Ollama Lifecycle")
    print("=" * 50)
    
    try:
        from ai.ollama_lifecycle import OllamaLifecycleManager
        
        # Create manager instance
        manager = OllamaLifecycleManager(
            ollama_host="http://localhost:11434",
            max_concurrent_models=2,
            memory_threshold_mb=1000
        )
        
        print("[OK] OllamaLifecycleManager created successfully")
        
        # Test system memory
        memory_info = await manager.get_system_memory()
        if memory_info:
            print(f"[OK] System memory: {memory_info.get('available_mb', 0)}MB available")
        else:
            print("[WARN] System memory info not available")
        
        # Test GPU metrics (will return None if nvidia-smi not available)
        gpu_metrics = await manager.get_gpu_metrics()
        if gpu_metrics:
            print(f"[OK] GPU metrics: {gpu_metrics.total_memory_mb}MB total, {gpu_metrics.utilization_percent}% utilization")
        else:
            print("[OK] GPU metrics not available (expected on systems without NVIDIA GPU)")
        
        # Test status retrieval
        status = await manager.get_status()
        print(f"[OK] Status retrieved: {len(status)} status fields")
        
        print("[PASS] Basic Ollama lifecycle test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic Ollama lifecycle test failed: {e}")
        return False

async def test_multi_model_basic():
    """Test basic multi-model functionality."""
    print("\n[TEST] Basic Multi-Model")
    print("=" * 50)
    
    try:
        from ai.multi_model import MultiModelManager
        
        # Create multi-model manager
        manager = MultiModelManager(mode="DEMO")
        print("[OK] MultiModelManager created successfully")
        
        # Test model configurations
        configs = manager.get_all_model_configs()
        print(f"[OK] Model configurations: {len(configs)} models configured")
        
        # Test model weights
        weights = manager.get_model_weights()
        print(f"[OK] Model weights: {len(weights)} models with weights")
        
        # Test adaptive weights
        adaptive_weights = manager.get_adaptive_weights()
        print(f"[OK] Adaptive weights: {len(adaptive_weights)} models with adaptive weights")
        
        print("[PASS] Basic multi-model test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic multi-model test failed: {e}")
        return False

async def test_lifecycle_integration():
    """Test lifecycle integration."""
    print("\n[TEST] Lifecycle Integration")
    print("=" * 50)
    
    try:
        from ai.ollama_lifecycle import get_lifecycle_manager
        from ai.multi_model import MultiModelManager
        
        # Get lifecycle manager
        lifecycle_manager = get_lifecycle_manager()
        print("[OK] Lifecycle manager retrieved successfully")
        
        # Create multi-model manager
        multi_model_manager = MultiModelManager(mode="DEMO")
        print("[OK] Multi-model manager created with lifecycle integration")
        
        # Test that lifecycle manager is integrated
        assert multi_model_manager.lifecycle_manager is not None
        print("[OK] Lifecycle manager is integrated with multi-model manager")
        
        # Test system status
        status = await multi_model_manager.get_system_status()
        print(f"[OK] System status retrieved: {len(status)} status fields")
        
        print("[PASS] Lifecycle integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Lifecycle integration test failed: {e}")
        return False

async def test_performance_features():
    """Test performance tracking features."""
    print("\n[TEST] Performance Features")
    print("=" * 50)
    
    try:
        from ai.ollama_lifecycle import OllamaLifecycleManager
        
        manager = OllamaLifecycleManager()
        
        # Test model load time tracking (empty initially)
        assert len(manager.model_load_times) == 0
        print("[OK] Model load time tracking initialized")
        
        # Test health check failure tracking
        assert manager.health_check_failures == 0
        print("[OK] Health check failure tracking initialized")
        
        # Test loaded models tracking
        assert len(manager.loaded_models) == 0
        print("[OK] Loaded models tracking initialized")
        
        # Test concurrent model limit
        assert manager.max_concurrent_models == 2
        print("[OK] Concurrent model limit configured")
        
        print("[PASS] Performance features test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Performance features test failed: {e}")
        return False

async def test_gpu_monitoring_basic():
    """Test basic GPU monitoring functionality."""
    print("\n[TEST] Basic GPU Monitoring")
    print("=" * 50)
    
    try:
        from ai.ollama_lifecycle import OllamaLifecycleManager
        
        manager = OllamaLifecycleManager()
        
        # Test GPU metrics (will return None if nvidia-smi not available)
        gpu_metrics = await manager.get_gpu_metrics()
        
        if gpu_metrics:
            print(f"[OK] GPU detected: {gpu_metrics.total_memory_mb}MB total memory")
            print(f"[OK] GPU utilization: {gpu_metrics.utilization_percent}%")
            if gpu_metrics.temperature_c:
                print(f"[OK] GPU temperature: {gpu_metrics.temperature_c}°C")
        else:
            print("[OK] GPU monitoring not available (expected on systems without NVIDIA GPU)")
        
        # Test memory threshold configuration
        assert manager.memory_threshold_mb == 1000
        print("[OK] Memory threshold configured")
        
        print("[PASS] Basic GPU monitoring test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic GPU monitoring test failed: {e}")
        return False

async def main():
    """Run all Phase 9 simple tests."""
    print("Phase 9 Simple Test - GPU & Ollama Lifecycle")
    print("=" * 60)
    
    tests = [
        ("Basic Ollama Lifecycle", test_ollama_lifecycle_basic),
        ("Basic Multi-Model", test_multi_model_basic),
        ("Lifecycle Integration", test_lifecycle_integration),
        ("Performance Features", test_performance_features),
        ("Basic GPU Monitoring", test_gpu_monitoring_basic),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[READY] Starting {test_name} test...")
        try:
            if await test_func():
                passed += 1
            else:
                print(f"[FAIL] {test_name} test failed")
        except Exception as e:
            print(f"[FAIL] {test_name} test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Phase 9 Simple Test Summary")
    print("=" * 60)
    
    for test_name, _ in tests:
        status = "[PASS]" if test_name in [tests[i][0] for i in range(passed)] else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Phase 9 GPU & Ollama lifecycle management is working correctly.")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
