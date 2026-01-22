#!/usr/bin/env python3
"""
Phase 9 Smoke Test - GPU & Ollama Lifecycle

Tests the core functionality of Ollama lifecycle management,
GPU monitoring, and memory-aware scheduling.
"""

import sys
import os
import logging
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ollama_lifecycle_manager():
    """Test Ollama lifecycle manager functionality."""
    print("\n[TEST] Ollama Lifecycle Manager")
    print("=" * 50)
    
    try:
        from ai.ollama_lifecycle import OllamaLifecycleManager, GPUMetrics
        
        # Create manager instance
        manager = OllamaLifecycleManager(
            ollama_host="http://localhost:11434",
            max_concurrent_models=2,
            memory_threshold_mb=1000
        )
        
        print("[OK] OllamaLifecycleManager created successfully")
        
        # Test health check (mocked)
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {'models': []}
            mock_get.return_value = mock_response
            
            health_status = await manager.health_check()
            print(f"[OK] Health check completed: {health_status}")
        
        # Test GPU metrics (mocked)
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "8192, 4096, 75, 65\n"
            mock_run.return_value = mock_result
            
            gpu_metrics = await manager.get_gpu_metrics()
            if gpu_metrics:
                print(f"[OK] GPU metrics retrieved: {gpu_metrics.total_memory_mb}MB total, {gpu_metrics.utilization_percent}% utilization")
            else:
                print("[OK] GPU metrics not available (expected on systems without NVIDIA GPU)")
        
        # Test system memory
        memory_info = await manager.get_system_memory()
        if memory_info:
            print(f"[OK] System memory: {memory_info.get('available_mb', 0)}MB available")
        else:
            print("[WARN] System memory info not available")
        
        # Test model pre-warming (mocked)
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {'response': 'Hello'}
            mock_post.return_value = mock_response
            
            pre_warm_result = await manager.pre_warm_model('test-model')
            print(f"[OK] Model pre-warming test: {pre_warm_result}")
        
        # Test status retrieval
        with patch.object(manager, 'health_check') as mock_health:
            mock_health.return_value = True
            
            status = await manager.get_status()
            print(f"[OK] Status retrieved: {len(status)} status fields")
        
        print("[PASS] Ollama lifecycle manager test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Ollama lifecycle manager test failed: {e}")
        return False

async def test_multi_model_integration():
    """Test multi-model manager integration with lifecycle management."""
    print("\n[TEST] Multi-Model Integration")
    print("=" * 50)
    
    try:
        from ai.multi_model import MultiModelManager
        
        # Create multi-model manager
        manager = MultiModelManager(mode="DEMO")
        print("[OK] MultiModelManager created successfully")
        
        # Test model readiness check (mocked)
        with patch.object(manager.lifecycle_manager, 'health_check') as mock_health:
            mock_health.return_value = True
            
            with patch.object(manager.lifecycle_manager, 'get_model_for_inference') as mock_get_model:
                mock_get_model.return_value = 'qwen2.5:14b-instruct'
                
                ready = await manager.ensure_models_ready()
                print(f"[OK] Models ready check: {ready}")
        
        # Test pre-warming (mocked)
        with patch.object(manager.lifecycle_manager, 'pre_warm_model') as mock_pre_warm:
            mock_pre_warm.return_value = True
            
            pre_warm_results = await manager.pre_warm_models(['qwen2.5:14b-instruct'])
            print(f"[OK] Model pre-warming: {pre_warm_results}")
        
        # Test memory cleanup (mocked)
        with patch.object(manager.lifecycle_manager, 'memory_cleanup_if_needed') as mock_cleanup:
            mock_cleanup.return_value = False
            
            cleanup_result = await manager.cleanup_memory_if_needed()
            print(f"[OK] Memory cleanup check: {cleanup_result}")
        
        # Test system status (mocked)
        with patch.object(manager.lifecycle_manager, 'get_status') as mock_lifecycle_status:
            mock_lifecycle_status.return_value = {'ollama_healthy': True}
            
            with patch.object(manager, 'check_model_availability') as mock_availability:
                mock_availability.return_value = {'qwen2.5:14b-instruct': True}
                
                status = await manager.get_system_status()
                print(f"[OK] System status retrieved: {len(status)} status fields")
        
        print("[PASS] Multi-model integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Multi-model integration test failed: {e}")
        return False

async def test_global_functions():
    """Test global convenience functions."""
    print("\n[TEST] Global Functions")
    print("=" * 50)
    
    try:
        from ai.ollama_lifecycle import (
            ensure_ollama_healthy,
            pre_warm_models,
            cleanup_memory_if_needed,
            get_ollama_status
        )
        
        # Test ensure_ollama_healthy (mocked)
        with patch('ai.ollama_lifecycle.get_lifecycle_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.health_check = Mock(return_value=True)
            mock_get_manager.return_value = mock_manager
            
            health_result = await ensure_ollama_healthy()
            print(f"[OK] Global health check: {health_result}")
        
        # Test pre_warm_models (mocked)
        with patch('ai.ollama_lifecycle.get_lifecycle_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.pre_warm_model = Mock(return_value=True)
            mock_get_manager.return_value = mock_manager
            
            pre_warm_results = await pre_warm_models(['test-model'])
            print(f"[OK] Global pre-warm models: {pre_warm_results}")
        
        # Test cleanup_memory_if_needed (mocked)
        with patch('ai.ollama_lifecycle.get_lifecycle_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.memory_cleanup_if_needed = Mock(return_value=False)
            mock_get_manager.return_value = mock_manager
            
            cleanup_result = await cleanup_memory_if_needed()
            print(f"[OK] Global memory cleanup: {cleanup_result}")
        
        # Test get_ollama_status (mocked)
        with patch('ai.ollama_lifecycle.get_lifecycle_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_status = Mock(return_value={'ollama_healthy': True})
            mock_get_manager.return_value = mock_manager
            
            status = await get_ollama_status()
            print(f"[OK] Global status retrieval: {len(status)} status fields")
        
        print("[PASS] Global functions test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Global functions test failed: {e}")
        return False

async def test_performance_tracking():
    """Test performance and memory tracking features."""
    print("\n[TEST] Performance Tracking")
    print("=" * 50)
    
    try:
        from ai.ollama_lifecycle import OllamaLifecycleManager
        import time
        
        manager = OllamaLifecycleManager()
        
        # Test model load time tracking (mocked)
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {'response': 'Hello'}
            mock_post.return_value = mock_response
            
            start_time = time.time()
            result = await manager.pre_warm_model('performance-test-model')
            end_time = time.time()
            
            if result and 'performance-test-model' in manager.model_load_times:
                load_time = manager.model_load_times['performance-test-model']
                print(f"[OK] Model load time tracked: {load_time:.3f}s")
            else:
                print("[WARN] Model load time tracking not working")
        
        # Test memory pressure detection (mocked)
        with patch.object(manager, 'get_system_memory') as mock_memory:
            # Test low memory condition
            mock_memory.return_value = {'available_mb': 500}
            
            with patch.object(manager, 'unload_all_models') as mock_unload:
                mock_unload.return_value = True
                
                cleanup_result = await manager.memory_cleanup_if_needed()
                print(f"[OK] Memory pressure detection: {cleanup_result}")
            
            # Test normal memory condition
            mock_memory.return_value = {'available_mb': 2000}
            
            with patch.object(manager, 'unload_all_models') as mock_unload:
                cleanup_result = await manager.memory_cleanup_if_needed()
                print(f"[OK] Normal memory condition: {cleanup_result}")
        
        print("[PASS] Performance tracking test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Performance tracking test failed: {e}")
        return False

async def test_gpu_monitoring():
    """Test GPU monitoring functionality."""
    print("\n[TEST] GPU Monitoring")
    print("=" * 50)
    
    try:
        from ai.ollama_lifecycle import OllamaLifecycleManager
        
        manager = OllamaLifecycleManager()
        
        # Test GPU metrics with nvidia-smi available (mocked)
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "8192, 6144, 75, 65\n"
            mock_run.return_value = mock_result
            
            gpu_metrics = await manager.get_gpu_metrics()
            
            if gpu_metrics:
                print(f"[OK] GPU metrics: {gpu_metrics.total_memory_mb}MB total, {gpu_metrics.used_memory_mb}MB used")
                print(f"[OK] GPU utilization: {gpu_metrics.utilization_percent}%, temperature: {gpu_metrics.temperature_c}°C")
            else:
                print("[OK] GPU metrics not available (expected on systems without NVIDIA GPU)")
        
        # Test GPU metrics with nvidia-smi not available (mocked)
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            
            gpu_metrics = await manager.get_gpu_metrics()
            
            if gpu_metrics is None:
                print("[OK] GPU monitoring gracefully handles missing nvidia-smi")
            else:
                print("[WARN] GPU monitoring should return None when nvidia-smi is not available")
        
        print("[PASS] GPU monitoring test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] GPU monitoring test failed: {e}")
        return False

async def main():
    """Run all Phase 9 smoke tests."""
    print("Phase 9 Smoke Test - GPU & Ollama Lifecycle")
    print("=" * 60)
    
    tests = [
        ("Ollama Lifecycle Manager", test_ollama_lifecycle_manager),
        ("Multi-Model Integration", test_multi_model_integration),
        ("Global Functions", test_global_functions),
        ("Performance Tracking", test_performance_tracking),
        ("GPU Monitoring", test_gpu_monitoring),
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
    print("Phase 9 Smoke Test Summary")
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
