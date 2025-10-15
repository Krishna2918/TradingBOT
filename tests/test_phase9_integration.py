"""
Phase 9 Integration Tests - GPU & Ollama Lifecycle

Tests the integration of Ollama lifecycle management, GPU monitoring,
and memory-aware scheduling in the multi-model system.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import json

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai.ollama_lifecycle import (
    OllamaLifecycleManager, 
    ModelInfo, 
    GPUMetrics,
    get_lifecycle_manager,
    ensure_ollama_healthy,
    pre_warm_models,
    cleanup_memory_if_needed,
    get_ollama_status
)
from ai.multi_model import MultiModelManager, ModelRole, ModelConfig

class TestOllamaLifecycleManager:
    """Test the Ollama lifecycle manager."""
    
    @pytest.fixture
    def lifecycle_manager(self):
        """Create a lifecycle manager instance for testing."""
        return OllamaLifecycleManager(
            ollama_host="http://localhost:11434",
            max_concurrent_models=2,
            memory_threshold_mb=1000,
            model_timeout_seconds=30
        )
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, lifecycle_manager):
        """Test successful health check."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {'models': []}
            mock_get.return_value = mock_response
            
            result = await lifecycle_manager.health_check()
            
            assert result is True
            assert lifecycle_manager.health_check_failures == 0
            assert lifecycle_manager.last_health_check is not None
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, lifecycle_manager):
        """Test health check failure."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            result = await lifecycle_manager.health_check()
            
            assert result is False
            assert lifecycle_manager.health_check_failures == 1
    
    @pytest.mark.asyncio
    async def test_get_available_models(self, lifecycle_manager):
        """Test getting available models."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                'models': [
                    {'name': 'qwen2.5:14b-instruct'},
                    {'name': 'llama3.1:8b-instruct'}
                ]
            }
            mock_get.return_value = mock_response
            
            models = await lifecycle_manager.get_available_models()
            
            assert len(models) == 2
            assert 'qwen2.5:14b-instruct' in models
            assert 'llama3.1:8b-instruct' in models
    
    @pytest.mark.asyncio
    async def test_get_gpu_metrics_success(self, lifecycle_manager):
        """Test successful GPU metrics retrieval."""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "8192, 4096, 75, 65\n"
            mock_run.return_value = mock_result
            
            metrics = await lifecycle_manager.get_gpu_metrics()
            
            assert metrics is not None
            assert metrics.total_memory_mb == 8192
            assert metrics.used_memory_mb == 4096
            assert metrics.free_memory_mb == 4096
            assert metrics.utilization_percent == 75.0
            assert metrics.temperature_c == 65.0
    
    @pytest.mark.asyncio
    async def test_get_gpu_metrics_not_available(self, lifecycle_manager):
        """Test GPU metrics when nvidia-smi is not available."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            
            metrics = await lifecycle_manager.get_gpu_metrics()
            
            assert metrics is None
    
    @pytest.mark.asyncio
    async def test_get_system_memory(self, lifecycle_manager):
        """Test system memory retrieval."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=16 * 1024 * 1024 * 1024,  # 16GB
                available=8 * 1024 * 1024 * 1024,  # 8GB
                used=8 * 1024 * 1024 * 1024,  # 8GB
                percent=50.0
            )
            
            memory_info = await lifecycle_manager.get_system_memory()
            
            assert memory_info['total_mb'] == 16384
            assert memory_info['available_mb'] == 8192
            assert memory_info['used_mb'] == 8192
            assert memory_info['percent_used'] == 50.0
    
    @pytest.mark.asyncio
    async def test_pre_warm_model_success(self, lifecycle_manager):
        """Test successful model pre-warming."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {'response': 'Hello'}
            mock_post.return_value = mock_response
            
            result = await lifecycle_manager.pre_warm_model('qwen2.5:14b-instruct')
            
            assert result is True
            assert 'qwen2.5:14b-instruct' in lifecycle_manager.loaded_models
            assert lifecycle_manager.loaded_models['qwen2.5:14b-instruct'].is_loaded is True
    
    @pytest.mark.asyncio
    async def test_pre_warm_model_failure(self, lifecycle_manager):
        """Test model pre-warming failure."""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Model not found")
            
            result = await lifecycle_manager.pre_warm_model('nonexistent:model')
            
            assert result is False
            assert 'nonexistent:model' not in lifecycle_manager.loaded_models
    
    @pytest.mark.asyncio
    async def test_concurrent_model_limit(self, lifecycle_manager):
        """Test concurrent model limit enforcement."""
        lifecycle_manager.max_concurrent_models = 1
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {'response': 'Hello'}
            mock_post.return_value = mock_response
            
            # Pre-warm first model
            result1 = await lifecycle_manager.pre_warm_model('model1')
            assert result1 is True
            
            # Pre-warm second model should trigger cleanup
            result2 = await lifecycle_manager.pre_warm_model('model2')
            assert result2 is True
            
            # Only one model should be loaded
            assert len(lifecycle_manager.loaded_models) == 1
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_triggered(self, lifecycle_manager):
        """Test memory cleanup when under pressure."""
        with patch.object(lifecycle_manager, 'get_system_memory') as mock_memory:
            mock_memory.return_value = {'available_mb': 500}  # Below threshold
            
            with patch.object(lifecycle_manager, 'unload_all_models') as mock_unload:
                mock_unload.return_value = True
                
                result = await lifecycle_manager.memory_cleanup_if_needed()
                
                assert result is True
                mock_unload.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_model_for_inference(self, lifecycle_manager):
        """Test getting model for inference."""
        # Add a loaded model
        lifecycle_manager.loaded_models['qwen2.5:14b-instruct'] = ModelInfo(
            name='qwen2.5:14b-instruct',
            size_gb=14.0,
            last_used=datetime.now(),
            is_loaded=True
        )
        
        preferred_models = ['qwen2.5:14b-instruct', 'llama3.1:8b-instruct']
        model = await lifecycle_manager.get_model_for_inference(preferred_models)
        
        assert model == 'qwen2.5:14b-instruct'
    
    @pytest.mark.asyncio
    async def test_get_status(self, lifecycle_manager):
        """Test getting comprehensive status."""
        with patch.object(lifecycle_manager, 'health_check') as mock_health:
            mock_health.return_value = True
            
            with patch.object(lifecycle_manager, 'get_system_memory') as mock_memory:
                mock_memory.return_value = {'available_mb': 8192}
                
                with patch.object(lifecycle_manager, 'get_gpu_metrics') as mock_gpu:
                    mock_gpu.return_value = None
                    
                    status = await lifecycle_manager.get_status()
                    
                    assert status['ollama_healthy'] is True
                    assert status['health_check_failures'] == 0
                    assert status['loaded_model_count'] == 0
                    assert status['system_memory']['available_mb'] == 8192
                    assert status['gpu_metrics'] is None

class TestMultiModelManagerIntegration:
    """Test multi-model manager integration with lifecycle management."""
    
    @pytest.fixture
    def multi_model_manager(self):
        """Create a multi-model manager instance for testing."""
        return MultiModelManager(mode="DEMO")
    
    @pytest.mark.asyncio
    async def test_ensure_models_ready_success(self, multi_model_manager):
        """Test successful model readiness check."""
        with patch.object(multi_model_manager.lifecycle_manager, 'health_check') as mock_health:
            mock_health.return_value = True
            
            with patch.object(multi_model_manager.lifecycle_manager, 'get_model_for_inference') as mock_get_model:
                mock_get_model.return_value = 'qwen2.5:14b-instruct'
                
                result = await multi_model_manager.ensure_models_ready()
                
                assert result is True
    
    @pytest.mark.asyncio
    async def test_ensure_models_ready_failure(self, multi_model_manager):
        """Test model readiness check failure."""
        with patch.object(multi_model_manager.lifecycle_manager, 'health_check') as mock_health:
            mock_health.return_value = False
                
            result = await multi_model_manager.ensure_models_ready()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_pre_warm_models(self, multi_model_manager):
        """Test pre-warming multiple models."""
        with patch.object(multi_model_manager.lifecycle_manager, 'pre_warm_model') as mock_pre_warm:
            mock_pre_warm.return_value = True
            
            results = await multi_model_manager.pre_warm_models(['qwen2.5:14b-instruct'])
            
            assert results['qwen2.5:14b-instruct'] is True
            mock_pre_warm.assert_called_once_with('qwen2.5:14b-instruct')
    
    @pytest.mark.asyncio
    async def test_cleanup_memory_if_needed(self, multi_model_manager):
        """Test memory cleanup."""
        with patch.object(multi_model_manager.lifecycle_manager, 'memory_cleanup_if_needed') as mock_cleanup:
            mock_cleanup.return_value = True
            
            result = await multi_model_manager.cleanup_memory_if_needed()
            
            assert result is True
            mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, multi_model_manager):
        """Test getting comprehensive system status."""
        with patch.object(multi_model_manager.lifecycle_manager, 'get_status') as mock_lifecycle_status:
            mock_lifecycle_status.return_value = {'ollama_healthy': True}
            
            with patch.object(multi_model_manager, 'check_model_availability') as mock_availability:
                mock_availability.return_value = {'qwen2.5:14b-instruct': True}
                
                status = await multi_model_manager.get_system_status()
                
                assert 'lifecycle_status' in status
                assert 'model_availability' in status
                assert 'model_configs' in status
                assert 'adaptive_weights' in status
                assert status['lifecycle_status']['ollama_healthy'] is True

class TestGlobalFunctions:
    """Test global convenience functions."""
    
    @pytest.mark.asyncio
    async def test_ensure_ollama_healthy(self):
        """Test global ensure_ollama_healthy function."""
        with patch('ai.ollama_lifecycle.get_lifecycle_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.health_check = AsyncMock(return_value=True)
            mock_get_manager.return_value = mock_manager
            
            result = await ensure_ollama_healthy()
            
            assert result is True
            mock_manager.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pre_warm_models_global(self):
        """Test global pre_warm_models function."""
        with patch('ai.ollama_lifecycle.get_lifecycle_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.pre_warm_model = AsyncMock(return_value=True)
            mock_get_manager.return_value = mock_manager
            
            results = await pre_warm_models(['qwen2.5:14b-instruct'])
            
            assert results['qwen2.5:14b-instruct'] is True
            mock_manager.pre_warm_model.assert_called_once_with('qwen2.5:14b-instruct')
    
    @pytest.mark.asyncio
    async def test_cleanup_memory_if_needed_global(self):
        """Test global cleanup_memory_if_needed function."""
        with patch('ai.ollama_lifecycle.get_lifecycle_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.memory_cleanup_if_needed = AsyncMock(return_value=True)
            mock_get_manager.return_value = mock_manager
            
            result = await cleanup_memory_if_needed()
            
            assert result is True
            mock_manager.memory_cleanup_if_needed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_ollama_status_global(self):
        """Test global get_ollama_status function."""
        with patch('ai.ollama_lifecycle.get_lifecycle_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_status = AsyncMock(return_value={'ollama_healthy': True})
            mock_get_manager.return_value = mock_manager
            
            status = await get_ollama_status()
            
            assert status['ollama_healthy'] is True
            mock_manager.get_status.assert_called_once()

class TestPerformanceAndMemory:
    """Test performance and memory management features."""
    
    @pytest.mark.asyncio
    async def test_model_load_time_tracking(self):
        """Test that model load times are tracked."""
        manager = OllamaLifecycleManager()
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {'response': 'Hello'}
            mock_post.return_value = mock_response
            
            start_time = time.time()
            result = await manager.pre_warm_model('test-model')
            end_time = time.time()
            
            assert result is True
            assert 'test-model' in manager.model_load_times
            assert manager.model_load_times['test-model'] > 0
            assert manager.model_load_times['test-model'] <= (end_time - start_time) + 0.1
    
    @pytest.mark.asyncio
    async def test_memory_pressure_detection(self):
        """Test memory pressure detection and response."""
        manager = OllamaLifecycleManager(memory_threshold_mb=1000)
        
        with patch.object(manager, 'get_system_memory') as mock_memory:
            # Test low memory condition
            mock_memory.return_value = {'available_mb': 500}
            
            with patch.object(manager, 'unload_all_models') as mock_unload:
                mock_unload.return_value = True
                
                result = await manager.memory_cleanup_if_needed()
                
                assert result is True
                mock_unload.assert_called_once()
            
            # Test normal memory condition
            mock_memory.return_value = {'available_mb': 2000}
            
            with patch.object(manager, 'unload_all_models') as mock_unload:
                result = await manager.memory_cleanup_if_needed()
                
                assert result is False
                mock_unload.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_gpu_memory_monitoring(self):
        """Test GPU memory monitoring integration."""
        manager = OllamaLifecycleManager()
        
        with patch.object(manager, 'get_gpu_metrics') as mock_gpu:
            # Test with GPU available
            mock_gpu.return_value = GPUMetrics(
                total_memory_mb=8192,
                used_memory_mb=6144,
                free_memory_mb=2048,
                utilization_percent=75.0,
                temperature_c=65.0
            )
            
            metrics = await manager.get_gpu_metrics()
            
            assert metrics is not None
            assert metrics.total_memory_mb == 8192
            assert metrics.free_memory_mb == 2048
            assert metrics.utilization_percent == 75.0
            
            # Test with GPU not available
            mock_gpu.return_value = None
            
            metrics = await manager.get_gpu_metrics()
            
            assert metrics is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
