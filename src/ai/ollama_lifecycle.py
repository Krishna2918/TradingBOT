"""
Ollama Lifecycle Management - GPU & Memory Optimization

This module manages Ollama model lifecycle including health checks, pre-warming,
memory management, and concurrent model call limits.
"""

import asyncio
import logging
import subprocess
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import psutil

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    size_gb: float
    last_used: datetime
    is_loaded: bool = False
    load_time: Optional[datetime] = None

@dataclass
class GPUMetrics:
    """GPU memory and utilization metrics."""
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_percent: float
    temperature_c: Optional[float] = None
    timestamp: datetime = None

class OllamaLifecycleManager:
    """Manages Ollama model lifecycle and GPU resources."""
    
    def __init__(self, 
                 ollama_host: str = "http://localhost:11434",
                 max_concurrent_models: int = 2,
                 memory_threshold_mb: int = 1000,
                 model_timeout_seconds: int = 30):
        """
        Initialize the Ollama lifecycle manager.
        
        Args:
            ollama_host: Ollama server URL
            max_concurrent_models: Maximum number of models to keep loaded
            memory_threshold_mb: Free memory threshold to trigger cleanup
            model_timeout_seconds: Timeout for model operations
        """
        self.ollama_host = ollama_host
        self.max_concurrent_models = max_concurrent_models
        self.memory_threshold_mb = memory_threshold_mb
        self.model_timeout_seconds = model_timeout_seconds
        
        # Model registry
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.model_queue: List[str] = []
        self.lock = asyncio.Lock()
        
        # Performance tracking
        self.health_check_failures = 0
        self.last_health_check = None
        self.model_load_times: Dict[str, float] = {}
        
        logger.info(f"OllamaLifecycleManager initialized with max_concurrent_models={max_concurrent_models}")
    
    async def health_check(self) -> bool:
        """
        Check if Ollama server is healthy and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            start_time = time.time()
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            response.raise_for_status()
            
            self.last_health_check = datetime.now()
            self.health_check_failures = 0
            
            health_time = time.time() - start_time
            logger.debug(f"Ollama health check passed in {health_time:.2f}s")
            return True
            
        except Exception as e:
            self.health_check_failures += 1
            logger.warning(f"Ollama health check failed (attempt {self.health_check_failures}): {e}")
            return False
    
    async def get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            logger.debug(f"Found {len(models)} available models: {models}")
            return models
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    async def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """
        Get GPU metrics using nvidia-smi (Windows compatible).
        
        Returns:
            GPUMetrics object or None if GPU not available
        """
        try:
            # Try nvidia-smi command
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                logger.debug("nvidia-smi not available or failed")
                return None
            
            # Parse output: memory.total,memory.used,utilization.gpu,temperature.gpu
            lines = result.stdout.strip().split('\n')
            if not lines or not lines[0]:
                return None
            
            values = lines[0].split(', ')
            if len(values) < 4:
                return None
            
            total_memory = int(values[0])
            used_memory = int(values[1])
            utilization = float(values[2])
            temperature = float(values[3]) if values[3] != 'N/A' else None
            
            metrics = GPUMetrics(
                total_memory_mb=total_memory,
                used_memory_mb=used_memory,
                free_memory_mb=total_memory - used_memory,
                utilization_percent=utilization,
                temperature_c=temperature,
                timestamp=datetime.now()
            )
            
            logger.debug(f"GPU metrics: {used_memory}/{total_memory}MB used, {utilization}% utilization")
            return metrics
            
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            logger.debug(f"GPU monitoring not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error getting GPU metrics: {e}")
            return None
    
    async def get_system_memory(self) -> Dict[str, int]:
        """
        Get system memory usage.
        
        Returns:
            Dictionary with memory metrics in MB
        """
        try:
            memory = psutil.virtual_memory()
            return {
                'total_mb': memory.total // (1024 * 1024),
                'available_mb': memory.available // (1024 * 1024),
                'used_mb': memory.used // (1024 * 1024),
                'percent_used': memory.percent
            }
        except Exception as e:
            logger.warning(f"Failed to get system memory: {e}")
            return {}
    
    async def pre_warm_model(self, model_name: str) -> bool:
        """
        Pre-warm a model by loading it into memory.
        
        Args:
            model_name: Name of the model to pre-warm
            
        Returns:
            True if successful, False otherwise
        """
        async with self.lock:
            if model_name in self.loaded_models:
                logger.debug(f"Model {model_name} already loaded")
                return True
            
            # Check if we're at the concurrent model limit
            if len(self.loaded_models) >= self.max_concurrent_models:
                await self._cleanup_oldest_model()
            
            try:
                start_time = time.time()
                logger.info(f"Pre-warming model: {model_name}")
                
                # Send a simple request to load the model
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "Hello",
                        "stream": False,
                        "options": {"num_predict": 1}
                    },
                    timeout=self.model_timeout_seconds
                )
                response.raise_for_status()
                
                load_time = time.time() - start_time
                self.model_load_times[model_name] = load_time
                
                # Register the model as loaded
                self.loaded_models[model_name] = ModelInfo(
                    name=model_name,
                    size_gb=0.0,  # We don't know the exact size
                    last_used=datetime.now(),
                    is_loaded=True,
                    load_time=datetime.now()
                )
                
                logger.info(f"Model {model_name} pre-warmed successfully in {load_time:.2f}s")
                return True
                
            except Exception as e:
                logger.error(f"Failed to pre-warm model {model_name}: {e}")
                return False
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if successful, False otherwise
        """
        async with self.lock:
            if model_name not in self.loaded_models:
                logger.debug(f"Model {model_name} not loaded")
                return True
            
            try:
                # Ollama doesn't have a direct unload API, but we can stop all models
                # This is a bit aggressive but ensures memory is freed
                response = requests.post(f"{self.ollama_host}/api/stop", timeout=10)
                response.raise_for_status()
                
                # Remove from our registry
                del self.loaded_models[model_name]
                
                logger.info(f"Model {model_name} unloaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload model {model_name}: {e}")
                return False
    
    async def unload_all_models(self) -> bool:
        """
        Unload all models from memory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(f"{self.ollama_host}/api/stop", timeout=10)
            response.raise_for_status()
            
            async with self.lock:
                self.loaded_models.clear()
            
            logger.info("All models unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload all models: {e}")
            return False
    
    async def _cleanup_oldest_model(self) -> None:
        """Remove the oldest unused model to make room for new ones."""
        if not self.loaded_models:
            return
        
        # Find the oldest model
        oldest_model = min(
            self.loaded_models.values(),
            key=lambda m: m.last_used
        )
        
        logger.info(f"Cleaning up oldest model: {oldest_model.name}")
        await self.unload_model(oldest_model.name)
    
    async def get_model_for_inference(self, preferred_models: List[str]) -> Optional[str]:
        """
        Get the best available model for inference.
        
        Args:
            preferred_models: List of preferred model names in order of preference
            
        Returns:
            Model name to use or None if none available
        """
        async with self.lock:
            # Check if any preferred models are already loaded
            for model_name in preferred_models:
                if model_name in self.loaded_models:
                    # Update last used time
                    self.loaded_models[model_name].last_used = datetime.now()
                    logger.debug(f"Using already loaded model: {model_name}")
                    return model_name
            
            # Try to pre-warm the first preferred model
            for model_name in preferred_models:
                if await self.pre_warm_model(model_name):
                    return model_name
            
            logger.warning("No preferred models available for inference")
            return None
    
    async def memory_cleanup_if_needed(self) -> bool:
        """
        Perform memory cleanup if system is under memory pressure.
        
        Returns:
            True if cleanup was performed, False otherwise
        """
        # Check system memory
        memory_info = await self.get_system_memory()
        if not memory_info:
            return False
        
        available_mb = memory_info.get('available_mb', 0)
        
        # Check GPU memory if available
        gpu_metrics = await self.get_gpu_metrics()
        if gpu_metrics and gpu_metrics.free_memory_mb < self.memory_threshold_mb:
            logger.warning(f"GPU memory low: {gpu_metrics.free_memory_mb}MB free")
            await self.unload_all_models()
            return True
        
        # Check system memory
        if available_mb < self.memory_threshold_mb:
            logger.warning(f"System memory low: {available_mb}MB available")
            await self.unload_all_models()
            return True
        
        return False
    
    async def get_status(self) -> Dict:
        """
        Get current status of the lifecycle manager.
        
        Returns:
            Dictionary with status information
        """
        memory_info = await self.get_system_memory()
        gpu_metrics = await self.get_gpu_metrics()
        
        return {
            'ollama_healthy': await self.health_check(),
            'health_check_failures': self.health_check_failures,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'loaded_models': list(self.loaded_models.keys()),
            'loaded_model_count': len(self.loaded_models),
            'max_concurrent_models': self.max_concurrent_models,
            'system_memory': memory_info,
            'gpu_metrics': {
                'total_memory_mb': gpu_metrics.total_memory_mb if gpu_metrics else None,
                'used_memory_mb': gpu_metrics.used_memory_mb if gpu_metrics else None,
                'free_memory_mb': gpu_metrics.free_memory_mb if gpu_metrics else None,
                'utilization_percent': gpu_metrics.utilization_percent if gpu_metrics else None,
                'temperature_c': gpu_metrics.temperature_c if gpu_metrics else None,
            } if gpu_metrics else None,
            'model_load_times': self.model_load_times,
            'timestamp': datetime.now().isoformat()
        }

# Global instance
_lifecycle_manager: Optional[OllamaLifecycleManager] = None

def get_lifecycle_manager() -> OllamaLifecycleManager:
    """Get the global lifecycle manager instance."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = OllamaLifecycleManager()
    return _lifecycle_manager

async def ensure_ollama_healthy() -> bool:
    """Ensure Ollama is healthy before proceeding with AI operations."""
    manager = get_lifecycle_manager()
    return await manager.health_check()

async def pre_warm_models(models: List[str]) -> Dict[str, bool]:
    """Pre-warm multiple models."""
    manager = get_lifecycle_manager()
    results = {}
    for model in models:
        results[model] = await manager.pre_warm_model(model)
    return results

async def cleanup_memory_if_needed() -> bool:
    """Clean up memory if under pressure."""
    manager = get_lifecycle_manager()
    return await manager.memory_cleanup_if_needed()

async def get_ollama_status() -> Dict:
    """Get comprehensive Ollama status."""
    manager = get_lifecycle_manager()
    return await manager.get_status()
