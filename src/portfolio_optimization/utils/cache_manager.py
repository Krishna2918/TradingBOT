"""
Cache management for the Portfolio Optimization Engine.

Provides intelligent caching to reduce API calls and improve performance
while ensuring data freshness and preventing resource conflicts.
"""

import time
import pickle
import hashlib
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from pathlib import Path
import threading

from ..config.settings import get_config
from .logger import get_logger


class CacheManager:
    """
    Thread-safe cache manager for optimization data.
    
    Provides intelligent caching with TTL, size limits, and automatic cleanup
    to reduce API calls and improve performance.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.config = get_config()
        self.logger = get_logger('cache_manager')
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path('cache') / 'portfolio_optimization'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for frequently accessed data
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        
        # Cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0
        }
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        with self._cache_lock:
            # Check memory cache first
            if key in self._memory_cache:
                cache_entry = self._memory_cache[key]
                if self._is_valid(cache_entry):
                    self._stats['hits'] += 1
                    self.logger.debug(f"Cache hit (memory): {key}")
                    return cache_entry['value']
                else:
                    # Remove expired entry
                    del self._memory_cache[key]
            
            # Check disk cache
            cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cache_entry = pickle.load(f)
                    
                    if self._is_valid(cache_entry):
                        # Move to memory cache for faster access
                        self._memory_cache[key] = cache_entry
                        self._stats['hits'] += 1
                        self.logger.debug(f"Cache hit (disk): {key}")
                        return cache_entry['value']
                    else:
                        # Remove expired file
                        cache_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Error reading cache file {cache_file}: {e}")
            
            self._stats['misses'] += 1
            self.logger.debug(f"Cache miss: {key}")
            return default
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None,
        persist_to_disk: bool = True
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds (None for default)
            persist_to_disk: Whether to persist to disk
        """
        if ttl_seconds is None:
            ttl_seconds = 3600  # Default 1 hour
        
        cache_entry = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl_seconds,
            'size_bytes': self._estimate_size(value)
        }
        
        with self._cache_lock:
            # Add to memory cache
            self._memory_cache[key] = cache_entry
            self._stats['size_bytes'] += cache_entry['size_bytes']
            
            # Persist to disk if requested
            if persist_to_disk:
                try:
                    cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_entry, f)
                except Exception as e:
                    self.logger.warning(f"Error writing cache file: {e}")
            
            # Check if we need to evict entries
            self._maybe_evict()
            
            self.logger.debug(f"Cache set: {key} (TTL: {ttl_seconds}s)")
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        with self._cache_lock:
            deleted = False
            
            # Remove from memory cache
            if key in self._memory_cache:
                cache_entry = self._memory_cache[key]
                self._stats['size_bytes'] -= cache_entry['size_bytes']
                del self._memory_cache[key]
                deleted = True
            
            # Remove from disk cache
            cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
            if cache_file.exists():
                cache_file.unlink()
                deleted = True
            
            if deleted:
                self.logger.debug(f"Cache delete: {key}")
            
            return deleted
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._cache_lock:
            # Clear memory cache
            self._memory_cache.clear()
            self._stats['size_bytes'] = 0
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Error deleting cache file {cache_file}: {e}")
            
            self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self._stats['evictions'],
                'memory_entries': len(self._memory_cache),
                'size_bytes': self._stats['size_bytes'],
                'size_mb': self._stats['size_bytes'] / (1024 * 1024)
            }
    
    def cached(self, ttl_seconds: int = 3600, key_func: Optional[Callable] = None):
        """
        Decorator for caching function results.
        
        Args:
            ttl_seconds: Time to live for cached result
            key_func: Function to generate cache key from args/kwargs
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}_{self._hash_args(args, kwargs)}"
                
                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl_seconds)
                return result
            
            return wrapper
        return decorator
    
    def _is_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        age = time.time() - cache_entry['timestamp']
        return age < cache_entry['ttl']
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _hash_args(self, args: tuple, kwargs: dict) -> str:
        """Generate hash for function arguments"""
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate size of object in bytes"""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            return 1024  # Default estimate
    
    def _maybe_evict(self) -> None:
        """Evict entries if cache is too large"""
        max_size_mb = self.config.performance.memory_limit_mb
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if self._stats['size_bytes'] > max_size_bytes:
            # Sort by timestamp (oldest first)
            entries = list(self._memory_cache.items())
            entries.sort(key=lambda x: x[1]['timestamp'])
            
            # Remove oldest entries until under limit
            while (self._stats['size_bytes'] > max_size_bytes * 0.8 and entries):
                key, cache_entry = entries.pop(0)
                if key in self._memory_cache:
                    self._stats['size_bytes'] -= cache_entry['size_bytes']
                    del self._memory_cache[key]
                    self._stats['evictions'] += 1
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired entries"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                with self._cache_lock:
                    # Clean memory cache
                    expired_keys = []
                    for key, cache_entry in self._memory_cache.items():
                        if not self._is_valid(cache_entry):
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        cache_entry = self._memory_cache[key]
                        self._stats['size_bytes'] -= cache_entry['size_bytes']
                        del self._memory_cache[key]
                    
                    # Clean disk cache
                    for cache_file in self.cache_dir.glob("*.pkl"):
                        try:
                            with open(cache_file, 'rb') as f:
                                cache_entry = pickle.load(f)
                            
                            if not self._is_valid(cache_entry):
                                cache_file.unlink()
                        except Exception:
                            # Remove corrupted files
                            cache_file.unlink()
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                        
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")


# Global cache instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager