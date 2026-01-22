# Performance Optimization Guide - Continuous Data Collection System

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [System Optimization](#system-optimization)
3. [Worker Pool Tuning](#worker-pool-tuning)
4. [API Rate Limiting Optimization](#api-rate-limiting-optimization)
5. [Storage and I/O Optimization](#storage-and-io-optimization)
6. [Memory Management](#memory-management)
7. [Network Optimization](#network-optimization)
8. [Monitoring and Metrics](#monitoring-and-metrics)
9. [Scaling Strategies](#scaling-strategies)
10. [Performance Benchmarking](#performance-benchmarking)

## Performance Overview

### Target Performance Metrics

| Metric | Minimum | Recommended | Optimal |
|--------|---------|-------------|---------|
| **Throughput** | 40 stocks/min | 60-70 stocks/min | 80+ stocks/min |
| **API Response Time** | <10 seconds | <5 seconds | <3 seconds |
| **Success Rate** | >85% | >95% | >98% |
| **CPU Usage** | <90% | <70% | <60% |
| **Memory Usage** | <90% | <70% | <60% |
| **Disk I/O Wait** | <20% | <10% | <5% |

### Performance Factors

**Primary Bottlenecks:**
1. API rate limits (75 requests/minute per key)
2. Network latency and connectivity
3. Data validation and processing time
4. Disk I/O for data storage
5. Memory allocation and garbage collection

**Optimization Priorities:**
1. Maximize API key utilization
2. Minimize network overhead
3. Optimize data processing pipeline
4. Reduce memory footprint
5. Improve storage efficiency

## System Optimization

### Hardware Optimization

#### CPU Configuration

**Optimal CPU Settings:**
```yaml
# config/production.yaml
system:
  max_workers: 8              # 1 worker per 1-2 CPU cores
  cpu_affinity: true          # Pin workers to specific cores
  nice_level: 0               # Normal priority (adjust if needed)
  thread_pool_size: 16        # 2x worker count
```

**CPU Affinity Setup:**
```python
# Implement CPU affinity for workers
import os
import psutil

def set_cpu_affinity():
    """Set CPU affinity for optimal performance."""
    cpu_count = psutil.cpu_count()
    worker_count = min(8, cpu_count)
    
    # Distribute workers across cores
    for worker_id in range(worker_count):
        cpu_core = worker_id % cpu_count
        os.sched_setaffinity(0, {cpu_core})
```

**CPU Governor Settings (Linux):**
```bash
# Set CPU governor to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify setting
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

#### Memory Configuration

**Memory Optimization Settings:**
```yaml
# config/production.yaml
memory_management:
  max_memory_per_worker: "2GB"    # Limit per worker
  gc_threshold: 1000              # Garbage collection frequency
  buffer_size: "100MB"            # I/O buffer size
  cache_size: "500MB"             # Data cache size
```

**System Memory Settings (Linux):**
```bash
# Optimize memory settings for long-running processes
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_ratio=15' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' | sudo tee -a /etc/sysctl.conf

# Apply settings
sudo sysctl -p
```

### Operating System Optimization

#### File System Optimization

**Mount Options for Data Directory:**
```bash
# Optimize mount options for performance
sudo mount -o remount,noatime,nodiratime /path/to/data/directory

# For permanent setting, add to /etc/fstab:
# /dev/sdb1 /path/to/data ext4 defaults,noatime,nodiratime 0 2
```

**File System Tuning:**
```bash
# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize ext4 file system
sudo tune2fs -o journal_data_writeback /dev/your_disk
```

#### Network Stack Optimization

**TCP Settings:**
```bash
# Optimize TCP settings for high-throughput applications
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 16777216' | sudo tee -a /etc/sysctl.conf

sudo sysctl -p
```

## Worker Pool Tuning

### Optimal Worker Count Calculation

**Dynamic Worker Calculation:**
```python
import psutil
import math

def calculate_optimal_workers():
    """Calculate optimal worker count based on system resources."""
    
    # Get system resources
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Calculate based on different constraints
    cpu_based = cpu_count  # 1 worker per core
    memory_based = int(memory_gb / 2)  # 2GB per worker
    api_based = 4 * 75 / 60 * 60  # 4 API keys * 75 req/min
    
    # Take the minimum to avoid bottlenecks
    optimal = min(cpu_based, memory_based, 16)  # Cap at 16
    
    print(f"CPU cores: {cpu_count}")
    print(f"Memory: {memory_gb:.1f} GB")
    print(f"CPU-based workers: {cpu_based}")
    print(f"Memory-based workers: {memory_based}")
    print(f"Recommended workers: {optimal}")
    
    return optimal

# Run calculation
optimal_workers = calculate_optimal_workers()
```

### Worker Pool Configuration

**Advanced Worker Pool Settings:**
```yaml
# config/production.yaml
worker_pool:
  max_workers: 8
  min_workers: 4
  worker_timeout: 300           # 5 minutes
  task_timeout: 120             # 2 minutes per stock
  queue_size: 100               # Task queue size
  
  # Dynamic scaling
  scale_up_threshold: 0.8       # Scale up when 80% busy
  scale_down_threshold: 0.3     # Scale down when 30% busy
  scale_check_interval: 60      # Check every minute
  
  # Worker lifecycle
  max_tasks_per_worker: 1000    # Restart worker after 1000 tasks
  worker_memory_limit: "2GB"    # Restart if memory exceeds limit
```

**Worker Pool Implementation:**
```python
import asyncio
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

class OptimizedWorkerPool:
    def __init__(self, config):
        self.config = config
        self.executor = None
        self.active_workers = 0
        self.task_queue = asyncio.Queue(maxsize=config.queue_size)
        
    async def start(self):
        """Start worker pool with optimal configuration."""
        optimal_workers = self.calculate_optimal_workers()
        self.executor = ThreadPoolExecutor(
            max_workers=optimal_workers,
            thread_name_prefix="DataCollector"
        )
        
        # Start monitoring task
        asyncio.create_task(self.monitor_performance())
    
    def calculate_optimal_workers(self) -> int:
        """Calculate optimal worker count dynamically."""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Consider current system load
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Adjust based on current load
        load_factor = 1.0
        if cpu_percent > 70:
            load_factor *= 0.8
        if memory_percent > 70:
            load_factor *= 0.8
            
        optimal = int(min(cpu_count, memory_gb / 2) * load_factor)
        return max(2, min(optimal, 16))  # Between 2 and 16 workers
    
    async def monitor_performance(self):
        """Monitor and adjust worker pool performance."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            # Get current metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            queue_size = self.task_queue.qsize()
            
            # Adjust worker count if needed
            if queue_size > 50 and cpu_percent < 70:
                # Scale up
                new_workers = min(self.executor._max_workers + 2, 16)
                self.executor._max_workers = new_workers
                
            elif queue_size < 10 and cpu_percent < 30:
                # Scale down
                new_workers = max(self.executor._max_workers - 1, 2)
                self.executor._max_workers = new_workers
```

### Batch Processing Optimization

**Optimal Batch Sizing:**
```python
def calculate_optimal_batch_size():
    """Calculate optimal batch size based on system performance."""
    
    # Base batch size
    base_batch_size = 20
    
    # Adjust based on system resources
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Scale batch size with resources
    cpu_factor = min(cpu_count / 8, 2.0)  # Max 2x scaling
    memory_factor = min(memory_gb / 16, 2.0)  # Max 2x scaling
    
    optimal_batch = int(base_batch_size * min(cpu_factor, memory_factor))
    
    # Consider API rate limits
    # With 4 keys at 75 req/min each = 300 req/min total
    # Target 60-70 stocks/min means ~4-5 requests per stock
    # So batch size should allow efficient API utilization
    
    return min(optimal_batch, 50)  # Cap at 50 for memory reasons

# Dynamic batch sizing
batch_size = calculate_optimal_batch_size()
```

## API Rate Limiting Optimization

### Advanced Rate Limiting Strategy

**Intelligent API Key Rotation:**
```python
import time
from collections import deque
from typing import Dict, List

class OptimizedRateLimiter:
    def __init__(self, api_keys: List[str], rate_limit: int = 75):
        self.api_keys = api_keys
        self.rate_limit = rate_limit  # requests per minute
        self.key_usage = {i: deque() for i in range(len(api_keys))}
        self.current_key_index = 0
        
        # Advanced tracking
        self.key_performance = {i: {'avg_response_time': 0, 'error_count': 0} 
                               for i in range(len(api_keys))}
        
    def get_best_available_key(self) -> int:
        """Get the best available API key based on usage and performance."""
        current_time = time.time()
        
        # Clean old usage records (older than 1 minute)
        for key_id in self.key_usage:
            while (self.key_usage[key_id] and 
                   current_time - self.key_usage[key_id][0] > 60):
                self.key_usage[key_id].popleft()
        
        # Find key with lowest usage and best performance
        best_key = None
        best_score = float('inf')
        
        for key_id in range(len(self.api_keys)):
            usage_count = len(self.key_usage[key_id])
            
            # Skip if at rate limit
            if usage_count >= self.rate_limit:
                continue
                
            # Calculate score (lower is better)
            usage_factor = usage_count / self.rate_limit
            performance_factor = self.key_performance[key_id]['avg_response_time']
            error_factor = self.key_performance[key_id]['error_count'] * 0.1
            
            score = usage_factor + performance_factor + error_factor
            
            if score < best_score:
                best_score = score
                best_key = key_id
        
        return best_key
    
    async def acquire_rate_limit(self) -> int:
        """Acquire rate limit slot and return best API key index."""
        while True:
            key_id = self.get_best_available_key()
            
            if key_id is not None:
                # Record usage
                self.key_usage[key_id].append(time.time())
                return key_id
            
            # All keys at limit, wait for next available slot
            await asyncio.sleep(1)
    
    def record_performance(self, key_id: int, response_time: float, 
                          success: bool):
        """Record API key performance metrics."""
        perf = self.key_performance[key_id]
        
        # Update average response time (exponential moving average)
        alpha = 0.1
        perf['avg_response_time'] = (alpha * response_time + 
                                   (1 - alpha) * perf['avg_response_time'])
        
        # Update error count
        if not success:
            perf['error_count'] += 1
        else:
            # Decay error count on success
            perf['error_count'] = max(0, perf['error_count'] - 0.1)
```

### Burst Handling and Smoothing

**Burst Management:**
```yaml
# config/production.yaml
rate_limiting:
  base_rate_limit: 73           # Slightly below 75 for safety
  burst_allowance: 5            # Allow short bursts
  burst_window: 10              # 10 second burst window
  smoothing_factor: 0.9         # Smooth out request timing
  
  # Adaptive rate limiting
  adaptive_enabled: true
  min_rate_limit: 60            # Never go below 60/min
  max_rate_limit: 75            # Never exceed 75/min
  adjustment_factor: 0.1        # 10% adjustments
```

**Implementation:**
```python
class AdaptiveRateLimiter:
    def __init__(self, config):
        self.base_rate = config.base_rate_limit
        self.current_rate = self.base_rate
        self.burst_tokens = config.burst_allowance
        self.max_burst = config.burst_allowance
        
        # Performance tracking
        self.success_rate = 1.0
        self.avg_response_time = 0.0
        
    async def adaptive_adjustment(self):
        """Adjust rate limits based on performance."""
        while True:
            await asyncio.sleep(60)  # Adjust every minute
            
            # Increase rate if performing well
            if self.success_rate > 0.98 and self.avg_response_time < 3.0:
                self.current_rate = min(self.current_rate * 1.05, 75)
                
            # Decrease rate if having issues
            elif self.success_rate < 0.95 or self.avg_response_time > 8.0:
                self.current_rate = max(self.current_rate * 0.95, 60)
            
            print(f"Adjusted rate limit to: {self.current_rate:.1f}/min")
```

## Storage and I/O Optimization

### Parquet File Optimization

**Optimal Parquet Settings:**
```yaml
# config/production.yaml
storage:
  format: "parquet"
  compression: "snappy"          # Fast compression/decompression
  compression_level: 6           # Balance size vs speed
  
  # Parquet-specific settings
  row_group_size: 50000         # Optimize for read performance
  page_size: 1048576            # 1MB pages
  dictionary_encoding: true      # Enable dictionary encoding
  
  # Write optimization
  write_batch_size: 1000        # Write in batches
  buffer_size: "100MB"          # Large write buffers
  sync_interval: 30             # Sync every 30 seconds
```

**Optimized Storage Implementation:**
```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

class OptimizedStorage:
    def __init__(self, config):
        self.config = config
        self.write_buffer = []
        self.buffer_size = config.write_batch_size
        
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for storage."""
        
        # Optimize data types
        for col in df.columns:
            if df[col].dtype == 'float64':
                # Use float32 if precision allows
                if df[col].max() < 3.4e38 and df[col].min() > -3.4e38:
                    df[col] = df[col].astype('float32')
                    
            elif df[col].dtype == 'int64':
                # Use smaller int types if possible
                if df[col].max() < 2147483647 and df[col].min() > -2147483648:
                    df[col] = df[col].astype('int32')
        
        # Sort by date for better compression
        if 'date' in df.columns:
            df = df.sort_values('date')
            
        return df
    
    async def write_optimized(self, symbol: str, data: pd.DataFrame):
        """Write data with optimization."""
        
        # Optimize DataFrame
        data = self.optimize_dataframe(data)
        
        # Create Parquet table with optimization
        table = pa.Table.from_pandas(data)
        
        # Write with optimal settings
        file_path = Path(self.config.data_directory) / f"{symbol}.parquet"
        
        pq.write_table(
            table, 
            file_path,
            compression='snappy',
            compression_level=6,
            use_dictionary=True,
            row_group_size=50000,
            data_page_size=1048576
        )
```

### Asynchronous I/O

**Async File Operations:**
```python
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncFileManager:
    def __init__(self, max_concurrent_writes: int = 10):
        self.write_semaphore = asyncio.Semaphore(max_concurrent_writes)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def write_data_async(self, symbol: str, data: pd.DataFrame):
        """Write data asynchronously."""
        async with self.write_semaphore:
            loop = asyncio.get_event_loop()
            
            # Run CPU-intensive operations in thread pool
            await loop.run_in_executor(
                self.executor,
                self._write_parquet_sync,
                symbol,
                data
            )
    
    def _write_parquet_sync(self, symbol: str, data: pd.DataFrame):
        """Synchronous Parquet write operation."""
        file_path = f"data/raw/{symbol}.parquet"
        data.to_parquet(
            file_path,
            compression='snappy',
            engine='pyarrow'
        )
```

### Disk I/O Optimization

**I/O Scheduling:**
```python
import asyncio
from collections import deque
from typing import Callable, Any

class IOScheduler:
    def __init__(self, max_concurrent_io: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent_io)
        self.write_queue = deque()
        self.read_queue = deque()
        
    async def schedule_write(self, operation: Callable, *args, **kwargs):
        """Schedule write operation with priority."""
        async with self.semaphore:
            # Batch small writes together
            if len(self.write_queue) < 10:
                self.write_queue.append((operation, args, kwargs))
            else:
                # Execute batch
                await self._execute_write_batch()
                
    async def _execute_write_batch(self):
        """Execute batched write operations."""
        batch = list(self.write_queue)
        self.write_queue.clear()
        
        # Execute all writes in batch
        tasks = []
        for operation, args, kwargs in batch:
            task = asyncio.create_task(operation(*args, **kwargs))
            tasks.append(task)
            
        await asyncio.gather(*tasks, return_exceptions=True)
```

## Memory Management

### Garbage Collection Optimization

**GC Tuning:**
```python
import gc
import psutil
import asyncio

class MemoryManager:
    def __init__(self, config):
        self.config = config
        self.memory_threshold = config.memory_threshold  # e.g., 0.8 (80%)
        self.gc_frequency = config.gc_frequency  # e.g., 1000 operations
        self.operation_count = 0
        
    async def start_monitoring(self):
        """Start memory monitoring task."""
        asyncio.create_task(self._monitor_memory())
        
    async def _monitor_memory(self):
        """Monitor memory usage and trigger cleanup."""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            memory_percent = psutil.virtual_memory().percent / 100
            
            if memory_percent > self.memory_threshold:
                await self._cleanup_memory()
                
    async def _cleanup_memory(self):
        """Perform memory cleanup."""
        print(f"Memory usage high ({psutil.virtual_memory().percent}%), cleaning up...")
        
        # Force garbage collection
        collected = gc.collect()
        print(f"Garbage collected: {collected} objects")
        
        # Clear caches if implemented
        await self._clear_caches()
        
    async def _clear_caches(self):
        """Clear application caches."""
        # Implement cache clearing logic
        pass
    
    def track_operation(self):
        """Track operations for periodic GC."""
        self.operation_count += 1
        
        if self.operation_count >= self.gc_frequency:
            gc.collect()
            self.operation_count = 0
```

### Memory Pool Management

**Object Pool Implementation:**
```python
from typing import Generic, TypeVar, List, Optional
import asyncio

T = TypeVar('T')

class ObjectPool(Generic[T]):
    def __init__(self, factory: Callable[[], T], max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self.pool: List[T] = []
        self.lock = asyncio.Lock()
        
    async def acquire(self) -> T:
        """Acquire object from pool."""
        async with self.lock:
            if self.pool:
                return self.pool.pop()
            else:
                return self.factory()
                
    async def release(self, obj: T):
        """Return object to pool."""
        async with self.lock:
            if len(self.pool) < self.max_size:
                # Reset object state if needed
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)

# Usage example
class DataFramePool:
    def __init__(self):
        self.pool = ObjectPool(lambda: pd.DataFrame(), max_size=50)
        
    async def get_dataframe(self) -> pd.DataFrame:
        return await self.pool.acquire()
        
    async def return_dataframe(self, df: pd.DataFrame):
        df.drop(df.index, inplace=True)  # Clear data
        await self.pool.release(df)
```

## Network Optimization

### Connection Pooling

**HTTP Connection Pool:**
```python
import aiohttp
import asyncio
from typing import Optional

class OptimizedHTTPClient:
    def __init__(self, config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def start(self):
        """Initialize HTTP client with optimized settings."""
        
        # Connection pool settings
        connector = aiohttp.TCPConnector(
            limit=100,              # Total connection pool size
            limit_per_host=20,      # Connections per host
            ttl_dns_cache=300,      # DNS cache TTL
            use_dns_cache=True,     # Enable DNS caching
            keepalive_timeout=30,   # Keep-alive timeout
            enable_cleanup_closed=True
        )
        
        # Timeout settings
        timeout = aiohttp.ClientTimeout(
            total=30,               # Total timeout
            connect=10,             # Connection timeout
            sock_read=20            # Socket read timeout
        )
        
        # Create session
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'DataCollector/1.0',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
        )
        
    async def close(self):
        """Close HTTP client."""
        if self.session:
            await self.session.close()
            
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Optimized GET request."""
        if not self.session:
            await self.start()
            
        return await self.session.get(url, **kwargs)
```

### Request Optimization

**Request Batching and Pipelining:**
```python
import asyncio
from typing import List, Dict, Any

class RequestBatcher:
    def __init__(self, batch_size: int = 10, batch_timeout: float = 1.0):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests = []
        self.batch_timer = None
        
    async def add_request(self, request_data: Dict[str, Any]) -> Any:
        """Add request to batch."""
        future = asyncio.Future()
        self.pending_requests.append((request_data, future))
        
        # Start batch timer if not already running
        if not self.batch_timer:
            self.batch_timer = asyncio.create_task(
                self._batch_timeout_handler()
            )
            
        # Execute batch if full
        if len(self.pending_requests) >= self.batch_size:
            await self._execute_batch()
            
        return await future
        
    async def _batch_timeout_handler(self):
        """Handle batch timeout."""
        await asyncio.sleep(self.batch_timeout)
        if self.pending_requests:
            await self._execute_batch()
            
    async def _execute_batch(self):
        """Execute batched requests."""
        if not self.pending_requests:
            return
            
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        self.batch_timer = None
        
        # Execute all requests concurrently
        tasks = []
        for request_data, future in batch:
            task = asyncio.create_task(
                self._execute_single_request(request_data, future)
            )
            tasks.append(task)
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _execute_single_request(self, request_data: Dict[str, Any], 
                                    future: asyncio.Future):
        """Execute single request."""
        try:
            # Implement actual request logic here
            result = await self._make_request(request_data)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
```

## Monitoring and Metrics

### Performance Metrics Collection

**Comprehensive Metrics:**
```python
import time
import psutil
import asyncio
from collections import defaultdict, deque
from typing import Dict, List

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(deque)
        self.start_time = time.time()
        
        # Performance counters
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # Timing metrics
        self.response_times = deque(maxlen=1000)
        self.processing_times = deque(maxlen=1000)
        
    async def start_monitoring(self):
        """Start performance monitoring."""
        asyncio.create_task(self._collect_system_metrics())
        asyncio.create_task(self._calculate_throughput())
        
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        while True:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_usage'].append(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].append(memory.percent)
            self.metrics['memory_available'].append(memory.available)
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.metrics['disk_read_bytes'].append(disk_io.read_bytes)
                self.metrics['disk_write_bytes'].append(disk_io.write_bytes)
            
            # Network I/O metrics
            net_io = psutil.net_io_counters()
            if net_io:
                self.metrics['network_sent'].append(net_io.bytes_sent)
                self.metrics['network_recv'].append(net_io.bytes_recv)
            
            # Keep only last 1000 measurements
            for metric_name in self.metrics:
                if len(self.metrics[metric_name]) > 1000:
                    self.metrics[metric_name].popleft()
                    
            await asyncio.sleep(5)  # Collect every 5 seconds
            
    async def _calculate_throughput(self):
        """Calculate throughput metrics."""
        while True:
            await asyncio.sleep(60)  # Calculate every minute
            
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Calculate rates
            throughput = self.success_count / (elapsed_time / 60)  # per minute
            success_rate = self.success_count / max(self.request_count, 1)
            
            # Average response time
            avg_response_time = (sum(self.response_times) / 
                               max(len(self.response_times), 1))
            
            print(f"Performance Metrics:")
            print(f"  Throughput: {throughput:.1f} stocks/minute")
            print(f"  Success Rate: {success_rate:.1%}")
            print(f"  Avg Response Time: {avg_response_time:.2f}s")
            print(f"  CPU Usage: {self.get_avg_metric('cpu_usage'):.1f}%")
            print(f"  Memory Usage: {self.get_avg_metric('memory_usage'):.1f}%")
            
    def record_request(self, response_time: float, success: bool):
        """Record request metrics."""
        self.request_count += 1
        self.response_times.append(response_time)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            
    def get_avg_metric(self, metric_name: str) -> float:
        """Get average value for a metric."""
        values = self.metrics.get(metric_name, [])
        return sum(values) / max(len(values), 1) if values else 0.0
```

### Real-time Performance Dashboard

**Dashboard Implementation:**
```python
import asyncio
import json
from datetime import datetime

class PerformanceDashboard:
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        
    async def start_dashboard(self, update_interval: int = 10):
        """Start real-time dashboard updates."""
        while True:
            await self._update_dashboard()
            await asyncio.sleep(update_interval)
            
    async def _update_dashboard(self):
        """Update dashboard with current metrics."""
        
        # Clear screen (for terminal dashboard)
        print("\033[2J\033[H")
        
        print("=" * 60)
        print("CONTINUOUS DATA COLLECTION - PERFORMANCE DASHBOARD")
        print("=" * 60)
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System metrics
        print("SYSTEM METRICS:")
        print(f"  CPU Usage:    {self.monitor.get_avg_metric('cpu_usage'):6.1f}%")
        print(f"  Memory Usage: {self.monitor.get_avg_metric('memory_usage'):6.1f}%")
        print(f"  Disk I/O:     {self._format_bytes(self.monitor.get_avg_metric('disk_write_bytes'))}/s")
        print()
        
        # Collection metrics
        elapsed_time = time.time() - self.monitor.start_time
        throughput = self.monitor.success_count / (elapsed_time / 60)
        success_rate = self.monitor.success_count / max(self.monitor.request_count, 1)
        
        print("COLLECTION METRICS:")
        print(f"  Throughput:   {throughput:6.1f} stocks/minute")
        print(f"  Success Rate: {success_rate:6.1%}")
        print(f"  Total Stocks: {self.monitor.success_count:6d}")
        print(f"  Failed:       {self.monitor.error_count:6d}")
        print()
        
        # Performance metrics
        if self.monitor.response_times:
            avg_response = sum(self.monitor.response_times) / len(self.monitor.response_times)
            print("PERFORMANCE METRICS:")
            print(f"  Avg Response: {avg_response:6.2f}s")
            print(f"  Min Response: {min(self.monitor.response_times):6.2f}s")
            print(f"  Max Response: {max(self.monitor.response_times):6.2f}s")
        
        print("=" * 60)
        
    def _format_bytes(self, bytes_value: float) -> str:
        """Format bytes for display."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024
        return f"{bytes_value:.1f}TB"
```

## Scaling Strategies

### Horizontal Scaling

**Multi-Instance Deployment:**
```yaml
# config/instance_1.yaml
instance:
  id: 1
  stock_range: [0, 350]        # First 350 stocks
  port: 8001
  
# config/instance_2.yaml  
instance:
  id: 2
  stock_range: [350, 700]      # Next 350 stocks
  port: 8002
  
# config/instance_3.yaml
instance:
  id: 3
  stock_range: [700, 1050]     # Next 350 stocks
  port: 8003
  
# config/instance_4.yaml
instance:
  id: 4
  stock_range: [1050, 1400]    # Last 350 stocks
  port: 8004
```

**Load Balancer Configuration:**
```python
import asyncio
import aiohttp
from typing import List, Dict

class LoadBalancer:
    def __init__(self, instances: List[Dict[str, str]]):
        self.instances = instances
        self.current_instance = 0
        
    async def distribute_work(self, stock_list: List[str]):
        """Distribute work across instances."""
        
        # Split stock list among instances
        chunk_size = len(stock_list) // len(self.instances)
        
        tasks = []
        for i, instance in enumerate(self.instances):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(self.instances) - 1 else len(stock_list)
            
            stock_chunk = stock_list[start_idx:end_idx]
            
            task = asyncio.create_task(
                self._start_instance_collection(instance, stock_chunk)
            )
            tasks.append(task)
            
        # Wait for all instances to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
        
    async def _start_instance_collection(self, instance: Dict[str, str], 
                                       stocks: List[str]):
        """Start collection on specific instance."""
        async with aiohttp.ClientSession() as session:
            url = f"http://{instance['host']}:{instance['port']}/start"
            data = {'stocks': stocks}
            
            async with session.post(url, json=data) as response:
                return await response.json()
```

### Vertical Scaling

**Dynamic Resource Allocation:**
```python
import psutil
import asyncio

class ResourceScaler:
    def __init__(self, config):
        self.config = config
        self.current_workers = config.initial_workers
        self.max_workers = config.max_workers
        self.min_workers = config.min_workers
        
    async def start_scaling(self):
        """Start dynamic resource scaling."""
        asyncio.create_task(self._monitor_and_scale())
        
    async def _monitor_and_scale(self):
        """Monitor system resources and scale accordingly."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            # Get current resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Scaling decisions
            if cpu_percent < 50 and memory_percent < 50:
                # Scale up if resources available
                new_workers = min(self.current_workers + 1, self.max_workers)
                if new_workers != self.current_workers:
                    await self._scale_workers(new_workers)
                    
            elif cpu_percent > 80 or memory_percent > 80:
                # Scale down if resources constrained
                new_workers = max(self.current_workers - 1, self.min_workers)
                if new_workers != self.current_workers:
                    await self._scale_workers(new_workers)
                    
    async def _scale_workers(self, new_count: int):
        """Scale worker count."""
        print(f"Scaling workers from {self.current_workers} to {new_count}")
        
        # Implement worker scaling logic
        # This would interact with your worker pool
        
        self.current_workers = new_count
```

## Performance Benchmarking

### Benchmark Suite

**Comprehensive Performance Test:**
```python
import asyncio
import time
import statistics
from typing import List, Dict, Any

class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
        
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        
        print("Starting Performance Benchmark...")
        
        # Test categories
        benchmarks = [
            ("API Performance", self._benchmark_api_performance),
            ("Data Processing", self._benchmark_data_processing),
            ("Storage Performance", self._benchmark_storage),
            ("Memory Usage", self._benchmark_memory),
            ("Concurrent Operations", self._benchmark_concurrency)
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"\nRunning {name} benchmark...")
            result = await benchmark_func()
            self.results[name] = result
            self._print_benchmark_result(name, result)
            
        return self.results
        
    async def _benchmark_api_performance(self) -> Dict[str, float]:
        """Benchmark API performance."""
        
        # Test API response times
        response_times = []
        
        for _ in range(50):  # 50 test requests
            start_time = time.time()
            
            # Simulate API request
            await asyncio.sleep(0.1)  # Replace with actual API call
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            
        return {
            'avg_response_time': statistics.mean(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'median_response_time': statistics.median(response_times),
            'std_dev': statistics.stdev(response_times)
        }
        
    async def _benchmark_data_processing(self) -> Dict[str, float]:
        """Benchmark data processing performance."""
        
        import pandas as pd
        import numpy as np
        
        # Generate test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=5000),
            'open': np.random.random(5000) * 100,
            'high': np.random.random(5000) * 100,
            'low': np.random.random(5000) * 100,
            'close': np.random.random(5000) * 100,
            'volume': np.random.randint(1000, 1000000, 5000)
        })
        
        processing_times = []
        
        for _ in range(20):  # 20 processing iterations
            start_time = time.time()
            
            # Simulate data processing
            processed_data = test_data.copy()
            processed_data['returns'] = processed_data['close'].pct_change()
            processed_data['sma_20'] = processed_data['close'].rolling(20).mean()
            processed_data = processed_data.dropna()
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
        return {
            'avg_processing_time': statistics.mean(processing_times),
            'records_per_second': 5000 / statistics.mean(processing_times)
        }
        
    async def _benchmark_storage(self) -> Dict[str, float]:
        """Benchmark storage performance."""
        
        import pandas as pd
        import tempfile
        import os
        
        # Generate test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=5000),
            'value': np.random.random(5000)
        })
        
        write_times = []
        read_times = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(10):  # 10 write/read cycles
                file_path = os.path.join(temp_dir, f'test_{i}.parquet')
                
                # Write benchmark
                start_time = time.time()
                test_data.to_parquet(file_path, compression='snappy')
                write_time = time.time() - start_time
                write_times.append(write_time)
                
                # Read benchmark
                start_time = time.time()
                read_data = pd.read_parquet(file_path)
                read_time = time.time() - start_time
                read_times.append(read_time)
                
        return {
            'avg_write_time': statistics.mean(write_times),
            'avg_read_time': statistics.mean(read_times),
            'write_throughput_mb_s': (test_data.memory_usage(deep=True).sum() / 1024 / 1024) / statistics.mean(write_times),
            'read_throughput_mb_s': (test_data.memory_usage(deep=True).sum() / 1024 / 1024) / statistics.mean(read_times)
        }
        
    async def _benchmark_memory(self) -> Dict[str, float]:
        """Benchmark memory usage."""
        
        import psutil
        import gc
        
        # Get baseline memory
        gc.collect()
        baseline_memory = psutil.virtual_memory().used
        
        # Allocate test data
        test_objects = []
        for _ in range(1000):
            test_objects.append([i for i in range(1000)])
            
        peak_memory = psutil.virtual_memory().used
        
        # Clean up
        del test_objects
        gc.collect()
        final_memory = psutil.virtual_memory().used
        
        return {
            'baseline_memory_mb': baseline_memory / 1024 / 1024,
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'memory_increase_mb': (peak_memory - baseline_memory) / 1024 / 1024,
            'memory_cleanup_mb': (peak_memory - final_memory) / 1024 / 1024
        }
        
    async def _benchmark_concurrency(self) -> Dict[str, float]:
        """Benchmark concurrent operations."""
        
        async def dummy_task(task_id: int) -> float:
            """Dummy async task for concurrency testing."""
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate work
            return time.time() - start_time
            
        # Test different concurrency levels
        concurrency_results = {}
        
        for concurrency in [1, 5, 10, 20, 50]:
            start_time = time.time()
            
            tasks = [dummy_task(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            concurrency_results[f'concurrency_{concurrency}'] = {
                'total_time': total_time,
                'tasks_per_second': concurrency / total_time
            }
            
        return concurrency_results
        
    def _print_benchmark_result(self, name: str, result: Dict[str, Any]):
        """Print benchmark results."""
        print(f"\n{name} Results:")
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value:.3f}")
            else:
                print(f"  {key}: {value:.3f}")

# Usage
async def run_benchmark():
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_full_benchmark()
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    # Print summary recommendations
    api_avg = results['API Performance']['avg_response_time']
    if api_avg < 3.0:
        print("✓ API Performance: Excellent")
    elif api_avg < 5.0:
        print("⚠ API Performance: Good")
    else:
        print("✗ API Performance: Needs optimization")
        
    # Add more summary analysis...

if __name__ == "__main__":
    asyncio.run(run_benchmark())
```

This comprehensive performance optimization guide provides detailed strategies for maximizing the efficiency of your continuous data collection system. Implement these optimizations incrementally and monitor their impact on system performance.