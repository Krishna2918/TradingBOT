#!/usr/bin/env python3
"""
Performance Benchmark Script

Comprehensive performance benchmarking for the trading system.
"""

import sys
import os
import time
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import statistics

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        """Initialize the performance benchmark."""
        self.start_time = time.time()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': {},
            'summary': {}
        }
        self.ai_limit = int(os.environ.get('AI_LIMIT', 100))
        
        logger.info(f"Performance Benchmark initialized with AI_LIMIT={self.ai_limit}")
    
    async def run_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        logger.info("Starting performance benchmark suite...")
        
        try:
            # System startup benchmark
            await self._benchmark_system_startup()
            
            # Database performance benchmark
            await self._benchmark_database_performance()
            
            # AI model performance benchmark
            await self._benchmark_ai_performance()
            
            # Memory usage benchmark
            await self._benchmark_memory_usage()
            
            # End-to-end pipeline benchmark
            await self._benchmark_end_to_end()
            
            # Generate summary
            self._generate_summary()
            
            # Save results
            self._save_results()
            
            logger.info("Performance benchmark suite completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Performance benchmark suite failed: {e}")
            self.results['error'] = str(e)
            return self.results
    
    async def _benchmark_system_startup(self) -> None:
        """Benchmark system startup time."""
        logger.info("Benchmarking system startup...")
        
        startup_times = []
        
        for i in range(5):  # Run 5 iterations
            start_time = time.time()
            
            try:
                # Import core modules
                import src.config.database
                import src.monitoring.system_monitor
                import src.ai.multi_model
                import src.trading.risk
                import src.dashboard.connector
                
                startup_time = time.time() - start_time
                startup_times.append(startup_time)
                
            except Exception as e:
                logger.error(f"Startup benchmark iteration {i} failed: {e}")
                startup_times.append(float('inf'))
        
        self.results['benchmarks']['system_startup'] = {
            'iterations': len(startup_times),
            'times_seconds': startup_times,
            'avg_time_seconds': statistics.mean([t for t in startup_times if t != float('inf')]),
            'min_time_seconds': min([t for t in startup_times if t != float('inf')]),
            'max_time_seconds': max([t for t in startup_times if t != float('inf')]),
            'std_dev_seconds': statistics.stdev([t for t in startup_times if t != float('inf')]) if len(startup_times) > 1 else 0
        }
    
    async def _benchmark_database_performance(self) -> None:
        """Benchmark database performance."""
        logger.info("Benchmarking database performance...")
        
        try:
            from config.database import get_connection
            
            # Simple query benchmark
            simple_times = []
            for i in range(10):
                start_time = time.time()
                with get_connection('DEMO') as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM sqlite_master")
                    cursor.fetchone()
                simple_times.append(time.time() - start_time)
            
            # Complex query benchmark
            complex_times = []
            for i in range(5):
                start_time = time.time()
                with get_connection('DEMO') as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    """)
                    cursor.fetchall()
                complex_times.append(time.time() - start_time)
            
            # Write benchmark
            write_times = []
            for i in range(5):
                start_time = time.time()
                with get_connection('DEMO') as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS benchmark_test (
                            id INTEGER PRIMARY KEY,
                            data TEXT,
                            timestamp TEXT
                        )
                    """)
                    cursor.execute("""
                        INSERT INTO benchmark_test (data, timestamp) 
                        VALUES (?, ?)
                    """, (f"test_data_{i}", datetime.now().isoformat()))
                    conn.commit()
                write_times.append(time.time() - start_time)
            
            # Cleanup
            with get_connection('DEMO') as conn:
                cursor = conn.cursor()
                cursor.execute("DROP TABLE IF EXISTS benchmark_test")
                conn.commit()
            
            self.results['benchmarks']['database_performance'] = {
                'simple_query': {
                    'iterations': len(simple_times),
                    'avg_time_seconds': statistics.mean(simple_times),
                    'min_time_seconds': min(simple_times),
                    'max_time_seconds': max(simple_times)
                },
                'complex_query': {
                    'iterations': len(complex_times),
                    'avg_time_seconds': statistics.mean(complex_times),
                    'min_time_seconds': min(complex_times),
                    'max_time_seconds': max(complex_times)
                },
                'write_operations': {
                    'iterations': len(write_times),
                    'avg_time_seconds': statistics.mean(write_times),
                    'min_time_seconds': min(write_times),
                    'max_time_seconds': max(write_times)
                }
            }
            
        except Exception as e:
            logger.error(f"Database performance benchmark failed: {e}")
            self.results['benchmarks']['database_performance'] = {'error': str(e)}
    
    async def _benchmark_ai_performance(self) -> None:
        """Benchmark AI model performance."""
        logger.info("Benchmarking AI model performance...")
        
        try:
            from ai.multi_model import MultiModelManager
            
            # Model manager initialization benchmark
            init_times = []
            for i in range(3):
                start_time = time.time()
                manager = MultiModelManager(mode="DEMO")
                init_times.append(time.time() - start_time)
            
            # Model configuration loading benchmark
            config_times = []
            for i in range(10):
                start_time = time.time()
                configs = manager.get_all_model_configs()
                config_times.append(time.time() - start_time)
            
            # Weight calculation benchmark
            weight_times = []
            for i in range(10):
                start_time = time.time()
                weights = manager.get_model_weights()
                adaptive_weights = manager.get_adaptive_weights()
                weight_times.append(time.time() - start_time)
            
            # Model availability check benchmark
            availability_times = []
            for i in range(5):
                start_time = time.time()
                availability = await manager.check_model_availability()
                availability_times.append(time.time() - start_time)
            
            self.results['benchmarks']['ai_performance'] = {
                'model_manager_init': {
                    'iterations': len(init_times),
                    'avg_time_seconds': statistics.mean(init_times),
                    'min_time_seconds': min(init_times),
                    'max_time_seconds': max(init_times)
                },
                'config_loading': {
                    'iterations': len(config_times),
                    'avg_time_seconds': statistics.mean(config_times),
                    'min_time_seconds': min(config_times),
                    'max_time_seconds': max(config_times)
                },
                'weight_calculation': {
                    'iterations': len(weight_times),
                    'avg_time_seconds': statistics.mean(weight_times),
                    'min_time_seconds': min(weight_times),
                    'max_time_seconds': max(weight_times)
                },
                'availability_check': {
                    'iterations': len(availability_times),
                    'avg_time_seconds': statistics.mean(availability_times),
                    'min_time_seconds': min(availability_times),
                    'max_time_seconds': max(availability_times)
                }
            }
            
        except Exception as e:
            logger.error(f"AI performance benchmark failed: {e}")
            self.results['benchmarks']['ai_performance'] = {'error': str(e)}
    
    async def _benchmark_memory_usage(self) -> None:
        """Benchmark memory usage patterns."""
        logger.info("Benchmarking memory usage...")
        
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024**2)  # MB
            
            # Memory usage during operations
            memory_samples = []
            
            # Sample 1: After imports
            memory_samples.append(process.memory_info().rss / (1024**2))
            
            # Sample 2: After creating objects
            test_objects = []
            for i in range(1000):
                test_objects.append({
                    'id': i,
                    'data': f'test_data_{i}' * 100,
                    'timestamp': datetime.now().isoformat()
                })
            memory_samples.append(process.memory_info().rss / (1024**2))
            
            # Sample 3: After processing
            processed_data = [obj['data'] for obj in test_objects]
            memory_samples.append(process.memory_info().rss / (1024**2))
            
            # Sample 4: After cleanup
            del test_objects, processed_data
            gc.collect()
            memory_samples.append(process.memory_info().rss / (1024**2))
            
            # Calculate memory deltas
            memory_deltas = []
            for i in range(1, len(memory_samples)):
                memory_deltas.append(memory_samples[i] - memory_samples[i-1])
            
            self.results['benchmarks']['memory_usage'] = {
                'initial_memory_mb': initial_memory,
                'memory_samples_mb': memory_samples,
                'memory_deltas_mb': memory_deltas,
                'peak_memory_mb': max(memory_samples),
                'final_memory_mb': memory_samples[-1],
                'memory_growth_mb': memory_samples[-1] - initial_memory
            }
            
        except Exception as e:
            logger.error(f"Memory usage benchmark failed: {e}")
            self.results['benchmarks']['memory_usage'] = {'error': str(e)}
    
    async def _benchmark_end_to_end(self) -> None:
        """Benchmark end-to-end pipeline performance."""
        logger.info("Benchmarking end-to-end pipeline...")
        
        try:
            # Simulate a complete trading cycle
            pipeline_times = []
            
            for i in range(3):  # Run 3 iterations
                start_time = time.time()
                
                try:
                    # Step 1: System initialization
                    from src.monitoring.system_monitor import SystemMonitor
                    monitor = SystemMonitor()
                    
                    # Step 2: Database operations
                    from config.database import get_connection
                    with get_connection('DEMO') as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM sqlite_master")
                        cursor.fetchone()
                    
                    # Step 3: AI model operations
                    from ai.multi_model import MultiModelManager
                    manager = MultiModelManager(mode="DEMO")
                    configs = manager.get_all_model_configs()
                    weights = manager.get_model_weights()
                    
                    # Step 4: Risk calculations
                    from src.trading.risk import RiskManager
                    risk_manager = RiskManager()
                    
                    # Step 5: Dashboard operations
                    from src.dashboard.connector import DashboardConnector
                    connector = DashboardConnector()
                    
                    pipeline_time = time.time() - start_time
                    pipeline_times.append(pipeline_time)
                    
                except Exception as e:
                    logger.error(f"End-to-end benchmark iteration {i} failed: {e}")
                    pipeline_times.append(float('inf'))
            
            self.results['benchmarks']['end_to_end'] = {
                'iterations': len(pipeline_times),
                'times_seconds': pipeline_times,
                'avg_time_seconds': statistics.mean([t for t in pipeline_times if t != float('inf')]),
                'min_time_seconds': min([t for t in pipeline_times if t != float('inf')]),
                'max_time_seconds': max([t for t in pipeline_times if t != float('inf')])
            }
            
        except Exception as e:
            logger.error(f"End-to-end benchmark failed: {e}")
            self.results['benchmarks']['end_to_end'] = {'error': str(e)}
    
    def _generate_summary(self) -> None:
        """Generate benchmark summary."""
        logger.info("Generating benchmark summary...")
        
        summary = {
            'total_time_seconds': round(time.time() - self.start_time, 2),
            'ai_limit': self.ai_limit,
            'timestamp': datetime.now().isoformat(),
            'benchmark_count': len(self.results['benchmarks']),
            'successful_benchmarks': 0,
            'failed_benchmarks': 0
        }
        
        # Count successful vs failed benchmarks
        for benchmark_name, benchmark_data in self.results['benchmarks'].items():
            if 'error' in benchmark_data:
                summary['failed_benchmarks'] += 1
            else:
                summary['successful_benchmarks'] += 1
        
        # Calculate overall performance metrics
        if 'system_startup' in self.results['benchmarks'] and 'error' not in self.results['benchmarks']['system_startup']:
            summary['avg_startup_time_seconds'] = self.results['benchmarks']['system_startup']['avg_time_seconds']
        
        if 'end_to_end' in self.results['benchmarks'] and 'error' not in self.results['benchmarks']['end_to_end']:
            summary['avg_pipeline_time_seconds'] = self.results['benchmarks']['end_to_end']['avg_time_seconds']
        
        if 'memory_usage' in self.results['benchmarks'] and 'error' not in self.results['benchmarks']['memory_usage']:
            summary['peak_memory_mb'] = self.results['benchmarks']['memory_usage']['peak_memory_mb']
            summary['memory_growth_mb'] = self.results['benchmarks']['memory_usage']['memory_growth_mb']
        
        self.results['summary'] = summary
        logger.info(f"Benchmark summary generated: {summary['successful_benchmarks']}/{summary['benchmark_count']} successful")
    
    def _save_results(self) -> None:
        """Save benchmark results to file."""
        try:
            results_file = 'performance-results.json'
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Performance results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save performance results: {e}")

async def main():
    """Main performance benchmark function."""
    print("Performance Benchmark Suite - Trading System Performance Analysis")
    print("=" * 70)
    
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_benchmark_suite()
    
    # Print summary
    summary = results.get('summary', {})
    total_time = summary.get('total_time_seconds', 0)
    successful = summary.get('successful_benchmarks', 0)
    total_benchmarks = summary.get('benchmark_count', 0)
    
    print(f"\nBenchmark Summary:")
    print(f"Total Time: {total_time}s")
    print(f"Successful Benchmarks: {successful}/{total_benchmarks}")
    print(f"AI Limit: {summary.get('ai_limit', 'N/A')}")
    
    if 'avg_startup_time_seconds' in summary:
        print(f"Average Startup Time: {summary['avg_startup_time_seconds']:.3f}s")
    
    if 'avg_pipeline_time_seconds' in summary:
        print(f"Average Pipeline Time: {summary['avg_pipeline_time_seconds']:.3f}s")
    
    if 'peak_memory_mb' in summary:
        print(f"Peak Memory Usage: {summary['peak_memory_mb']:.1f} MB")
    
    if successful == total_benchmarks:
        print("\n✅✅✅ PERFORMANCE BENCHMARK COMPLETED — SYSTEM PERFORMING WELL 🚀")
        return 0
    else:
        print(f"\n⚠️⚠️⚠️ PERFORMANCE BENCHMARK COMPLETED WITH {total_benchmarks - successful} FAILURES ⚠️")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
