#!/usr/bin/env python3
"""
Automated Maintenance and Cleanup Procedures for Continuous Data Collection System

This script provides automated maintenance tasks including log rotation,
storage management, performance optimization, and system cleanup.
"""

import asyncio
import os
import shutil
import gzip
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse
import psutil
import glob

from continuous_data_collection.core.config import Config
from continuous_data_collection.core.state_manager import StateManager


class AutomatedMaintenance:
    """
    Automated maintenance system for production deployment.
    
    Features:
    - Log rotation and compression
    - Storage cleanup and optimization
    - Database maintenance
    - Performance optimization
    - System health checks
    - Backup management
    """
    
    def __init__(self, config_path: str = "config/production.yaml"):
        """
        Initialize automated maintenance system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.state_manager = StateManager(
            state_file="system_state.json",
            backup_dir="backups"
        )
        
        # Maintenance settings
        self.log_retention_days = self.config.get('maintenance', {}).get('log_retention_days', 30)
        self.backup_retention_days = self.config.get('maintenance', {}).get('backup_retention_days', 7)
        self.data_retention_days = self.config.get('maintenance', {}).get('data_retention_days', 365)
        self.compress_logs_older_than_days = self.config.get('maintenance', {}).get('compress_logs_days', 7)
        
        # Storage thresholds
        self.disk_warning_threshold = 85.0  # Percent
        self.disk_critical_threshold = 95.0  # Percent
        self.cleanup_target_free_space = 20.0  # Percent
        
        # Database settings
        self.db_vacuum_threshold_mb = 100  # Vacuum if DB is larger than this
        self.db_analyze_interval_days = 7
        
        self.logger = logging.getLogger(__name__)
        
    async def run_full_maintenance(self) -> Dict[str, Any]:
        """
        Run complete maintenance cycle.
        
        Returns:
            Dictionary with maintenance results
        """
        self.logger.info("Starting full maintenance cycle...")
        start_time = datetime.utcnow()
        
        results = {
            'start_time': start_time.isoformat(),
            'tasks': {},
            'errors': [],
            'warnings': []
        }
        
        # Run maintenance tasks
        maintenance_tasks = [
            ('log_rotation', self._rotate_logs),
            ('log_compression', self._compress_old_logs),
            ('storage_cleanup', self._cleanup_storage),
            ('database_maintenance', self._maintain_databases),
            ('backup_cleanup', self._cleanup_backups),
            ('performance_optimization', self._optimize_performance),
            ('system_health_check', self._check_system_health),
            ('state_backup', self._backup_system_state)
        ]
        
        for task_name, task_func in maintenance_tasks:
            try:
                self.logger.info(f"Running maintenance task: {task_name}")
                task_result = await task_func()
                results['tasks'][task_name] = {
                    'status': 'success',
                    'result': task_result,
                    'timestamp': datetime.utcnow().isoformat()
                }
                self.logger.info(f"Completed maintenance task: {task_name}")
                
            except Exception as e:
                error_msg = f"Error in {task_name}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                results['tasks'][task_name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        # Calculate total time
        end_time = datetime.utcnow()
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Generate summary
        successful_tasks = sum(1 for task in results['tasks'].values() if task['status'] == 'success')
        total_tasks = len(results['tasks'])
        
        results['summary'] = {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': total_tasks - successful_tasks,
            'success_rate': (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        }
        
        self.logger.info(f"Maintenance cycle completed in {results['duration_seconds']:.2f} seconds")
        self.logger.info(f"Tasks: {successful_tasks}/{total_tasks} successful")
        
        return results
        
    async def _rotate_logs(self) -> Dict[str, Any]:
        """Rotate log files."""
        results = {
            'rotated_files': [],
            'total_size_mb': 0,
            'errors': []
        }
        
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return results
            
        # Find log files to rotate
        for log_file in logs_dir.glob("*.log"):
            try:
                # Check if file is large enough to rotate (>10MB)
                file_size = log_file.stat().st_size
                if file_size > 10 * 1024 * 1024:  # 10MB
                    # Create rotated filename with timestamp
                    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    rotated_name = f"{log_file.stem}_{timestamp}.log"
                    rotated_path = logs_dir / rotated_name
                    
                    # Move current log to rotated name
                    shutil.move(str(log_file), str(rotated_path))
                    
                    # Create new empty log file
                    log_file.touch()
                    
                    results['rotated_files'].append({
                        'original': str(log_file),
                        'rotated': str(rotated_path),
                        'size_mb': file_size / (1024 * 1024)
                    })
                    
                    results['total_size_mb'] += file_size / (1024 * 1024)
                    
            except Exception as e:
                error_msg = f"Error rotating {log_file}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
                
        return results
        
    async def _compress_old_logs(self) -> Dict[str, Any]:
        """Compress old log files."""
        results = {
            'compressed_files': [],
            'space_saved_mb': 0,
            'errors': []
        }
        
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return results
            
        cutoff_date = datetime.utcnow() - timedelta(days=self.compress_logs_older_than_days)
        
        # Find log files to compress
        for log_file in logs_dir.glob("*.log*"):
            if log_file.suffix == '.gz':
                continue  # Already compressed
                
            try:
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_mtime < cutoff_date:
                    original_size = log_file.stat().st_size
                    
                    # Compress the file
                    compressed_path = log_file.with_suffix(log_file.suffix + '.gz')
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                            
                    # Remove original file
                    log_file.unlink()
                    
                    compressed_size = compressed_path.stat().st_size
                    space_saved = (original_size - compressed_size) / (1024 * 1024)
                    
                    results['compressed_files'].append({
                        'file': str(log_file),
                        'compressed': str(compressed_path),
                        'original_size_mb': original_size / (1024 * 1024),
                        'compressed_size_mb': compressed_size / (1024 * 1024),
                        'space_saved_mb': space_saved
                    })
                    
                    results['space_saved_mb'] += space_saved
                    
            except Exception as e:
                error_msg = f"Error compressing {log_file}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
                
        return results
        
    async def _cleanup_storage(self) -> Dict[str, Any]:
        """Clean up storage and manage disk space."""
        results = {
            'disk_usage': {},
            'cleaned_files': [],
            'space_freed_mb': 0,
            'warnings': [],
            'errors': []
        }
        
        # Check disk usage
        disk_usage = psutil.disk_usage('.')
        disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
        
        results['disk_usage'] = {
            'total_gb': disk_usage.total / (1024**3),
            'used_gb': disk_usage.used / (1024**3),
            'free_gb': disk_usage.free / (1024**3),
            'used_percent': disk_usage_percent
        }
        
        # Check if cleanup is needed
        if disk_usage_percent > self.disk_warning_threshold:
            results['warnings'].append(f"Disk usage is {disk_usage_percent:.1f}% (warning threshold: {self.disk_warning_threshold}%)")
            
            if disk_usage_percent > self.disk_critical_threshold:
                results['warnings'].append("Critical disk usage - performing aggressive cleanup")
                await self._aggressive_cleanup(results)
            else:
                await self._standard_cleanup(results)
                
        # Clean old log files
        await self._cleanup_old_logs(results)
        
        # Clean temporary files
        await self._cleanup_temp_files(results)
        
        return results
        
    async def _standard_cleanup(self, results: Dict[str, Any]) -> None:
        """Perform standard cleanup procedures."""
        # Clean old compressed logs
        logs_dir = Path("logs")
        if logs_dir.exists():
            cutoff_date = datetime.utcnow() - timedelta(days=self.log_retention_days)
            
            for log_file in logs_dir.glob("*.gz"):
                try:
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        file_size = log_file.stat().st_size
                        log_file.unlink()
                        
                        results['cleaned_files'].append({
                            'file': str(log_file),
                            'size_mb': file_size / (1024 * 1024),
                            'reason': 'old_log'
                        })
                        
                        results['space_freed_mb'] += file_size / (1024 * 1024)
                        
                except Exception as e:
                    results['errors'].append(f"Error cleaning {log_file}: {str(e)}")
                    
    async def _aggressive_cleanup(self, results: Dict[str, Any]) -> None:
        """Perform aggressive cleanup when disk space is critical."""
        # More aggressive log cleanup
        logs_dir = Path("logs")
        if logs_dir.exists():
            # Keep only last 7 days of logs
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            for log_file in logs_dir.glob("*"):
                try:
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        file_size = log_file.stat().st_size
                        log_file.unlink()
                        
                        results['cleaned_files'].append({
                            'file': str(log_file),
                            'size_mb': file_size / (1024 * 1024),
                            'reason': 'aggressive_cleanup'
                        })
                        
                        results['space_freed_mb'] += file_size / (1024 * 1024)
                        
                except Exception as e:
                    results['errors'].append(f"Error in aggressive cleanup of {log_file}: {str(e)}")
                    
        # Clean old data files if configured
        data_dir = Path("data")
        if data_dir.exists():
            cutoff_date = datetime.utcnow() - timedelta(days=self.data_retention_days)
            
            for data_file in data_dir.glob("**/*.parquet"):
                try:
                    file_mtime = datetime.fromtimestamp(data_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        file_size = data_file.stat().st_size
                        data_file.unlink()
                        
                        results['cleaned_files'].append({
                            'file': str(data_file),
                            'size_mb': file_size / (1024 * 1024),
                            'reason': 'old_data'
                        })
                        
                        results['space_freed_mb'] += file_size / (1024 * 1024)
                        
                except Exception as e:
                    results['errors'].append(f"Error cleaning old data {data_file}: {str(e)}")
                    
    async def _cleanup_old_logs(self, results: Dict[str, Any]) -> None:
        """Clean up old log files based on retention policy."""
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return
            
        cutoff_date = datetime.utcnow() - timedelta(days=self.log_retention_days)
        
        for log_file in logs_dir.glob("*"):
            try:
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_mtime < cutoff_date:
                    file_size = log_file.stat().st_size
                    log_file.unlink()
                    
                    results['cleaned_files'].append({
                        'file': str(log_file),
                        'size_mb': file_size / (1024 * 1024),
                        'reason': 'retention_policy'
                    })
                    
                    results['space_freed_mb'] += file_size / (1024 * 1024)
                    
            except Exception as e:
                results['errors'].append(f"Error cleaning log {log_file}: {str(e)}")
                
    async def _cleanup_temp_files(self, results: Dict[str, Any]) -> None:
        """Clean up temporary files."""
        temp_patterns = [
            "*.tmp",
            "*.temp",
            "*.lock",
            "*.pid",
            "__pycache__/**/*",
            ".pytest_cache/**/*"
        ]
        
        for pattern in temp_patterns:
            for temp_file in Path(".").glob(pattern):
                try:
                    if temp_file.is_file():
                        file_size = temp_file.stat().st_size
                        temp_file.unlink()
                        
                        results['cleaned_files'].append({
                            'file': str(temp_file),
                            'size_mb': file_size / (1024 * 1024),
                            'reason': 'temp_file'
                        })
                        
                        results['space_freed_mb'] += file_size / (1024 * 1024)
                        
                except Exception as e:
                    results['errors'].append(f"Error cleaning temp file {temp_file}: {str(e)}")
                    
    async def _maintain_databases(self) -> Dict[str, Any]:
        """Maintain SQLite databases."""
        results = {
            'databases': [],
            'total_space_freed_mb': 0,
            'errors': []
        }
        
        # Find SQLite databases
        db_files = list(Path(".").glob("*.db")) + list(Path(".").glob("**/*.db"))
        
        for db_file in db_files:
            try:
                db_result = await self._maintain_single_database(db_file)
                results['databases'].append(db_result)
                results['total_space_freed_mb'] += db_result.get('space_freed_mb', 0)
                
            except Exception as e:
                error_msg = f"Error maintaining database {db_file}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
                
        return results
        
    async def _maintain_single_database(self, db_file: Path) -> Dict[str, Any]:
        """Maintain a single SQLite database."""
        result = {
            'database': str(db_file),
            'original_size_mb': 0,
            'final_size_mb': 0,
            'space_freed_mb': 0,
            'operations': []
        }
        
        try:
            # Get original size
            original_size = db_file.stat().st_size
            result['original_size_mb'] = original_size / (1024 * 1024)
            
            # Connect to database
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # Check if vacuum is needed
            if original_size > self.db_vacuum_threshold_mb * 1024 * 1024:
                cursor.execute("VACUUM")
                result['operations'].append('vacuum')
                
            # Update statistics
            cursor.execute("ANALYZE")
            result['operations'].append('analyze')
            
            # Clean up old records if this is the alerts database
            if 'alerts' in db_file.name:
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                cursor.execute(
                    "DELETE FROM alerts WHERE created_at < ? AND status = 'resolved'",
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    result['operations'].append(f'deleted_{deleted_count}_old_alerts')
                    
            conn.commit()
            conn.close()
            
            # Get final size
            final_size = db_file.stat().st_size
            result['final_size_mb'] = final_size / (1024 * 1024)
            result['space_freed_mb'] = (original_size - final_size) / (1024 * 1024)
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
        
    async def _cleanup_backups(self) -> Dict[str, Any]:
        """Clean up old backup files."""
        results = {
            'cleaned_backups': [],
            'space_freed_mb': 0,
            'errors': []
        }
        
        backups_dir = Path("backups")
        if not backups_dir.exists():
            return results
            
        cutoff_date = datetime.utcnow() - timedelta(days=self.backup_retention_days)
        
        for backup_file in backups_dir.glob("*"):
            try:
                file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_mtime < cutoff_date:
                    file_size = backup_file.stat().st_size
                    backup_file.unlink()
                    
                    results['cleaned_backups'].append({
                        'file': str(backup_file),
                        'size_mb': file_size / (1024 * 1024),
                        'age_days': (datetime.utcnow() - file_mtime).days
                    })
                    
                    results['space_freed_mb'] += file_size / (1024 * 1024)
                    
            except Exception as e:
                error_msg = f"Error cleaning backup {backup_file}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
                
        return results
        
    async def _optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance."""
        results = {
            'optimizations': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Memory optimization
            if memory.percent > 80:
                results['warnings'].append(f"High memory usage: {memory.percent:.1f}%")
                # Force garbage collection
                import gc
                gc.collect()
                results['optimizations'].append('forced_garbage_collection')
                
            # CPU optimization
            if cpu_percent > 90:
                results['warnings'].append(f"High CPU usage: {cpu_percent:.1f}%")
                
            # Check for zombie processes
            zombie_count = 0
            for proc in psutil.process_iter(['pid', 'status']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        zombie_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            if zombie_count > 0:
                results['warnings'].append(f"Found {zombie_count} zombie processes")
                
            # Optimize Python bytecode
            import py_compile
            import compileall
            
            # Compile Python files for better performance
            compileall.compile_dir('.', quiet=1)
            results['optimizations'].append('compiled_python_bytecode')
            
        except Exception as e:
            error_msg = f"Error in performance optimization: {str(e)}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
            
        return results
        
    async def _check_system_health(self) -> Dict[str, Any]:
        """Perform system health checks."""
        results = {
            'health_status': 'healthy',
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check disk space
            disk_usage = psutil.disk_usage('.')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            results['checks']['disk_usage'] = {
                'status': 'healthy' if disk_percent < 85 else 'warning' if disk_percent < 95 else 'critical',
                'usage_percent': disk_percent,
                'free_gb': disk_usage.free / (1024**3)
            }
            
            # Check memory
            memory = psutil.virtual_memory()
            results['checks']['memory'] = {
                'status': 'healthy' if memory.percent < 80 else 'warning' if memory.percent < 90 else 'critical',
                'usage_percent': memory.percent,
                'available_gb': memory.available / (1024**3)
            }
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            results['checks']['cpu'] = {
                'status': 'healthy' if cpu_percent < 80 else 'warning' if cpu_percent < 90 else 'critical',
                'usage_percent': cpu_percent
            }
            
            # Check file system
            results['checks']['filesystem'] = await self._check_filesystem_health()
            
            # Check process health
            results['checks']['processes'] = await self._check_process_health()
            
            # Determine overall health
            critical_checks = [check for check in results['checks'].values() 
                             if isinstance(check, dict) and check.get('status') == 'critical']
            warning_checks = [check for check in results['checks'].values() 
                            if isinstance(check, dict) and check.get('status') == 'warning']
            
            if critical_checks:
                results['health_status'] = 'critical'
            elif warning_checks:
                results['health_status'] = 'warning'
                
        except Exception as e:
            error_msg = f"Error in system health check: {str(e)}"
            results['errors'].append(error_msg)
            results['health_status'] = 'error'
            self.logger.error(error_msg)
            
        return results
        
    async def _check_filesystem_health(self) -> Dict[str, Any]:
        """Check filesystem health."""
        result = {
            'status': 'healthy',
            'issues': []
        }
        
        try:
            # Check for required directories
            required_dirs = ['logs', 'data', 'backups', 'config']
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    result['issues'].append(f"Missing directory: {dir_name}")
                    result['status'] = 'warning'
                elif not os.access(dir_path, os.W_OK):
                    result['issues'].append(f"Directory not writable: {dir_name}")
                    result['status'] = 'critical'
                    
            # Check for corrupted files
            for db_file in Path(".").glob("*.db"):
                try:
                    conn = sqlite3.connect(str(db_file))
                    conn.execute("PRAGMA integrity_check")
                    conn.close()
                except Exception as e:
                    result['issues'].append(f"Database corruption detected: {db_file}")
                    result['status'] = 'critical'
                    
        except Exception as e:
            result['issues'].append(f"Filesystem check error: {str(e)}")
            result['status'] = 'error'
            
        return result
        
    async def _check_process_health(self) -> Dict[str, Any]:
        """Check process health."""
        result = {
            'status': 'healthy',
            'running_processes': 0,
            'zombie_processes': 0,
            'high_memory_processes': []
        }
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'status', 'memory_percent']):
                try:
                    result['running_processes'] += 1
                    
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        result['zombie_processes'] += 1
                        
                    if proc.info['memory_percent'] > 10:  # More than 10% memory
                        result['high_memory_processes'].append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'memory_percent': proc.info['memory_percent']
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            if result['zombie_processes'] > 0:
                result['status'] = 'warning'
                
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            
        return result
        
    async def _backup_system_state(self) -> Dict[str, Any]:
        """Backup system state."""
        try:
            backup_result = await self.state_manager.backup_state()
            return {
                'status': 'success',
                'backup_file': backup_result.get('backup_file', 'unknown'),
                'backup_size_mb': backup_result.get('backup_size_mb', 0)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


async def main():
    """Main function for running maintenance."""
    parser = argparse.ArgumentParser(description='Automated Maintenance System')
    parser.add_argument('--config', default='config/production.yaml',
                       help='Configuration file path')
    parser.add_argument('--task', choices=[
        'full', 'logs', 'storage', 'database', 'backups', 'performance', 'health'
    ], default='full', help='Maintenance task to run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create maintenance system
    maintenance = AutomatedMaintenance(args.config)
    
    try:
        if args.task == 'full':
            results = await maintenance.run_full_maintenance()
        elif args.task == 'logs':
            results = {
                'log_rotation': await maintenance._rotate_logs(),
                'log_compression': await maintenance._compress_old_logs()
            }
        elif args.task == 'storage':
            results = {'storage_cleanup': await maintenance._cleanup_storage()}
        elif args.task == 'database':
            results = {'database_maintenance': await maintenance._maintain_databases()}
        elif args.task == 'backups':
            results = {'backup_cleanup': await maintenance._cleanup_backups()}
        elif args.task == 'performance':
            results = {'performance_optimization': await maintenance._optimize_performance()}
        elif args.task == 'health':
            results = {'system_health_check': await maintenance._check_system_health()}
        else:
            results = {'error': f'Unknown task: {args.task}'}
            
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))
            
    except Exception as e:
        logging.error(f"Maintenance error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)