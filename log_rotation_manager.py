#!/usr/bin/env python3
"""
Log Rotation and Storage Management for Continuous Data Collection System

This script provides comprehensive log rotation, compression, and storage management
with configurable retention policies and automatic cleanup.
"""

import os
import gzip
import shutil
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse
import threading
import signal
import glob
import re


class LogRotationManager:
    """
    Comprehensive log rotation and storage management system.
    
    Features:
    - Automatic log rotation based on size and time
    - Log compression with configurable algorithms
    - Retention policy enforcement
    - Storage usage monitoring
    - Performance-optimized operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize log rotation manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Rotation settings
        self.max_file_size_mb = self.config.get('max_file_size_mb', 50)
        self.max_age_days = self.config.get('max_age_days', 7)
        self.retention_days = self.config.get('retention_days', 30)
        self.compress_after_days = self.config.get('compress_after_days', 1)
        
        # Directories
        self.log_directories = self.config.get('log_directories', ['logs'])
        self.archive_directory = self.config.get('archive_directory', 'logs/archive')
        
        # Compression settings
        self.compression_enabled = self.config.get('compression_enabled', True)
        self.compression_level = self.config.get('compression_level', 6)
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 10)
        self.max_concurrent_operations = self.config.get('max_concurrent_operations', 3)
        
        # Monitoring
        self.storage_warning_threshold_gb = self.config.get('storage_warning_threshold_gb', 10)
        self.storage_critical_threshold_gb = self.config.get('storage_critical_threshold_gb', 5)
        
        # State
        self.is_running = False
        self.rotation_thread: Optional[threading.Thread] = None
        self.stats = {
            'rotations_performed': 0,
            'files_compressed': 0,
            'files_deleted': 0,
            'space_saved_mb': 0,
            'last_rotation': None,
            'errors': []
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure archive directory exists
        Path(self.archive_directory).mkdir(parents=True, exist_ok=True)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'max_file_size_mb': 50,
            'max_age_days': 7,
            'retention_days': 30,
            'compress_after_days': 1,
            'log_directories': ['logs'],
            'archive_directory': 'logs/archive',
            'compression_enabled': True,
            'compression_level': 6,
            'batch_size': 10,
            'max_concurrent_operations': 3,
            'storage_warning_threshold_gb': 10,
            'storage_critical_threshold_gb': 5,
            'rotation_interval_minutes': 60,
            'file_patterns': ['*.log', '*.out', '*.err']
        }
        
    def start_automatic_rotation(self, interval_minutes: int = 60) -> None:
        """
        Start automatic log rotation in background thread.
        
        Args:
            interval_minutes: Rotation check interval in minutes
        """
        if self.is_running:
            self.logger.warning("Log rotation is already running")
            return
            
        self.is_running = True
        self.rotation_thread = threading.Thread(
            target=self._rotation_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self.rotation_thread.start()
        
        self.logger.info(f"Started automatic log rotation (interval: {interval_minutes} minutes)")
        
    def stop_automatic_rotation(self) -> None:
        """Stop automatic log rotation."""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.rotation_thread:
            self.rotation_thread.join(timeout=30)
            
        self.logger.info("Stopped automatic log rotation")
        
    def _rotation_loop(self, interval_minutes: int) -> None:
        """Main rotation loop running in background thread."""
        while self.is_running:
            try:
                self.rotate_logs()
                time.sleep(interval_minutes * 60)
            except Exception as e:
                self.logger.error(f"Error in rotation loop: {e}")
                self.stats['errors'].append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e),
                    'operation': 'rotation_loop'
                })
                time.sleep(60)  # Wait 1 minute before retrying
                
    def rotate_logs(self) -> Dict[str, Any]:
        """
        Perform log rotation for all configured directories.
        
        Returns:
            Dictionary with rotation results
        """
        self.logger.info("Starting log rotation cycle...")
        start_time = time.time()
        
        results = {
            'start_time': datetime.utcnow().isoformat(),
            'directories_processed': 0,
            'files_rotated': 0,
            'files_compressed': 0,
            'files_deleted': 0,
            'space_saved_mb': 0,
            'errors': [],
            'warnings': []
        }
        
        for log_dir in self.log_directories:
            try:
                dir_results = self._rotate_directory(log_dir)
                results['directories_processed'] += 1
                results['files_rotated'] += dir_results['files_rotated']
                results['files_compressed'] += dir_results['files_compressed']
                results['files_deleted'] += dir_results['files_deleted']
                results['space_saved_mb'] += dir_results['space_saved_mb']
                results['errors'].extend(dir_results['errors'])
                results['warnings'].extend(dir_results['warnings'])
                
            except Exception as e:
                error_msg = f"Error processing directory {log_dir}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                
        # Update global stats
        self.stats['rotations_performed'] += 1
        self.stats['files_compressed'] += results['files_compressed']
        self.stats['files_deleted'] += results['files_deleted']
        self.stats['space_saved_mb'] += results['space_saved_mb']
        self.stats['last_rotation'] = datetime.utcnow().isoformat()
        self.stats['errors'].extend(results['errors'])
        
        # Calculate duration
        duration = time.time() - start_time
        results['duration_seconds'] = duration
        results['end_time'] = datetime.utcnow().isoformat()
        
        self.logger.info(f"Log rotation completed in {duration:.2f} seconds")
        self.logger.info(f"Rotated: {results['files_rotated']}, "
                        f"Compressed: {results['files_compressed']}, "
                        f"Deleted: {results['files_deleted']}, "
                        f"Space saved: {results['space_saved_mb']:.2f} MB")
        
        return results
        
    def _rotate_directory(self, directory: str) -> Dict[str, Any]:
        """
        Rotate logs in a specific directory.
        
        Args:
            directory: Directory path to process
            
        Returns:
            Dictionary with directory rotation results
        """
        results = {
            'directory': directory,
            'files_rotated': 0,
            'files_compressed': 0,
            'files_deleted': 0,
            'space_saved_mb': 0,
            'errors': [],
            'warnings': []
        }
        
        log_dir = Path(directory)
        if not log_dir.exists():
            results['warnings'].append(f"Directory does not exist: {directory}")
            return results
            
        # Find files to process
        files_to_rotate = self._find_files_to_rotate(log_dir)
        files_to_compress = self._find_files_to_compress(log_dir)
        files_to_delete = self._find_files_to_delete(log_dir)
        
        # Rotate files
        for file_path in files_to_rotate:
            try:
                if self._rotate_file(file_path):
                    results['files_rotated'] += 1
            except Exception as e:
                error_msg = f"Error rotating {file_path}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
                
        # Compress files
        if self.compression_enabled:
            for file_path in files_to_compress:
                try:
                    space_saved = self._compress_file(file_path)
                    if space_saved > 0:
                        results['files_compressed'] += 1
                        results['space_saved_mb'] += space_saved
                except Exception as e:
                    error_msg = f"Error compressing {file_path}: {str(e)}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    
        # Delete old files
        for file_path in files_to_delete:
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                file_path.unlink()
                results['files_deleted'] += 1
                results['space_saved_mb'] += file_size_mb
                self.logger.debug(f"Deleted old file: {file_path}")
            except Exception as e:
                error_msg = f"Error deleting {file_path}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
                
        return results
        
    def _find_files_to_rotate(self, directory: Path) -> List[Path]:
        """Find files that need rotation."""
        files_to_rotate = []
        
        for pattern in self.config.get('file_patterns', ['*.log']):
            for file_path in directory.glob(pattern):
                if file_path.is_file() and not file_path.name.endswith('.gz'):
                    # Check size
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    if file_size_mb > self.max_file_size_mb:
                        files_to_rotate.append(file_path)
                        continue
                        
                    # Check age
                    file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.days > self.max_age_days:
                        files_to_rotate.append(file_path)
                        
        return files_to_rotate
        
    def _find_files_to_compress(self, directory: Path) -> List[Path]:
        """Find files that need compression."""
        files_to_compress = []
        
        if not self.compression_enabled:
            return files_to_compress
            
        cutoff_date = datetime.now() - timedelta(days=self.compress_after_days)
        
        # Look for rotated files (with timestamp suffix)
        for file_path in directory.glob("*"):
            if (file_path.is_file() and 
                not file_path.name.endswith('.gz') and
                self._is_rotated_file(file_path)):
                
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_date:
                    files_to_compress.append(file_path)
                    
        return files_to_compress
        
    def _find_files_to_delete(self, directory: Path) -> List[Path]:
        """Find files that should be deleted based on retention policy."""
        files_to_delete = []
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for file_path in directory.glob("*"):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_date:
                    files_to_delete.append(file_path)
                    
        return files_to_delete
        
    def _is_rotated_file(self, file_path: Path) -> bool:
        """Check if a file is a rotated log file."""
        # Look for timestamp pattern in filename
        timestamp_pattern = r'\d{8}_\d{6}'  # YYYYMMDD_HHMMSS
        return bool(re.search(timestamp_pattern, file_path.name))
        
    def _rotate_file(self, file_path: Path) -> bool:
        """
        Rotate a single file.
        
        Args:
            file_path: Path to file to rotate
            
        Returns:
            True if rotation was successful
        """
        try:
            # Generate rotated filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            
            # Move to archive directory
            archive_path = Path(self.archive_directory) / rotated_name
            
            # Ensure archive directory exists
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(file_path), str(archive_path))
            
            # Create new empty file with same name
            file_path.touch()
            
            # Set appropriate permissions
            os.chmod(file_path, 0o644)
            
            self.logger.debug(f"Rotated {file_path} to {archive_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rotate {file_path}: {e}")
            return False
            
    def _compress_file(self, file_path: Path) -> float:
        """
        Compress a file using gzip.
        
        Args:
            file_path: Path to file to compress
            
        Returns:
            Space saved in MB
        """
        try:
            original_size = file_path.stat().st_size
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            # Compress file
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb', compresslevel=self.compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            # Remove original file
            file_path.unlink()
            
            # Calculate space saved
            compressed_size = compressed_path.stat().st_size
            space_saved_mb = (original_size - compressed_size) / (1024 * 1024)
            
            self.logger.debug(f"Compressed {file_path} (saved {space_saved_mb:.2f} MB)")
            return space_saved_mb
            
        except Exception as e:
            self.logger.error(f"Failed to compress {file_path}: {e}")
            return 0.0
            
    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        usage_stats = {
            'directories': {},
            'total_size_mb': 0,
            'total_files': 0,
            'compressed_files': 0,
            'warnings': []
        }
        
        for log_dir in self.log_directories:
            dir_path = Path(log_dir)
            if not dir_path.exists():
                continue
                
            dir_stats = {
                'size_mb': 0,
                'files': 0,
                'compressed_files': 0,
                'oldest_file': None,
                'newest_file': None
            }
            
            oldest_time = None
            newest_time = None
            
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    dir_stats['size_mb'] += file_size / (1024 * 1024)
                    dir_stats['files'] += 1
                    
                    if file_path.name.endswith('.gz'):
                        dir_stats['compressed_files'] += 1
                        
                    # Track oldest and newest files
                    if oldest_time is None or file_mtime < oldest_time:
                        oldest_time = file_mtime
                        dir_stats['oldest_file'] = {
                            'path': str(file_path),
                            'modified': file_mtime.isoformat()
                        }
                        
                    if newest_time is None or file_mtime > newest_time:
                        newest_time = file_mtime
                        dir_stats['newest_file'] = {
                            'path': str(file_path),
                            'modified': file_mtime.isoformat()
                        }
                        
            usage_stats['directories'][log_dir] = dir_stats
            usage_stats['total_size_mb'] += dir_stats['size_mb']
            usage_stats['total_files'] += dir_stats['files']
            usage_stats['compressed_files'] += dir_stats['compressed_files']
            
        # Check storage thresholds
        total_size_gb = usage_stats['total_size_mb'] / 1024
        
        if total_size_gb > self.storage_critical_threshold_gb:
            usage_stats['warnings'].append(
                f"Critical storage usage: {total_size_gb:.2f} GB "
                f"(threshold: {self.storage_critical_threshold_gb} GB)"
            )
        elif total_size_gb > self.storage_warning_threshold_gb:
            usage_stats['warnings'].append(
                f"High storage usage: {total_size_gb:.2f} GB "
                f"(threshold: {self.storage_warning_threshold_gb} GB)"
            )
            
        return usage_stats
        
    def cleanup_old_files(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Clean up files older than specified days.
        
        Args:
            days: Number of days (uses retention_days if not specified)
            
        Returns:
            Cleanup results
        """
        cleanup_days = days or self.retention_days
        cutoff_date = datetime.now() - timedelta(days=cleanup_days)
        
        results = {
            'cutoff_date': cutoff_date.isoformat(),
            'files_deleted': 0,
            'space_freed_mb': 0,
            'errors': []
        }
        
        for log_dir in self.log_directories:
            dir_path = Path(log_dir)
            if not dir_path.exists():
                continue
                
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    try:
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_mtime < cutoff_date:
                            file_size_mb = file_path.stat().st_size / (1024 * 1024)
                            file_path.unlink()
                            
                            results['files_deleted'] += 1
                            results['space_freed_mb'] += file_size_mb
                            
                            self.logger.debug(f"Deleted old file: {file_path}")
                            
                    except Exception as e:
                        error_msg = f"Error deleting {file_path}: {str(e)}"
                        results['errors'].append(error_msg)
                        self.logger.error(error_msg)
                        
        self.logger.info(f"Cleanup completed: deleted {results['files_deleted']} files, "
                        f"freed {results['space_freed_mb']:.2f} MB")
        
        return results
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get rotation statistics."""
        stats = self.stats.copy()
        stats['storage_usage'] = self.get_storage_usage()
        stats['configuration'] = self.config
        stats['is_running'] = self.is_running
        
        return stats
        
    def export_configuration(self, file_path: str) -> None:
        """Export current configuration to file."""
        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        self.logger.info(f"Configuration exported to {file_path}")
        
    def load_configuration(self, file_path: str) -> None:
        """Load configuration from file."""
        with open(file_path, 'r') as f:
            self.config = json.load(f)
        self.logger.info(f"Configuration loaded from {file_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Log Rotation Manager')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--action', choices=[
        'rotate', 'compress', 'cleanup', 'status', 'start-daemon', 'stop-daemon'
    ], default='rotate', help='Action to perform')
    parser.add_argument('--days', type=int, help='Days for cleanup operation')
    parser.add_argument('--daemon-interval', type=int, default=60,
                       help='Daemon check interval in minutes')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            
    # Create manager
    manager = LogRotationManager(config)
    
    try:
        if args.action == 'rotate':
            results = manager.rotate_logs()
        elif args.action == 'cleanup':
            results = manager.cleanup_old_files(args.days)
        elif args.action == 'status':
            results = manager.get_statistics()
        elif args.action == 'start-daemon':
            manager.start_automatic_rotation(args.daemon_interval)
            print(f"Started log rotation daemon (interval: {args.daemon_interval} minutes)")
            print("Press Ctrl+C to stop...")
            try:
                while manager.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping daemon...")
                manager.stop_automatic_rotation()
            return
        elif args.action == 'stop-daemon':
            manager.stop_automatic_rotation()
            results = {'message': 'Daemon stopped'}
        else:
            results = {'error': f'Unknown action: {args.action}'}
            
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))
            
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())