#!/usr/bin/env python3
"""
Production Monitoring Startup Script for Continuous Data Collection System

This script starts all production monitoring components including the dashboard,
automated maintenance, log rotation, and performance monitoring.
"""

import asyncio
import logging
import signal
import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import time
from datetime import datetime

# Import monitoring components
from production_monitoring_dashboard import ProductionMonitoringDashboard
from automated_maintenance import AutomatedMaintenance
from log_rotation_manager import LogRotationManager
from performance_capacity_monitor import PerformanceCapacityMonitor


class ProductionMonitoringOrchestrator:
    """
    Orchestrator for all production monitoring components.
    
    Features:
    - Centralized startup and shutdown of all monitoring services
    - Health monitoring of monitoring components themselves
    - Graceful error handling and recovery
    - Configuration management
    - Service dependency management
    """
    
    def __init__(self, config_path: str = "config/production.yaml"):
        """
        Initialize production monitoring orchestrator.
        
        Args:
            config_path: Path to main configuration file
        """
        self.config_path = config_path
        self.services = {}
        self.is_running = False
        self.shutdown_requested = False
        
        # Service configurations
        self.service_configs = {
            'dashboard': 'config/production.yaml',
            'maintenance': 'config/maintenance.json',
            'log_rotation': 'config/maintenance.json',
            'performance_monitor': 'config/production.yaml'
        }
        
        # Service startup order (dependencies)
        self.startup_order = [
            'log_rotation',
            'performance_monitor',
            'maintenance',
            'dashboard'
        ]
        
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        asyncio.create_task(self.stop_all_services())
        
    async def start_all_services(self) -> Dict[str, Any]:
        """
        Start all monitoring services in proper order.
        
        Returns:
            Dictionary with startup results
        """
        self.logger.info("Starting production monitoring services...")
        start_time = datetime.utcnow()
        
        results = {
            'start_time': start_time.isoformat(),
            'services': {},
            'errors': [],
            'warnings': []
        }
        
        # Ensure required directories exist
        self._ensure_directories()
        
        # Start services in order
        for service_name in self.startup_order:
            try:
                self.logger.info(f"Starting service: {service_name}")
                service_result = await self._start_service(service_name)
                
                results['services'][service_name] = {
                    'status': 'started' if service_result else 'failed',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                if service_result:
                    self.logger.info(f"Successfully started service: {service_name}")
                else:
                    error_msg = f"Failed to start service: {service_name}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                    
            except Exception as e:
                error_msg = f"Error starting service {service_name}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                results['services'][service_name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        # Calculate startup time
        end_time = datetime.utcnow()
        results['end_time'] = end_time.isoformat()
        results['startup_duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Determine overall status
        successful_services = sum(1 for service in results['services'].values() 
                                if service['status'] == 'started')
        total_services = len(results['services'])
        
        results['summary'] = {
            'total_services': total_services,
            'successful_services': successful_services,
            'failed_services': total_services - successful_services,
            'success_rate': (successful_services / total_services) * 100 if total_services > 0 else 0
        }
        
        if successful_services == total_services:
            self.is_running = True
            self.logger.info("All monitoring services started successfully")
        else:
            self.logger.warning(f"Started {successful_services}/{total_services} services")
            
        return results
        
    async def _start_service(self, service_name: str) -> bool:
        """
        Start a specific service.
        
        Args:
            service_name: Name of the service to start
            
        Returns:
            True if service started successfully
        """
        try:
            config_path = self.service_configs.get(service_name)
            
            if service_name == 'dashboard':
                dashboard = ProductionMonitoringDashboard(config_path)
                await dashboard.start()
                self.services['dashboard'] = dashboard
                return True
                
            elif service_name == 'maintenance':
                # Load maintenance config
                maintenance_config = None
                if config_path and Path(config_path).exists():
                    with open(config_path, 'r') as f:
                        maintenance_config = json.load(f)
                        
                maintenance = AutomatedMaintenance(config_path)
                # Start maintenance in background (it will run periodically)
                self.services['maintenance'] = maintenance
                return True
                
            elif service_name == 'log_rotation':
                # Load log rotation config
                log_config = None
                if config_path and Path(config_path).exists():
                    with open(config_path, 'r') as f:
                        full_config = json.load(f)
                        log_config = full_config.get('log_rotation', {})
                        
                log_manager = LogRotationManager(log_config)
                log_manager.start_automatic_rotation(60)  # 60-minute intervals
                self.services['log_rotation'] = log_manager
                return True
                
            elif service_name == 'performance_monitor':
                # Load performance monitor config
                perf_config = None
                if config_path and Path(config_path).exists():
                    with open(config_path, 'r') as f:
                        perf_config = json.load(f)
                        
                perf_monitor = PerformanceCapacityMonitor(perf_config)
                perf_monitor.start_monitoring()
                self.services['performance_monitor'] = perf_monitor
                return True
                
            else:
                self.logger.error(f"Unknown service: {service_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting {service_name}: {e}")
            return False
            
    async def stop_all_services(self) -> Dict[str, Any]:
        """
        Stop all monitoring services gracefully.
        
        Returns:
            Dictionary with shutdown results
        """
        self.logger.info("Stopping all monitoring services...")
        start_time = datetime.utcnow()
        
        results = {
            'stop_time': start_time.isoformat(),
            'services': {},
            'errors': []
        }
        
        # Stop services in reverse order
        for service_name in reversed(self.startup_order):
            if service_name in self.services:
                try:
                    self.logger.info(f"Stopping service: {service_name}")
                    await self._stop_service(service_name)
                    
                    results['services'][service_name] = {
                        'status': 'stopped',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    self.logger.info(f"Successfully stopped service: {service_name}")
                    
                except Exception as e:
                    error_msg = f"Error stopping service {service_name}: {str(e)}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                    results['services'][service_name] = {
                        'status': 'error',
                        'error': str(e),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
        # Calculate shutdown time
        end_time = datetime.utcnow()
        results['end_time'] = end_time.isoformat()
        results['shutdown_duration_seconds'] = (end_time - start_time).total_seconds()
        
        self.is_running = False
        self.logger.info("All monitoring services stopped")
        
        return results
        
    async def _stop_service(self, service_name: str) -> None:
        """
        Stop a specific service.
        
        Args:
            service_name: Name of the service to stop
        """
        service = self.services.get(service_name)
        if not service:
            return
            
        try:
            if service_name == 'dashboard':
                await service.stop()
            elif service_name == 'maintenance':
                # Maintenance service doesn't need explicit stopping
                pass
            elif service_name == 'log_rotation':
                service.stop_automatic_rotation()
            elif service_name == 'performance_monitor':
                service.stop_monitoring()
                
            # Remove from services dict
            del self.services[service_name]
            
        except Exception as e:
            self.logger.error(f"Error stopping {service_name}: {e}")
            raise
            
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        required_dirs = [
            'logs',
            'logs/archive',
            'backups',
            'config',
            'reports',
            'data'
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    async def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'orchestrator_running': self.is_running,
            'services': {}
        }
        
        for service_name, service in self.services.items():
            try:
                if service_name == 'dashboard':
                    service_status = service.get_dashboard_status()
                elif service_name == 'maintenance':
                    service_status = {'status': 'running', 'type': 'maintenance'}
                elif service_name == 'log_rotation':
                    service_status = service.get_statistics()
                elif service_name == 'performance_monitor':
                    service_status = service.get_current_performance()
                else:
                    service_status = {'status': 'unknown'}
                    
                status['services'][service_name] = service_status
                
            except Exception as e:
                status['services'][service_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                
        return status
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        health = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_health': 'healthy',
            'services': {},
            'issues': []
        }
        
        for service_name in self.services:
            try:
                # Basic health check - service exists and is responsive
                service = self.services[service_name]
                
                if service_name == 'dashboard':
                    # Check if dashboard is responsive
                    status = service.get_dashboard_status()
                    is_healthy = status.get('status') != 'error'
                elif service_name == 'log_rotation':
                    # Check if log rotation is working
                    stats = service.get_statistics()
                    is_healthy = stats.get('is_running', False)
                elif service_name == 'performance_monitor':
                    # Check if performance monitor is collecting data
                    perf_data = service.get_current_performance()
                    is_healthy = 'error' not in perf_data
                else:
                    is_healthy = True
                    
                health['services'][service_name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'responsive': True
                }
                
                if not is_healthy:
                    health['overall_health'] = 'degraded'
                    health['issues'].append(f"Service {service_name} is unhealthy")
                    
            except Exception as e:
                health['services'][service_name] = {
                    'status': 'error',
                    'responsive': False,
                    'error': str(e)
                }
                health['overall_health'] = 'degraded'
                health['issues'].append(f"Service {service_name} health check failed: {str(e)}")
                
        return health
        
    async def run_maintenance_cycle(self) -> Dict[str, Any]:
        """Run a manual maintenance cycle."""
        if 'maintenance' not in self.services:
            return {'error': 'Maintenance service not running'}
            
        try:
            maintenance_service = self.services['maintenance']
            results = await maintenance_service.run_full_maintenance()
            return results
        except Exception as e:
            return {'error': f'Maintenance cycle failed: {str(e)}'}
            
    async def main_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info("Starting main monitoring loop...")
        
        while self.is_running and not self.shutdown_requested:
            try:
                # Perform periodic health checks
                health = await self.health_check()
                
                if health['overall_health'] != 'healthy':
                    self.logger.warning(f"System health degraded: {health['issues']}")
                    
                # Sleep for a minute before next check
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
                
        self.logger.info("Main monitoring loop stopped")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Production Monitoring Orchestrator')
    parser.add_argument('--config', default='config/production.yaml',
                       help='Configuration file path')
    parser.add_argument('--action', choices=[
        'start', 'stop', 'status', 'health', 'maintenance'
    ], default='start', help='Action to perform')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon (background process)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if args.daemon:
        # Log to file when running as daemon
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format=log_format,
            handlers=[
                logging.FileHandler('logs/monitoring_orchestrator.log'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format=log_format
        )
        
    # Create orchestrator
    orchestrator = ProductionMonitoringOrchestrator(args.config)
    
    try:
        if args.action == 'start':
            # Start all services
            startup_results = await orchestrator.start_all_services()
            print(json.dumps(startup_results, indent=2))
            
            if orchestrator.is_running:
                print("All services started successfully. Running main loop...")
                print("Press Ctrl+C to stop all services.")
                
                try:
                    await orchestrator.main_loop()
                except KeyboardInterrupt:
                    print("\nReceived interrupt signal, shutting down...")
                    
                # Stop all services
                shutdown_results = await orchestrator.stop_all_services()
                print("Shutdown results:")
                print(json.dumps(shutdown_results, indent=2))
            else:
                print("Failed to start all services. Check logs for details.")
                return 1
                
        elif args.action == 'stop':
            # This would typically connect to a running instance
            print("Stop action not implemented for external control")
            return 1
            
        elif args.action == 'status':
            # This would typically connect to a running instance
            print("Status action not implemented for external control")
            return 1
            
        elif args.action == 'health':
            # This would typically connect to a running instance
            print("Health action not implemented for external control")
            return 1
            
        elif args.action == 'maintenance':
            # Run maintenance cycle
            maintenance_results = await orchestrator.run_maintenance_cycle()
            print(json.dumps(maintenance_results, indent=2))
            
    except Exception as e:
        logging.error(f"Orchestrator error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)