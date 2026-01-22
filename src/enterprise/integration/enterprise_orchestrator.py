"""
Enterprise Orchestrator

This module implements a comprehensive enterprise orchestrator for managing
system components, integration configurations, and enterprise-grade system
orchestration for production deployment and management.

Author: AI Trading System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import sqlite3
from pathlib import Path
import asyncio
import threading
import time
import subprocess
import psutil
import requests
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ComponentStatus(Enum):
    """Component status levels."""
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

class OrchestrationStatus(Enum):
    """Orchestration status levels."""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"

class ComponentType(Enum):
    """Component types."""
    TRADING_SYSTEM = "TRADING_SYSTEM"
    API_GATEWAY = "API_GATEWAY"
    DATABASE = "DATABASE"
    AI_MODELS = "AI_MODELS"
    MARKET_DATA = "MARKET_DATA"
    MONITORING = "MONITORING"
    ALERTING = "ALERTING"
    DASHBOARD = "DASHBOARD"
    BACKUP = "BACKUP"
    SECURITY = "SECURITY"

@dataclass
class SystemComponent:
    """System component definition."""
    component_id: str
    component_type: ComponentType
    name: str
    description: str
    host: str
    port: int
    health_check_url: str
    dependencies: List[str]
    startup_order: int
    shutdown_order: int
    restart_policy: str
    resource_limits: Dict[str, Any]
    environment_variables: Dict[str, str]
    status: ComponentStatus = ComponentStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    startup_time: Optional[datetime] = None
    shutdown_time: Optional[datetime] = None
    error_count: int = 0
    restart_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class IntegrationConfig:
    """Integration configuration definition."""
    config_id: str
    name: str
    description: str
    components: List[str]
    startup_sequence: List[str]
    shutdown_sequence: List[str]
    health_check_interval: int
    restart_policy: str
    monitoring_enabled: bool
    alerting_enabled: bool
    backup_enabled: bool
    scaling_enabled: bool
    created_at: datetime = field(default_factory=datetime.now)

class EnterpriseOrchestrator:
    """
    Comprehensive enterprise orchestrator system.
    
    Features:
    - System component management
    - Integration configuration management
    - Health monitoring and management
    - Automated startup and shutdown sequences
    - Component dependency management
    - Resource monitoring and management
    - Error handling and recovery
    - Enterprise-grade orchestration
    """
    
    def __init__(self, db_path: str = "data/enterprise_orchestrator.db"):
        """
        Initialize enterprise orchestrator.
        
        Args:
            db_path: Path to enterprise orchestrator database
        """
        self.db_path = db_path
        self.components: Dict[str, SystemComponent] = {}
        self.integration_configs: Dict[str, IntegrationConfig] = {}
        self.orchestration_status = OrchestrationStatus.STOPPED
        self.orchestration_thread = None
        self.health_check_thread = None
        self.running = False
        
        # Orchestration configuration
        self.orchestration_config = {
            'health_check_interval': 30,  # seconds
            'startup_timeout': 300,       # seconds
            'shutdown_timeout': 120,      # seconds
            'restart_delay': 10,          # seconds
            'max_restart_attempts': 3,
            'dependency_check_interval': 5,  # seconds
            'resource_check_interval': 60    # seconds
        }
        
        # Initialize database
        self._init_database()
        
        # Load default components and configurations
        self._load_default_components()
        self._load_default_integration_configs()
        
        logger.info("Enterprise Orchestrator initialized")
    
    def _init_database(self) -> None:
        """Initialize enterprise orchestrator database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create system components table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_components (
                component_id TEXT PRIMARY KEY,
                component_type TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                host TEXT NOT NULL,
                port INTEGER NOT NULL,
                health_check_url TEXT NOT NULL,
                dependencies TEXT NOT NULL,
                startup_order INTEGER NOT NULL,
                shutdown_order INTEGER NOT NULL,
                restart_policy TEXT NOT NULL,
                resource_limits TEXT NOT NULL,
                environment_variables TEXT NOT NULL,
                status TEXT NOT NULL,
                last_health_check TEXT,
                startup_time TEXT,
                shutdown_time TEXT,
                error_count INTEGER NOT NULL,
                restart_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create integration configurations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS integration_configs (
                config_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                components TEXT NOT NULL,
                startup_sequence TEXT NOT NULL,
                shutdown_sequence TEXT NOT NULL,
                health_check_interval INTEGER NOT NULL,
                restart_policy TEXT NOT NULL,
                monitoring_enabled INTEGER NOT NULL,
                alerting_enabled INTEGER NOT NULL,
                backup_enabled INTEGER NOT NULL,
                scaling_enabled INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create orchestration events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orchestration_events (
                event_id TEXT PRIMARY KEY,
                component_id TEXT,
                event_type TEXT NOT NULL,
                event_message TEXT,
                event_data TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_components(self) -> None:
        """Load default system components."""
        default_components = [
            SystemComponent(
                component_id="COMP_001",
                component_type=ComponentType.DATABASE,
                name="Trading Database",
                description="Primary trading system database",
                host="localhost",
                port=5432,
                health_check_url="tcp://localhost:5432",
                dependencies=[],
                startup_order=1,
                shutdown_order=10,
                restart_policy="always",
                resource_limits={"cpu": "2", "memory": "4Gi", "disk": "100Gi"},
                environment_variables={"POSTGRES_DB": "trading", "POSTGRES_USER": "trader"}
            ),
            SystemComponent(
                component_id="COMP_002",
                component_type=ComponentType.MARKET_DATA,
                name="Market Data Service",
                description="Real-time market data feed service",
                host="localhost",
                port=7000,
                health_check_url="http://localhost:7000/health",
                dependencies=["COMP_001"],
                startup_order=2,
                shutdown_order=9,
                restart_policy="always",
                resource_limits={"cpu": "1", "memory": "2Gi", "disk": "10Gi"},
                environment_variables={"MARKET_DATA_API_KEY": "api_key", "FEED_TYPE": "real_time"}
            ),
            SystemComponent(
                component_id="COMP_003",
                component_type=ComponentType.AI_MODELS,
                name="AI Model Service",
                description="AI model prediction and analysis service",
                host="localhost",
                port=9000,
                health_check_url="http://localhost:9000/health",
                dependencies=["COMP_001", "COMP_002"],
                startup_order=3,
                shutdown_order=8,
                restart_policy="always",
                resource_limits={"cpu": "4", "memory": "8Gi", "disk": "20Gi"},
                environment_variables={"MODEL_PATH": "/models", "GPU_ENABLED": "true"}
            ),
            SystemComponent(
                component_id="COMP_004",
                component_type=ComponentType.API_GATEWAY,
                name="API Gateway",
                description="API gateway and routing service",
                host="localhost",
                port=8080,
                health_check_url="http://localhost:8080/health",
                dependencies=["COMP_001", "COMP_002", "COMP_003"],
                startup_order=4,
                shutdown_order=7,
                restart_policy="always",
                resource_limits={"cpu": "1", "memory": "2Gi", "disk": "5Gi"},
                environment_variables={"API_VERSION": "v1", "RATE_LIMIT": "1000"}
            ),
            SystemComponent(
                component_id="COMP_005",
                component_type=ComponentType.TRADING_SYSTEM,
                name="Trading Engine",
                description="Core trading system engine",
                host="localhost",
                port=8000,
                health_check_url="http://localhost:8000/health",
                dependencies=["COMP_001", "COMP_002", "COMP_003", "COMP_004"],
                startup_order=5,
                shutdown_order=6,
                restart_policy="always",
                resource_limits={"cpu": "2", "memory": "4Gi", "disk": "10Gi"},
                environment_variables={"TRADING_MODE": "live", "RISK_LIMIT": "100000"}
            ),
            SystemComponent(
                component_id="COMP_006",
                component_type=ComponentType.MONITORING,
                name="Monitoring Service",
                description="System monitoring and metrics collection",
                host="localhost",
                port=3000,
                health_check_url="http://localhost:3000/health",
                dependencies=[],
                startup_order=6,
                shutdown_order=5,
                restart_policy="always",
                resource_limits={"cpu": "0.5", "memory": "1Gi", "disk": "5Gi"},
                environment_variables={"METRICS_INTERVAL": "30", "ALERT_ENABLED": "true"}
            ),
            SystemComponent(
                component_id="COMP_007",
                component_type=ComponentType.ALERTING,
                name="Alerting Service",
                description="Alert management and notification service",
                host="localhost",
                port=4000,
                health_check_url="http://localhost:4000/health",
                dependencies=["COMP_006"],
                startup_order=7,
                shutdown_order=4,
                restart_policy="always",
                resource_limits={"cpu": "0.5", "memory": "1Gi", "disk": "2Gi"},
                environment_variables={"ALERT_CHANNELS": "email,slack", "ESCALATION_ENABLED": "true"}
            ),
            SystemComponent(
                component_id="COMP_008",
                component_type=ComponentType.DASHBOARD,
                name="Dashboard Service",
                description="Web dashboard and visualization service",
                host="localhost",
                port=5000,
                health_check_url="http://localhost:5000/health",
                dependencies=["COMP_006", "COMP_007"],
                startup_order=8,
                shutdown_order=3,
                restart_policy="always",
                resource_limits={"cpu": "0.5", "memory": "1Gi", "disk": "3Gi"},
                environment_variables={"DASHBOARD_THEME": "dark", "REFRESH_INTERVAL": "30"}
            ),
            SystemComponent(
                component_id="COMP_009",
                component_type=ComponentType.BACKUP,
                name="Backup Service",
                description="Automated backup and recovery service",
                host="localhost",
                port=6000,
                health_check_url="http://localhost:6000/health",
                dependencies=["COMP_001"],
                startup_order=9,
                shutdown_order=2,
                restart_policy="always",
                resource_limits={"cpu": "0.5", "memory": "1Gi", "disk": "50Gi"},
                environment_variables={"BACKUP_SCHEDULE": "daily", "RETENTION_DAYS": "30"}
            ),
            SystemComponent(
                component_id="COMP_010",
                component_type=ComponentType.SECURITY,
                name="Security Service",
                description="Security monitoring and threat detection",
                host="localhost",
                port=7001,
                health_check_url="http://localhost:7001/health",
                dependencies=[],
                startup_order=10,
                shutdown_order=1,
                restart_policy="always",
                resource_limits={"cpu": "1", "memory": "2Gi", "disk": "10Gi"},
                environment_variables={"SECURITY_LEVEL": "high", "THREAT_DETECTION": "enabled"}
            )
        ]
        
        for component in default_components:
            self.add_component(component)
    
    def _load_default_integration_configs(self) -> None:
        """Load default integration configurations."""
        default_configs = [
            IntegrationConfig(
                config_id="CONFIG_001",
                name="Full System Integration",
                description="Complete system integration with all components",
                components=["COMP_001", "COMP_002", "COMP_003", "COMP_004", "COMP_005", 
                           "COMP_006", "COMP_007", "COMP_008", "COMP_009", "COMP_010"],
                startup_sequence=["COMP_001", "COMP_002", "COMP_003", "COMP_004", "COMP_005", 
                                "COMP_006", "COMP_007", "COMP_008", "COMP_009", "COMP_010"],
                shutdown_sequence=["COMP_010", "COMP_009", "COMP_008", "COMP_007", "COMP_006", 
                                 "COMP_005", "COMP_004", "COMP_003", "COMP_002", "COMP_001"],
                health_check_interval=30,
                restart_policy="always",
                monitoring_enabled=True,
                alerting_enabled=True,
                backup_enabled=True,
                scaling_enabled=True
            ),
            IntegrationConfig(
                config_id="CONFIG_002",
                name="Core Trading System",
                description="Core trading system components only",
                components=["COMP_001", "COMP_002", "COMP_003", "COMP_004", "COMP_005"],
                startup_sequence=["COMP_001", "COMP_002", "COMP_003", "COMP_004", "COMP_005"],
                shutdown_sequence=["COMP_005", "COMP_004", "COMP_003", "COMP_002", "COMP_001"],
                health_check_interval=30,
                restart_policy="always",
                monitoring_enabled=True,
                alerting_enabled=True,
                backup_enabled=True,
                scaling_enabled=False
            ),
            IntegrationConfig(
                config_id="CONFIG_003",
                name="Monitoring and Alerting",
                description="Monitoring and alerting services only",
                components=["COMP_006", "COMP_007", "COMP_008"],
                startup_sequence=["COMP_006", "COMP_007", "COMP_008"],
                shutdown_sequence=["COMP_008", "COMP_007", "COMP_006"],
                health_check_interval=60,
                restart_policy="always",
                monitoring_enabled=True,
                alerting_enabled=True,
                backup_enabled=False,
                scaling_enabled=False
            )
        ]
        
        for config in default_configs:
            self.add_integration_config(config)
    
    def add_component(self, component: SystemComponent) -> None:
        """
        Add system component.
        
        Args:
            component: System component definition
        """
        self.components[component.component_id] = component
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO system_components 
            (component_id, component_type, name, description, host, port, health_check_url,
             dependencies, startup_order, shutdown_order, restart_policy, resource_limits,
             environment_variables, status, last_health_check, startup_time, shutdown_time,
             error_count, restart_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            component.component_id, component.component_type.value, component.name, component.description,
            component.host, component.port, component.health_check_url, json.dumps(component.dependencies),
            component.startup_order, component.shutdown_order, component.restart_policy,
            json.dumps(component.resource_limits), json.dumps(component.environment_variables),
            component.status.value, component.last_health_check.isoformat() if component.last_health_check else None,
            component.startup_time.isoformat() if component.startup_time else None,
            component.shutdown_time.isoformat() if component.shutdown_time else None,
            component.error_count, component.restart_count, component.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added component: {component.component_id} - {component.name}")
    
    def add_integration_config(self, config: IntegrationConfig) -> None:
        """
        Add integration configuration.
        
        Args:
            config: Integration configuration definition
        """
        self.integration_configs[config.config_id] = config
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO integration_configs 
            (config_id, name, description, components, startup_sequence, shutdown_sequence,
             health_check_interval, restart_policy, monitoring_enabled, alerting_enabled,
             backup_enabled, scaling_enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            config.config_id, config.name, config.description, json.dumps(config.components),
            json.dumps(config.startup_sequence), json.dumps(config.shutdown_sequence),
            config.health_check_interval, config.restart_policy, config.monitoring_enabled,
            config.alerting_enabled, config.backup_enabled, config.scaling_enabled,
            config.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added integration config: {config.config_id} - {config.name}")
    
    def start_orchestration(self, config_id: str) -> bool:
        """
        Start system orchestration.
        
        Args:
            config_id: Integration configuration ID
            
        Returns:
            True if orchestration started successfully
        """
        if config_id not in self.integration_configs:
            logger.error(f"Integration config {config_id} not found")
            return False
        
        if self.orchestration_status != OrchestrationStatus.STOPPED:
            logger.warning("Orchestration is already running")
            return False
        
        try:
            self.orchestration_status = OrchestrationStatus.INITIALIZING
            config = self.integration_configs[config_id]
            
            # Start orchestration thread
            self.running = True
            self.orchestration_thread = threading.Thread(
                target=self._orchestration_loop, 
                args=(config,), 
                daemon=True
            )
            self.orchestration_thread.start()
            
            # Start health check thread
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop, 
                daemon=True
            )
            self.health_check_thread.start()
            
            logger.info(f"Started orchestration with config: {config_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting orchestration: {e}")
            self.orchestration_status = OrchestrationStatus.ERROR
            return False
    
    def stop_orchestration(self) -> bool:
        """
        Stop system orchestration.
        
        Returns:
            True if orchestration stopped successfully
        """
        try:
            self.orchestration_status = OrchestrationStatus.STOPPING
            self.running = False
            
            # Wait for threads to finish
            if self.orchestration_thread:
                self.orchestration_thread.join(timeout=30)
            if self.health_check_thread:
                self.health_check_thread.join(timeout=30)
            
            self.orchestration_status = OrchestrationStatus.STOPPED
            logger.info("Stopped orchestration")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping orchestration: {e}")
            self.orchestration_status = OrchestrationStatus.ERROR
            return False
    
    def _orchestration_loop(self, config: IntegrationConfig) -> None:
        """Main orchestration loop."""
        try:
            self.orchestration_status = OrchestrationStatus.RUNNING
            
            # Start components in sequence
            self._start_components(config.startup_sequence)
            
            # Main orchestration loop
            while self.running:
                # Check component health
                self._check_component_health()
                
                # Handle component failures
                self._handle_component_failures()
                
                # Check dependencies
                self._check_dependencies()
                
                # Resource monitoring
                self._monitor_resources()
                
                time.sleep(self.orchestration_config['health_check_interval'])
            
            # Stop components in sequence
            self._stop_components(config.shutdown_sequence)
            
        except Exception as e:
            logger.error(f"Error in orchestration loop: {e}")
            self.orchestration_status = OrchestrationStatus.ERROR
    
    def _start_components(self, startup_sequence: List[str]) -> None:
        """Start components in sequence."""
        for component_id in startup_sequence:
            if component_id not in self.components:
                logger.warning(f"Component {component_id} not found")
                continue
            
            component = self.components[component_id]
            
            # Check dependencies
            if not self._check_component_dependencies(component):
                logger.error(f"Dependencies not met for component {component_id}")
                continue
            
            # Start component
            if self._start_component(component):
                logger.info(f"Started component: {component_id}")
            else:
                logger.error(f"Failed to start component: {component_id}")
    
    def _stop_components(self, shutdown_sequence: List[str]) -> None:
        """Stop components in sequence."""
        for component_id in shutdown_sequence:
            if component_id not in self.components:
                continue
            
            component = self.components[component_id]
            
            # Stop component
            if self._stop_component(component):
                logger.info(f"Stopped component: {component_id}")
            else:
                logger.error(f"Failed to stop component: {component_id}")
    
    def _start_component(self, component: SystemComponent) -> bool:
        """Start a single component."""
        try:
            component.status = ComponentStatus.STARTING
            component.startup_time = datetime.now()
            
            # Simulate component startup
            # In a real implementation, this would start the actual service
            time.sleep(2)  # Simulate startup time
            
            # Check if component is healthy
            if self._check_component_health(component):
                component.status = ComponentStatus.RUNNING
                component.error_count = 0
                return True
            else:
                component.status = ComponentStatus.ERROR
                component.error_count += 1
                return False
                
        except Exception as e:
            logger.error(f"Error starting component {component.component_id}: {e}")
            component.status = ComponentStatus.ERROR
            component.error_count += 1
            return False
    
    def _stop_component(self, component: SystemComponent) -> bool:
        """Stop a single component."""
        try:
            component.status = ComponentStatus.STOPPING
            component.shutdown_time = datetime.now()
            
            # Simulate component shutdown
            # In a real implementation, this would stop the actual service
            time.sleep(1)  # Simulate shutdown time
            
            component.status = ComponentStatus.STOPPED
            return True
            
        except Exception as e:
            logger.error(f"Error stopping component {component.component_id}: {e}")
            component.status = ComponentStatus.ERROR
            return False
    
    def _check_component_health(self, component: SystemComponent = None) -> bool:
        """Check component health."""
        if component:
            return self._check_single_component_health(component)
        else:
            # Check all components
            for comp in self.components.values():
                self._check_single_component_health(comp)
            return True
    
    def _check_single_component_health(self, component: SystemComponent) -> bool:
        """Check health of a single component."""
        try:
            component.last_health_check = datetime.now()
            
            # Simulate health check
            # In a real implementation, this would make actual health check requests
            if component.health_check_url.startswith('http'):
                # HTTP health check
                response = requests.get(component.health_check_url, timeout=5)
                is_healthy = response.status_code == 200
            elif component.health_check_url.startswith('tcp'):
                # TCP health check
                host_port = component.health_check_url.replace('tcp://', '')
                if ':' in host_port:
                    host, port = host_port.split(':')
                    port = int(port)
                else:
                    host = host_port
                    port = component.port
                
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                is_healthy = result == 0
            else:
                # Default to healthy for simulation
                is_healthy = True
            
            if is_healthy:
                if component.status == ComponentStatus.ERROR:
                    component.status = ComponentStatus.RUNNING
                    component.error_count = 0
                return True
            else:
                component.status = ComponentStatus.ERROR
                component.error_count += 1
                return False
                
        except Exception as e:
            logger.error(f"Health check failed for component {component.component_id}: {e}")
            component.status = ComponentStatus.ERROR
            component.error_count += 1
            return False
    
    def _check_component_dependencies(self, component: SystemComponent) -> bool:
        """Check if component dependencies are met."""
        for dep_id in component.dependencies:
            if dep_id not in self.components:
                logger.error(f"Dependency {dep_id} not found for component {component.component_id}")
                return False
            
            dep_component = self.components[dep_id]
            if dep_component.status != ComponentStatus.RUNNING:
                logger.warning(f"Dependency {dep_id} not running for component {component.component_id}")
                return False
        
        return True
    
    def _handle_component_failures(self) -> None:
        """Handle component failures and restarts."""
        for component in self.components.values():
            if component.status == ComponentStatus.ERROR:
                if component.restart_policy == "always" and component.restart_count < self.orchestration_config['max_restart_attempts']:
                    logger.info(f"Restarting failed component: {component.component_id}")
                    time.sleep(self.orchestration_config['restart_delay'])
                    
                    if self._start_component(component):
                        component.restart_count += 1
                        logger.info(f"Successfully restarted component: {component.component_id}")
                    else:
                        logger.error(f"Failed to restart component: {component.component_id}")
    
    def _check_dependencies(self) -> None:
        """Check component dependencies."""
        for component in self.components.values():
            if component.status == ComponentStatus.RUNNING:
                if not self._check_component_dependencies(component):
                    logger.warning(f"Dependencies not met for running component {component.component_id}")
                    # In a real implementation, this might trigger a restart or shutdown
    
    def _monitor_resources(self) -> None:
        """Monitor system resources."""
        try:
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check if resources are within limits
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                logger.warning(f"High disk usage: {disk.percent}%")
                
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
    
    def _health_check_loop(self) -> None:
        """Health check loop."""
        while self.running:
            try:
                self._check_component_health()
                time.sleep(self.orchestration_config['health_check_interval'])
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(5)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status.
        
        Returns:
            System status dictionary
        """
        component_statuses = {}
        for component_id, component in self.components.items():
            component_statuses[component_id] = {
                'name': component.name,
                'type': component.component_type.value,
                'status': component.status.value,
                'host': component.host,
                'port': component.port,
                'last_health_check': component.last_health_check.isoformat() if component.last_health_check else None,
                'startup_time': component.startup_time.isoformat() if component.startup_time else None,
                'error_count': component.error_count,
                'restart_count': component.restart_count
            }
        
        return {
            'orchestration_status': self.orchestration_status.value,
            'total_components': len(self.components),
            'running_components': len([c for c in self.components.values() if c.status == ComponentStatus.RUNNING]),
            'error_components': len([c for c in self.components.values() if c.status == ComponentStatus.ERROR]),
            'components': component_statuses,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_component_status(self, component_id: str) -> Dict[str, Any]:
        """
        Get component status.
        
        Args:
            component_id: Component ID
            
        Returns:
            Component status dictionary
        """
        if component_id not in self.components:
            return {"error": f"Component {component_id} not found"}
        
        component = self.components[component_id]
        
        return {
            'component_id': component.component_id,
            'name': component.name,
            'type': component.component_type.value,
            'status': component.status.value,
            'host': component.host,
            'port': component.port,
            'health_check_url': component.health_check_url,
            'dependencies': component.dependencies,
            'startup_order': component.startup_order,
            'shutdown_order': component.shutdown_order,
            'restart_policy': component.restart_policy,
            'resource_limits': component.resource_limits,
            'environment_variables': component.environment_variables,
            'last_health_check': component.last_health_check.isoformat() if component.last_health_check else None,
            'startup_time': component.startup_time.isoformat() if component.startup_time else None,
            'shutdown_time': component.shutdown_time.isoformat() if component.shutdown_time else None,
            'error_count': component.error_count,
            'restart_count': component.restart_count,
            'created_at': component.created_at.isoformat()
        }
    
    def restart_component(self, component_id: str) -> bool:
        """
        Restart a component.
        
        Args:
            component_id: Component ID
            
        Returns:
            True if component restarted successfully
        """
        if component_id not in self.components:
            logger.error(f"Component {component_id} not found")
            return False
        
        component = self.components[component_id]
        
        # Stop component
        if self._stop_component(component):
            time.sleep(2)  # Wait before restart
            
            # Start component
            if self._start_component(component):
                logger.info(f"Successfully restarted component: {component_id}")
                return True
            else:
                logger.error(f"Failed to restart component: {component_id}")
                return False
        else:
            logger.error(f"Failed to stop component for restart: {component_id}")
            return False
    
    def get_integration_configs(self) -> Dict[str, Any]:
        """
        Get integration configurations.
        
        Returns:
            Integration configurations dictionary
        """
        configs = {}
        for config_id, config in self.integration_configs.items():
            configs[config_id] = {
                'name': config.name,
                'description': config.description,
                'components': config.components,
                'startup_sequence': config.startup_sequence,
                'shutdown_sequence': config.shutdown_sequence,
                'health_check_interval': config.health_check_interval,
                'restart_policy': config.restart_policy,
                'monitoring_enabled': config.monitoring_enabled,
                'alerting_enabled': config.alerting_enabled,
                'backup_enabled': config.backup_enabled,
                'scaling_enabled': config.scaling_enabled,
                'created_at': config.created_at.isoformat()
            }
        
        return {
            'configurations': configs,
            'total_configs': len(self.integration_configs)
        }
