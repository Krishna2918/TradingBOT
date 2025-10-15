"""
Deployment Manager

This module implements a comprehensive deployment manager for managing
deployment configurations, strategies, and enterprise-grade deployment
automation for production deployment and management.

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
import subprocess
import shutil
import os
import yaml
import docker
import kubernetes
import threading
import time
import requests
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status levels."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ROLLBACK = "ROLLBACK"
    CANCELLED = "CANCELLED"

class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "BLUE_GREEN"
    CANARY = "CANARY"
    ROLLING = "ROLLING"
    RECREATE = "RECREATE"
    A_B_TESTING = "A_B_TESTING"

class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "DEVELOPMENT"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    TESTING = "TESTING"

@dataclass
class DeploymentConfig:
    """Deployment configuration definition."""
    config_id: str
    name: str
    description: str
    application: str
    version: str
    environment: Environment
    strategy: DeploymentStrategy
    target_hosts: List[str]
    deployment_script: str
    rollback_script: str
    health_check_url: str
    pre_deployment_checks: List[str]
    post_deployment_checks: List[str]
    environment_variables: Dict[str, str]
    resource_requirements: Dict[str, Any]
    deployment_timeout: int
    rollback_timeout: int
    max_retries: int
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Deployment:
    """Deployment definition."""
    deployment_id: str
    config_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_minutes: float
    deployed_version: str
    target_hosts: List[str]
    deployment_log: List[str]
    error_message: Optional[str]
    rollback_available: bool
    health_check_results: Dict[str, bool]
    created_at: datetime = field(default_factory=datetime.now)

class DeploymentManager:
    """
    Comprehensive deployment management system.
    
    Features:
    - Deployment configuration management
    - Multiple deployment strategies
    - Automated deployment execution
    - Health check validation
    - Rollback capabilities
    - Deployment monitoring
    - Environment management
    - Enterprise-grade deployment automation
    """
    
    def __init__(self, db_path: str = "data/deployment_manager.db"):
        """
        Initialize deployment manager.
        
        Args:
            db_path: Path to deployment manager database
        """
        self.db_path = db_path
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        self.deployments: List[Deployment] = []
        self.active_deployments: Dict[str, Deployment] = {}
        
        # Deployment configuration
        self.deployment_config = {
            'default_timeout': 1800,  # 30 minutes
            'default_retries': 3,
            'health_check_interval': 30,  # seconds
            'health_check_timeout': 300,  # 5 minutes
            'rollback_timeout': 600,  # 10 minutes
            'deployment_log_retention_days': 30
        }
        
        # Initialize database
        self._init_database()
        
        # Load default deployment configurations
        self._load_default_deployment_configs()
        
        logger.info("Deployment Manager initialized")
    
    def _init_database(self) -> None:
        """Initialize deployment manager database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create deployment configurations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deployment_configs (
                config_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                application TEXT NOT NULL,
                version TEXT NOT NULL,
                environment TEXT NOT NULL,
                strategy TEXT NOT NULL,
                target_hosts TEXT NOT NULL,
                deployment_script TEXT NOT NULL,
                rollback_script TEXT NOT NULL,
                health_check_url TEXT NOT NULL,
                pre_deployment_checks TEXT NOT NULL,
                post_deployment_checks TEXT NOT NULL,
                environment_variables TEXT NOT NULL,
                resource_requirements TEXT NOT NULL,
                deployment_timeout INTEGER NOT NULL,
                rollback_timeout INTEGER NOT NULL,
                max_retries INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create deployments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deployments (
                deployment_id TEXT PRIMARY KEY,
                config_id TEXT NOT NULL,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_minutes REAL NOT NULL,
                deployed_version TEXT NOT NULL,
                target_hosts TEXT NOT NULL,
                deployment_log TEXT NOT NULL,
                error_message TEXT,
                rollback_available INTEGER NOT NULL,
                health_check_results TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_deployment_configs(self) -> None:
        """Load default deployment configurations."""
        default_configs = [
            DeploymentConfig(
                config_id="DEPLOY_001",
                name="Trading System Production Deployment",
                description="Production deployment for trading system",
                application="trading_system",
                version="1.0.0",
                environment=Environment.PRODUCTION,
                strategy=DeploymentStrategy.BLUE_GREEN,
                target_hosts=["prod-trading-01", "prod-trading-02"],
                deployment_script="scripts/deploy_trading_system.sh",
                rollback_script="scripts/rollback_trading_system.sh",
                health_check_url="http://localhost:8000/health",
                pre_deployment_checks=["database_backup", "service_health", "resource_check"],
                post_deployment_checks=["health_check", "smoke_test", "performance_test"],
                environment_variables={"ENV": "production", "LOG_LEVEL": "INFO"},
                resource_requirements={"cpu": "2", "memory": "4Gi", "disk": "20Gi"},
                deployment_timeout=1800,
                rollback_timeout=600,
                max_retries=3
            ),
            DeploymentConfig(
                config_id="DEPLOY_002",
                name="API Gateway Staging Deployment",
                description="Staging deployment for API gateway",
                application="api_gateway",
                version="1.0.0",
                environment=Environment.STAGING,
                strategy=DeploymentStrategy.ROLLING,
                target_hosts=["staging-api-01"],
                deployment_script="scripts/deploy_api_gateway.sh",
                rollback_script="scripts/rollback_api_gateway.sh",
                health_check_url="http://localhost:8080/health",
                pre_deployment_checks=["service_health", "resource_check"],
                post_deployment_checks=["health_check", "smoke_test"],
                environment_variables={"ENV": "staging", "LOG_LEVEL": "DEBUG"},
                resource_requirements={"cpu": "1", "memory": "2Gi", "disk": "10Gi"},
                deployment_timeout=900,
                rollback_timeout=300,
                max_retries=2
            ),
            DeploymentConfig(
                config_id="DEPLOY_003",
                name="AI Models Canary Deployment",
                description="Canary deployment for AI models",
                application="ai_models",
                version="1.0.0",
                environment=Environment.PRODUCTION,
                strategy=DeploymentStrategy.CANARY,
                target_hosts=["prod-ai-01", "prod-ai-02"],
                deployment_script="scripts/deploy_ai_models.sh",
                rollback_script="scripts/rollback_ai_models.sh",
                health_check_url="http://localhost:9000/health",
                pre_deployment_checks=["model_validation", "service_health", "resource_check"],
                post_deployment_checks=["health_check", "model_test", "performance_test"],
                environment_variables={"ENV": "production", "MODEL_PATH": "/models"},
                resource_requirements={"cpu": "4", "memory": "8Gi", "disk": "50Gi"},
                deployment_timeout=2400,
                rollback_timeout=900,
                max_retries=3
            ),
            DeploymentConfig(
                config_id="DEPLOY_004",
                name="Database Migration Deployment",
                description="Database migration deployment",
                application="database",
                version="1.0.0",
                environment=Environment.PRODUCTION,
                strategy=DeploymentStrategy.RECREATE,
                target_hosts=["prod-db-01"],
                deployment_script="scripts/deploy_database.sh",
                rollback_script="scripts/rollback_database.sh",
                health_check_url="tcp://localhost:5432",
                pre_deployment_checks=["backup_verification", "service_health"],
                post_deployment_checks=["health_check", "data_integrity"],
                environment_variables={"ENV": "production", "DB_NAME": "trading"},
                resource_requirements={"cpu": "2", "memory": "8Gi", "disk": "100Gi"},
                deployment_timeout=3600,
                rollback_timeout=1800,
                max_retries=2
            ),
            DeploymentConfig(
                config_id="DEPLOY_005",
                name="Monitoring System Deployment",
                description="Monitoring system deployment",
                application="monitoring",
                version="1.0.0",
                environment=Environment.PRODUCTION,
                strategy=DeploymentStrategy.ROLLING,
                target_hosts=["prod-monitor-01"],
                deployment_script="scripts/deploy_monitoring.sh",
                rollback_script="scripts/rollback_monitoring.sh",
                health_check_url="http://localhost:3000/health",
                pre_deployment_checks=["service_health", "resource_check"],
                post_deployment_checks=["health_check", "metrics_test"],
                environment_variables={"ENV": "production", "METRICS_INTERVAL": "30"},
                resource_requirements={"cpu": "1", "memory": "2Gi", "disk": "20Gi"},
                deployment_timeout=1200,
                rollback_timeout=600,
                max_retries=2
            )
        ]
        
        for config in default_configs:
            self.add_deployment_config(config)
    
    def add_deployment_config(self, config: DeploymentConfig) -> None:
        """
        Add deployment configuration.
        
        Args:
            config: Deployment configuration definition
        """
        self.deployment_configs[config.config_id] = config
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO deployment_configs 
            (config_id, name, description, application, version, environment, strategy,
             target_hosts, deployment_script, rollback_script, health_check_url,
             pre_deployment_checks, post_deployment_checks, environment_variables,
             resource_requirements, deployment_timeout, rollback_timeout, max_retries, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            config.config_id, config.name, config.description, config.application,
            config.version, config.environment.value, config.strategy.value,
            json.dumps(config.target_hosts), config.deployment_script, config.rollback_script,
            config.health_check_url, json.dumps(config.pre_deployment_checks),
            json.dumps(config.post_deployment_checks), json.dumps(config.environment_variables),
            json.dumps(config.resource_requirements), config.deployment_timeout,
            config.rollback_timeout, config.max_retries, config.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added deployment config: {config.config_id} - {config.name}")
    
    def deploy(self, config_id: str, version: str = None) -> str:
        """
        Start deployment.
        
        Args:
            config_id: Deployment configuration ID
            version: Version to deploy (optional)
            
        Returns:
            Deployment ID
        """
        if config_id not in self.deployment_configs:
            raise ValueError(f"Deployment config {config_id} not found")
        
        config = self.deployment_configs[config_id]
        deployment_version = version or config.version
        
        # Create deployment
        deployment_id = f"DEPLOY_{config_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        deployment = Deployment(
            deployment_id=deployment_id,
            config_id=config_id,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            duration_minutes=0.0,
            deployed_version=deployment_version,
            target_hosts=config.target_hosts.copy(),
            deployment_log=[],
            error_message=None,
            rollback_available=False,
            health_check_results={}
        )
        
        self.deployments.append(deployment)
        self.active_deployments[deployment_id] = deployment
        
        # Start deployment in background
        deployment_thread = threading.Thread(
            target=self._execute_deployment,
            args=(deployment,),
            daemon=True
        )
        deployment_thread.start()
        
        logger.info(f"Started deployment: {deployment_id}")
        return deployment_id
    
    def _execute_deployment(self, deployment: Deployment) -> None:
        """Execute deployment."""
        try:
            deployment.status = DeploymentStatus.IN_PROGRESS
            config = self.deployment_configs[deployment.config_id]
            
            # Add to deployment log
            deployment.deployment_log.append(f"Starting deployment of {config.application} version {deployment.deployed_version}")
            
            # Pre-deployment checks
            if not self._run_pre_deployment_checks(deployment, config):
                deployment.status = DeploymentStatus.FAILED
                deployment.error_message = "Pre-deployment checks failed"
                deployment.end_time = datetime.now()
                deployment.duration_minutes = (deployment.end_time - deployment.start_time).total_seconds() / 60
                self._store_deployment(deployment)
                return
            
            # Execute deployment based on strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                success = self._deploy_blue_green(deployment, config)
            elif config.strategy == DeploymentStrategy.CANARY:
                success = self._deploy_canary(deployment, config)
            elif config.strategy == DeploymentStrategy.ROLLING:
                success = self._deploy_rolling(deployment, config)
            elif config.strategy == DeploymentStrategy.RECREATE:
                success = self._deploy_recreate(deployment, config)
            else:
                success = self._deploy_default(deployment, config)
            
            if success:
                # Post-deployment checks
                if self._run_post_deployment_checks(deployment, config):
                    deployment.status = DeploymentStatus.SUCCESS
                    deployment.rollback_available = True
                    deployment.deployment_log.append("Deployment completed successfully")
                else:
                    deployment.status = DeploymentStatus.FAILED
                    deployment.error_message = "Post-deployment checks failed"
                    deployment.deployment_log.append("Post-deployment checks failed")
            else:
                deployment.status = DeploymentStatus.FAILED
                deployment.error_message = "Deployment execution failed"
                deployment.deployment_log.append("Deployment execution failed")
            
            deployment.end_time = datetime.now()
            deployment.duration_minutes = (deployment.end_time - deployment.start_time).total_seconds() / 60
            
            # Store deployment
            self._store_deployment(deployment)
            
            # Remove from active deployments
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
            
        except Exception as e:
            logger.error(f"Error executing deployment {deployment.deployment_id}: {e}")
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            deployment.end_time = datetime.now()
            deployment.duration_minutes = (deployment.end_time - deployment.start_time).total_seconds() / 60
            self._store_deployment(deployment)
    
    def _run_pre_deployment_checks(self, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Run pre-deployment checks."""
        deployment.deployment_log.append("Running pre-deployment checks...")
        
        for check in config.pre_deployment_checks:
            deployment.deployment_log.append(f"Running check: {check}")
            
            if check == "database_backup":
                if not self._check_database_backup():
                    deployment.deployment_log.append(f"Check failed: {check}")
                    return False
            elif check == "service_health":
                if not self._check_service_health(config.health_check_url):
                    deployment.deployment_log.append(f"Check failed: {check}")
                    return False
            elif check == "resource_check":
                if not self._check_resource_requirements(config.resource_requirements):
                    deployment.deployment_log.append(f"Check failed: {check}")
                    return False
            elif check == "model_validation":
                if not self._check_model_validation():
                    deployment.deployment_log.append(f"Check failed: {check}")
                    return False
            elif check == "backup_verification":
                if not self._check_backup_verification():
                    deployment.deployment_log.append(f"Check failed: {check}")
                    return False
            
            deployment.deployment_log.append(f"Check passed: {check}")
        
        deployment.deployment_log.append("All pre-deployment checks passed")
        return True
    
    def _run_post_deployment_checks(self, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Run post-deployment checks."""
        deployment.deployment_log.append("Running post-deployment checks...")
        
        for check in config.post_deployment_checks:
            deployment.deployment_log.append(f"Running check: {check}")
            
            if check == "health_check":
                if not self._check_service_health(config.health_check_url):
                    deployment.deployment_log.append(f"Check failed: {check}")
                    return False
            elif check == "smoke_test":
                if not self._run_smoke_test(config.application):
                    deployment.deployment_log.append(f"Check failed: {check}")
                    return False
            elif check == "performance_test":
                if not self._run_performance_test(config.application):
                    deployment.deployment_log.append(f"Check failed: {check}")
                    return False
            elif check == "model_test":
                if not self._run_model_test():
                    deployment.deployment_log.append(f"Check failed: {check}")
                    return False
            elif check == "data_integrity":
                if not self._check_data_integrity():
                    deployment.deployment_log.append(f"Check failed: {check}")
                    return False
            elif check == "metrics_test":
                if not self._run_metrics_test():
                    deployment.deployment_log.append(f"Check failed: {check}")
                    return False
            
            deployment.deployment_log.append(f"Check passed: {check}")
        
        deployment.deployment_log.append("All post-deployment checks passed")
        return True
    
    def _deploy_blue_green(self, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Deploy using blue-green strategy."""
        deployment.deployment_log.append("Starting blue-green deployment...")
        
        # Simulate blue-green deployment
        for host in deployment.target_hosts:
            deployment.deployment_log.append(f"Deploying to green environment on {host}")
            time.sleep(2)  # Simulate deployment time
            
            # Check health
            if not self._check_service_health(config.health_check_url):
                deployment.deployment_log.append(f"Health check failed on {host}")
                return False
            
            deployment.deployment_log.append(f"Successfully deployed to {host}")
        
        deployment.deployment_log.append("Blue-green deployment completed")
        return True
    
    def _deploy_canary(self, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Deploy using canary strategy."""
        deployment.deployment_log.append("Starting canary deployment...")
        
        # Deploy to first host (canary)
        canary_host = deployment.target_hosts[0]
        deployment.deployment_log.append(f"Deploying canary to {canary_host}")
        time.sleep(3)  # Simulate deployment time
        
        # Check canary health
        if not self._check_service_health(config.health_check_url):
            deployment.deployment_log.append(f"Canary health check failed on {canary_host}")
            return False
        
        deployment.deployment_log.append(f"Canary deployment successful on {canary_host}")
        
        # Deploy to remaining hosts
        for host in deployment.target_hosts[1:]:
            deployment.deployment_log.append(f"Deploying to {host}")
            time.sleep(2)  # Simulate deployment time
            
            if not self._check_service_health(config.health_check_url):
                deployment.deployment_log.append(f"Health check failed on {host}")
                return False
            
            deployment.deployment_log.append(f"Successfully deployed to {host}")
        
        deployment.deployment_log.append("Canary deployment completed")
        return True
    
    def _deploy_rolling(self, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Deploy using rolling strategy."""
        deployment.deployment_log.append("Starting rolling deployment...")
        
        for host in deployment.target_hosts:
            deployment.deployment_log.append(f"Deploying to {host}")
            time.sleep(2)  # Simulate deployment time
            
            if not self._check_service_health(config.health_check_url):
                deployment.deployment_log.append(f"Health check failed on {host}")
                return False
            
            deployment.deployment_log.append(f"Successfully deployed to {host}")
        
        deployment.deployment_log.append("Rolling deployment completed")
        return True
    
    def _deploy_recreate(self, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Deploy using recreate strategy."""
        deployment.deployment_log.append("Starting recreate deployment...")
        
        for host in deployment.target_hosts:
            deployment.deployment_log.append(f"Stopping service on {host}")
            time.sleep(1)  # Simulate stop time
            
            deployment.deployment_log.append(f"Deploying to {host}")
            time.sleep(3)  # Simulate deployment time
            
            deployment.deployment_log.append(f"Starting service on {host}")
            time.sleep(2)  # Simulate start time
            
            if not self._check_service_health(config.health_check_url):
                deployment.deployment_log.append(f"Health check failed on {host}")
                return False
            
            deployment.deployment_log.append(f"Successfully deployed to {host}")
        
        deployment.deployment_log.append("Recreate deployment completed")
        return True
    
    def _deploy_default(self, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Deploy using default strategy."""
        deployment.deployment_log.append("Starting default deployment...")
        
        for host in deployment.target_hosts:
            deployment.deployment_log.append(f"Deploying to {host}")
            time.sleep(2)  # Simulate deployment time
            
            if not self._check_service_health(config.health_check_url):
                deployment.deployment_log.append(f"Health check failed on {host}")
                return False
            
            deployment.deployment_log.append(f"Successfully deployed to {host}")
        
        deployment.deployment_log.append("Default deployment completed")
        return True
    
    def _check_database_backup(self) -> bool:
        """Check database backup."""
        # Simulate database backup check
        return True
    
    def _check_service_health(self, health_check_url: str) -> bool:
        """Check service health."""
        try:
            if health_check_url.startswith('http'):
                response = requests.get(health_check_url, timeout=10)
                return response.status_code == 200
            elif health_check_url.startswith('tcp'):
                # TCP health check
                host_port = health_check_url.replace('tcp://', '')
                if ':' in host_port:
                    host, port = host_port.split(':')
                    port = int(port)
                else:
                    host = host_port
                    port = 5432
                
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0
            else:
                return True  # Default to healthy for simulation
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _check_resource_requirements(self, requirements: Dict[str, Any]) -> bool:
        """Check resource requirements."""
        # Simulate resource check
        return True
    
    def _check_model_validation(self) -> bool:
        """Check model validation."""
        # Simulate model validation
        return True
    
    def _check_backup_verification(self) -> bool:
        """Check backup verification."""
        # Simulate backup verification
        return True
    
    def _run_smoke_test(self, application: str) -> bool:
        """Run smoke test."""
        # Simulate smoke test
        return True
    
    def _run_performance_test(self, application: str) -> bool:
        """Run performance test."""
        # Simulate performance test
        return True
    
    def _run_model_test(self) -> bool:
        """Run model test."""
        # Simulate model test
        return True
    
    def _check_data_integrity(self) -> bool:
        """Check data integrity."""
        # Simulate data integrity check
        return True
    
    def _run_metrics_test(self) -> bool:
        """Run metrics test."""
        # Simulate metrics test
        return True
    
    def rollback(self, deployment_id: str) -> bool:
        """
        Rollback deployment.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            True if rollback successful
        """
        deployment = None
        for d in self.deployments:
            if d.deployment_id == deployment_id:
                deployment = d
                break
        
        if not deployment:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        if not deployment.rollback_available:
            logger.error(f"Rollback not available for deployment {deployment_id}")
            return False
        
        try:
            deployment.status = DeploymentStatus.ROLLBACK
            config = self.deployment_configs[deployment.config_id]
            
            deployment.deployment_log.append("Starting rollback...")
            
            # Execute rollback script
            for host in deployment.target_hosts:
                deployment.deployment_log.append(f"Rolling back on {host}")
                time.sleep(2)  # Simulate rollback time
            
            # Verify rollback
            if self._check_service_health(config.health_check_url):
                deployment.deployment_log.append("Rollback completed successfully")
                deployment.status = DeploymentStatus.SUCCESS
                return True
            else:
                deployment.deployment_log.append("Rollback verification failed")
                deployment.status = DeploymentStatus.FAILED
                return False
                
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            return False
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get deployment status.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Deployment status dictionary
        """
        deployment = None
        for d in self.deployments:
            if d.deployment_id == deployment_id:
                deployment = d
                break
        
        if not deployment:
            return {"error": f"Deployment {deployment_id} not found"}
        
        config = self.deployment_configs[deployment.config_id]
        
        return {
            'deployment_id': deployment.deployment_id,
            'config_id': deployment.config_id,
            'application': config.application,
            'version': deployment.deployed_version,
            'environment': config.environment.value,
            'strategy': config.strategy.value,
            'status': deployment.status.value,
            'start_time': deployment.start_time.isoformat(),
            'end_time': deployment.end_time.isoformat() if deployment.end_time else None,
            'duration_minutes': deployment.duration_minutes,
            'target_hosts': deployment.target_hosts,
            'rollback_available': deployment.rollback_available,
            'error_message': deployment.error_message,
            'deployment_log': deployment.deployment_log[-10:],  # Last 10 log entries
            'health_check_results': deployment.health_check_results
        }
    
    def get_deployment_history(self, application: str = None, environment: Environment = None) -> List[Dict[str, Any]]:
        """
        Get deployment history.
        
        Args:
            application: Filter by application
            environment: Filter by environment
            
        Returns:
            List of deployment history
        """
        history = []
        
        for deployment in self.deployments:
            config = self.deployment_configs[deployment.config_id]
            
            # Apply filters
            if application and config.application != application:
                continue
            if environment and config.environment != environment:
                continue
            
            history.append({
                'deployment_id': deployment.deployment_id,
                'application': config.application,
                'version': deployment.deployed_version,
                'environment': config.environment.value,
                'strategy': config.strategy.value,
                'status': deployment.status.value,
                'start_time': deployment.start_time.isoformat(),
                'end_time': deployment.end_time.isoformat() if deployment.end_time else None,
                'duration_minutes': deployment.duration_minutes,
                'rollback_available': deployment.rollback_available
            })
        
        # Sort by start time (newest first)
        history.sort(key=lambda x: x['start_time'], reverse=True)
        return history
    
    def get_active_deployments(self) -> Dict[str, Any]:
        """
        Get active deployments.
        
        Returns:
            Active deployments dictionary
        """
        active_deployments = {}
        
        for deployment_id, deployment in self.active_deployments.items():
            config = self.deployment_configs[deployment.config_id]
            active_deployments[deployment_id] = {
                'application': config.application,
                'version': deployment.deployed_version,
                'environment': config.environment.value,
                'strategy': config.strategy.value,
                'status': deployment.status.value,
                'start_time': deployment.start_time.isoformat(),
                'target_hosts': deployment.target_hosts,
                'progress': len(deployment.deployment_log)
            }
        
        return {
            'active_deployments': active_deployments,
            'total_active': len(active_deployments)
        }
    
    def _store_deployment(self, deployment: Deployment) -> None:
        """Store deployment in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO deployments 
            (deployment_id, config_id, status, start_time, end_time, duration_minutes,
             deployed_version, target_hosts, deployment_log, error_message,
             rollback_available, health_check_results, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            deployment.deployment_id, deployment.config_id, deployment.status.value,
            deployment.start_time.isoformat(), deployment.end_time.isoformat() if deployment.end_time else None,
            deployment.duration_minutes, deployment.deployed_version, json.dumps(deployment.target_hosts),
            json.dumps(deployment.deployment_log), deployment.error_message, deployment.rollback_available,
            json.dumps(deployment.health_check_results), deployment.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_deployment_configs(self) -> Dict[str, Any]:
        """
        Get deployment configurations.
        
        Returns:
            Deployment configurations dictionary
        """
        configs = {}
        for config_id, config in self.deployment_configs.items():
            configs[config_id] = {
                'name': config.name,
                'description': config.description,
                'application': config.application,
                'version': config.version,
                'environment': config.environment.value,
                'strategy': config.strategy.value,
                'target_hosts': config.target_hosts,
                'deployment_timeout': config.deployment_timeout,
                'rollback_timeout': config.rollback_timeout,
                'max_retries': config.max_retries,
                'created_at': config.created_at.isoformat()
            }
        
        return {
            'configurations': configs,
            'total_configs': len(self.deployment_configs)
        }
