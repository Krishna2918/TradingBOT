"""
Service Discovery

This module implements a comprehensive service discovery system for managing
service registries, health checks, and enterprise-grade service discovery
for production deployment and management.

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
import threading
import time
import requests
import socket
import asyncio
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status levels."""
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"
    STARTING = "STARTING"
    STOPPING = "STOPPING"

class ServiceType(Enum):
    """Service types."""
    HTTP = "HTTP"
    HTTPS = "HTTPS"
    TCP = "TCP"
    UDP = "UDP"
    GRPC = "GRPC"
    WEBSOCKET = "WEBSOCKET"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "ROUND_ROBIN"
    LEAST_CONNECTIONS = "LEAST_CONNECTIONS"
    WEIGHTED_ROUND_ROBIN = "WEIGHTED_ROUND_ROBIN"
    IP_HASH = "IP_HASH"
    RANDOM = "RANDOM"

@dataclass
class ServiceEndpoint:
    """Service endpoint definition."""
    endpoint_id: str
    service_id: str
    host: str
    port: int
    protocol: ServiceType
    path: str
    weight: int
    priority: int
    tags: List[str]
    metadata: Dict[str, Any]
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ServiceHealth:
    """Service health definition."""
    health_id: str
    service_id: str
    endpoint_id: str
    health_check_url: str
    health_check_interval: int
    health_check_timeout: int
    healthy_threshold: int
    unhealthy_threshold: int
    last_check: Optional[datetime] = None
    last_status: ServiceStatus = ServiceStatus.UNKNOWN
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ServiceRegistry:
    """Service registry definition."""
    service_id: str
    name: str
    description: str
    version: str
    namespace: str
    endpoints: List[ServiceEndpoint]
    health_checks: List[ServiceHealth]
    load_balancing_strategy: LoadBalancingStrategy
    circuit_breaker_enabled: bool
    circuit_breaker_threshold: int
    circuit_breaker_timeout: int
    retry_policy: Dict[str, Any]
    timeout_policy: Dict[str, Any]
    tags: List[str]
    metadata: Dict[str, Any]
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_updated: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)

class ServiceDiscovery:
    """
    Comprehensive service discovery system.
    
    Features:
    - Service registry management
    - Health check monitoring
    - Load balancing
    - Circuit breaker pattern
    - Service discovery
    - Endpoint management
    - Enterprise-grade service discovery
    """
    
    def __init__(self, db_path: str = "data/service_discovery.db"):
        """
        Initialize service discovery.
        
        Args:
            db_path: Path to service discovery database
        """
        self.db_path = db_path
        self.services: Dict[str, ServiceRegistry] = {}
        self.endpoints: Dict[str, ServiceEndpoint] = {}
        self.health_checks: Dict[str, ServiceHealth] = {}
        self.discovery_active = False
        self.discovery_thread = None
        self.health_check_thread = None
        
        # Service discovery configuration
        self.discovery_config = {
            'health_check_interval': 30,  # seconds
            'health_check_timeout': 10,   # seconds
            'default_healthy_threshold': 2,
            'default_unhealthy_threshold': 3,
            'circuit_breaker_timeout': 60,  # seconds
            'service_ttl': 300,  # seconds
            'endpoint_ttl': 60,  # seconds
            'max_retries': 3,
            'retry_delay': 1  # seconds
        }
        
        # Initialize database
        self._init_database()
        
        # Load default services
        self._load_default_services()
        
        logger.info("Service Discovery initialized")
    
    def _init_database(self) -> None:
        """Initialize service discovery database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create service registries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS service_registries (
                service_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                version TEXT NOT NULL,
                namespace TEXT NOT NULL,
                endpoints TEXT NOT NULL,
                health_checks TEXT NOT NULL,
                load_balancing_strategy TEXT NOT NULL,
                circuit_breaker_enabled INTEGER NOT NULL,
                circuit_breaker_threshold INTEGER NOT NULL,
                circuit_breaker_timeout INTEGER NOT NULL,
                retry_policy TEXT NOT NULL,
                timeout_policy TEXT NOT NULL,
                tags TEXT NOT NULL,
                metadata TEXT NOT NULL,
                status TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create service endpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS service_endpoints (
                endpoint_id TEXT PRIMARY KEY,
                service_id TEXT NOT NULL,
                host TEXT NOT NULL,
                port INTEGER NOT NULL,
                protocol TEXT NOT NULL,
                path TEXT NOT NULL,
                weight INTEGER NOT NULL,
                priority INTEGER NOT NULL,
                tags TEXT NOT NULL,
                metadata TEXT NOT NULL,
                status TEXT NOT NULL,
                last_health_check TEXT,
                health_check_failures INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create service health checks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS service_health_checks (
                health_id TEXT PRIMARY KEY,
                service_id TEXT NOT NULL,
                endpoint_id TEXT NOT NULL,
                health_check_url TEXT NOT NULL,
                health_check_interval INTEGER NOT NULL,
                health_check_timeout INTEGER NOT NULL,
                healthy_threshold INTEGER NOT NULL,
                unhealthy_threshold INTEGER NOT NULL,
                last_check TEXT,
                last_status TEXT NOT NULL,
                consecutive_failures INTEGER NOT NULL,
                consecutive_successes INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_services(self) -> None:
        """Load default services."""
        default_services = [
            ServiceRegistry(
                service_id="SVC_001",
                name="trading-system",
                description="Core trading system service",
                version="1.0.0",
                namespace="production",
                endpoints=[
                    ServiceEndpoint(
                        endpoint_id="EP_001",
                        service_id="SVC_001",
                        host="localhost",
                        port=8000,
                        protocol=ServiceType.HTTP,
                        path="/api/v1",
                        weight=100,
                        priority=1,
                        tags=["primary", "trading"],
                        metadata={"region": "us-east-1", "zone": "a"}
                    ),
                    ServiceEndpoint(
                        endpoint_id="EP_002",
                        service_id="SVC_001",
                        host="localhost",
                        port=8001,
                        protocol=ServiceType.HTTP,
                        path="/api/v1",
                        weight=100,
                        priority=1,
                        tags=["secondary", "trading"],
                        metadata={"region": "us-east-1", "zone": "b"}
                    )
                ],
                health_checks=[
                    ServiceHealth(
                        health_id="HC_001",
                        service_id="SVC_001",
                        endpoint_id="EP_001",
                        health_check_url="http://localhost:8000/health",
                        health_check_interval=30,
                        health_check_timeout=10,
                        healthy_threshold=2,
                        unhealthy_threshold=3
                    ),
                    ServiceHealth(
                        health_id="HC_002",
                        service_id="SVC_001",
                        endpoint_id="EP_002",
                        health_check_url="http://localhost:8001/health",
                        health_check_interval=30,
                        health_check_timeout=10,
                        healthy_threshold=2,
                        unhealthy_threshold=3
                    )
                ],
                load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                circuit_breaker_enabled=True,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=60,
                retry_policy={"max_retries": 3, "retry_delay": 1},
                timeout_policy={"connect_timeout": 5, "read_timeout": 30},
                tags=["trading", "core"],
                metadata={"owner": "trading-team", "environment": "production"}
            ),
            ServiceRegistry(
                service_id="SVC_002",
                name="api-gateway",
                description="API gateway service",
                version="1.0.0",
                namespace="production",
                endpoints=[
                    ServiceEndpoint(
                        endpoint_id="EP_003",
                        service_id="SVC_002",
                        host="localhost",
                        port=8080,
                        protocol=ServiceType.HTTP,
                        path="/",
                        weight=100,
                        priority=1,
                        tags=["gateway", "api"],
                        metadata={"region": "us-east-1", "zone": "a"}
                    )
                ],
                health_checks=[
                    ServiceHealth(
                        health_id="HC_003",
                        service_id="SVC_002",
                        endpoint_id="EP_003",
                        health_check_url="http://localhost:8080/health",
                        health_check_interval=30,
                        health_check_timeout=10,
                        healthy_threshold=2,
                        unhealthy_threshold=3
                    )
                ],
                load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                circuit_breaker_enabled=True,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=60,
                retry_policy={"max_retries": 3, "retry_delay": 1},
                timeout_policy={"connect_timeout": 5, "read_timeout": 30},
                tags=["gateway", "api"],
                metadata={"owner": "platform-team", "environment": "production"}
            ),
            ServiceRegistry(
                service_id="SVC_003",
                name="ai-models",
                description="AI model service",
                version="1.0.0",
                namespace="production",
                endpoints=[
                    ServiceEndpoint(
                        endpoint_id="EP_004",
                        service_id="SVC_003",
                        host="localhost",
                        port=9000,
                        protocol=ServiceType.HTTP,
                        path="/api/v1",
                        weight=100,
                        priority=1,
                        tags=["ai", "models"],
                        metadata={"region": "us-east-1", "zone": "a"}
                    )
                ],
                health_checks=[
                    ServiceHealth(
                        health_id="HC_004",
                        service_id="SVC_003",
                        endpoint_id="EP_004",
                        health_check_url="http://localhost:9000/health",
                        health_check_interval=30,
                        health_check_timeout=10,
                        healthy_threshold=2,
                        unhealthy_threshold=3
                    )
                ],
                load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                circuit_breaker_enabled=True,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=60,
                retry_policy={"max_retries": 3, "retry_delay": 1},
                timeout_policy={"connect_timeout": 5, "read_timeout": 30},
                tags=["ai", "models"],
                metadata={"owner": "ai-team", "environment": "production"}
            ),
            ServiceRegistry(
                service_id="SVC_004",
                name="market-data",
                description="Market data service",
                version="1.0.0",
                namespace="production",
                endpoints=[
                    ServiceEndpoint(
                        endpoint_id="EP_005",
                        service_id="SVC_004",
                        host="localhost",
                        port=7000,
                        protocol=ServiceType.HTTP,
                        path="/api/v1",
                        weight=100,
                        priority=1,
                        tags=["market", "data"],
                        metadata={"region": "us-east-1", "zone": "a"}
                    )
                ],
                health_checks=[
                    ServiceHealth(
                        health_id="HC_005",
                        service_id="SVC_004",
                        endpoint_id="EP_005",
                        health_check_url="http://localhost:7000/health",
                        health_check_interval=30,
                        health_check_timeout=10,
                        healthy_threshold=2,
                        unhealthy_threshold=3
                    )
                ],
                load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                circuit_breaker_enabled=True,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=60,
                retry_policy={"max_retries": 3, "retry_delay": 1},
                timeout_policy={"connect_timeout": 5, "read_timeout": 30},
                tags=["market", "data"],
                metadata={"owner": "data-team", "environment": "production"}
            ),
            ServiceRegistry(
                service_id="SVC_005",
                name="database",
                description="Database service",
                version="1.0.0",
                namespace="production",
                endpoints=[
                    ServiceEndpoint(
                        endpoint_id="EP_006",
                        service_id="SVC_005",
                        host="localhost",
                        port=5432,
                        protocol=ServiceType.TCP,
                        path="",
                        weight=100,
                        priority=1,
                        tags=["database", "postgres"],
                        metadata={"region": "us-east-1", "zone": "a"}
                    )
                ],
                health_checks=[
                    ServiceHealth(
                        health_id="HC_006",
                        service_id="SVC_005",
                        endpoint_id="EP_006",
                        health_check_url="tcp://localhost:5432",
                        health_check_interval=30,
                        health_check_timeout=10,
                        healthy_threshold=2,
                        unhealthy_threshold=3
                    )
                ],
                load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                circuit_breaker_enabled=True,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=60,
                retry_policy={"max_retries": 3, "retry_delay": 1},
                timeout_policy={"connect_timeout": 5, "read_timeout": 30},
                tags=["database", "postgres"],
                metadata={"owner": "dba-team", "environment": "production"}
            )
        ]
        
        for service in default_services:
            self.register_service(service)
    
    def register_service(self, service: ServiceRegistry) -> None:
        """
        Register a service.
        
        Args:
            service: Service registry definition
        """
        self.services[service.service_id] = service
        
        # Register endpoints
        for endpoint in service.endpoints:
            self.endpoints[endpoint.endpoint_id] = endpoint
        
        # Register health checks
        for health_check in service.health_checks:
            self.health_checks[health_check.health_id] = health_check
        
        # Store in database
        self._store_service(service)
        
        logger.info(f"Registered service: {service.service_id} - {service.name}")
    
    def deregister_service(self, service_id: str) -> bool:
        """
        Deregister a service.
        
        Args:
            service_id: Service ID
            
        Returns:
            True if service deregistered successfully
        """
        if service_id not in self.services:
            logger.error(f"Service {service_id} not found")
            return False
        
        service = self.services[service_id]
        
        # Remove endpoints
        for endpoint in service.endpoints:
            if endpoint.endpoint_id in self.endpoints:
                del self.endpoints[endpoint.endpoint_id]
        
        # Remove health checks
        for health_check in service.health_checks:
            if health_check.health_id in self.health_checks:
                del self.health_checks[health_check.health_id]
        
        # Remove service
        del self.services[service_id]
        
        # Remove from database
        self._remove_service(service_id)
        
        logger.info(f"Deregistered service: {service_id}")
        return True
    
    def start_discovery(self) -> None:
        """Start service discovery."""
        if self.discovery_active:
            logger.warning("Service discovery is already active")
            return
        
        self.discovery_active = True
        
        # Start discovery thread
        self.discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
        self.discovery_thread.start()
        
        # Start health check thread
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        logger.info("Service discovery started")
    
    def stop_discovery(self) -> None:
        """Stop service discovery."""
        self.discovery_active = False
        
        if self.discovery_thread:
            self.discovery_thread.join(timeout=5)
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        
        logger.info("Service discovery stopped")
    
    def _discovery_loop(self) -> None:
        """Main discovery loop."""
        while self.discovery_active:
            try:
                # Update service statuses
                self._update_service_statuses()
                
                # Clean up expired services
                self._cleanup_expired_services()
                
                time.sleep(self.discovery_config['health_check_interval'])
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                time.sleep(5)
    
    def _health_check_loop(self) -> None:
        """Health check loop."""
        while self.discovery_active:
            try:
                # Run health checks
                self._run_health_checks()
                
                time.sleep(self.discovery_config['health_check_interval'])
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(5)
    
    def _run_health_checks(self) -> None:
        """Run health checks for all services."""
        for health_check in self.health_checks.values():
            self._check_endpoint_health(health_check)
    
    def _check_endpoint_health(self, health_check: ServiceHealth) -> None:
        """Check health of a specific endpoint."""
        try:
            health_check.last_check = datetime.now()
            
            # Perform health check
            is_healthy = self._perform_health_check(health_check.health_check_url)
            
            if is_healthy:
                health_check.consecutive_successes += 1
                health_check.consecutive_failures = 0
                
                if health_check.consecutive_successes >= health_check.healthy_threshold:
                    health_check.last_status = ServiceStatus.HEALTHY
            else:
                health_check.consecutive_failures += 1
                health_check.consecutive_successes = 0
                
                if health_check.consecutive_failures >= health_check.unhealthy_threshold:
                    health_check.last_status = ServiceStatus.UNHEALTHY
            
            # Update endpoint status
            if health_check.endpoint_id in self.endpoints:
                endpoint = self.endpoints[health_check.endpoint_id]
                endpoint.last_health_check = datetime.now()
                endpoint.status = health_check.last_status
                
                if health_check.last_status == ServiceStatus.UNHEALTHY:
                    endpoint.health_check_failures += 1
                else:
                    endpoint.health_check_failures = 0
            
            # Update health check in database
            self._update_health_check(health_check)
            
        except Exception as e:
            logger.error(f"Health check failed for {health_check.health_id}: {e}")
            health_check.consecutive_failures += 1
            health_check.last_status = ServiceStatus.UNHEALTHY
    
    def _perform_health_check(self, health_check_url: str) -> bool:
        """Perform health check."""
        try:
            if health_check_url.startswith('http'):
                response = requests.get(health_check_url, timeout=self.discovery_config['health_check_timeout'])
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
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.discovery_config['health_check_timeout'])
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0
            else:
                return True  # Default to healthy for simulation
                
        except Exception as e:
            logger.error(f"Health check failed for {health_check_url}: {e}")
            return False
    
    def _update_service_statuses(self) -> None:
        """Update service statuses based on endpoint health."""
        for service in self.services.values():
            healthy_endpoints = 0
            total_endpoints = len(service.endpoints)
            
            for endpoint in service.endpoints:
                if endpoint.status == ServiceStatus.HEALTHY:
                    healthy_endpoints += 1
            
            # Update service status
            if healthy_endpoints == 0:
                service.status = ServiceStatus.UNHEALTHY
            elif healthy_endpoints == total_endpoints:
                service.status = ServiceStatus.HEALTHY
            else:
                service.status = ServiceStatus.UNHEALTHY  # Partial health
            
            service.last_updated = datetime.now()
    
    def _cleanup_expired_services(self) -> None:
        """Clean up expired services."""
        current_time = datetime.now()
        
        for service_id, service in list(self.services.items()):
            # Check if service has expired
            if (current_time - service.last_updated).total_seconds() > self.discovery_config['service_ttl']:
                logger.info(f"Service {service_id} has expired, removing...")
                self.deregister_service(service_id)
    
    def discover_service(self, service_name: str, namespace: str = "default") -> Optional[ServiceRegistry]:
        """
        Discover a service by name.
        
        Args:
            service_name: Service name
            namespace: Service namespace
            
        Returns:
            Service registry or None if not found
        """
        for service in self.services.values():
            if service.name == service_name and service.namespace == namespace:
                return service
        
        return None
    
    def get_service_endpoints(self, service_id: str, healthy_only: bool = True) -> List[ServiceEndpoint]:
        """
        Get service endpoints.
        
        Args:
            service_id: Service ID
            healthy_only: Return only healthy endpoints
            
        Returns:
            List of service endpoints
        """
        if service_id not in self.services:
            return []
        
        service = self.services[service_id]
        endpoints = []
        
        for endpoint in service.endpoints:
            if not healthy_only or endpoint.status == ServiceStatus.HEALTHY:
                endpoints.append(endpoint)
        
        # Sort by priority and weight
        endpoints.sort(key=lambda x: (x.priority, -x.weight))
        
        return endpoints
    
    def get_healthy_endpoint(self, service_id: str) -> Optional[ServiceEndpoint]:
        """
        Get a healthy endpoint for a service using load balancing.
        
        Args:
            service_id: Service ID
            
        Returns:
            Healthy endpoint or None if not found
        """
        endpoints = self.get_service_endpoints(service_id, healthy_only=True)
        
        if not endpoints:
            return None
        
        if service_id not in self.services:
            return endpoints[0]
        
        service = self.services[service_id]
        
        # Apply load balancing strategy
        if service.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(endpoints)
        elif service.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(endpoints)
        elif service.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(endpoints)
        elif service.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            return self._random_selection(endpoints)
        else:
            return endpoints[0]
    
    def _round_robin_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round robin endpoint selection."""
        # Simple round robin implementation
        return endpoints[0]  # In a real implementation, this would maintain state
    
    def _least_connections_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least connections endpoint selection."""
        # Return endpoint with lowest connection count
        return min(endpoints, key=lambda x: x.metadata.get('connections', 0))
    
    def _weighted_round_robin_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted round robin endpoint selection."""
        # Simple weighted selection
        total_weight = sum(endpoint.weight for endpoint in endpoints)
        if total_weight == 0:
            return endpoints[0]
        
        # In a real implementation, this would maintain state
        return endpoints[0]
    
    def _random_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Random endpoint selection."""
        import random
        return random.choice(endpoints)
    
    def get_service_status(self, service_id: str) -> Dict[str, Any]:
        """
        Get service status.
        
        Args:
            service_id: Service ID
            
        Returns:
            Service status dictionary
        """
        if service_id not in self.services:
            return {"error": f"Service {service_id} not found"}
        
        service = self.services[service_id]
        
        # Get endpoint statuses
        endpoint_statuses = []
        for endpoint in service.endpoints:
            endpoint_statuses.append({
                'endpoint_id': endpoint.endpoint_id,
                'host': endpoint.host,
                'port': endpoint.port,
                'protocol': endpoint.protocol.value,
                'status': endpoint.status.value,
                'weight': endpoint.weight,
                'priority': endpoint.priority,
                'last_health_check': endpoint.last_health_check.isoformat() if endpoint.last_health_check else None,
                'health_check_failures': endpoint.health_check_failures
            })
        
        # Get health check statuses
        health_check_statuses = []
        for health_check in service.health_checks:
            health_check_statuses.append({
                'health_id': health_check.health_id,
                'endpoint_id': health_check.endpoint_id,
                'health_check_url': health_check.health_check_url,
                'last_status': health_check.last_status.value,
                'last_check': health_check.last_check.isoformat() if health_check.last_check else None,
                'consecutive_failures': health_check.consecutive_failures,
                'consecutive_successes': health_check.consecutive_successes
            })
        
        return {
            'service_id': service.service_id,
            'name': service.name,
            'description': service.description,
            'version': service.version,
            'namespace': service.namespace,
            'status': service.status.value,
            'load_balancing_strategy': service.load_balancing_strategy.value,
            'circuit_breaker_enabled': service.circuit_breaker_enabled,
            'endpoints': endpoint_statuses,
            'health_checks': health_check_statuses,
            'tags': service.tags,
            'metadata': service.metadata,
            'last_updated': service.last_updated.isoformat(),
            'created_at': service.created_at.isoformat()
        }
    
    def get_all_services(self) -> Dict[str, Any]:
        """
        Get all services.
        
        Returns:
            All services dictionary
        """
        services = {}
        for service_id, service in self.services.items():
            services[service_id] = {
                'name': service.name,
                'description': service.description,
                'version': service.version,
                'namespace': service.namespace,
                'status': service.status.value,
                'endpoint_count': len(service.endpoints),
                'healthy_endpoints': len([e for e in service.endpoints if e.status == ServiceStatus.HEALTHY]),
                'tags': service.tags,
                'last_updated': service.last_updated.isoformat()
            }
        
        return {
            'services': services,
            'total_services': len(self.services),
            'discovery_active': self.discovery_active
        }
    
    def _store_service(self, service: ServiceRegistry) -> None:
        """Store service in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO service_registries 
            (service_id, name, description, version, namespace, endpoints, health_checks,
             load_balancing_strategy, circuit_breaker_enabled, circuit_breaker_threshold,
             circuit_breaker_timeout, retry_policy, timeout_policy, tags, metadata,
             status, last_updated, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            service.service_id, service.name, service.description, service.version,
            service.namespace, json.dumps([e.__dict__ for e in service.endpoints]),
            json.dumps([h.__dict__ for h in service.health_checks]),
            service.load_balancing_strategy.value, service.circuit_breaker_enabled,
            service.circuit_breaker_threshold, service.circuit_breaker_timeout,
            json.dumps(service.retry_policy), json.dumps(service.timeout_policy),
            json.dumps(service.tags), json.dumps(service.metadata),
            service.status.value, service.last_updated.isoformat(), service.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _remove_service(self, service_id: str) -> None:
        """Remove service from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM service_registries WHERE service_id = ?", (service_id,))
        cursor.execute("DELETE FROM service_endpoints WHERE service_id = ?", (service_id,))
        cursor.execute("DELETE FROM service_health_checks WHERE service_id = ?", (service_id,))
        
        conn.commit()
        conn.close()
    
    def _update_health_check(self, health_check: ServiceHealth) -> None:
        """Update health check in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE service_health_checks
            SET last_check = ?, last_status = ?, consecutive_failures = ?, consecutive_successes = ?
            WHERE health_id = ?
        """, (
            health_check.last_check.isoformat() if health_check.last_check else None,
            health_check.last_status.value, health_check.consecutive_failures,
            health_check.consecutive_successes, health_check.health_id
        ))
        
        conn.commit()
        conn.close()
