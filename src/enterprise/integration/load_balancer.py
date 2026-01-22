"""
Load Balancer

This module implements a comprehensive load balancer for managing
backend servers, load balancing strategies, and enterprise-grade
load balancing for production deployment and management.

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
import hashlib
import random
import asyncio
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "ROUND_ROBIN"
    LEAST_CONNECTIONS = "LEAST_CONNECTIONS"
    WEIGHTED_ROUND_ROBIN = "WEIGHTED_ROUND_ROBIN"
    IP_HASH = "IP_HASH"
    RANDOM = "RANDOM"
    LEAST_RESPONSE_TIME = "LEAST_RESPONSE_TIME"
    CONSISTENT_HASH = "CONSISTENT_HASH"

class ServerStatus(Enum):
    """Server status levels."""
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    DRAINING = "DRAINING"
    MAINTENANCE = "MAINTENANCE"

class HealthCheckType(Enum):
    """Health check types."""
    HTTP = "HTTP"
    HTTPS = "HTTPS"
    TCP = "TCP"
    UDP = "UDP"
    CUSTOM = "CUSTOM"

@dataclass
class HealthCheck:
    """Health check definition."""
    health_check_id: str
    server_id: str
    check_type: HealthCheckType
    url: str
    interval: int  # seconds
    timeout: int   # seconds
    healthy_threshold: int
    unhealthy_threshold: int
    last_check: Optional[datetime] = None
    last_status: ServerStatus = ServerStatus.UNHEALTHY
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    response_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class BackendServer:
    """Backend server definition."""
    server_id: str
    host: str
    port: int
    weight: int
    priority: int
    max_connections: int
    current_connections: int
    status: ServerStatus = ServerStatus.UNHEALTHY
    health_checks: List[HealthCheck] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0
    last_health_check: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class LoadBalancer:
    """Load balancer definition."""
    lb_id: str
    name: str
    description: str
    frontend_host: str
    frontend_port: int
    backend_servers: List[BackendServer]
    strategy: LoadBalancingStrategy
    health_check_interval: int
    session_affinity: bool
    session_timeout: int
    max_retries: int
    retry_timeout: int
    circuit_breaker_enabled: bool
    circuit_breaker_threshold: int
    circuit_breaker_timeout: int
    sticky_sessions: bool
    sticky_cookie_name: str
    sticky_cookie_timeout: int
    created_at: datetime = field(default_factory=datetime.now)

class LoadBalancer:
    """
    Comprehensive load balancer system.
    
    Features:
    - Multiple load balancing strategies
    - Health check monitoring
    - Session affinity
    - Circuit breaker pattern
    - Backend server management
    - Performance monitoring
    - Enterprise-grade load balancing
    """
    
    def __init__(self, db_path: str = "data/load_balancer.db"):
        """
        Initialize load balancer.
        
        Args:
            db_path: Path to load balancer database
        """
        self.db_path = db_path
        self.load_balancers: Dict[str, LoadBalancer] = {}
        self.backend_servers: Dict[str, BackendServer] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.balancing_active = False
        self.balancing_thread = None
        self.health_check_thread = None
        
        # Load balancer configuration
        self.lb_config = {
            'health_check_interval': 30,  # seconds
            'health_check_timeout': 10,   # seconds
            'default_healthy_threshold': 2,
            'default_unhealthy_threshold': 3,
            'max_response_time': 5.0,     # seconds
            'circuit_breaker_timeout': 60,  # seconds
            'session_timeout': 3600,      # seconds
            'max_retries': 3,
            'retry_timeout': 5,           # seconds
            'response_time_window': 100   # number of response times to track
        }
        
        # Round robin state
        self.round_robin_state: Dict[str, int] = {}
        
        # Session affinity state
        self.session_affinity_state: Dict[str, str] = {}
        
        # Circuit breaker state
        self.circuit_breaker_state: Dict[str, Dict[str, Any]] = {}
        
        # Initialize database
        self._init_database()
        
        # Load default load balancers
        self._load_default_load_balancers()
        
        logger.info("Load Balancer initialized")
    
    def _init_database(self) -> None:
        """Initialize load balancer database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create load balancers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS load_balancers (
                lb_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                frontend_host TEXT NOT NULL,
                frontend_port INTEGER NOT NULL,
                backend_servers TEXT NOT NULL,
                strategy TEXT NOT NULL,
                health_check_interval INTEGER NOT NULL,
                session_affinity INTEGER NOT NULL,
                session_timeout INTEGER NOT NULL,
                max_retries INTEGER NOT NULL,
                retry_timeout INTEGER NOT NULL,
                circuit_breaker_enabled INTEGER NOT NULL,
                circuit_breaker_threshold INTEGER NOT NULL,
                circuit_breaker_timeout INTEGER NOT NULL,
                sticky_sessions INTEGER NOT NULL,
                sticky_cookie_name TEXT,
                sticky_cookie_timeout INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create backend servers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backend_servers (
                server_id TEXT PRIMARY KEY,
                host TEXT NOT NULL,
                port INTEGER NOT NULL,
                weight INTEGER NOT NULL,
                priority INTEGER NOT NULL,
                max_connections INTEGER NOT NULL,
                current_connections INTEGER NOT NULL,
                status TEXT NOT NULL,
                response_times TEXT NOT NULL,
                error_count INTEGER NOT NULL,
                last_health_check TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create health checks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_checks (
                health_check_id TEXT PRIMARY KEY,
                server_id TEXT NOT NULL,
                check_type TEXT NOT NULL,
                url TEXT NOT NULL,
                interval INTEGER NOT NULL,
                timeout INTEGER NOT NULL,
                healthy_threshold INTEGER NOT NULL,
                unhealthy_threshold INTEGER NOT NULL,
                last_check TEXT,
                last_status TEXT NOT NULL,
                consecutive_failures INTEGER NOT NULL,
                consecutive_successes INTEGER NOT NULL,
                response_time REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_load_balancers(self) -> None:
        """Load default load balancers."""
        default_servers = [
            BackendServer(
                server_id="SRV_001",
                host="localhost",
                port=8000,
                weight=100,
                priority=1,
                max_connections=1000,
                current_connections=0,
                health_checks=[
                    HealthCheck(
                        health_check_id="HC_001",
                        server_id="SRV_001",
                        check_type=HealthCheckType.HTTP,
                        url="http://localhost:8000/health",
                        interval=30,
                        timeout=10,
                        healthy_threshold=2,
                        unhealthy_threshold=3
                    )
                ]
            ),
            BackendServer(
                server_id="SRV_002",
                host="localhost",
                port=8001,
                weight=100,
                priority=1,
                max_connections=1000,
                current_connections=0,
                health_checks=[
                    HealthCheck(
                        health_check_id="HC_002",
                        server_id="SRV_002",
                        check_type=HealthCheckType.HTTP,
                        url="http://localhost:8001/health",
                        interval=30,
                        timeout=10,
                        healthy_threshold=2,
                        unhealthy_threshold=3
                    )
                ]
            ),
            BackendServer(
                server_id="SRV_003",
                host="localhost",
                port=8002,
                weight=50,
                priority=2,
                max_connections=500,
                current_connections=0,
                health_checks=[
                    HealthCheck(
                        health_check_id="HC_003",
                        server_id="SRV_003",
                        check_type=HealthCheckType.HTTP,
                        url="http://localhost:8002/health",
                        interval=30,
                        timeout=10,
                        healthy_threshold=2,
                        unhealthy_threshold=3
                    )
                ]
            )
        ]
        
        default_lb = LoadBalancer(
            lb_id="LB_001",
            name="Trading System Load Balancer",
            description="Load balancer for trading system",
            frontend_host="localhost",
            frontend_port=8080,
            backend_servers=default_servers,
            strategy=LoadBalancingStrategy.ROUND_ROBIN,
            health_check_interval=30,
            session_affinity=True,
            session_timeout=3600,
            max_retries=3,
            retry_timeout=5,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60,
            sticky_sessions=True,
            sticky_cookie_name="LB_SESSION",
            sticky_cookie_timeout=3600
        )
        
        self.add_load_balancer(default_lb)
    
    def add_load_balancer(self, load_balancer: LoadBalancer) -> None:
        """
        Add load balancer.
        
        Args:
            load_balancer: Load balancer definition
        """
        self.load_balancers[load_balancer.lb_id] = load_balancer
        
        # Add backend servers
        for server in load_balancer.backend_servers:
            self.backend_servers[server.server_id] = server
            
            # Add health checks
            for health_check in server.health_checks:
                self.health_checks[health_check.health_check_id] = health_check
        
        # Initialize round robin state
        self.round_robin_state[load_balancer.lb_id] = 0
        
        # Initialize circuit breaker state
        self.circuit_breaker_state[load_balancer.lb_id] = {
            'failures': 0,
            'last_failure': None,
            'state': 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        }
        
        # Store in database
        self._store_load_balancer(load_balancer)
        
        logger.info(f"Added load balancer: {load_balancer.lb_id} - {load_balancer.name}")
    
    def remove_load_balancer(self, lb_id: str) -> bool:
        """
        Remove load balancer.
        
        Args:
            lb_id: Load balancer ID
            
        Returns:
            True if load balancer removed successfully
        """
        if lb_id not in self.load_balancers:
            logger.error(f"Load balancer {lb_id} not found")
            return False
        
        load_balancer = self.load_balancers[lb_id]
        
        # Remove backend servers
        for server in load_balancer.backend_servers:
            if server.server_id in self.backend_servers:
                del self.backend_servers[server.server_id]
            
            # Remove health checks
            for health_check in server.health_checks:
                if health_check.health_check_id in self.health_checks:
                    del self.health_checks[health_check.health_check_id]
        
        # Remove load balancer
        del self.load_balancers[lb_id]
        
        # Clean up state
        if lb_id in self.round_robin_state:
            del self.round_robin_state[lb_id]
        if lb_id in self.circuit_breaker_state:
            del self.circuit_breaker_state[lb_id]
        
        # Remove from database
        self._remove_load_balancer(lb_id)
        
        logger.info(f"Removed load balancer: {lb_id}")
        return True
    
    def start_load_balancing(self) -> None:
        """Start load balancing."""
        if self.balancing_active:
            logger.warning("Load balancing is already active")
            return
        
        self.balancing_active = True
        
        # Start balancing thread
        self.balancing_thread = threading.Thread(target=self._balancing_loop, daemon=True)
        self.balancing_thread.start()
        
        # Start health check thread
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        logger.info("Load balancing started")
    
    def stop_load_balancing(self) -> None:
        """Stop load balancing."""
        self.balancing_active = False
        
        if self.balancing_thread:
            self.balancing_thread.join(timeout=5)
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        
        logger.info("Load balancing stopped")
    
    def _balancing_loop(self) -> None:
        """Main balancing loop."""
        while self.balancing_active:
            try:
                # Update server statuses
                self._update_server_statuses()
                
                # Clean up expired sessions
                self._cleanup_expired_sessions()
                
                time.sleep(self.lb_config['health_check_interval'])
                
            except Exception as e:
                logger.error(f"Error in balancing loop: {e}")
                time.sleep(5)
    
    def _health_check_loop(self) -> None:
        """Health check loop."""
        while self.balancing_active:
            try:
                # Run health checks
                self._run_health_checks()
                
                time.sleep(self.lb_config['health_check_interval'])
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(5)
    
    def _run_health_checks(self) -> None:
        """Run health checks for all servers."""
        for health_check in self.health_checks.values():
            self._check_server_health(health_check)
    
    def _check_server_health(self, health_check: HealthCheck) -> None:
        """Check health of a specific server."""
        try:
            health_check.last_check = datetime.now()
            start_time = time.time()
            
            # Perform health check
            is_healthy = self._perform_health_check(health_check.url, health_check.check_type)
            
            response_time = time.time() - start_time
            health_check.response_time = response_time
            
            if is_healthy:
                health_check.consecutive_successes += 1
                health_check.consecutive_failures = 0
                
                if health_check.consecutive_successes >= health_check.healthy_threshold:
                    health_check.last_status = ServerStatus.HEALTHY
            else:
                health_check.consecutive_failures += 1
                health_check.consecutive_successes = 0
                
                if health_check.consecutive_failures >= health_check.unhealthy_threshold:
                    health_check.last_status = ServerStatus.UNHEALTHY
            
            # Update server status
            if health_check.server_id in self.backend_servers:
                server = self.backend_servers[health_check.server_id]
                server.last_health_check = datetime.now()
                server.status = health_check.last_status
                
                # Update response times
                server.response_times.append(response_time)
                if len(server.response_times) > self.lb_config['response_time_window']:
                    server.response_times = server.response_times[-self.lb_config['response_time_window']:]
                
                if health_check.last_status == ServerStatus.UNHEALTHY:
                    server.error_count += 1
                else:
                    server.error_count = 0
            
            # Update health check in database
            self._update_health_check(health_check)
            
        except Exception as e:
            logger.error(f"Health check failed for {health_check.health_check_id}: {e}")
            health_check.consecutive_failures += 1
            health_check.last_status = ServerStatus.UNHEALTHY
    
    def _perform_health_check(self, url: str, check_type: HealthCheckType) -> bool:
        """Perform health check."""
        try:
            if check_type in [HealthCheckType.HTTP, HealthCheckType.HTTPS]:
                response = requests.get(url, timeout=self.lb_config['health_check_timeout'])
                return response.status_code == 200
            elif check_type == HealthCheckType.TCP:
                # TCP health check
                if url.startswith('tcp://'):
                    host_port = url.replace('tcp://', '')
                else:
                    host_port = url
                
                if ':' in host_port:
                    host, port = host_port.split(':')
                    port = int(port)
                else:
                    host = host_port
                    port = 80
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.lb_config['health_check_timeout'])
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0
            else:
                return True  # Default to healthy for simulation
                
        except Exception as e:
            logger.error(f"Health check failed for {url}: {e}")
            return False
    
    def _update_server_statuses(self) -> None:
        """Update server statuses based on health checks."""
        for server in self.backend_servers.values():
            # Check if server has any healthy health checks
            healthy_checks = 0
            total_checks = 0
            
            for health_check in self.health_checks.values():
                if health_check.server_id == server.server_id:
                    total_checks += 1
                    if health_check.last_status == ServerStatus.HEALTHY:
                        healthy_checks += 1
            
            # Update server status
            if total_checks == 0:
                server.status = ServerStatus.UNHEALTHY
            elif healthy_checks == 0:
                server.status = ServerStatus.UNHEALTHY
            elif healthy_checks == total_checks:
                server.status = ServerStatus.HEALTHY
            else:
                server.status = ServerStatus.UNHEALTHY  # Partial health
    
    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, server_id in self.session_affinity_state.items():
            # In a real implementation, this would check session timeout
            # For simulation, we'll just clean up old sessions
            pass
        
        for session_id in expired_sessions:
            del self.session_affinity_state[session_id]
    
    def get_backend_server(self, lb_id: str, client_ip: str = None, session_id: str = None) -> Optional[BackendServer]:
        """
        Get backend server using load balancing strategy.
        
        Args:
            lb_id: Load balancer ID
            client_ip: Client IP address
            session_id: Session ID for affinity
            
        Returns:
            Backend server or None if not found
        """
        if lb_id not in self.load_balancers:
            logger.error(f"Load balancer {lb_id} not found")
            return None
        
        load_balancer = self.load_balancers[lb_id]
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(lb_id):
            logger.warning(f"Circuit breaker is open for load balancer {lb_id}")
            return None
        
        # Get healthy servers
        healthy_servers = []
        for server in load_balancer.backend_servers:
            if (server.status == ServerStatus.HEALTHY and 
                server.current_connections < server.max_connections):
                healthy_servers.append(server)
        
        if not healthy_servers:
            logger.warning(f"No healthy servers available for load balancer {lb_id}")
            return None
        
        # Check session affinity
        if load_balancer.session_affinity and session_id:
            if session_id in self.session_affinity_state:
                server_id = self.session_affinity_state[session_id]
                for server in healthy_servers:
                    if server.server_id == server_id:
                        return server
        
        # Apply load balancing strategy
        selected_server = None
        
        if load_balancer.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected_server = self._round_robin_selection(lb_id, healthy_servers)
        elif load_balancer.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected_server = self._least_connections_selection(healthy_servers)
        elif load_balancer.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            selected_server = self._weighted_round_robin_selection(lb_id, healthy_servers)
        elif load_balancer.strategy == LoadBalancingStrategy.IP_HASH:
            selected_server = self._ip_hash_selection(healthy_servers, client_ip)
        elif load_balancer.strategy == LoadBalancingStrategy.RANDOM:
            selected_server = self._random_selection(healthy_servers)
        elif load_balancer.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            selected_server = self._least_response_time_selection(healthy_servers)
        elif load_balancer.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            selected_server = self._consistent_hash_selection(healthy_servers, client_ip)
        else:
            selected_server = healthy_servers[0]
        
        if selected_server:
            # Update connection count
            selected_server.current_connections += 1
            
            # Update session affinity
            if load_balancer.session_affinity and session_id:
                self.session_affinity_state[session_id] = selected_server.server_id
            
            # Update circuit breaker on success
            self._update_circuit_breaker_success(lb_id)
        
        return selected_server
    
    def _round_robin_selection(self, lb_id: str, servers: List[BackendServer]) -> BackendServer:
        """Round robin server selection."""
        if lb_id not in self.round_robin_state:
            self.round_robin_state[lb_id] = 0
        
        selected_server = servers[self.round_robin_state[lb_id] % len(servers)]
        self.round_robin_state[lb_id] = (self.round_robin_state[lb_id] + 1) % len(servers)
        
        return selected_server
    
    def _least_connections_selection(self, servers: List[BackendServer]) -> BackendServer:
        """Least connections server selection."""
        return min(servers, key=lambda x: x.current_connections)
    
    def _weighted_round_robin_selection(self, lb_id: str, servers: List[BackendServer]) -> BackendServer:
        """Weighted round robin server selection."""
        # Simple weighted selection based on server weight
        total_weight = sum(server.weight for server in servers)
        if total_weight == 0:
            return servers[0]
        
        # In a real implementation, this would maintain weighted state
        return servers[0]
    
    def _ip_hash_selection(self, servers: List[BackendServer], client_ip: str) -> BackendServer:
        """IP hash server selection."""
        if not client_ip:
            return servers[0]
        
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return servers[hash_value % len(servers)]
    
    def _random_selection(self, servers: List[BackendServer]) -> BackendServer:
        """Random server selection."""
        return random.choice(servers)
    
    def _least_response_time_selection(self, servers: List[BackendServer]) -> BackendServer:
        """Least response time server selection."""
        return min(servers, key=lambda x: np.mean(x.response_times) if x.response_times else float('inf'))
    
    def _consistent_hash_selection(self, servers: List[BackendServer], client_ip: str) -> BackendServer:
        """Consistent hash server selection."""
        if not client_ip:
            return servers[0]
        
        # Simple consistent hash implementation
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return servers[hash_value % len(servers)]
    
    def _is_circuit_breaker_open(self, lb_id: str) -> bool:
        """Check if circuit breaker is open."""
        if lb_id not in self.circuit_breaker_state:
            return False
        
        state = self.circuit_breaker_state[lb_id]
        
        if state['state'] == 'OPEN':
            # Check if timeout has passed
            if state['last_failure']:
                time_since_failure = (datetime.now() - state['last_failure']).total_seconds()
                if time_since_failure > self.lb_config['circuit_breaker_timeout']:
                    state['state'] = 'HALF_OPEN'
                    return False
            return True
        
        return False
    
    def _update_circuit_breaker_success(self, lb_id: str) -> None:
        """Update circuit breaker on success."""
        if lb_id not in self.circuit_breaker_state:
            return
        
        state = self.circuit_breaker_state[lb_id]
        
        if state['state'] == 'HALF_OPEN':
            state['state'] = 'CLOSED'
            state['failures'] = 0
    
    def _update_circuit_breaker_failure(self, lb_id: str) -> None:
        """Update circuit breaker on failure."""
        if lb_id not in self.circuit_breaker_state:
            return
        
        state = self.circuit_breaker_state[lb_id]
        state['failures'] += 1
        state['last_failure'] = datetime.now()
        
        if state['failures'] >= self.lb_config['circuit_breaker_threshold']:
            state['state'] = 'OPEN'
    
    def release_connection(self, server_id: str) -> None:
        """
        Release connection from server.
        
        Args:
            server_id: Server ID
        """
        if server_id in self.backend_servers:
            server = self.backend_servers[server_id]
            if server.current_connections > 0:
                server.current_connections -= 1
    
    def get_load_balancer_status(self, lb_id: str) -> Dict[str, Any]:
        """
        Get load balancer status.
        
        Args:
            lb_id: Load balancer ID
            
        Returns:
            Load balancer status dictionary
        """
        if lb_id not in self.load_balancers:
            return {"error": f"Load balancer {lb_id} not found"}
        
        load_balancer = self.load_balancers[lb_id]
        
        # Get server statuses
        server_statuses = []
        for server in load_balancer.backend_servers:
            server_statuses.append({
                'server_id': server.server_id,
                'host': server.host,
                'port': server.port,
                'weight': server.weight,
                'priority': server.priority,
                'status': server.status.value,
                'current_connections': server.current_connections,
                'max_connections': server.max_connections,
                'error_count': server.error_count,
                'avg_response_time': np.mean(server.response_times) if server.response_times else 0.0,
                'last_health_check': server.last_health_check.isoformat() if server.last_health_check else None
            })
        
        # Get circuit breaker status
        circuit_breaker_status = self.circuit_breaker_state.get(lb_id, {})
        
        return {
            'lb_id': load_balancer.lb_id,
            'name': load_balancer.name,
            'description': load_balancer.description,
            'frontend_host': load_balancer.frontend_host,
            'frontend_port': load_balancer.frontend_port,
            'strategy': load_balancer.strategy.value,
            'session_affinity': load_balancer.session_affinity,
            'circuit_breaker_enabled': load_balancer.circuit_breaker_enabled,
            'servers': server_statuses,
            'total_servers': len(load_balancer.backend_servers),
            'healthy_servers': len([s for s in load_balancer.backend_servers if s.status == ServerStatus.HEALTHY]),
            'circuit_breaker_status': circuit_breaker_status,
            'balancing_active': self.balancing_active,
            'created_at': load_balancer.created_at.isoformat()
        }
    
    def get_all_load_balancers(self) -> Dict[str, Any]:
        """
        Get all load balancers.
        
        Returns:
            All load balancers dictionary
        """
        load_balancers = {}
        for lb_id, lb in self.load_balancers.items():
            load_balancers[lb_id] = {
                'name': lb.name,
                'description': lb.description,
                'frontend_host': lb.frontend_host,
                'frontend_port': lb.frontend_port,
                'strategy': lb.strategy.value,
                'total_servers': len(lb.backend_servers),
                'healthy_servers': len([s for s in lb.backend_servers if s.status == ServerStatus.HEALTHY]),
                'session_affinity': lb.session_affinity,
                'circuit_breaker_enabled': lb.circuit_breaker_enabled,
                'created_at': lb.created_at.isoformat()
            }
        
        return {
            'load_balancers': load_balancers,
            'total_load_balancers': len(self.load_balancers),
            'balancing_active': self.balancing_active
        }
    
    def _store_load_balancer(self, load_balancer: LoadBalancer) -> None:
        """Store load balancer in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO load_balancers 
            (lb_id, name, description, frontend_host, frontend_port, backend_servers,
             strategy, health_check_interval, session_affinity, session_timeout,
             max_retries, retry_timeout, circuit_breaker_enabled, circuit_breaker_threshold,
             circuit_breaker_timeout, sticky_sessions, sticky_cookie_name, sticky_cookie_timeout, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            load_balancer.lb_id, load_balancer.name, load_balancer.description,
            load_balancer.frontend_host, load_balancer.frontend_port,
            json.dumps([s.__dict__ for s in load_balancer.backend_servers]),
            load_balancer.strategy.value, load_balancer.health_check_interval,
            load_balancer.session_affinity, load_balancer.session_timeout,
            load_balancer.max_retries, load_balancer.retry_timeout,
            load_balancer.circuit_breaker_enabled, load_balancer.circuit_breaker_threshold,
            load_balancer.circuit_breaker_timeout, load_balancer.sticky_sessions,
            load_balancer.sticky_cookie_name, load_balancer.sticky_cookie_timeout,
            load_balancer.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _remove_load_balancer(self, lb_id: str) -> None:
        """Remove load balancer from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM load_balancers WHERE lb_id = ?", (lb_id,))
        cursor.execute("DELETE FROM backend_servers WHERE server_id IN (SELECT server_id FROM load_balancers WHERE lb_id = ?)", (lb_id,))
        cursor.execute("DELETE FROM health_checks WHERE server_id IN (SELECT server_id FROM load_balancers WHERE lb_id = ?)", (lb_id,))
        
        conn.commit()
        conn.close()
    
    def _update_health_check(self, health_check: HealthCheck) -> None:
        """Update health check in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE health_checks
            SET last_check = ?, last_status = ?, consecutive_failures = ?, 
                consecutive_successes = ?, response_time = ?
            WHERE health_check_id = ?
        """, (
            health_check.last_check.isoformat() if health_check.last_check else None,
            health_check.last_status.value, health_check.consecutive_failures,
            health_check.consecutive_successes, health_check.response_time,
            health_check.health_check_id
        ))
        
        conn.commit()
        conn.close()
