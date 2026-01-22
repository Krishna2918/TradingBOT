"""
Comprehensive Unit Tests for Enterprise Integration System

This module contains comprehensive unit tests for the enterprise integration,
deployment management, service discovery, and load balancing systems.

Author: AI Trading System
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import sqlite3
import tempfile
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from enterprise.integration.enterprise_orchestrator import (
    EnterpriseOrchestrator, SystemComponent, IntegrationConfig,
    ComponentStatus, OrchestrationStatus, ComponentType
)
from enterprise.integration.deployment_manager import (
    DeploymentManager, DeploymentConfig, Deployment,
    DeploymentStatus, DeploymentStrategy, Environment
)
from enterprise.integration.service_discovery import (
    ServiceDiscovery, ServiceRegistry, ServiceEndpoint, ServiceHealth,
    ServiceStatus, ServiceType, LoadBalancingStrategy
)
from enterprise.integration.load_balancer import (
    LoadBalancer, BackendServer, HealthCheck,
    ServerStatus, HealthCheckType, LoadBalancingStrategy
)

class TestEnterpriseOrchestrator:
    """Test cases for enterprise orchestrator system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def orchestrator(self, temp_db):
        """Create enterprise orchestrator instance."""
        return EnterpriseOrchestrator(db_path=temp_db)
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test enterprise orchestrator initialization."""
        assert orchestrator is not None
        assert orchestrator.db_path is not None
        assert len(orchestrator.components) > 0  # Should have default components
        assert len(orchestrator.integration_configs) > 0  # Should have default configs
        assert orchestrator.orchestration_status == OrchestrationStatus.STOPPED
        assert orchestrator.running == False
    
    def test_add_component(self, orchestrator):
        """Test adding system component."""
        component = SystemComponent(
            component_id="TEST_COMP_001",
            component_type=ComponentType.TRADING_SYSTEM,
            name="Test Trading System",
            description="Test trading system component",
            host="localhost",
            port=8000,
            health_check_url="http://localhost:8000/health",
            dependencies=[],
            startup_order=1,
            shutdown_order=10,
            restart_policy="always",
            resource_limits={"cpu": "2", "memory": "4Gi"},
            environment_variables={"ENV": "test"}
        )
        
        orchestrator.add_component(component)
        
        assert "TEST_COMP_001" in orchestrator.components
        assert orchestrator.components["TEST_COMP_001"].name == "Test Trading System"
    
    def test_add_integration_config(self, orchestrator):
        """Test adding integration configuration."""
        config = IntegrationConfig(
            config_id="TEST_CONFIG_001",
            name="Test Integration",
            description="Test integration configuration",
            components=["TEST_COMP_001"],
            startup_sequence=["TEST_COMP_001"],
            shutdown_sequence=["TEST_COMP_001"],
            health_check_interval=30,
            restart_policy="always",
            monitoring_enabled=True,
            alerting_enabled=True,
            backup_enabled=True,
            scaling_enabled=True
        )
        
        orchestrator.add_integration_config(config)
        
        assert "TEST_CONFIG_001" in orchestrator.integration_configs
        assert orchestrator.integration_configs["TEST_CONFIG_001"].name == "Test Integration"
    
    def test_start_orchestration(self, orchestrator):
        """Test starting orchestration."""
        config_id = list(orchestrator.integration_configs.keys())[0]
        result = orchestrator.start_orchestration(config_id)
        
        assert result == True
        assert orchestrator.orchestration_status == OrchestrationStatus.RUNNING
        assert orchestrator.running == True
        
        # Stop orchestration
        orchestrator.stop_orchestration()
        assert orchestrator.orchestration_status == OrchestrationStatus.STOPPED
        assert orchestrator.running == False
    
    def test_get_system_status(self, orchestrator):
        """Test getting system status."""
        status = orchestrator.get_system_status()
        
        assert status is not None
        assert 'orchestration_status' in status
        assert 'total_components' in status
        assert 'running_components' in status
        assert 'error_components' in status
        assert 'components' in status
        assert status['total_components'] > 0
    
    def test_get_component_status(self, orchestrator):
        """Test getting component status."""
        component_id = list(orchestrator.components.keys())[0]
        status = orchestrator.get_component_status(component_id)
        
        assert status is not None
        assert 'component_id' in status
        assert 'name' in status
        assert 'type' in status
        assert 'status' in status
        assert status['component_id'] == component_id
    
    def test_restart_component(self, orchestrator):
        """Test restarting component."""
        component_id = list(orchestrator.components.keys())[0]
        result = orchestrator.restart_component(component_id)
        
        assert result == True
    
    def test_get_integration_configs(self, orchestrator):
        """Test getting integration configurations."""
        configs = orchestrator.get_integration_configs()
        
        assert configs is not None
        assert 'configurations' in configs
        assert 'total_configs' in configs
        assert configs['total_configs'] > 0

class TestDeploymentManager:
    """Test cases for deployment manager system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def deployment_manager(self, temp_db):
        """Create deployment manager instance."""
        return DeploymentManager(db_path=temp_db)
    
    def test_deployment_manager_initialization(self, deployment_manager):
        """Test deployment manager initialization."""
        assert deployment_manager is not None
        assert deployment_manager.db_path is not None
        assert len(deployment_manager.deployment_configs) > 0  # Should have default configs
        assert len(deployment_manager.deployments) == 0
        assert len(deployment_manager.active_deployments) == 0
    
    def test_add_deployment_config(self, deployment_manager):
        """Test adding deployment configuration."""
        config = DeploymentConfig(
            config_id="TEST_DEPLOY_001",
            name="Test Deployment",
            description="Test deployment configuration",
            application="test_app",
            version="1.0.0",
            environment=Environment.DEVELOPMENT,
            strategy=DeploymentStrategy.ROLLING,
            target_hosts=["test-host-01"],
            deployment_script="scripts/deploy_test.sh",
            rollback_script="scripts/rollback_test.sh",
            health_check_url="http://localhost:8000/health",
            pre_deployment_checks=["service_health"],
            post_deployment_checks=["health_check"],
            environment_variables={"ENV": "test"},
            resource_requirements={"cpu": "1", "memory": "2Gi"},
            deployment_timeout=900,
            rollback_timeout=300,
            max_retries=2
        )
        
        deployment_manager.add_deployment_config(config)
        
        assert "TEST_DEPLOY_001" in deployment_manager.deployment_configs
        assert deployment_manager.deployment_configs["TEST_DEPLOY_001"].name == "Test Deployment"
    
    def test_deploy(self, deployment_manager):
        """Test starting deployment."""
        config_id = list(deployment_manager.deployment_configs.keys())[0]
        deployment_id = deployment_manager.deploy(config_id)
        
        assert deployment_id is not None
        assert deployment_id.startswith("DEPLOY_")
        assert len(deployment_manager.deployments) > 0
        assert deployment_id in deployment_manager.active_deployments
    
    def test_get_deployment_status(self, deployment_manager):
        """Test getting deployment status."""
        config_id = list(deployment_manager.deployment_configs.keys())[0]
        deployment_id = deployment_manager.deploy(config_id)
        
        # Wait a bit for deployment to start
        import time
        time.sleep(1)
        
        status = deployment_manager.get_deployment_status(deployment_id)
        
        assert status is not None
        assert 'deployment_id' in status
        assert 'config_id' in status
        assert 'application' in status
        assert 'version' in status
        assert 'status' in status
        assert status['deployment_id'] == deployment_id
    
    def test_get_deployment_history(self, deployment_manager):
        """Test getting deployment history."""
        history = deployment_manager.get_deployment_history()
        
        assert isinstance(history, list)
        # History may be empty initially
    
    def test_get_active_deployments(self, deployment_manager):
        """Test getting active deployments."""
        active = deployment_manager.get_active_deployments()
        
        assert active is not None
        assert 'active_deployments' in active
        assert 'total_active' in active
        assert isinstance(active['active_deployments'], dict)
    
    def test_rollback(self, deployment_manager):
        """Test rollback deployment."""
        config_id = list(deployment_manager.deployment_configs.keys())[0]
        deployment_id = deployment_manager.deploy(config_id)
        
        # Wait for deployment to complete
        import time
        time.sleep(5)
        
        # Check if rollback is available
        status = deployment_manager.get_deployment_status(deployment_id)
        if status.get('rollback_available'):
            result = deployment_manager.rollback(deployment_id)
            assert result == True
    
    def test_get_deployment_configs(self, deployment_manager):
        """Test getting deployment configurations."""
        configs = deployment_manager.get_deployment_configs()
        
        assert configs is not None
        assert 'configurations' in configs
        assert 'total_configs' in configs
        assert configs['total_configs'] > 0

class TestServiceDiscovery:
    """Test cases for service discovery system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def service_discovery(self, temp_db):
        """Create service discovery instance."""
        return ServiceDiscovery(db_path=temp_db)
    
    def test_service_discovery_initialization(self, service_discovery):
        """Test service discovery initialization."""
        assert service_discovery is not None
        assert service_discovery.db_path is not None
        assert len(service_discovery.services) > 0  # Should have default services
        assert len(service_discovery.endpoints) > 0  # Should have default endpoints
        assert len(service_discovery.health_checks) > 0  # Should have default health checks
        assert service_discovery.discovery_active == False
    
    def test_register_service(self, service_discovery):
        """Test registering service."""
        service = ServiceRegistry(
            service_id="TEST_SVC_001",
            name="test-service",
            description="Test service",
            version="1.0.0",
            namespace="test",
            endpoints=[
                ServiceEndpoint(
                    endpoint_id="TEST_EP_001",
                    service_id="TEST_SVC_001",
                    host="localhost",
                    port=8000,
                    protocol=ServiceType.HTTP,
                    path="/api/v1",
                    weight=100,
                    priority=1,
                    tags=["test"],
                    metadata={"env": "test"}
                )
            ],
            health_checks=[
                ServiceHealth(
                    health_id="TEST_HC_001",
                    service_id="TEST_SVC_001",
                    endpoint_id="TEST_EP_001",
                    health_check_url="http://localhost:8000/health",
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
            retry_policy={"max_retries": 3},
            timeout_policy={"timeout": 30},
            tags=["test"],
            metadata={"owner": "test-team"}
        )
        
        service_discovery.register_service(service)
        
        assert "TEST_SVC_001" in service_discovery.services
        assert "TEST_EP_001" in service_discovery.endpoints
        assert "TEST_HC_001" in service_discovery.health_checks
        assert service_discovery.services["TEST_SVC_001"].name == "test-service"
    
    def test_deregister_service(self, service_discovery):
        """Test deregistering service."""
        service_id = list(service_discovery.services.keys())[0]
        result = service_discovery.deregister_service(service_id)
        
        assert result == True
        assert service_id not in service_discovery.services
    
    def test_start_discovery(self, service_discovery):
        """Test starting service discovery."""
        service_discovery.start_discovery()
        
        assert service_discovery.discovery_active == True
        assert service_discovery.discovery_thread is not None
        assert service_discovery.health_check_thread is not None
        
        # Stop discovery
        service_discovery.stop_discovery()
        assert service_discovery.discovery_active == False
    
    def test_discover_service(self, service_discovery):
        """Test discovering service."""
        service = service_discovery.discover_service("trading-system", "production")
        
        assert service is not None
        assert service.name == "trading-system"
        assert service.namespace == "production"
    
    def test_get_service_endpoints(self, service_discovery):
        """Test getting service endpoints."""
        service_id = list(service_discovery.services.keys())[0]
        endpoints = service_discovery.get_service_endpoints(service_id)
        
        assert isinstance(endpoints, list)
        assert len(endpoints) > 0
    
    def test_get_healthy_endpoint(self, service_discovery):
        """Test getting healthy endpoint."""
        service_id = list(service_discovery.services.keys())[0]
        endpoint = service_discovery.get_healthy_endpoint(service_id)
        
        # Endpoint may be None if no healthy endpoints
        if endpoint:
            assert endpoint.service_id == service_id
    
    def test_get_service_status(self, service_discovery):
        """Test getting service status."""
        service_id = list(service_discovery.services.keys())[0]
        status = service_discovery.get_service_status(service_id)
        
        assert status is not None
        assert 'service_id' in status
        assert 'name' in status
        assert 'status' in status
        assert 'endpoints' in status
        assert 'health_checks' in status
        assert status['service_id'] == service_id
    
    def test_get_all_services(self, service_discovery):
        """Test getting all services."""
        services = service_discovery.get_all_services()
        
        assert services is not None
        assert 'services' in services
        assert 'total_services' in services
        assert 'discovery_active' in services
        assert services['total_services'] > 0

class TestLoadBalancer:
    """Test cases for load balancer system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def load_balancer(self, temp_db):
        """Create load balancer instance."""
        return LoadBalancer(db_path=temp_db)
    
    def test_load_balancer_initialization(self, load_balancer):
        """Test load balancer initialization."""
        assert load_balancer is not None
        assert load_balancer.db_path is not None
        assert len(load_balancer.load_balancers) > 0  # Should have default load balancer
        assert len(load_balancer.backend_servers) > 0  # Should have default servers
        assert len(load_balancer.health_checks) > 0  # Should have default health checks
        assert load_balancer.balancing_active == False
    
    def test_add_load_balancer(self, load_balancer):
        """Test adding load balancer."""
        servers = [
            BackendServer(
                server_id="TEST_SRV_001",
                host="localhost",
                port=8000,
                weight=100,
                priority=1,
                max_connections=1000,
                current_connections=0,
                health_checks=[
                    HealthCheck(
                        health_check_id="TEST_HC_001",
                        server_id="TEST_SRV_001",
                        check_type=HealthCheckType.HTTP,
                        url="http://localhost:8000/health",
                        interval=30,
                        timeout=10,
                        healthy_threshold=2,
                        unhealthy_threshold=3
                    )
                ]
            )
        ]
        
        lb = LoadBalancer(
            lb_id="TEST_LB_001",
            name="Test Load Balancer",
            description="Test load balancer",
            frontend_host="localhost",
            frontend_port=8080,
            backend_servers=servers,
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
            sticky_cookie_name="TEST_SESSION",
            sticky_cookie_timeout=3600
        )
        
        load_balancer.add_load_balancer(lb)
        
        assert "TEST_LB_001" in load_balancer.load_balancers
        assert "TEST_SRV_001" in load_balancer.backend_servers
        assert "TEST_HC_001" in load_balancer.health_checks
        assert load_balancer.load_balancers["TEST_LB_001"].name == "Test Load Balancer"
    
    def test_remove_load_balancer(self, load_balancer):
        """Test removing load balancer."""
        lb_id = list(load_balancer.load_balancers.keys())[0]
        result = load_balancer.remove_load_balancer(lb_id)
        
        assert result == True
        assert lb_id not in load_balancer.load_balancers
    
    def test_start_load_balancing(self, load_balancer):
        """Test starting load balancing."""
        load_balancer.start_load_balancing()
        
        assert load_balancer.balancing_active == True
        assert load_balancer.balancing_thread is not None
        assert load_balancer.health_check_thread is not None
        
        # Stop load balancing
        load_balancer.stop_load_balancing()
        assert load_balancer.balancing_active == False
    
    def test_get_backend_server(self, load_balancer):
        """Test getting backend server."""
        lb_id = list(load_balancer.load_balancers.keys())[0]
        server = load_balancer.get_backend_server(lb_id)
        
        # Server may be None if no healthy servers
        if server:
            assert server.server_id in load_balancer.backend_servers
    
    def test_release_connection(self, load_balancer):
        """Test releasing connection."""
        server_id = list(load_balancer.backend_servers.keys())[0]
        initial_connections = load_balancer.backend_servers[server_id].current_connections
        
        load_balancer.release_connection(server_id)
        
        # Connections should not go below 0
        assert load_balancer.backend_servers[server_id].current_connections >= 0
    
    def test_get_load_balancer_status(self, load_balancer):
        """Test getting load balancer status."""
        lb_id = list(load_balancer.load_balancers.keys())[0]
        status = load_balancer.get_load_balancer_status(lb_id)
        
        assert status is not None
        assert 'lb_id' in status
        assert 'name' in status
        assert 'strategy' in status
        assert 'servers' in status
        assert 'total_servers' in status
        assert 'healthy_servers' in status
        assert status['lb_id'] == lb_id
    
    def test_get_all_load_balancers(self, load_balancer):
        """Test getting all load balancers."""
        lbs = load_balancer.get_all_load_balancers()
        
        assert lbs is not None
        assert 'load_balancers' in lbs
        assert 'total_load_balancers' in lbs
        assert 'balancing_active' in lbs
        assert lbs['total_load_balancers'] > 0

class TestIntegration:
    """Integration tests for the enterprise integration system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    def test_end_to_end_enterprise_integration_workflow(self, temp_db):
        """Test complete enterprise integration workflow."""
        # Initialize all components
        orchestrator = EnterpriseOrchestrator(db_path=temp_db)
        deployment_manager = DeploymentManager(db_path=temp_db)
        service_discovery = ServiceDiscovery(db_path=temp_db)
        load_balancer = LoadBalancer(db_path=temp_db)
        
        # 1. Start orchestration
        config_id = list(orchestrator.integration_configs.keys())[0]
        orchestration_started = orchestrator.start_orchestration(config_id)
        
        # 2. Start service discovery
        service_discovery.start_discovery()
        
        # 3. Start load balancing
        load_balancer.start_load_balancing()
        
        # 4. Get system status
        system_status = orchestrator.get_system_status()
        
        # 5. Get service status
        service_id = list(service_discovery.services.keys())[0]
        service_status = service_discovery.get_service_status(service_id)
        
        # 6. Get load balancer status
        lb_id = list(load_balancer.load_balancers.keys())[0]
        lb_status = load_balancer.get_load_balancer_status(lb_id)
        
        # 7. Start deployment
        deploy_config_id = list(deployment_manager.deployment_configs.keys())[0]
        deployment_id = deployment_manager.deploy(deploy_config_id)
        
        # 8. Get deployment status
        deployment_status = deployment_manager.get_deployment_status(deployment_id)
        
        # 9. Stop all services
        orchestrator.stop_orchestration()
        service_discovery.stop_discovery()
        load_balancer.stop_load_balancing()
        
        # Verify all components worked together
        assert orchestration_started == True
        assert system_status is not None
        assert service_status is not None
        assert lb_status is not None
        assert deployment_id is not None
        assert deployment_status is not None
    
    def test_enterprise_integration_with_service_discovery(self, temp_db):
        """Test enterprise integration with service discovery."""
        orchestrator = EnterpriseOrchestrator(db_path=temp_db)
        service_discovery = ServiceDiscovery(db_path=temp_db)
        
        # Start orchestration
        config_id = list(orchestrator.integration_configs.keys())[0]
        orchestrator.start_orchestration(config_id)
        
        # Start service discovery
        service_discovery.start_discovery()
        
        # Get system status
        system_status = orchestrator.get_system_status()
        
        # Get service status
        service_id = list(service_discovery.services.keys())[0]
        service_status = service_discovery.get_service_status(service_id)
        
        # Discover service
        discovered_service = service_discovery.discover_service("trading-system", "production")
        
        # Get healthy endpoint
        healthy_endpoint = service_discovery.get_healthy_endpoint(service_id)
        
        # Stop services
        orchestrator.stop_orchestration()
        service_discovery.stop_discovery()
        
        # Verify integration
        assert system_status is not None
        assert service_status is not None
        assert discovered_service is not None
        # healthy_endpoint may be None if no healthy endpoints
    
    def test_enterprise_integration_with_load_balancing(self, temp_db):
        """Test enterprise integration with load balancing."""
        service_discovery = ServiceDiscovery(db_path=temp_db)
        load_balancer = LoadBalancer(db_path=temp_db)
        
        # Start service discovery
        service_discovery.start_discovery()
        
        # Start load balancing
        load_balancer.start_load_balancing()
        
        # Get service status
        service_id = list(service_discovery.services.keys())[0]
        service_status = service_discovery.get_service_status(service_id)
        
        # Get load balancer status
        lb_id = list(load_balancer.load_balancers.keys())[0]
        lb_status = load_balancer.get_load_balancer_status(lb_id)
        
        # Get backend server
        backend_server = load_balancer.get_backend_server(lb_id)
        
        # Release connection
        if backend_server:
            load_balancer.release_connection(backend_server.server_id)
        
        # Stop services
        service_discovery.stop_discovery()
        load_balancer.stop_load_balancing()
        
        # Verify integration
        assert service_status is not None
        assert lb_status is not None
        # backend_server may be None if no healthy servers

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
