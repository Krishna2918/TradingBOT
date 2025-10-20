"""
Resource Manager for Agentic AI System

The Resource Manager is the central intelligence that:
- Monitors system resources (CPU, memory, GPU)
- Dynamically activates/deactivates agents based on availability and need
- Learns which agents to prioritize based on market conditions
- Maintains agent priority queue
- Logs all resource allocation decisions

This is the "brain" that ensures optimal resource utilization across all agents.
"""

import logging
import time
import asyncio
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json

from .base_agent import BaseAgent, AgentPriority, AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    available_memory_mb: float
    active_agents: int
    total_agents: int


@dataclass
class AgentActivationRecord:
    """Record of agent activation/deactivation"""
    timestamp: datetime
    agent_id: str
    action: str  # 'activate' or 'deactivate'
    reason: str
    system_cpu: float
    system_memory: float


class ResourceManager:
    """
    Intelligent Resource Manager for dynamic agent allocation.
    
    Key responsibilities:
    1. Monitor system resources in real-time
    2. Activate/deactivate agents based on resource availability
    3. Learn optimal agent activation patterns
    4. Maintain agent priority queue
    5. Handle emergency resource situations
    """
    
    def __init__(
        self,
        cpu_threshold_critical: float = 85.0,
        cpu_threshold_warning: float = 70.0,
        memory_threshold_critical: float = 80.0,
        memory_threshold_warning: float = 60.0,
        learning_enabled: bool = True
    ):
        self.cpu_threshold_critical = cpu_threshold_critical
        self.cpu_threshold_warning = cpu_threshold_warning
        self.memory_threshold_critical = memory_threshold_critical
        self.memory_threshold_warning = memory_threshold_warning
        self.learning_enabled = learning_enabled
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.active_agents: List[str] = []
        self.inactive_agents: List[str] = []
        
        # Resource tracking
        self.resource_history: deque = deque(maxlen=1000)
        self.activation_history: List[AgentActivationRecord] = []
        
        # Learning data
        self.activation_patterns: Dict[str, Any] = {}
        self.performance_data: Dict[str, List[float]] = {}
        
        # State
        self.is_running = False
        self.emergency_mode = False
        self.start_time = datetime.now()
        
        logger.info("Resource Manager initialized")
        logger.info(f"  CPU Warning: {cpu_threshold_warning}%, Critical: {cpu_threshold_critical}%")
        logger.info(f"  Memory Warning: {memory_threshold_warning}%, Critical: {memory_threshold_critical}%")
    
    async def start(self) -> bool:
        """
        Start the Resource Manager and activate all registered agents.
        
        Returns:
            True if startup successful
        """
        try:
            logger.info("Starting Resource Manager...")
            self.is_running = True
            
            # Activate all registered agents based on priority
            activation_results = []
            
            # First activate CRITICAL priority agents
            critical_agents = [agent_id for agent_id, agent in self.agents.items() 
                             if agent.priority == AgentPriority.CRITICAL]
            
            for agent_id in critical_agents:
                try:
                    success = await self.activate_agent(agent_id, "System startup - CRITICAL priority")
                    activation_results.append((agent_id, success))
                    if success:
                        logger.info(f"✓ Started CRITICAL agent: {agent_id}")
                    else:
                        logger.error(f"✗ Failed to start CRITICAL agent: {agent_id}")
                except Exception as e:
                    logger.error(f"Error starting CRITICAL agent {agent_id}: {e}")
                    activation_results.append((agent_id, False))
            
            # Then activate IMPORTANT priority agents
            important_agents = [agent_id for agent_id, agent in self.agents.items() 
                              if agent.priority == AgentPriority.IMPORTANT]
            
            for agent_id in important_agents:
                try:
                    success = await self.activate_agent(agent_id, "System startup - IMPORTANT priority")
                    activation_results.append((agent_id, success))
                    if success:
                        logger.info(f"✓ Started IMPORTANT agent: {agent_id}")
                    else:
                        logger.warning(f"⚠ Failed to start IMPORTANT agent: {agent_id}")
                except Exception as e:
                    logger.error(f"Error starting IMPORTANT agent {agent_id}: {e}")
                    activation_results.append((agent_id, False))
            
            # Finally activate OPTIONAL priority agents if resources allow
            optional_agents = [agent_id for agent_id, agent in self.agents.items() 
                             if agent.priority == AgentPriority.OPTIONAL]
            
            for agent_id in optional_agents:
                try:
                    success = await self.activate_agent(agent_id, "System startup - OPTIONAL priority")
                    activation_results.append((agent_id, success))
                    if success:
                        logger.info(f"✓ Started OPTIONAL agent: {agent_id}")
                    else:
                        logger.info(f"ℹ Skipped OPTIONAL agent (insufficient resources): {agent_id}")
                except Exception as e:
                    logger.error(f"Error starting OPTIONAL agent {agent_id}: {e}")
                    activation_results.append((agent_id, False))
            
            # Summary
            successful = sum(1 for _, success in activation_results if success)
            total = len(activation_results)
            
            logger.info(f"Resource Manager startup complete: {successful}/{total} agents activated")
            
            if successful > 0:
                logger.info("✓ Resource Manager started successfully")
                return True
            else:
                logger.error("✗ Resource Manager startup failed - no agents activated")
                return False
                
        except Exception as e:
            logger.error(f"Error starting Resource Manager: {e}")
            self.is_running = False
            return False
    
    # ========================================================================
    # AGENT REGISTRATION
    # ========================================================================
    
    def register_agent(self, agent: BaseAgent) -> bool:
        """
        Register a new agent with the Resource Manager.
        
        Args:
            agent: Agent to register
            
        Returns:
            True if registration successful
        """
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.agent_id} already registered")
            return False
        
        self.agents[agent.agent_id] = agent
        self.inactive_agents.append(agent.agent_id)
        
        logger.info(f"Registered agent: {agent.name} ({agent.agent_id}) with priority {agent.priority.name}")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False
        
        # Deactivate if active
        if agent_id in self.active_agents:
            self.deactivate_agent(agent_id, "Agent unregistered")
        
        # Remove from registry
        del self.agents[agent_id]
        if agent_id in self.inactive_agents:
            self.inactive_agents.remove(agent_id)
        
        logger.info(f"Unregistered agent: {agent_id}")
        return True
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_active_agents(self) -> List[str]:
        """Get list of active agent IDs"""
        return self.active_agents.copy()
    
    # ========================================================================
    # RESOURCE MONITORING
    # ========================================================================
    
    def get_system_resources(self) -> ResourceSnapshot:
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / (1024 * 1024),
            available_memory_mb=memory.available / (1024 * 1024),
            active_agents=len(self.active_agents),
            total_agents=len(self.agents)
        )
        
        # Add to history
        self.resource_history.append(snapshot)
        
        return snapshot
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get detailed resource status"""
        snapshot = self.get_system_resources()
        
        # Determine status level
        cpu_status = "normal"
        if snapshot.cpu_percent > self.cpu_threshold_critical:
            cpu_status = "critical"
        elif snapshot.cpu_percent > self.cpu_threshold_warning:
            cpu_status = "warning"
        
        memory_status = "normal"
        if snapshot.memory_percent > self.memory_threshold_critical:
            memory_status = "critical"
        elif snapshot.memory_percent > self.memory_threshold_warning:
            memory_status = "warning"
        
        return {
            'timestamp': snapshot.timestamp.isoformat(),
            'cpu': {
                'percent': snapshot.cpu_percent,
                'status': cpu_status,
                'threshold_warning': self.cpu_threshold_warning,
                'threshold_critical': self.cpu_threshold_critical
            },
            'memory': {
                'percent': snapshot.memory_percent,
                'used_mb': snapshot.memory_mb,
                'available_mb': snapshot.available_memory_mb,
                'status': memory_status,
                'threshold_warning': self.memory_threshold_warning,
                'threshold_critical': self.memory_threshold_critical
            },
            'agents': {
                'active': len(self.active_agents),
                'inactive': len(self.inactive_agents),
                'total': len(self.agents)
            },
            'emergency_mode': self.emergency_mode
        }
    
    # ========================================================================
    # AGENT ACTIVATION/DEACTIVATION
    # ========================================================================
    
    async def activate_agent(self, agent_id: str, reason: str = "Manual activation") -> bool:
        """
        Activate an agent.
        
        Args:
            agent_id: ID of agent to activate
            reason: Reason for activation
            
        Returns:
            True if activation successful
        """
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        if agent_id in self.active_agents:
            logger.warning(f"Agent {agent_id} is already active")
            return True
        
        agent = self.agents[agent_id]
        
        # Check if we have enough resources
        resources = self.get_system_resources()
        if not self._can_activate_agent(agent, resources):
            logger.warning(f"Insufficient resources to activate {agent.name}")
            return False
        
        # Initialize agent
        try:
            initialized = await agent.initialize()
            if not initialized:
                logger.error(f"Failed to initialize agent {agent.name}")
                return False
        except Exception as e:
            logger.error(f"Error initializing agent {agent.name}: {e}")
            return False
        
        # Move to active list
        self.inactive_agents.remove(agent_id)
        self.active_agents.append(agent_id)
        agent.status = AgentStatus.IDLE
        
        # Log activation
        self._log_activation(agent_id, 'activate', reason, resources)
        
        logger.info(f"✓ Activated agent: {agent.name} ({reason})")
        return True
    
    async def deactivate_agent(self, agent_id: str, reason: str = "Manual deactivation") -> bool:
        """
        Deactivate an agent.
        
        Args:
            agent_id: ID of agent to deactivate
            reason: Reason for deactivation
            
        Returns:
            True if deactivation successful
        """
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        if agent_id not in self.active_agents:
            logger.warning(f"Agent {agent_id} is not active")
            return True
        
        agent = self.agents[agent_id]
        
        # Shutdown agent
        try:
            shutdown = await agent.shutdown()
            if not shutdown:
                logger.warning(f"Agent {agent.name} shutdown returned False")
        except Exception as e:
            logger.error(f"Error shutting down agent {agent.name}: {e}")
        
        # Move to inactive list
        self.active_agents.remove(agent_id)
        self.inactive_agents.append(agent_id)
        agent.status = AgentStatus.STOPPED
        
        # Log deactivation
        resources = self.get_system_resources()
        self._log_activation(agent_id, 'deactivate', reason, resources)
        
        logger.info(f"✓ Deactivated agent: {agent.name} ({reason})")
        return True
    
    def _can_activate_agent(self, agent: BaseAgent, resources: ResourceSnapshot) -> bool:
        """Check if there are enough resources to activate an agent"""
        # Check CPU
        if resources.cpu_percent + agent.resource_requirements.min_cpu_percent > self.cpu_threshold_critical:
            logger.warning(f"Insufficient CPU to activate {agent.name}")
            return False
        
        # Check memory
        if resources.available_memory_mb < agent.resource_requirements.min_memory_mb:
            logger.warning(f"Insufficient memory to activate {agent.name}")
            return False
        
        return True
    
    def _log_activation(self, agent_id: str, action: str, reason: str, resources: ResourceSnapshot):
        """Log agent activation/deactivation"""
        record = AgentActivationRecord(
            timestamp=datetime.now(),
            agent_id=agent_id,
            action=action,
            reason=reason,
            system_cpu=resources.cpu_percent,
            system_memory=resources.memory_percent
        )
        
        self.activation_history.append(record)
        
        # Save to file
        try:
            log_entry = {
                'timestamp': record.timestamp.isoformat(),
                'agent_id': record.agent_id,
                'action': record.action,
                'reason': record.reason,
                'system_cpu': record.system_cpu,
                'system_memory': record.system_memory
            }
            
            with open('logs/agent_activations.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Error logging activation: {e}")
    
    # ========================================================================
    # INTELLIGENT AGENT MANAGEMENT
    # ========================================================================
    
    async def optimize_agent_allocation(self) -> Dict[str, Any]:
        """
        Intelligently optimize agent allocation based on current resources.
        
        Returns:
            Optimization result
        """
        resources = self.get_system_resources()
        actions_taken = []
        
        # Emergency mode check
        if resources.cpu_percent > self.cpu_threshold_critical or \
           resources.memory_percent > self.memory_threshold_critical:
            logger.warning("EMERGENCY MODE: Critical resource usage detected")
            self.emergency_mode = True
            actions_taken.extend(await self._handle_emergency_mode())
        else:
            self.emergency_mode = False
        
        # Normal optimization
        if not self.emergency_mode:
            # Check if we should activate more agents
            if resources.cpu_percent < self.cpu_threshold_warning and \
               resources.memory_percent < self.memory_threshold_warning:
                actions_taken.extend(await self._activate_pending_agents(resources))
            
            # Check if we should deactivate low-priority agents
            elif resources.cpu_percent > self.cpu_threshold_warning or \
                 resources.memory_percent > self.memory_threshold_warning:
                actions_taken.extend(await self._deactivate_optional_agents(resources))
        
        return {
            'timestamp': datetime.now().isoformat(),
            'emergency_mode': self.emergency_mode,
            'resources': {
                'cpu_percent': resources.cpu_percent,
                'memory_percent': resources.memory_percent
            },
            'agents': {
                'active': len(self.active_agents),
                'inactive': len(self.inactive_agents)
            },
            'actions_taken': actions_taken
        }
    
    async def _handle_emergency_mode(self) -> List[str]:
        """Handle critical resource situation"""
        actions = []
        
        # Keep only CRITICAL priority agents
        for agent_id in self.active_agents.copy():
            agent = self.agents[agent_id]
            if agent.priority != AgentPriority.CRITICAL:
                await self.deactivate_agent(agent_id, "Emergency mode - resource conservation")
                actions.append(f"Deactivated {agent.name} (emergency mode)")
        
        logger.warning(f"Emergency mode: {len(actions)} agents deactivated")
        return actions
    
    async def _activate_pending_agents(self, resources: ResourceSnapshot) -> List[str]:
        """Activate pending agents based on priority and resources"""
        actions = []
        
        # Sort inactive agents by priority
        inactive_sorted = sorted(
            self.inactive_agents,
            key=lambda aid: self.agents[aid].priority.value
        )
        
        for agent_id in inactive_sorted:
            agent = self.agents[agent_id]
            
            # Check if we still have resources
            current_resources = self.get_system_resources()
            if current_resources.cpu_percent > self.cpu_threshold_warning:
                break
            
            if self._can_activate_agent(agent, current_resources):
                success = await self.activate_agent(agent_id, "Automatic optimization - resources available")
                if success:
                    actions.append(f"Activated {agent.name}")
        
        return actions
    
    async def _deactivate_optional_agents(self, resources: ResourceSnapshot) -> List[str]:
        """Deactivate OPTIONAL priority agents to free resources"""
        actions = []
        
        # Deactivate OPTIONAL priority agents first
        for agent_id in self.active_agents.copy():
            agent = self.agents[agent_id]
            if agent.priority == AgentPriority.OPTIONAL:
                await self.deactivate_agent(agent_id, "Resource conservation - high usage detected")
                actions.append(f"Deactivated {agent.name} (resource conservation)")
        
        return actions
    
    # ========================================================================
    # EMERGENCY CONTROLS
    # ========================================================================
    
    async def emergency_stop_all(self) -> Dict[str, Any]:
        """Emergency stop all non-critical agents"""
        logger.critical("EMERGENCY STOP ALL AGENTS INITIATED")
        
        stopped = []
        for agent_id in self.active_agents.copy():
            agent = self.agents[agent_id]
            if agent.priority != AgentPriority.CRITICAL:
                await self.deactivate_agent(agent_id, "EMERGENCY STOP")
                stopped.append(agent.name)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'action': 'emergency_stop_all',
            'agents_stopped': stopped,
            'critical_agents_remaining': [
                self.agents[aid].name for aid in self.active_agents
            ]
        }
    
    # ========================================================================
    # STATUS & METRICS
    # ========================================================================
    
    def get_all_agents_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            agent_id: agent.get_status()
            for agent_id, agent in self.agents.items()
        }
    
    def get_agent_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for specific agent"""
        agent = self.get_agent(agent_id)
        if agent:
            return agent.get_status()
        return None
    
    def get_resource_manager_status(self) -> Dict[str, Any]:
        """Get comprehensive Resource Manager status"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'is_running': self.is_running,
            'emergency_mode': self.emergency_mode,
            'thresholds': {
                'cpu_warning': self.cpu_threshold_warning,
                'cpu_critical': self.cpu_threshold_critical,
                'memory_warning': self.memory_threshold_warning,
                'memory_critical': self.memory_threshold_critical
            },
            'resources': self.get_resource_status(),
            'agents': {
                'total': len(self.agents),
                'active': len(self.active_agents),
                'inactive': len(self.inactive_agents),
                'active_list': [self.agents[aid].name for aid in self.active_agents],
                'inactive_list': [self.agents[aid].name for aid in self.inactive_agents]
            },
            'history': {
                'activations_total': len([a for a in self.activation_history if a.action == 'activate']),
                'deactivations_total': len([a for a in self.activation_history if a.action == 'deactivate']),
                'resource_snapshots': len(self.resource_history)
            }
        }
    
    def __repr__(self) -> str:
        return f"<ResourceManager: {len(self.active_agents)}/{len(self.agents)} agents active>"


