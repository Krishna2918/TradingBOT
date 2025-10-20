"""
Base Agent Class for Agentic AI System

This module provides the foundation for all specialized agents in the TradingBOT system.
Each agent has standardized interfaces for resource management, communication, and health monitoring.
"""

import logging
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import psutil

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent operational status"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    ACTIVE = "active"
    PROCESSING = "processing"
    TRAINING = "training"
    ERROR = "error"
    STOPPED = "stopped"


class AgentPriority(Enum):
    """Agent priority levels for resource allocation"""
    CRITICAL = 1  # Must always run (Risk, Monitoring, Execution)
    IMPORTANT = 2  # Run when resources available (Portfolio, Market Analysis)
    OPTIONAL = 3  # Run only when abundant resources (Learning)


@dataclass
class AgentMetrics:
    """Metrics tracked for each agent"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    avg_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceRequirements:
    """Resource requirements for an agent"""
    min_cpu_percent: float = 5.0  # Minimum CPU required
    min_memory_mb: float = 100.0  # Minimum memory required
    max_cpu_percent: float = 30.0  # Maximum CPU to use
    max_memory_mb: float = 500.0  # Maximum memory to use


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Agentic AI system.
    
    Provides standardized interfaces for:
    - Resource management
    - Communication
    - Health monitoring
    - Task execution
    - Performance tracking
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        priority: AgentPriority = AgentPriority.IMPORTANT,
        resource_requirements: Optional[ResourceRequirements] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.priority = priority
        self.resource_requirements = resource_requirements or ResourceRequirements()
        
        self.status = AgentStatus.INITIALIZING
        self.metrics = AgentMetrics()
        self.start_time = datetime.now()
        self.current_task: Optional[str] = None
        self.task_queue: List[Dict[str, Any]] = []
        
        # Process tracking
        self.process = psutil.Process()
        
        logger.info(f"Agent {self.name} ({self.agent_id}) initialized with priority {self.priority.name}")
    
    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single task.
        
        Args:
            task: Task data to process
            
        Returns:
            Task result
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the agent's resources and dependencies.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the agent.
        
        Returns:
            True if shutdown successful
        """
        pass
    
    # ========================================================================
    # RESOURCE MANAGEMENT
    # ========================================================================
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            self.metrics.cpu_usage = cpu_percent
            self.metrics.memory_usage = memory_mb
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'cpu_available': self.resource_requirements.max_cpu_percent - cpu_percent,
                'memory_available': self.resource_requirements.max_memory_mb - memory_mb
            }
        except Exception as e:
            logger.error(f"Error getting resource usage for {self.name}: {e}")
            return {'cpu_percent': 0.0, 'memory_mb': 0.0}
    
    def can_accept_task(self) -> bool:
        """Check if agent can accept new tasks based on resources"""
        usage = self.get_resource_usage()
        
        # Check if within resource limits
        if usage['cpu_percent'] > self.resource_requirements.max_cpu_percent:
            logger.warning(f"Agent {self.name} CPU usage ({usage['cpu_percent']}%) exceeds limit")
            return False
        
        if usage['memory_mb'] > self.resource_requirements.max_memory_mb:
            logger.warning(f"Agent {self.name} memory usage ({usage['memory_mb']}MB) exceeds limit")
            return False
        
        return True
    
    # ========================================================================
    # HEALTH & STATUS
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        success_rate = 0.0
        
        if self.metrics.tasks_completed + self.metrics.tasks_failed > 0:
            success_rate = self.metrics.tasks_completed / (
                self.metrics.tasks_completed + self.metrics.tasks_failed
            )
        
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'status': self.status.value,
            'priority': self.priority.name,
            'uptime_seconds': uptime,
            'current_task': self.current_task,
            'queue_depth': len(self.task_queue),
            'metrics': {
                'tasks_completed': self.metrics.tasks_completed,
                'tasks_failed': self.metrics.tasks_failed,
                'success_rate': success_rate,
                'avg_response_time': self.metrics.avg_response_time,
                'total_execution_time': self.metrics.total_execution_time,
                'messages_sent': self.metrics.messages_sent,
                'messages_received': self.metrics.messages_received
            },
            'resources': self.get_resource_usage(),
            'last_activity': self.metrics.last_activity.isoformat()
        }
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy"""
        if self.status == AgentStatus.ERROR:
            return False
        
        # Check if agent is responding
        time_since_activity = (datetime.now() - self.metrics.last_activity).total_seconds()
        if time_since_activity > 300:  # 5 minutes
            logger.warning(f"Agent {self.name} has been inactive for {time_since_activity}s")
            return False
        
        # Check resource usage
        usage = self.get_resource_usage()
        if usage['cpu_percent'] > self.resource_requirements.max_cpu_percent * 1.5:
            logger.warning(f"Agent {self.name} CPU usage critically high: {usage['cpu_percent']}%")
            return False
        
        return True
    
    def get_uptime(self) -> float:
        """Get agent uptime in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
    
    # ========================================================================
    # TASK MANAGEMENT
    # ========================================================================
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task with performance tracking.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        self.current_task = task.get('type', 'unknown')
        self.status = AgentStatus.PROCESSING
        start_time = time.time()
        
        try:
            result = await self.process_task(task)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics.tasks_completed += 1
            self.metrics.total_execution_time += execution_time
            
            # Update average response time
            total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
            self.metrics.avg_response_time = self.metrics.total_execution_time / total_tasks
            
            self.metrics.last_activity = datetime.now()
            self.status = AgentStatus.IDLE
            self.current_task = None
            
            logger.info(f"Agent {self.name} completed task in {execution_time:.2f}s")
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'agent_id': self.agent_id
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.tasks_failed += 1
            self.metrics.last_activity = datetime.now()
            self.status = AgentStatus.ERROR
            self.current_task = None
            
            logger.error(f"Agent {self.name} task failed after {execution_time:.2f}s: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'agent_id': self.agent_id
            }
    
    def add_to_queue(self, task: Dict[str, Any]) -> bool:
        """Add task to agent's queue"""
        if not self.can_accept_task():
            logger.warning(f"Agent {self.name} cannot accept task - resource limits reached")
            return False
        
        self.task_queue.append(task)
        logger.info(f"Task added to {self.name} queue. Queue depth: {len(self.task_queue)}")
        return True
    
    def get_queue_depth(self) -> int:
        """Get current queue depth"""
        return len(self.task_queue)
    
    def get_current_task(self) -> Optional[str]:
        """Get current task being processed"""
        return self.current_task
    
    # ========================================================================
    # COMMUNICATION
    # ========================================================================
    
    def on_message_sent(self):
        """Track outgoing message"""
        self.metrics.messages_sent += 1
        self.metrics.last_activity = datetime.now()
    
    def on_message_received(self):
        """Track incoming message"""
        self.metrics.messages_received += 1
        self.metrics.last_activity = datetime.now()
    
    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        return self.get_resource_usage()['cpu_percent']
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.get_resource_usage()['memory_mb']
    
    def get_completion_rate(self) -> float:
        """Get task completion rate (tasks per second)"""
        uptime = self.get_uptime()
        if uptime > 0:
            return self.metrics.tasks_completed / uptime
        return 0.0
    
    def get_avg_response_time(self) -> float:
        """Get average response time"""
        return self.metrics.avg_response_time
    
    def get_success_rate(self) -> float:
        """Get task success rate"""
        total = self.metrics.tasks_completed + self.metrics.tasks_failed
        if total > 0:
            return self.metrics.tasks_completed / total
        return 0.0
    
    def __repr__(self) -> str:
        return f"<Agent {self.name} ({self.agent_id}) - {self.status.value}>"


