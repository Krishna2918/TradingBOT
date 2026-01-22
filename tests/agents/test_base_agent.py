"""
Tests for BaseAgent

Tests the core functionality of the base agent class including:
- Initialization
- Resource tracking
- Task execution
- Health monitoring
- Performance metrics
"""

import pytest
import asyncio
from typing import Dict, Any

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.base_agent import BaseAgent, AgentStatus, AgentPriority, ResourceRequirements


class TestAgent(BaseAgent):
    """Test agent for testing base agent functionality"""
    
    def __init__(self):
        super().__init__(
            agent_id='test_agent',
            name='Test Agent',
            priority=AgentPriority.IMPORTANT
        )
        self.initialized = False
    
    async def initialize(self) -> bool:
        self.initialized = True
        self.status = AgentStatus.IDLE
        return True
    
    async def shutdown(self) -> bool:
        self.initialized = False
        self.status = AgentStatus.STOPPED
        return True
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Simple task processing for testing"""
        await asyncio.sleep(0.01)  # Simulate work
        return {'result': 'success', 'input': task}


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization"""
    agent = TestAgent()
    
    assert agent.agent_id == 'test_agent'
    assert agent.name == 'Test Agent'
    assert agent.priority == AgentPriority.IMPORTANT
    assert agent.status == AgentStatus.INITIALIZING
    
    # Initialize
    success = await agent.initialize()
    assert success is True
    assert agent.initialized is True
    assert agent.status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_agent_task_execution():
    """Test agent task execution"""
    agent = TestAgent()
    await agent.initialize()
    
    task = {'type': 'test_task', 'data': 'test_data'}
    result = await agent.execute_task(task)
    
    assert result['success'] is True
    assert 'result' in result
    assert result['execution_time'] > 0
    assert agent.metrics.tasks_completed == 1
    assert agent.metrics.tasks_failed == 0


@pytest.mark.asyncio
async def test_agent_resource_tracking():
    """Test agent resource usage tracking"""
    agent = TestAgent()
    await agent.initialize()
    
    # Execute a task to generate some resource usage
    await agent.execute_task({'type': 'test'})
    
    # Check resource usage
    usage = agent.get_resource_usage()
    assert 'cpu_percent' in usage
    assert 'memory_mb' in usage
    assert usage['cpu_percent'] >= 0
    assert usage['memory_mb'] > 0


@pytest.mark.asyncio
async def test_agent_status():
    """Test agent status reporting"""
    agent = TestAgent()
    await agent.initialize()
    
    status = agent.get_status()
    
    assert status['agent_id'] == 'test_agent'
    assert status['name'] == 'Test Agent'
    assert status['status'] == AgentStatus.IDLE.value
    assert status['priority'] == AgentPriority.IMPORTANT.name
    assert 'uptime_seconds' in status
    assert 'metrics' in status
    assert 'resources' in status


@pytest.mark.asyncio
async def test_agent_health_check():
    """Test agent health checking"""
    agent = TestAgent()
    await agent.initialize()
    
    # Fresh agent should be healthy
    assert agent.is_healthy() is True
    
    # Agent in error state should not be healthy
    agent.status = AgentStatus.ERROR
    assert agent.is_healthy() is False


@pytest.mark.asyncio
async def test_agent_metrics_tracking():
    """Test agent performance metrics tracking"""
    agent = TestAgent()
    await agent.initialize()
    
    # Execute multiple tasks
    for i in range(5):
        await agent.execute_task({'id': i})
    
    # Check metrics
    assert agent.metrics.tasks_completed == 5
    assert agent.metrics.avg_response_time > 0
    assert agent.get_success_rate() == 1.0
    assert agent.get_completion_rate() > 0


@pytest.mark.asyncio
async def test_agent_queue_management():
    """Test agent task queue management"""
    agent = TestAgent()
    await agent.initialize()
    
    # Add tasks to queue
    task1 = {'id': 1}
    task2 = {'id': 2}
    
    success1 = agent.add_to_queue(task1)
    success2 = agent.add_to_queue(task2)
    
    assert success1 is True
    assert success2 is True
    assert agent.get_queue_depth() == 2


@pytest.mark.asyncio
async def test_agent_shutdown():
    """Test agent shutdown"""
    agent = TestAgent()
    await agent.initialize()
    
    assert agent.initialized is True
    
    success = await agent.shutdown()
    assert success is True
    assert agent.status == AgentStatus.STOPPED
    assert agent.initialized is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


