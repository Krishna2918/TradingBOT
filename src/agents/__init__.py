"""
Agentic AI System

This module provides an intelligent agent-based architecture for the TradingBOT system.
Agents are dynamically managed by the Resource Manager based on system resources and trading needs.

Core Components:
- BaseAgent: Foundation for all agents
- ResourceManager: Central intelligence for resource allocation
- RiskManagementAgent: Risk assessment and validation (CRITICAL priority)

Future Agents (to be implemented):
- MonitoringAgent: System health and proactive monitoring
- ExecutionAgent: Intelligent order execution
- PortfolioAgent: Portfolio optimization
- MarketAnalysisAgent: Market regime detection
- LearningAgent: Continuous improvement
"""

from .base_agent import BaseAgent, AgentStatus, AgentPriority, AgentMetrics, ResourceRequirements
from .resource_manager import ResourceManager, ResourceSnapshot, AgentActivationRecord
from .risk_agent import RiskManagementAgent
from .monitoring_agent import MonitoringAgent, SystemHealthMetrics, Alert
from .execution_agent import ExecutionAgent, ExecutionMetrics, ExecutionQuality
from .portfolio_agent import PortfolioAgent, PortfolioMetrics, PositionAllocation, RebalancingRecommendation
from .market_analysis_agent import MarketAnalysisAgent, MarketMetrics, MarketRegime, TrendStrength, RegimeChange
from .learning_agent import LearningAgent, LearningMetrics, TradingPattern, PerformanceInsight

__all__ = [
    'BaseAgent',
    'AgentStatus',
    'AgentPriority',
    'AgentMetrics',
    'ResourceRequirements',
    'ResourceManager',
    'ResourceSnapshot',
    'AgentActivationRecord',
    'RiskManagementAgent',
    'MonitoringAgent',
    'SystemHealthMetrics',
    'Alert',
    'ExecutionAgent',
    'ExecutionMetrics',
    'ExecutionQuality',
    'PortfolioAgent',
    'PortfolioMetrics',
    'PositionAllocation',
    'RebalancingRecommendation',
    'MarketAnalysisAgent',
    'MarketMetrics',
    'MarketRegime',
    'TrendStrength',
    'RegimeChange',
    'LearningAgent',
    'LearningMetrics',
    'TradingPattern',
    'PerformanceInsight',
]

__version__ = '1.0.0'
