"""
Learning Agent

OPTIONAL priority agent that continuously learns from trading performance,
market conditions, and system behavior to improve decision-making over time.

Responsibilities:
- Performance analysis and learning
- Model adaptation and optimization
- Pattern recognition and memory
- Strategy evolution
- Knowledge base management
- Predictive model improvement
"""

import logging
import asyncio
import numpy as np
import json
import pickle
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .base_agent import BaseAgent, AgentPriority, AgentStatus, ResourceRequirements

logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """Learning performance metrics"""
    timestamp: datetime
    learning_episodes: int
    accuracy_improvement: float
    prediction_accuracy: float
    model_performance: float
    adaptation_rate: float
    knowledge_base_size: int
    patterns_learned: int


@dataclass
class TradingPattern:
    """Trading pattern learned by the agent"""
    pattern_id: str
    pattern_type: str  # 'bullish', 'bearish', 'reversal', 'continuation'
    market_conditions: Dict[str, Any]
    success_rate: float
    confidence: float
    frequency: int
    last_seen: datetime
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PerformanceInsight:
    """Performance insight derived from learning"""
    insight_id: str
    category: str  # 'timing', 'sizing', 'selection', 'risk'
    description: str
    confidence: float
    impact_score: float
    evidence: List[Dict[str, Any]]
    created_at: datetime


class LearningAgent(BaseAgent):
    """
    Learning Agent - OPTIONAL priority
    
    Continuously learns and adapts to improve trading performance.
    """
    
    def __init__(self):
        super().__init__(
            agent_id='learning_agent',
            name='Learning Agent',
            priority=AgentPriority.OPTIONAL,  # Optional - runs when resources available
            resource_requirements=ResourceRequirements(
                min_cpu_percent=15.0,
                min_memory_mb=300.0,
                max_cpu_percent=40.0,
                max_memory_mb=800.0
            )
        )
        
        # Learning state
        self.learning_active = False
        self.knowledge_base: Dict[str, Any] = {}
        self.trading_patterns: Dict[str, TradingPattern] = {}
        self.performance_insights: List[PerformanceInsight] = []
        self.learning_history: List[LearningMetrics] = []
        
        # Learning parameters
        self.min_pattern_confidence = 0.7
        self.min_success_rate = 0.6
        self.learning_rate = 0.01
        self.memory_size = 10000
        self.adaptation_threshold = 0.05
        
        # Performance tracking
        self.learning_episodes = 0
        self.patterns_discovered = 0
        self.insights_generated = 0
        self.model_improvements = 0
        self.accuracy_baseline = 0.5
        
        # Learning data storage
        self.trading_examples: deque = deque(maxlen=self.memory_size)
        self.market_conditions_history: deque = deque(maxlen=self.memory_size)
        self.performance_history: deque = deque(maxlen=self.memory_size)
        
        # Learning algorithms state
        self.pattern_recognition_model = None
        self.performance_prediction_model = None
        self.adaptation_model = None
    
    async def initialize(self) -> bool:
        """Initialize learning systems"""
        try:
            self.status = AgentStatus.IDLE
            self.learning_active = True
            
            # Load existing knowledge base
            await self._load_knowledge_base()
            
            # Start background learning
            asyncio.create_task(self._background_learning())
            
            logger.info("Learning Agent initialized and background learning started")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Learning Agent: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown learning systems"""
        logger.info("Learning Agent shutting down")
        
        # Save knowledge base
        await self._save_knowledge_base()
        
        self.learning_active = False
        self.status = AgentStatus.STOPPED
        return True
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process learning tasks.
        
        Task types:
        - 'learn_from_trade': Learn from a completed trade
        - 'analyze_performance': Analyze trading performance
        - 'discover_patterns': Discover new trading patterns
        - 'generate_insights': Generate performance insights
        - 'adapt_models': Adapt prediction models
        - 'get_learning_progress': Get learning progress metrics
        - 'predict_performance': Predict future performance
        - 'get_knowledge_summary': Get knowledge base summary
        """
        task_type = task.get('type')
        
        if task_type == 'learn_from_trade':
            return await self._learn_from_trade(task)
        elif task_type == 'analyze_performance':
            return await self._analyze_performance(task.get('timeframe', '1M'))
        elif task_type == 'discover_patterns':
            return await self._discover_patterns()
        elif task_type == 'generate_insights':
            return await self._generate_insights()
        elif task_type == 'adapt_models':
            return await self._adapt_models()
        elif task_type == 'get_learning_progress':
            return await self._get_learning_progress()
        elif task_type == 'predict_performance':
            return await self._predict_performance(task.get('horizon', '1W'))
        elif task_type == 'get_knowledge_summary':
            return await self._get_knowledge_summary()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _background_learning(self):
        """Background learning loop"""
        while self.learning_active:
            try:
                # Perform learning tasks every 10 minutes
                await self._discover_patterns()
                await self._generate_insights()
                await self._adapt_models()
                await asyncio.sleep(600)  # 10 minutes
            except Exception as e:
                logger.error(f"Background learning error: {e}")
                await asyncio.sleep(1200)  # Wait longer on error
    
    async def _learn_from_trade(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from a completed trade"""
        try:
            trade_data = task.get('trade_data', {})
            
            # Extract trade information
            symbol = trade_data.get('symbol')
            action = trade_data.get('action')  # 'buy' or 'sell'
            quantity = trade_data.get('quantity')
            entry_price = trade_data.get('entry_price')
            exit_price = trade_data.get('exit_price')
            pnl = trade_data.get('pnl', 0)
            market_conditions = trade_data.get('market_conditions', {})
            decision_factors = trade_data.get('decision_factors', {})
            
            # Create trading example
            trading_example = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'market_conditions': market_conditions,
                'decision_factors': decision_factors,
                'success': pnl > 0,
                'return_percent': (pnl / (entry_price * quantity)) * 100 if entry_price and quantity else 0
            }
            
            # Store in memory
            self.trading_examples.append(trading_example)
            self.market_conditions_history.append(market_conditions)
            self.performance_history.append(pnl)
            
            # Update learning metrics
            self.learning_episodes += 1
            
            # Check for pattern recognition
            pattern_match = self._recognize_pattern(trading_example)
            
            # Update knowledge base
            self._update_knowledge_base(trading_example)
            
            return {
                'success': True,
                'learning_episode': self.learning_episodes,
                'pattern_recognized': pattern_match is not None,
                'pattern_id': pattern_match.pattern_id if pattern_match else None,
                'knowledge_updated': True,
                'memory_size': len(self.trading_examples)
            }
            
        except Exception as e:
            logger.error(f"Learning from trade failed: {e}")
            return {'error': str(e)}
    
    def _recognize_pattern(self, trading_example: Dict[str, Any]) -> Optional[TradingPattern]:
        """Recognize if trading example matches existing patterns"""
        for pattern in self.trading_patterns.values():
            if self._matches_pattern(trading_example, pattern):
                # Update pattern with new example
                pattern.examples.append(trading_example)
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
                
                # Recalculate success rate
                successful_examples = [ex for ex in pattern.examples if ex['success']]
                pattern.success_rate = len(successful_examples) / len(pattern.examples)
                
                return pattern
        
        return None
    
    def _matches_pattern(self, trading_example: Dict[str, Any], pattern: TradingPattern) -> bool:
        """Check if trading example matches a pattern"""
        # Simple pattern matching (in real implementation, would use ML)
        market_conditions = trading_example.get('market_conditions', {})
        pattern_conditions = pattern.market_conditions
        
        # Check if market conditions match
        matches = 0
        total_checks = 0
        
        for key, value in pattern_conditions.items():
            if key in market_conditions:
                total_checks += 1
                if abs(market_conditions[key] - value) < 0.1:  # 10% tolerance
                    matches += 1
        
        # Pattern matches if 70% of conditions match
        return matches / total_checks >= 0.7 if total_checks > 0 else False
    
    def _update_knowledge_base(self, trading_example: Dict[str, Any]):
        """Update knowledge base with new trading example"""
        symbol = trading_example.get('symbol')
        action = trading_example.get('action')
        
        if symbol not in self.knowledge_base:
            self.knowledge_base[symbol] = {
                'total_trades': 0,
                'successful_trades': 0,
                'total_pnl': 0,
                'avg_return': 0,
                'patterns': []
            }
        
        symbol_data = self.knowledge_base[symbol]
        symbol_data['total_trades'] += 1
        
        if trading_example['success']:
            symbol_data['successful_trades'] += 1
        
        symbol_data['total_pnl'] += trading_example['pnl']
        symbol_data['avg_return'] = symbol_data['total_pnl'] / symbol_data['total_trades']
    
    async def _analyze_performance(self, timeframe: str) -> Dict[str, Any]:
        """Analyze trading performance over timeframe"""
        try:
            if not self.trading_examples:
                return {'error': 'No trading data available'}
            
            # Filter examples by timeframe
            cutoff_time = datetime.now() - timedelta(days=30 if timeframe == '1M' else 7 if timeframe == '1W' else 1)
            recent_examples = [ex for ex in self.trading_examples if ex['timestamp'] > cutoff_time]
            
            if not recent_examples:
                return {'error': f'No trading data for {timeframe}'}
            
            # Calculate performance metrics
            total_trades = len(recent_examples)
            successful_trades = len([ex for ex in recent_examples if ex['success']])
            success_rate = successful_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(ex['pnl'] for ex in recent_examples)
            avg_return = np.mean([ex['return_percent'] for ex in recent_examples])
            return_std = np.std([ex['return_percent'] for ex in recent_examples])
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = avg_return / return_std if return_std > 0 else 0
            
            # Performance by symbol
            symbol_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'success_rate': 0})
            for ex in recent_examples:
                symbol = ex['symbol']
                symbol_performance[symbol]['trades'] += 1
                symbol_performance[symbol]['pnl'] += ex['pnl']
                if ex['success']:
                    symbol_performance[symbol]['success_rate'] += 1
            
            # Calculate success rates
            for symbol, data in symbol_performance.items():
                data['success_rate'] = data['success_rate'] / data['trades'] if data['trades'] > 0 else 0
            
            return {
                'timeframe': timeframe,
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'success_rate': success_rate,
                'total_pnl': total_pnl,
                'avg_return_percent': avg_return,
                'return_volatility': return_std,
                'sharpe_ratio': sharpe_ratio,
                'symbol_performance': dict(symbol_performance),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {'error': str(e)}
    
    async def _discover_patterns(self) -> Dict[str, Any]:
        """Discover new trading patterns"""
        try:
            if len(self.trading_examples) < 10:
                return {'error': 'Insufficient trading data for pattern discovery'}
            
            new_patterns = []
            
            # Group examples by market conditions
            condition_groups = defaultdict(list)
            for example in self.trading_examples:
                conditions = example.get('market_conditions', {})
                # Create a simplified condition key
                condition_key = f"{conditions.get('volatility', 0):.2f}_{conditions.get('trend', 'neutral')}"
                condition_groups[condition_key].append(example)
            
            # Analyze each group for patterns
            for condition_key, examples in condition_groups.items():
                if len(examples) >= 5:  # Minimum examples for pattern
                    success_rate = len([ex for ex in examples if ex['success']]) / len(examples)
                    
                    if success_rate >= self.min_success_rate:
                        # Create new pattern
                        pattern_id = f"pattern_{len(self.trading_patterns) + 1}"
                        
                        # Calculate average market conditions
                        avg_conditions = {}
                        for key in examples[0]['market_conditions'].keys():
                            values = [ex['market_conditions'][key] for ex in examples if key in ex['market_conditions']]
                            if values:
                                avg_conditions[key] = np.mean(values)
                        
                        pattern = TradingPattern(
                            pattern_id=pattern_id,
                            pattern_type=self._classify_pattern_type(examples),
                            market_conditions=avg_conditions,
                            success_rate=success_rate,
                            confidence=min(0.95, success_rate + 0.1),
                            frequency=len(examples),
                            last_seen=datetime.now(),
                            examples=examples[-10:]  # Keep last 10 examples
                        )
                        
                        self.trading_patterns[pattern_id] = pattern
                        new_patterns.append(pattern)
                        self.patterns_discovered += 1
            
            return {
                'new_patterns_discovered': len(new_patterns),
                'total_patterns': len(self.trading_patterns),
                'patterns': [
                    {
                        'pattern_id': pattern.pattern_id,
                        'pattern_type': pattern.pattern_type,
                        'success_rate': pattern.success_rate,
                        'confidence': pattern.confidence,
                        'frequency': pattern.frequency,
                        'market_conditions': pattern.market_conditions
                    }
                    for pattern in new_patterns
                ],
                'discovery_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pattern discovery failed: {e}")
            return {'error': str(e)}
    
    def _classify_pattern_type(self, examples: List[Dict[str, Any]]) -> str:
        """Classify pattern type based on examples"""
        # Simple classification based on returns
        returns = [ex['return_percent'] for ex in examples]
        avg_return = np.mean(returns)
        
        if avg_return > 2:
            return 'bullish'
        elif avg_return < -2:
            return 'bearish'
        elif abs(avg_return) < 1:
            return 'neutral'
        else:
            return 'mixed'
    
    async def _generate_insights(self) -> Dict[str, Any]:
        """Generate performance insights"""
        try:
            if not self.trading_examples:
                return {'error': 'No trading data available'}
            
            new_insights = []
            
            # Analyze timing patterns
            timing_insight = self._analyze_timing_patterns()
            if timing_insight:
                new_insights.append(timing_insight)
            
            # Analyze sizing patterns
            sizing_insight = self._analyze_sizing_patterns()
            if sizing_insight:
                new_insights.append(sizing_insight)
            
            # Analyze selection patterns
            selection_insight = self._analyze_selection_patterns()
            if selection_insight:
                new_insights.append(selection_insight)
            
            # Store insights
            for insight in new_insights:
                self.performance_insights.append(insight)
                self.insights_generated += 1
            
            return {
                'new_insights_generated': len(new_insights),
                'total_insights': len(self.performance_insights),
                'insights': [
                    {
                        'insight_id': insight.insight_id,
                        'category': insight.category,
                        'description': insight.description,
                        'confidence': insight.confidence,
                        'impact_score': insight.impact_score
                    }
                    for insight in new_insights
                ],
                'generation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_timing_patterns(self) -> Optional[PerformanceInsight]:
        """Analyze timing patterns in trades"""
        if len(self.trading_examples) < 20:
            return None
        
        # Analyze performance by time of day
        hourly_performance = defaultdict(list)
        for example in self.trading_examples:
            hour = example['timestamp'].hour
            hourly_performance[hour].append(example['return_percent'])
        
        # Find best performing hour
        best_hour = None
        best_avg_return = -float('inf')
        
        for hour, returns in hourly_performance.items():
            if len(returns) >= 3:  # Minimum trades
                avg_return = np.mean(returns)
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    best_hour = hour
        
        if best_hour is not None and best_avg_return > 1.0:  # 1% threshold
            return PerformanceInsight(
                insight_id=f"timing_insight_{len(self.performance_insights) + 1}",
                category='timing',
                description=f"Best trading performance occurs at {best_hour}:00 with {best_avg_return:.2f}% average return",
                confidence=0.8,
                impact_score=0.7,
                evidence=[{'hour': best_hour, 'avg_return': best_avg_return, 'trades': len(hourly_performance[best_hour])}],
                created_at=datetime.now()
            )
        
        return None
    
    def _analyze_sizing_patterns(self) -> Optional[PerformanceInsight]:
        """Analyze position sizing patterns"""
        if len(self.trading_examples) < 20:
            return None
        
        # Analyze performance by position size
        size_groups = {'small': [], 'medium': [], 'large': []}
        
        for example in self.trading_examples:
            quantity = example.get('quantity', 0)
            if quantity < 50:
                size_groups['small'].append(example['return_percent'])
            elif quantity < 200:
                size_groups['medium'].append(example['return_percent'])
            else:
                size_groups['large'].append(example['return_percent'])
        
        # Find best performing size group
        best_size = None
        best_avg_return = -float('inf')
        
        for size, returns in size_groups.items():
            if len(returns) >= 5:  # Minimum trades
                avg_return = np.mean(returns)
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    best_size = size
        
        if best_size is not None and best_avg_return > 0.5:  # 0.5% threshold
            return PerformanceInsight(
                insight_id=f"sizing_insight_{len(self.performance_insights) + 1}",
                category='sizing',
                description=f"Best performance with {best_size} position sizes: {best_avg_return:.2f}% average return",
                confidence=0.7,
                impact_score=0.6,
                evidence=[{'size': best_size, 'avg_return': best_avg_return, 'trades': len(size_groups[best_size])}],
                created_at=datetime.now()
            )
        
        return None
    
    def _analyze_selection_patterns(self) -> Optional[PerformanceInsight]:
        """Analyze symbol selection patterns"""
        if len(self.trading_examples) < 20:
            return None
        
        # Analyze performance by symbol
        symbol_performance = defaultdict(list)
        for example in self.trading_examples:
            symbol = example.get('symbol', 'unknown')
            symbol_performance[symbol].append(example['return_percent'])
        
        # Find best performing symbol
        best_symbol = None
        best_avg_return = -float('inf')
        
        for symbol, returns in symbol_performance.items():
            if len(returns) >= 3:  # Minimum trades
                avg_return = np.mean(returns)
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    best_symbol = symbol
        
        if best_symbol is not None and best_avg_return > 1.0:  # 1% threshold
            return PerformanceInsight(
                insight_id=f"selection_insight_{len(self.performance_insights) + 1}",
                category='selection',
                description=f"Best performance with {best_symbol}: {best_avg_return:.2f}% average return",
                confidence=0.8,
                impact_score=0.8,
                evidence=[{'symbol': best_symbol, 'avg_return': best_avg_return, 'trades': len(symbol_performance[best_symbol])}],
                created_at=datetime.now()
            )
        
        return None
    
    async def _adapt_models(self) -> Dict[str, Any]:
        """Adapt prediction models based on recent performance"""
        try:
            if len(self.trading_examples) < 50:
                return {'error': 'Insufficient data for model adaptation'}
            
            # Calculate recent performance
            recent_examples = list(self.trading_examples)[-20:]  # Last 20 trades
            recent_accuracy = len([ex for ex in recent_examples if ex['success']]) / len(recent_examples)
            
            # Check if adaptation is needed
            accuracy_change = recent_accuracy - self.accuracy_baseline
            
            if abs(accuracy_change) > self.adaptation_threshold:
                # Perform model adaptation (placeholder)
                adaptation_performed = True
                self.model_improvements += 1
                
                # Update baseline
                self.accuracy_baseline = recent_accuracy
                
                return {
                    'adaptation_performed': adaptation_performed,
                    'accuracy_change': accuracy_change,
                    'new_accuracy': recent_accuracy,
                    'model_improvements': self.model_improvements,
                    'adaptation_timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'adaptation_performed': False,
                    'accuracy_change': accuracy_change,
                    'reason': 'Change below adaptation threshold',
                    'adaptation_timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Model adaptation failed: {e}")
            return {'error': str(e)}
    
    async def _get_learning_progress(self) -> Dict[str, Any]:
        """Get learning progress metrics"""
        if not self.learning_history:
            return {'error': 'No learning history available'}
        
        latest_metrics = self.learning_history[-1]
        
        return {
            'learning_episodes': self.learning_episodes,
            'patterns_discovered': self.patterns_discovered,
            'insights_generated': self.insights_generated,
            'model_improvements': self.model_improvements,
            'current_accuracy': self.accuracy_baseline,
            'knowledge_base_size': len(self.knowledge_base),
            'trading_patterns': len(self.trading_patterns),
            'performance_insights': len(self.performance_insights),
            'memory_usage': len(self.trading_examples),
            'learning_active': self.learning_active,
            'progress_timestamp': datetime.now().isoformat()
        }
    
    async def _predict_performance(self, horizon: str) -> Dict[str, Any]:
        """Predict future performance based on learning"""
        try:
            if not self.trading_examples:
                return {'error': 'No trading data available'}
            
            # Simple prediction based on recent trends
            recent_examples = list(self.trading_examples)[-10:]  # Last 10 trades
            recent_accuracy = len([ex for ex in recent_examples if ex['success']]) / len(recent_examples)
            recent_avg_return = np.mean([ex['return_percent'] for ex in recent_examples])
            
            # Predict based on patterns
            pattern_confidence = 0.0
            if self.trading_patterns:
                avg_pattern_success = np.mean([p.success_rate for p in self.trading_patterns.values()])
                pattern_confidence = avg_pattern_success
            
            # Combine recent performance and pattern confidence
            predicted_accuracy = (recent_accuracy + pattern_confidence) / 2
            predicted_return = recent_avg_return * 0.8  # Conservative estimate
            
            return {
                'horizon': horizon,
                'predicted_accuracy': predicted_accuracy,
                'predicted_avg_return': predicted_return,
                'confidence': min(0.9, predicted_accuracy),
                'factors': {
                    'recent_accuracy': recent_accuracy,
                    'pattern_confidence': pattern_confidence,
                    'recent_avg_return': recent_avg_return
                },
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance prediction failed: {e}")
            return {'error': str(e)}
    
    async def _get_knowledge_summary(self) -> Dict[str, Any]:
        """Get knowledge base summary"""
        return {
            'knowledge_base_size': len(self.knowledge_base),
            'trading_patterns': len(self.trading_patterns),
            'performance_insights': len(self.performance_insights),
            'learning_episodes': self.learning_episodes,
            'memory_usage': len(self.trading_examples),
            'top_symbols': sorted(
                [(symbol, data['total_trades']) for symbol, data in self.knowledge_base.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'top_patterns': [
                {
                    'pattern_id': pattern.pattern_id,
                    'type': pattern.pattern_type,
                    'success_rate': pattern.success_rate,
                    'frequency': pattern.frequency
                }
                for pattern in sorted(
                    self.trading_patterns.values(),
                    key=lambda p: p.success_rate,
                    reverse=True
                )[:5]
            ],
            'recent_insights': [
                {
                    'category': insight.category,
                    'description': insight.description,
                    'confidence': insight.confidence
                }
                for insight in sorted(
                    self.performance_insights,
                    key=lambda i: i.created_at,
                    reverse=True
                )[:5]
            ],
            'summary_timestamp': datetime.now().isoformat()
        }
    
    async def _load_knowledge_base(self):
        """Load knowledge base from storage"""
        try:
            # In real implementation, would load from persistent storage
            logger.info("Knowledge base loaded (placeholder)")
        except Exception as e:
            logger.warning(f"Failed to load knowledge base: {e}")
    
    async def _save_knowledge_base(self):
        """Save knowledge base to storage"""
        try:
            # In real implementation, would save to persistent storage
            logger.info("Knowledge base saved (placeholder)")
        except Exception as e:
            logger.warning(f"Failed to save knowledge base: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status with learning-specific metrics"""
        base_status = super().get_status()
        
        # Add learning-specific metrics
        base_status['learning_metrics'] = {
            'learning_active': self.learning_active,
            'learning_episodes': self.learning_episodes,
            'patterns_discovered': self.patterns_discovered,
            'insights_generated': self.insights_generated,
            'model_improvements': self.model_improvements,
            'accuracy_baseline': self.accuracy_baseline,
            'knowledge_base_size': len(self.knowledge_base),
            'trading_patterns': len(self.trading_patterns),
            'performance_insights': len(self.performance_insights),
            'memory_usage': len(self.trading_examples),
            'learning_rate': self.learning_rate,
            'min_pattern_confidence': self.min_pattern_confidence,
            'min_success_rate': self.min_success_rate,
            'adaptation_threshold': self.adaptation_threshold
        }
        
        return base_status
