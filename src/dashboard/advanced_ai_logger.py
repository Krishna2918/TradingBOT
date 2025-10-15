#!/usr/bin/env python3
"""
Advanced AI Activity Logger - Comprehensive logging system for AI trading decisions
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import time

@dataclass
class AIComponentStatus:
    """Status of an AI component"""
    name: str
    status: str  # 'active', 'idle', 'error', 'processing'
    last_activity: str
    performance_metrics: Dict[str, float]
    current_task: str

@dataclass
class TradingDecision:
    """Detailed trading decision"""
    timestamp: str
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD', 'PASS'
    confidence: float
    reasoning: str
    ai_models_used: List[str]
    market_conditions: Dict[str, Any]
    risk_assessment: Dict[str, float]
    position_size: float
    expected_return: float

@dataclass
class AIActivityLog:
    """Comprehensive AI activity log entry"""
    timestamp: str
    component: str
    activity_type: str  # 'decision', 'analysis', 'training', 'error'
    details: Dict[str, Any]
    performance_metrics: Dict[str, float]
    duration_ms: float

class AdvancedAILogger:
    """Advanced AI activity logger with real-time tracking"""
    
    def __init__(self, log_dir: str = "logs/ai_activity"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Component status tracking
        self.component_status: Dict[str, AIComponentStatus] = {}
        self.trading_decisions: List[TradingDecision] = []
        self.activity_logs: List[AIActivityLog] = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        # Initialize components
        self._initialize_components()
        
        # Start background logging
        self._start_background_logging()
    
    def _initialize_components(self):
        """Initialize AI component status tracking"""
        components = [
            'MasterOrchestrator', 'ModelSelector', 'PerformanceLearner',
            'MarketAnalyzer', 'CrossValidator', 'FeaturePipeline',
            'RiskManager', 'PositionManager', 'OrderExecutor',
            'DataPipeline', 'SystemMonitor'
        ]
        
        for component in components:
            self.component_status[component] = AIComponentStatus(
                name=component,
                status='idle',
                last_activity=datetime.now().isoformat(),
                performance_metrics={},
                current_task='Initialized'
            )
    
    def log_component_activity(self, component: str, activity_type: str, 
                             details: Dict[str, Any], duration_ms: float = 0):
        """Log AI component activity"""
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # Update component status
            if component in self.component_status:
                self.component_status[component].status = 'active'
                self.component_status[component].last_activity = timestamp
                self.component_status[component].current_task = details.get('task', 'Processing')
            
            # Create activity log
            activity_log = AIActivityLog(
                timestamp=timestamp,
                component=component,
                activity_type=activity_type,
                details=details,
                performance_metrics=details.get('metrics', {}),
                duration_ms=duration_ms
            )
            
            self.activity_logs.append(activity_log)
            
            # Keep only last 1000 logs
            if len(self.activity_logs) > 1000:
                self.activity_logs = self.activity_logs[-1000:]
    
    def log_trading_decision(self, symbol: str, action: str, confidence: float,
                           reasoning: str, ai_models: List[str], 
                           market_conditions: Dict[str, Any],
                           risk_assessment: Dict[str, float],
                           position_size: float, expected_return: float):
        """Log detailed trading decision"""
        with self.lock:
            decision = TradingDecision(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                ai_models_used=ai_models,
                market_conditions=market_conditions,
                risk_assessment=risk_assessment,
                position_size=position_size,
                expected_return=expected_return
            )
            
            self.trading_decisions.append(decision)
            
            # Keep only last 500 decisions
            if len(self.trading_decisions) > 500:
                self.trading_decisions = self.trading_decisions[-500:]
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log system performance metrics"""
        with self.lock:
            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
            self.performance_history.append(performance_entry)
            
            # Keep only last 1000 entries
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
    
    def get_recent_activity(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent AI activity logs"""
        with self.lock:
            return [asdict(log) for log in self.activity_logs[-limit:]]
    
    def get_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trading decisions"""
        with self.lock:
            return [asdict(decision) for decision in self.trading_decisions[-limit:]]
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all AI components"""
        with self.lock:
            return {name: asdict(status) for name, status in self.component_status.items()}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self.lock:
            if not self.performance_history:
                return {}
            
            recent_metrics = self.performance_history[-10:]  # Last 10 entries
            
            summary = {
                'total_decisions': len(self.trading_decisions),
                'recent_activity_count': len(self.activity_logs),
                'component_count': len(self.component_status),
                'avg_decision_confidence': 0,
                'success_rate': 0,
                'active_components': 0
            }
            
            # Calculate average confidence
            if self.trading_decisions:
                confidences = [d.confidence for d in self.trading_decisions[-50:]]
                summary['avg_decision_confidence'] = sum(confidences) / len(confidences)
            
            # Calculate success rate (simplified)
            if self.trading_decisions:
                profitable_decisions = sum(1 for d in self.trading_decisions 
                                         if d.expected_return > 0)
                summary['success_rate'] = profitable_decisions / len(self.trading_decisions)
            
            # Count active components
            summary['active_components'] = sum(1 for status in self.component_status.values() 
                                             if status.status == 'active')
            
            return summary
    
    def _start_background_logging(self):
        """Start background logging thread"""
        def background_logger():
            while True:
                try:
                    self._save_logs_to_file()
                    time.sleep(30)  # Save every 30 seconds
                except Exception as e:
                    logging.error(f"Background logging error: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=background_logger, daemon=True)
        thread.start()
    
    def _save_logs_to_file(self):
        """Save logs to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save activity logs
            activity_file = self.log_dir / f"activity_{timestamp}.json"
            with open(activity_file, 'w') as f:
                json.dump([asdict(log) for log in self.activity_logs[-100:]], f, indent=2)
            
            # Save trading decisions
            decisions_file = self.log_dir / f"decisions_{timestamp}.json"
            with open(decisions_file, 'w') as f:
                json.dump([asdict(decision) for decision in self.trading_decisions[-50:]], f, indent=2)
            
            # Save component status
            status_file = self.log_dir / f"status_{timestamp}.json"
            with open(status_file, 'w') as f:
                json.dump(self.get_component_status(), f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving logs: {e}")
    
    def clear_logs(self):
        """Clear all logs for fresh session"""
        with self.lock:
            self.activity_logs.clear()
            self.trading_decisions.clear()
            self.performance_history.clear()
            
            # Reset component status
            for component in self.component_status.values():
                component.status = 'idle'
                component.current_task = 'Ready'
                component.performance_metrics.clear()

# Global logger instance
advanced_ai_logger = AdvancedAILogger()
