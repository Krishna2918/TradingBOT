"""
Adaptive Configuration Management System

This module implements dynamic parameter adjustment based on trading performance,
market conditions, and learning outcomes.
"""

import logging
import json
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
from pathlib import Path

from src.config.mode_manager import get_current_mode, get_mode_config

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    """Types of configurable parameters."""
    POSITION_SIZE = "POSITION_SIZE"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    CONFIDENCE_THRESHOLD = "CONFIDENCE_THRESHOLD"
    RISK_TOLERANCE = "RISK_TOLERANCE"
    MARKET_SENTIMENT_WEIGHT = "MARKET_SENTIMENT_WEIGHT"
    TECHNICAL_ANALYSIS_WEIGHT = "TECHNICAL_ANALYSIS_WEIGHT"
    FUNDAMENTAL_ANALYSIS_WEIGHT = "FUNDAMENTAL_ANALYSIS_WEIGHT"

class AdjustmentTrigger(Enum):
    """Triggers for parameter adjustment."""
    PERFORMANCE_DECLINE = "PERFORMANCE_DECLINE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    MARKET_REGIME_CHANGE = "MARKET_REGIME_CHANGE"
    LEARNING_INSIGHT = "LEARNING_INSIGHT"
    RISK_THRESHOLD_BREACH = "RISK_THRESHOLD_BREACH"
    SCHEDULED_OPTIMIZATION = "SCHEDULED_OPTIMIZATION"

@dataclass
class ParameterConfig:
    """Configuration for a single parameter."""
    name: str
    parameter_type: ParameterType
    current_value: float
    min_value: float
    max_value: float
    default_value: float
    adjustment_step: float
    learning_rate: float
    last_updated: datetime
    performance_history: List[float]
    adjustment_count: int
    is_adaptive: bool

@dataclass
class AdjustmentRecord:
    """Record of a parameter adjustment."""
    parameter_name: str
    old_value: float
    new_value: float
    trigger: AdjustmentTrigger
    reason: str
    expected_impact: str
    timestamp: datetime
    performance_before: float
    performance_after: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for parameter evaluation."""
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    trade_count: int
    avg_trade_duration: float
    volatility: float
    timestamp: datetime

class AdaptiveConfigurationManager:
    """Manages adaptive configuration parameters."""
    
    def __init__(self, mode: str = None):
        self.mode = mode or get_current_mode()
        self.mode_config = get_mode_config()
        self.parameters: Dict[str, ParameterConfig] = {}
        self.adjustment_history: List[AdjustmentRecord] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.db_path = self._get_database_path()
        self._initialize_parameters()
        self._load_from_database()
        
        logger.info(f"Adaptive Configuration Manager initialized for {self.mode} mode")
    
    def _get_database_path(self) -> str:
        """Get database path based on mode."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        return str(data_dir / f"adaptive_config_{self.mode.lower()}.db")
    
    def _initialize_parameters(self):
        """Initialize default parameter configurations."""
        base_config = getattr(self.mode_config, "adaptive_config", {})
        
        self.parameters = {
            "position_size": ParameterConfig(
                name="position_size",
                parameter_type=ParameterType.POSITION_SIZE,
                current_value=base_config.get("position_size", 0.1) if isinstance(base_config, dict) else 0.1,
                min_value=0.01,
                max_value=0.25,
                default_value=0.1,
                adjustment_step=0.01,
                learning_rate=0.1,
                last_updated=datetime.now(),
                performance_history=[],
                adjustment_count=0,
                is_adaptive=True
            ),
            "stop_loss": ParameterConfig(
                name="stop_loss",
                parameter_type=ParameterType.STOP_LOSS,
                current_value=base_config.get("stop_loss", 0.05) if isinstance(base_config, dict) else 0.05,
                min_value=0.01,
                max_value=0.15,
                default_value=0.05,
                adjustment_step=0.005,
                learning_rate=0.15,
                last_updated=datetime.now(),
                performance_history=[],
                adjustment_count=0,
                is_adaptive=True
            ),
            "take_profit": ParameterConfig(
                name="take_profit",
                parameter_type=ParameterType.TAKE_PROFIT,
                current_value=base_config.get("take_profit", 0.10) if isinstance(base_config, dict) else 0.10,
                min_value=0.02,
                max_value=0.30,
                default_value=0.10,
                adjustment_step=0.01,
                learning_rate=0.12,
                last_updated=datetime.now(),
                performance_history=[],
                adjustment_count=0,
                is_adaptive=True
            ),
            "confidence_threshold": ParameterConfig(
                name="confidence_threshold",
                parameter_type=ParameterType.CONFIDENCE_THRESHOLD,
                current_value=base_config.get("confidence_threshold", 0.7) if isinstance(base_config, dict) else 0.7,
                min_value=0.5,
                max_value=0.95,
                default_value=0.7,
                adjustment_step=0.05,
                learning_rate=0.08,
                last_updated=datetime.now(),
                performance_history=[],
                adjustment_count=0,
                is_adaptive=True
            ),
            "risk_tolerance": ParameterConfig(
                name="risk_tolerance",
                parameter_type=ParameterType.RISK_TOLERANCE,
                current_value=base_config.get("risk_tolerance", 0.02) if isinstance(base_config, dict) else 0.02,
                min_value=0.005,
                max_value=0.05,
                default_value=0.02,
                adjustment_step=0.002,
                learning_rate=0.1,
                last_updated=datetime.now(),
                performance_history=[],
                adjustment_count=0,
                is_adaptive=True
            ),
            "market_sentiment_weight": ParameterConfig(
                name="market_sentiment_weight",
                parameter_type=ParameterType.MARKET_SENTIMENT_WEIGHT,
                current_value=base_config.get("market_sentiment_weight", 0.3) if isinstance(base_config, dict) else 0.3,
                min_value=0.1,
                max_value=0.6,
                default_value=0.3,
                adjustment_step=0.05,
                learning_rate=0.1,
                last_updated=datetime.now(),
                performance_history=[],
                adjustment_count=0,
                is_adaptive=True
            ),
            "technical_analysis_weight": ParameterConfig(
                name="technical_analysis_weight",
                parameter_type=ParameterType.TECHNICAL_ANALYSIS_WEIGHT,
                current_value=base_config.get("technical_analysis_weight", 0.4) if isinstance(base_config, dict) else 0.4,
                min_value=0.2,
                max_value=0.7,
                default_value=0.4,
                adjustment_step=0.05,
                learning_rate=0.1,
                last_updated=datetime.now(),
                performance_history=[],
                adjustment_count=0,
                is_adaptive=True
            ),
            "fundamental_analysis_weight": ParameterConfig(
                name="fundamental_analysis_weight",
                parameter_type=ParameterType.FUNDAMENTAL_ANALYSIS_WEIGHT,
                current_value=base_config.get("fundamental_analysis_weight", 0.3) if isinstance(base_config, dict) else 0.3,
                min_value=0.1,
                max_value=0.6,
                default_value=0.3,
                adjustment_step=0.05,
                learning_rate=0.1,
                last_updated=datetime.now(),
                performance_history=[],
                adjustment_count=0,
                is_adaptive=True
            )
        }
    
    def _load_from_database(self):
        """Load configuration from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS parameter_configs (
                        name TEXT PRIMARY KEY,
                        current_value REAL,
                        last_updated TEXT,
                        performance_history TEXT,
                        adjustment_count INTEGER
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS adjustment_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        parameter_name TEXT,
                        old_value REAL,
                        new_value REAL,
                        trigger TEXT,
                        reason TEXT,
                        expected_impact TEXT,
                        timestamp TEXT,
                        performance_before REAL,
                        performance_after REAL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        win_rate REAL,
                        profit_factor REAL,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        total_return REAL,
                        trade_count INTEGER,
                        avg_trade_duration REAL,
                        volatility REAL,
                        timestamp TEXT
                    )
                """)
                
                # Load parameter configs
                cursor = conn.execute("SELECT * FROM parameter_configs")
                for row in cursor.fetchall():
                    name, current_value, last_updated, perf_history, adj_count = row
                    if name in self.parameters:
                        self.parameters[name].current_value = current_value
                        self.parameters[name].last_updated = datetime.fromisoformat(last_updated)
                        self.parameters[name].performance_history = json.loads(perf_history) if perf_history else []
                        self.parameters[name].adjustment_count = adj_count
                
                # Load adjustment history
                cursor = conn.execute("SELECT * FROM adjustment_history ORDER BY timestamp DESC LIMIT 100")
                for row in cursor.fetchall():
                    _, param_name, old_val, new_val, trigger, reason, impact, timestamp, perf_before, perf_after = row
                    self.adjustment_history.append(AdjustmentRecord(
                        parameter_name=param_name,
                        old_value=old_val,
                        new_value=new_val,
                        trigger=AdjustmentTrigger(trigger),
                        reason=reason,
                        expected_impact=impact,
                        timestamp=datetime.fromisoformat(timestamp),
                        performance_before=perf_before,
                        performance_after=perf_after
                    ))
                
                # Load performance metrics
                cursor = conn.execute("SELECT * FROM performance_metrics ORDER BY timestamp DESC LIMIT 50")
                for row in cursor.fetchall():
                    _, win_rate, profit_factor, sharpe, drawdown, total_return, trade_count, avg_duration, volatility, timestamp = row
                    self.performance_history.append(PerformanceMetrics(
                        win_rate=win_rate,
                        profit_factor=profit_factor,
                        sharpe_ratio=sharpe,
                        max_drawdown=drawdown,
                        total_return=total_return,
                        trade_count=trade_count,
                        avg_trade_duration=avg_duration,
                        volatility=volatility,
                        timestamp=datetime.fromisoformat(timestamp)
                    ))
                
                logger.info(f"Loaded configuration from database: {len(self.parameters)} parameters, {len(self.adjustment_history)} adjustments")
                
        except Exception as e:
            logger.error(f"Error loading configuration from database: {e}")
    
    def _save_to_database(self):
        """Save configuration to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Save parameter configs
                for name, config in self.parameters.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO parameter_configs 
                        (name, current_value, last_updated, performance_history, adjustment_count)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        name,
                        config.current_value,
                        config.last_updated.isoformat(),
                        json.dumps(config.performance_history),
                        config.adjustment_count
                    ))
                
                # Save latest adjustment if any
                if self.adjustment_history:
                    latest = self.adjustment_history[-1]
                    conn.execute("""
                        INSERT INTO adjustment_history 
                        (parameter_name, old_value, new_value, trigger, reason, expected_impact, 
                         timestamp, performance_before, performance_after)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        latest.parameter_name,
                        latest.old_value,
                        latest.new_value,
                        latest.trigger.value,
                        latest.reason,
                        latest.expected_impact,
                        latest.timestamp.isoformat(),
                        latest.performance_before,
                        latest.performance_after
                    ))
                
                # Save latest performance metrics if any
                if self.performance_history:
                    latest = self.performance_history[-1]
                    conn.execute("""
                        INSERT INTO performance_metrics 
                        (win_rate, profit_factor, sharpe_ratio, max_drawdown, total_return, 
                         trade_count, avg_trade_duration, volatility, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        latest.win_rate,
                        latest.profit_factor,
                        latest.sharpe_ratio,
                        latest.max_drawdown,
                        latest.total_return,
                        latest.trade_count,
                        latest.avg_trade_duration,
                        latest.volatility,
                        latest.timestamp.isoformat()
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving configuration to database: {e}")
    
    def get_parameter(self, name: str) -> Optional[float]:
        """Get current value of a parameter."""
        if name in self.parameters:
            return self.parameters[name].current_value
        return None
    
    def get_all_parameters(self) -> Dict[str, float]:
        """Get all current parameter values."""
        return {name: config.current_value for name, config in self.parameters.items()}
    
    def update_performance_metrics(self, metrics: PerformanceMetrics):
        """Update performance metrics and trigger adjustments if needed."""
        self.performance_history.append(metrics)
        
        # Keep only last 50 metrics
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
        
        # Check for adjustment triggers
        self._check_adjustment_triggers(metrics)
        
        # Save to database
        self._save_to_database()
        
        logger.info(f"Updated performance metrics: Win Rate={metrics.win_rate:.3f}, "
                   f"Profit Factor={metrics.profit_factor:.3f}, Sharpe={metrics.sharpe_ratio:.3f}")
    
    def _check_adjustment_triggers(self, metrics: PerformanceMetrics):
        """Check if any parameters need adjustment based on performance."""
        if len(self.performance_history) < 5:
            return  # Need at least 5 data points for meaningful analysis
        
        # Check for performance decline
        recent_metrics = self.performance_history[-3:]
        older_metrics = self.performance_history[-6:-3] if len(self.performance_history) >= 6 else []
        
        if older_metrics:
            recent_avg_win_rate = np.mean([m.win_rate for m in recent_metrics])
            older_avg_win_rate = np.mean([m.win_rate for m in older_metrics])
            
            if recent_avg_win_rate < older_avg_win_rate - 0.1:  # 10% decline
                self._adjust_confidence_threshold(AdjustmentTrigger.PERFORMANCE_DECLINE, 
                                                "Win rate declined significantly")
        
        # Check for high volatility
        if metrics.volatility > 0.3:  # High volatility threshold
            self._adjust_risk_tolerance(AdjustmentTrigger.HIGH_VOLATILITY, 
                                      "High market volatility detected")
        
        # Check for drawdown
        if metrics.max_drawdown > 0.15:  # 15% drawdown threshold
            self._adjust_position_size(AdjustmentTrigger.RISK_THRESHOLD_BREACH, 
                                     "Maximum drawdown exceeded")
    
    def _adjust_confidence_threshold(self, trigger: AdjustmentTrigger, reason: str):
        """Adjust confidence threshold based on performance."""
        param = self.parameters["confidence_threshold"]
        old_value = param.current_value
        
        # Increase confidence threshold if performance is declining
        new_value = min(param.max_value, old_value + param.adjustment_step)
        
        if new_value != old_value:
            self._apply_adjustment(param, old_value, new_value, trigger, reason,
                                 "Higher confidence threshold should improve trade quality")
    
    def _adjust_risk_tolerance(self, trigger: AdjustmentTrigger, reason: str):
        """Adjust risk tolerance based on market conditions."""
        param = self.parameters["risk_tolerance"]
        old_value = param.current_value
        
        # Decrease risk tolerance in high volatility
        new_value = max(param.min_value, old_value - param.adjustment_step)
        
        if new_value != old_value:
            self._apply_adjustment(param, old_value, new_value, trigger, reason,
                                 "Lower risk tolerance should reduce volatility exposure")
    
    def _adjust_position_size(self, trigger: AdjustmentTrigger, reason: str):
        """Adjust position size based on risk metrics."""
        param = self.parameters["position_size"]
        old_value = param.current_value
        
        # Decrease position size if drawdown is high
        new_value = max(param.min_value, old_value - param.adjustment_step)
        
        if new_value != old_value:
            self._apply_adjustment(param, old_value, new_value, trigger, reason,
                                 "Smaller position size should reduce drawdown risk")
    
    def _apply_adjustment(self, param: ParameterConfig, old_value: float, new_value: float,
                         trigger: AdjustmentTrigger, reason: str, expected_impact: str):
        """Apply a parameter adjustment."""
        param.current_value = new_value
        param.last_updated = datetime.now()
        param.adjustment_count += 1
        
        # Record the adjustment
        adjustment = AdjustmentRecord(
            parameter_name=param.name,
            old_value=old_value,
            new_value=new_value,
            trigger=trigger,
            reason=reason,
            expected_impact=expected_impact,
            timestamp=datetime.now(),
            performance_before=self.performance_history[-1].win_rate if self.performance_history else 0.0
        )
        
        self.adjustment_history.append(adjustment)
        
        # Keep only last 100 adjustments
        if len(self.adjustment_history) > 100:
            self.adjustment_history = self.adjustment_history[-100:]
        
        logger.info(f"Adjusted {param.name}: {old_value:.4f} → {new_value:.4f} "
                   f"(Trigger: {trigger.value}, Reason: {reason})")
        
        # Save to database
        self._save_to_database()
    
    def get_adjustment_summary(self) -> Dict[str, Any]:
        """Get summary of recent adjustments."""
        recent_adjustments = [adj for adj in self.adjustment_history 
                            if adj.timestamp > datetime.now() - timedelta(days=7)]
        
        return {
            "total_adjustments": len(self.adjustment_history),
            "recent_adjustments": len(recent_adjustments),
            "parameters_modified": len(set(adj.parameter_name for adj in recent_adjustments)),
            "most_adjusted_parameter": max(self.parameters.items(), 
                                         key=lambda x: x[1].adjustment_count)[0] if self.parameters else None,
            "last_adjustment": self.adjustment_history[-1].timestamp.isoformat() if self.adjustment_history else None
        }
    
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        for param in self.parameters.values():
            param.current_value = param.default_value
            param.last_updated = datetime.now()
            param.adjustment_count = 0
            param.performance_history = []
        
        self.adjustment_history = []
        self.performance_history = []
        
        self._save_to_database()
        logger.info("Reset all parameters to default values")
    
    def get_performance_trend(self, days: int = 7) -> Dict[str, Any]:
        """Get performance trend over specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [m for m in self.performance_history if m.timestamp > cutoff_date]
        
        if len(recent_metrics) < 2:
            return {"trend": "insufficient_data", "metrics": []}
        
        # Calculate trends
        win_rates = [m.win_rate for m in recent_metrics]
        profit_factors = [m.profit_factor for m in recent_metrics]
        sharpe_ratios = [m.sharpe_ratio for m in recent_metrics]
        
        win_rate_trend = "improving" if win_rates[-1] > win_rates[0] else "declining"
        profit_factor_trend = "improving" if profit_factors[-1] > profit_factors[0] else "declining"
        sharpe_trend = "improving" if sharpe_ratios[-1] > sharpe_ratios[0] else "declining"
        
        return {
            "trend": "mixed" if len(set([win_rate_trend, profit_factor_trend, sharpe_trend])) > 1 else win_rate_trend,
            "win_rate_trend": win_rate_trend,
            "profit_factor_trend": profit_factor_trend,
            "sharpe_trend": sharpe_trend,
            "metrics_count": len(recent_metrics),
            "latest_win_rate": win_rates[-1],
            "latest_profit_factor": profit_factors[-1],
            "latest_sharpe": sharpe_ratios[-1]
        }

# Global instance
_config_manager = None

def get_adaptive_config_manager(mode: str = None) -> AdaptiveConfigurationManager:
    """Get the global adaptive configuration manager instance."""
    global _config_manager
    if _config_manager is None or (mode and _config_manager.mode != mode):
        _config_manager = AdaptiveConfigurationManager(mode)
    return _config_manager

def get_parameter(name: str, mode: str = None) -> Optional[float]:
    """Get a parameter value."""
    return get_adaptive_config_manager(mode).get_parameter(name)

def get_all_parameters(mode: str = None) -> Dict[str, float]:
    """Get all parameter values."""
    return get_adaptive_config_manager(mode).get_all_parameters()

def update_performance_metrics(metrics: PerformanceMetrics, mode: str = None):
    """Update performance metrics."""
    get_adaptive_config_manager(mode).update_performance_metrics(metrics)

def get_adjustment_summary(mode: str = None) -> Dict[str, Any]:
    """Get adjustment summary."""
    return get_adaptive_config_manager(mode).get_adjustment_summary()

def get_performance_trend(days: int = 7, mode: str = None) -> Dict[str, Any]:
    """Get performance trend."""
    return get_adaptive_config_manager(mode).get_performance_trend(days)
