"""
Adaptive Confidence Threshold System

This module implements a self-adjusting confidence threshold that learns from
trading performance and automatically adjusts to optimize returns.
"""

import logging
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import os

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Performance metrics for confidence adjustment."""
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    TOTAL_RETURN = "total_return"

@dataclass
class ConfidenceThreshold:
    """Dynamic confidence threshold configuration."""
    current_threshold: float
    min_threshold: float = 0.1
    max_threshold: float = 0.9
    adjustment_step: float = 0.05
    performance_window: int = 30  # days
    min_trades_for_adjustment: int = 10
    last_adjustment: datetime = None
    adjustment_history: List[Dict] = None
    
    def __post_init__(self):
        if self.adjustment_history is None:
            self.adjustment_history = []

@dataclass
class TradeResult:
    """Individual trade result for learning."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    exit_price: Optional[float] = None
    quantity: int = 0
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration_hours: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PerformanceStats:
    """Performance statistics for confidence adjustment."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_confidence: float
    period_start: datetime
    period_end: datetime

class AdaptiveConfidenceManager:
    """Manages dynamic confidence threshold adjustment."""
    
    def __init__(self, db_path: str = "data/trading_state.db"):
        """Initialize adaptive confidence manager."""
        self.db_path = db_path
        self.confidence_config = ConfidenceThreshold(
            current_threshold=0.3,
            min_threshold=0.1,
            max_threshold=0.9,
            adjustment_step=0.05
        )
        self._ensure_db_exists()
        self._create_schema()
        self._load_config()
    
    def _ensure_db_exists(self):
        """Ensure database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _create_schema(self):
        """Create database schema for adaptive confidence."""
        conn = sqlite3.connect(self.db_path)
        
        # Trade results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR,
                action VARCHAR,
                confidence DOUBLE,
                entry_price DOUBLE,
                exit_price DOUBLE,
                quantity INTEGER,
                pnl DOUBLE,
                pnl_pct DOUBLE,
                duration_hours DOUBLE,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Confidence adjustments table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS confidence_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_threshold DOUBLE,
                new_threshold DOUBLE,
                reason VARCHAR,
                performance_stats TEXT,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Performance metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start TIMESTAMP,
                period_end TIMESTAMP,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate DOUBLE,
                total_pnl DOUBLE,
                total_pnl_pct DOUBLE,
                profit_factor DOUBLE,
                sharpe_ratio DOUBLE,
                max_drawdown DOUBLE,
                avg_confidence DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Adaptive confidence database schema created")
    
    def _load_config(self):
        """Load confidence configuration from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get latest adjustment
            result = conn.execute("""
                SELECT new_threshold, timestamp 
                FROM confidence_adjustments 
                ORDER BY timestamp DESC 
                LIMIT 1
            """).fetchone()
            
            if result:
                self.confidence_config.current_threshold = result[0]
                self.confidence_config.last_adjustment = datetime.fromisoformat(result[1])
            
            conn.close()
            logger.info(f"Loaded confidence threshold: {self.confidence_config.current_threshold}")
            
        except Exception as e:
            logger.error(f"Error loading confidence config: {e}")
    
    def _save_config(self):
        """Save confidence configuration to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Save current threshold as adjustment
            conn.execute("""
                INSERT INTO confidence_adjustments 
                (old_threshold, new_threshold, reason, timestamp)
                VALUES (?, ?, ?, ?)
            """, [
                self.confidence_config.current_threshold,
                self.confidence_config.current_threshold,
                "System initialization",
                datetime.now()
            ])
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving confidence config: {e}")
    
    def record_trade_result(self, trade_result: TradeResult):
        """Record a trade result for learning."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute("""
                INSERT INTO trade_results 
                (symbol, action, confidence, entry_price, exit_price, quantity, 
                 pnl, pnl_pct, duration_hours, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                trade_result.symbol,
                trade_result.action,
                trade_result.confidence,
                trade_result.entry_price,
                trade_result.exit_price,
                trade_result.quantity,
                trade_result.pnl,
                trade_result.pnl_pct,
                trade_result.duration_hours,
                trade_result.timestamp
            ])
            
            conn.commit()
            conn.close()
            
            if trade_result.pnl_pct is not None:
                logger.info(f"Recorded trade result for {trade_result.symbol}: {trade_result.pnl_pct:.2%} P&L")
            else:
                logger.info(f"Recorded trade result for {trade_result.symbol}: P&L not calculated")
            
        except Exception as e:
            logger.error(f"Error recording trade result: {e}")
    
    def calculate_performance_stats(self, days_back: int = None) -> PerformanceStats:
        """Calculate performance statistics for the specified period."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Determine date range
            if days_back:
                start_date = datetime.now() - timedelta(days=days_back)
                where_clause = "WHERE timestamp >= ?"
                params = [start_date]
            else:
                where_clause = ""
                params = []
            
            # Get trade data
            if where_clause:
                query = f"""
                    SELECT confidence, pnl, pnl_pct, timestamp
                    FROM trade_results 
                    {where_clause}
                    AND pnl IS NOT NULL
                    ORDER BY timestamp
                """
            else:
                query = """
                    SELECT confidence, pnl, pnl_pct, timestamp
                    FROM trade_results 
                    WHERE pnl IS NOT NULL
                    ORDER BY timestamp
                """
            
            results = conn.execute(query, params).fetchall()
            conn.close()
            
            if not results:
                return PerformanceStats(
                    total_trades=0, winning_trades=0, losing_trades=0,
                    win_rate=0.0, total_pnl=0.0, total_pnl_pct=0.0,
                    profit_factor=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
                    avg_confidence=0.0,
                    period_start=datetime.now(), period_end=datetime.now()
                )
            
            # Calculate statistics
            confidences = [r[0] for r in results]
            pnls = [r[1] for r in results]
            pnl_pcts = [r[2] for r in results]
            timestamps = [datetime.fromisoformat(r[3]) for r in results]
            
            total_trades = len(results)
            winning_trades = sum(1 for pnl in pnls if pnl > 0)
            losing_trades = sum(1 for pnl in pnls if pnl < 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            total_pnl = sum(pnls)
            total_pnl_pct = sum(pnl_pcts)
            
            # Profit factor
            gross_profit = sum(pnl for pnl in pnls if pnl > 0)
            gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Sharpe ratio (simplified)
            if len(pnl_pcts) > 1:
                sharpe_ratio = np.mean(pnl_pcts) / np.std(pnl_pcts) if np.std(pnl_pcts) > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # Max drawdown
            cumulative_returns = np.cumsum(pnl_pcts)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            avg_confidence = np.mean(confidences)
            
            return PerformanceStats(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                avg_confidence=avg_confidence,
                period_start=min(timestamps),
                period_end=max(timestamps)
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return PerformanceStats(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, total_pnl=0.0, total_pnl_pct=0.0,
                profit_factor=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
                avg_confidence=0.0,
                period_start=datetime.now(), period_end=datetime.now()
            )
    
    def should_adjust_threshold(self) -> bool:
        """Determine if confidence threshold should be adjusted."""
        # Check if enough time has passed since last adjustment
        if self.confidence_config.last_adjustment:
            time_since_adjustment = datetime.now() - self.confidence_config.last_adjustment
            if time_since_adjustment < timedelta(hours=6):  # Minimum 6 hours between adjustments
                return False
        
        # Check if we have enough trade data
        stats = self.calculate_performance_stats(self.confidence_config.performance_window)
        if stats.total_trades < self.confidence_config.min_trades_for_adjustment:
            return False
        
        return True
    
    def calculate_optimal_threshold(self) -> Tuple[float, str]:
        """Calculate optimal confidence threshold based on performance."""
        stats = self.calculate_performance_stats(self.confidence_config.performance_window)
        
        # Performance-based adjustment logic
        adjustment_reasons = []
        new_threshold = self.confidence_config.current_threshold
        
        # Rule 1: If win rate is too low, increase threshold (be more selective)
        if stats.win_rate < 0.4:  # Less than 40% win rate
            new_threshold = min(
                self.confidence_config.max_threshold,
                new_threshold + self.confidence_config.adjustment_step
            )
            adjustment_reasons.append(f"Low win rate ({stats.win_rate:.1%})")
        
        # Rule 2: If win rate is high but profit factor is low, increase threshold
        elif stats.win_rate > 0.6 and stats.profit_factor < 1.2:
            new_threshold = min(
                self.confidence_config.max_threshold,
                new_threshold + self.confidence_config.adjustment_step
            )
            adjustment_reasons.append(f"High win rate but low profit factor ({stats.profit_factor:.2f})")
        
        # Rule 3: If performance is excellent, decrease threshold (be more aggressive)
        elif stats.win_rate > 0.7 and stats.profit_factor > 1.5 and stats.sharpe_ratio > 1.0:
            new_threshold = max(
                self.confidence_config.min_threshold,
                new_threshold - self.confidence_config.adjustment_step
            )
            adjustment_reasons.append(f"Excellent performance (WR: {stats.win_rate:.1%}, PF: {stats.profit_factor:.2f})")
        
        # Rule 4: If max drawdown is too high, increase threshold
        elif stats.max_drawdown > 0.15:  # More than 15% drawdown
            new_threshold = min(
                self.confidence_config.max_threshold,
                new_threshold + self.confidence_config.adjustment_step
            )
            adjustment_reasons.append(f"High drawdown ({stats.max_drawdown:.1%})")
        
        # Rule 5: If we're being too conservative (high threshold, low volume), decrease
        elif (new_threshold > 0.7 and stats.total_trades < 20 and 
              stats.win_rate > 0.5 and stats.profit_factor > 1.0):
            new_threshold = max(
                self.confidence_config.min_threshold,
                new_threshold - self.confidence_config.adjustment_step
            )
            adjustment_reasons.append("Too conservative - reducing threshold for more opportunities")
        
        # Ensure threshold stays within bounds
        new_threshold = max(
            self.confidence_config.min_threshold,
            min(self.confidence_config.max_threshold, new_threshold)
        )
        
        reason = "; ".join(adjustment_reasons) if adjustment_reasons else "No adjustment needed"
        return new_threshold, reason
    
    def adjust_confidence_threshold(self) -> bool:
        """Adjust confidence threshold based on performance."""
        if not self.should_adjust_threshold():
            return False
        
        old_threshold = self.confidence_config.current_threshold
        new_threshold, reason = self.calculate_optimal_threshold()
        
        # Only adjust if there's a meaningful change
        if abs(new_threshold - old_threshold) < 0.01:
            return False
        
        # Update configuration
        self.confidence_config.current_threshold = new_threshold
        self.confidence_config.last_adjustment = datetime.now()
        
        # Record adjustment
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Save performance stats
            stats = self.calculate_performance_stats(self.confidence_config.performance_window)
            conn.execute("""
                INSERT INTO performance_metrics 
                (period_start, period_end, total_trades, winning_trades, losing_trades,
                 win_rate, total_pnl, total_pnl_pct, profit_factor, sharpe_ratio, 
                 max_drawdown, avg_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                stats.period_start, stats.period_end, stats.total_trades,
                stats.winning_trades, stats.losing_trades, stats.win_rate,
                stats.total_pnl, stats.total_pnl_pct, stats.profit_factor,
                stats.sharpe_ratio, stats.max_drawdown, stats.avg_confidence
            ])
            
            # Save adjustment
            conn.execute("""
                INSERT INTO confidence_adjustments 
                (old_threshold, new_threshold, reason, performance_stats, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, [
                old_threshold, new_threshold, reason, 
                json.dumps(asdict(stats)), datetime.now()
            ])
            
            conn.commit()
            conn.close()
            
            # Update adjustment history
            self.confidence_config.adjustment_history.append({
                'timestamp': datetime.now(),
                'old_threshold': old_threshold,
                'new_threshold': new_threshold,
                'reason': reason
            })
            
            logger.info(f"Adjusted confidence threshold: {old_threshold:.3f} → {new_threshold:.3f} ({reason})")
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting confidence threshold: {e}")
            return False
    
    def get_current_threshold(self) -> float:
        """Get current confidence threshold."""
        return self.confidence_config.current_threshold
    
    def get_threshold_info(self) -> Dict:
        """Get detailed threshold information."""
        stats = self.calculate_performance_stats(self.confidence_config.performance_window)
        
        return {
            'current_threshold': self.confidence_config.current_threshold,
            'min_threshold': self.confidence_config.min_threshold,
            'max_threshold': self.confidence_config.max_threshold,
            'last_adjustment': self.confidence_config.last_adjustment,
            'adjustment_count': len(self.confidence_config.adjustment_history),
            'recent_performance': {
                'total_trades': stats.total_trades,
                'win_rate': stats.win_rate,
                'profit_factor': stats.profit_factor,
                'sharpe_ratio': stats.sharpe_ratio,
                'max_drawdown': stats.max_drawdown,
                'avg_confidence': stats.avg_confidence
            },
            'should_adjust': self.should_adjust_threshold()
        }
    
    def simulate_trade_result(self, symbol: str, action: str, confidence: float, 
                            entry_price: float, exit_price: float, quantity: int = 100):
        """Simulate a trade result for testing purposes."""
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = (exit_price - entry_price) / entry_price
        duration_hours = 24.0  # Simulate 24-hour hold
        
        trade_result = TradeResult(
            symbol=symbol,
            action=action,
            confidence=confidence,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_hours=duration_hours
        )
        
        self.record_trade_result(trade_result)
        return trade_result

# Global adaptive confidence manager
adaptive_confidence = AdaptiveConfidenceManager()

def get_confidence_threshold() -> float:
    """Get current dynamic confidence threshold."""
    return adaptive_confidence.get_current_threshold()

def adjust_confidence_threshold() -> bool:
    """Adjust confidence threshold based on performance."""
    return adaptive_confidence.adjust_confidence_threshold()

def record_trade_result(symbol: str, action: str, confidence: float, 
                       entry_price: float, exit_price: float = None, 
                       quantity: int = 100, pnl: float = None, pnl_pct: float = None):
    """Record a trade result for learning."""
    trade_result = TradeResult(
        symbol=symbol,
        action=action,
        confidence=confidence,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        pnl=pnl,
        pnl_pct=pnl_pct
    )
    
    adaptive_confidence.record_trade_result(trade_result)

def get_confidence_info() -> Dict:
    """Get detailed confidence threshold information."""
    return adaptive_confidence.get_threshold_info()
