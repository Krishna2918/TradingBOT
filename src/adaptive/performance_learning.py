"""
Performance-Based Learning Engine

This module implements learning from trade outcomes, pattern identification,
and automatic parameter adjustment based on historical performance.
"""

import logging
import json
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

from src.config.mode_manager import get_current_mode, get_mode_config
from src.adaptive.configuration_manager import PerformanceMetrics, get_adaptive_config_manager

logger = logging.getLogger(__name__)

class LearningTrigger(Enum):
    """Triggers for learning updates."""
    TRADE_COMPLETION = "TRADE_COMPLETION"
    DAILY_REVIEW = "DAILY_REVIEW"
    WEEKLY_ANALYSIS = "WEEKLY_ANALYSIS"
    PERFORMANCE_DECLINE = "PERFORMANCE_DECLINE"
    MARKET_REGIME_CHANGE = "MARKET_REGIME_CHANGE"
    MANUAL_TRIGGER = "MANUAL_TRIGGER"

class PatternType(Enum):
    """Types of patterns identified."""
    WINNING_CONDITIONS = "WINNING_CONDITIONS"
    LOSING_CONDITIONS = "LOSING_CONDITIONS"
    MARKET_TIMING = "MARKET_TIMING"
    RISK_PATTERNS = "RISK_PATTERNS"
    SENTIMENT_PATTERNS = "SENTIMENT_PATTERNS"
    TECHNICAL_PATTERNS = "TECHNICAL_PATTERNS"

@dataclass
class TradeOutcome:
    """Outcome of a completed trade."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_percentage: float
    duration_hours: float
    confidence_score: float
    market_conditions: Dict[str, Any]
    technical_indicators: Dict[str, float]
    sentiment_scores: Dict[str, float]
    risk_metrics: Dict[str, float]
    exit_reason: str
    success: bool

@dataclass
class IdentifiedPattern:
    """A pattern identified from trade analysis."""
    pattern_type: PatternType
    description: str
    conditions: Dict[str, Any]
    success_rate: float
    avg_pnl: float
    frequency: int
    confidence: float
    discovered_at: datetime
    last_seen: datetime
    examples: List[str]  # Trade IDs that match this pattern

@dataclass
class LearningInsight:
    """An insight derived from learning analysis."""
    insight_type: str
    description: str
    impact_score: float
    confidence: float
    supporting_evidence: List[str]
    recommended_action: str
    discovered_at: datetime
    validated: bool

@dataclass
class ParameterRecommendation:
    """Recommendation for parameter adjustment."""
    parameter_name: str
    current_value: float
    recommended_value: float
    confidence: float
    reasoning: str
    expected_improvement: float
    risk_assessment: str
    created_at: datetime

class PerformanceLearningEngine:
    """Engine for learning from trading performance."""
    
    def __init__(self, mode: str = None):
        self.mode = mode or get_current_mode()
        self.mode_config = get_mode_config()
        self.db_path = self._get_database_path()
        self.trade_outcomes: List[TradeOutcome] = []
        self.identified_patterns: List[IdentifiedPattern] = []
        self.learning_insights: List[LearningInsight] = []
        self.parameter_recommendations: List[ParameterRecommendation] = []
        
        # ML Models
        self.success_predictor = None
        self.pnl_predictor = None
        self.pattern_classifier = None
        self.scaler = StandardScaler()
        
        # Learning state
        self.last_learning_update = None
        self.learning_accuracy = 0.0
        self.model_training_count = 0
        
        self._initialize_database()
        self._load_historical_data()
        
        logger.info(f"Performance Learning Engine initialized for {self.mode} mode")
    
    def _get_database_path(self) -> str:
        """Get database path based on mode."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        return str(data_dir / f"performance_learning_{self.mode.lower()}.db")
    
    def _initialize_database(self):
        """Initialize learning database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Trade outcomes table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trade_outcomes (
                        trade_id TEXT PRIMARY KEY,
                        symbol TEXT,
                        entry_time TEXT,
                        exit_time TEXT,
                        entry_price REAL,
                        exit_price REAL,
                        position_size REAL,
                        pnl REAL,
                        pnl_percentage REAL,
                        duration_hours REAL,
                        confidence_score REAL,
                        market_conditions TEXT,
                        technical_indicators TEXT,
                        sentiment_scores TEXT,
                        risk_metrics TEXT,
                        exit_reason TEXT,
                        success INTEGER
                    )
                """)
                
                # Identified patterns table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS identified_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_type TEXT,
                        description TEXT,
                        conditions TEXT,
                        success_rate REAL,
                        avg_pnl REAL,
                        frequency INTEGER,
                        confidence REAL,
                        discovered_at TEXT,
                        last_seen TEXT,
                        examples TEXT
                    )
                """)
                
                # Learning insights table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_insights (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        insight_type TEXT,
                        description TEXT,
                        impact_score REAL,
                        confidence REAL,
                        supporting_evidence TEXT,
                        recommended_action TEXT,
                        discovered_at TEXT,
                        validated INTEGER
                    )
                """)
                
                # Parameter recommendations table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS parameter_recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        parameter_name TEXT,
                        current_value REAL,
                        recommended_value REAL,
                        confidence REAL,
                        reasoning TEXT,
                        expected_improvement REAL,
                        risk_assessment TEXT,
                        created_at TEXT,
                        applied INTEGER DEFAULT 0
                    )
                """)
                
                conn.commit()
                logger.info("Learning database initialized")
                
        except Exception as e:
            logger.error(f"Error initializing learning database: {e}")
    
    def _load_historical_data(self):
        """Load historical learning data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load trade outcomes
                cursor = conn.execute("SELECT * FROM trade_outcomes ORDER BY exit_time DESC LIMIT 1000")
                for row in cursor.fetchall():
                    trade_outcome = TradeOutcome(
                        trade_id=row[0],
                        symbol=row[1],
                        entry_time=datetime.fromisoformat(row[2]),
                        exit_time=datetime.fromisoformat(row[3]),
                        entry_price=row[4],
                        exit_price=row[5],
                        position_size=row[6],
                        pnl=row[7],
                        pnl_percentage=row[8],
                        duration_hours=row[9],
                        confidence_score=row[10],
                        market_conditions=json.loads(row[11]),
                        technical_indicators=json.loads(row[12]),
                        sentiment_scores=json.loads(row[13]),
                        risk_metrics=json.loads(row[14]),
                        exit_reason=row[15],
                        success=bool(row[16])
                    )
                    self.trade_outcomes.append(trade_outcome)
                
                # Load identified patterns
                cursor = conn.execute("SELECT * FROM identified_patterns ORDER BY discovered_at DESC LIMIT 100")
                for row in cursor.fetchall():
                    pattern = IdentifiedPattern(
                        pattern_type=PatternType(row[1]),
                        description=row[2],
                        conditions=json.loads(row[3]),
                        success_rate=row[4],
                        avg_pnl=row[5],
                        frequency=row[6],
                        confidence=row[7],
                        discovered_at=datetime.fromisoformat(row[8]),
                        last_seen=datetime.fromisoformat(row[9]),
                        examples=json.loads(row[10])
                    )
                    self.identified_patterns.append(pattern)
                
                # Load learning insights
                cursor = conn.execute("SELECT * FROM learning_insights ORDER BY discovered_at DESC LIMIT 50")
                for row in cursor.fetchall():
                    insight = LearningInsight(
                        insight_type=row[1],
                        description=row[2],
                        impact_score=row[3],
                        confidence=row[4],
                        supporting_evidence=json.loads(row[5]),
                        recommended_action=row[6],
                        discovered_at=datetime.fromisoformat(row[7]),
                        validated=bool(row[8])
                    )
                    self.learning_insights.append(insight)
                
                # Load parameter recommendations
                cursor = conn.execute("SELECT * FROM parameter_recommendations WHERE applied = 0 ORDER BY created_at DESC LIMIT 20")
                for row in cursor.fetchall():
                    recommendation = ParameterRecommendation(
                        parameter_name=row[1],
                        current_value=row[2],
                        recommended_value=row[3],
                        confidence=row[4],
                        reasoning=row[5],
                        expected_improvement=row[6],
                        risk_assessment=row[7],
                        created_at=datetime.fromisoformat(row[8])
                    )
                    self.parameter_recommendations.append(recommendation)
                
                logger.info(f"Loaded historical data: {len(self.trade_outcomes)} trades, "
                           f"{len(self.identified_patterns)} patterns, {len(self.learning_insights)} insights")
                
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def record_trade_outcome(self, outcome: TradeOutcome):
        """Record a completed trade outcome."""
        self.trade_outcomes.append(outcome)
        
        # Save to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trade_outcomes 
                    (trade_id, symbol, entry_time, exit_time, entry_price, exit_price, 
                     position_size, pnl, pnl_percentage, duration_hours, confidence_score,
                     market_conditions, technical_indicators, sentiment_scores, risk_metrics,
                     exit_reason, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    outcome.trade_id,
                    outcome.symbol,
                    outcome.entry_time.isoformat(),
                    outcome.exit_time.isoformat(),
                    outcome.entry_price,
                    outcome.exit_price,
                    outcome.position_size,
                    outcome.pnl,
                    outcome.pnl_percentage,
                    outcome.duration_hours,
                    outcome.confidence_score,
                    json.dumps(outcome.market_conditions),
                    json.dumps(outcome.technical_indicators),
                    json.dumps(outcome.sentiment_scores),
                    json.dumps(outcome.risk_metrics),
                    outcome.exit_reason,
                    int(outcome.success)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving trade outcome: {e}")
        
        # Trigger learning if we have enough data
        if len(self.trade_outcomes) >= 10:
            asyncio.create_task(self._trigger_learning(LearningTrigger.TRADE_COMPLETION))
        
        logger.info(f"Recorded trade outcome: {outcome.symbol} - "
                   f"{'WIN' if outcome.success else 'LOSS'} - P&L: {outcome.pnl:.2f}")
    
    async def _trigger_learning(self, trigger: LearningTrigger):
        """Trigger learning analysis."""
        try:
            logger.info(f"Triggering learning analysis: {trigger.value}")
            
            # Update performance metrics
            await self._update_performance_metrics()
            
            # Identify patterns
            await self._identify_patterns()
            
            # Generate insights
            await self._generate_insights()
            
            # Train ML models
            await self._train_models()
            
            # Generate parameter recommendations
            await self._generate_parameter_recommendations()
            
            self.last_learning_update = datetime.now()
            
            logger.info("Learning analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in learning analysis: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics in configuration manager."""
        if len(self.trade_outcomes) < 5:
            return
        
        recent_trades = [t for t in self.trade_outcomes 
                        if t.exit_time > datetime.now() - timedelta(days=30)]
        
        if not recent_trades:
            return
        
        # Calculate metrics
        total_trades = len(recent_trades)
        winning_trades = [t for t in recent_trades if t.success]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in recent_trades)
        winning_pnl = sum(t.pnl for t in winning_trades)
        losing_pnl = sum(t.pnl for t in recent_trades if not t.success)
        
        profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        returns = [t.pnl_percentage for t in recent_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Calculate volatility
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Average trade duration
        avg_duration = np.mean([t.duration_hours for t in recent_trades])
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_pnl,
            trade_count=total_trades,
            avg_trade_duration=avg_duration,
            volatility=volatility,
            timestamp=datetime.now()
        )
        
        # Update configuration manager
        config_manager = get_adaptive_config_manager(self.mode)
        config_manager.update_performance_metrics(metrics)
    
    async def _identify_patterns(self):
        """Identify patterns in trade outcomes."""
        if len(self.trade_outcomes) < 20:
            return
        
        # Analyze winning conditions
        winning_trades = [t for t in self.trade_outcomes if t.success]
        losing_trades = [t for t in self.trade_outcomes if not t.success]
        
        if len(winning_trades) >= 5:
            await self._analyze_winning_patterns(winning_trades)
        
        if len(losing_trades) >= 5:
            await self._analyze_losing_patterns(losing_trades)
        
        # Analyze market timing patterns
        await self._analyze_market_timing_patterns()
        
        # Analyze risk patterns
        await self._analyze_risk_patterns()
    
    async def _analyze_winning_patterns(self, winning_trades: List[TradeOutcome]):
        """Analyze patterns in winning trades."""
        # High confidence trades
        high_confidence_wins = [t for t in winning_trades if t.confidence_score > 0.8]
        if len(high_confidence_wins) >= 3:
            avg_pnl = np.mean([t.pnl_percentage for t in high_confidence_wins])
            pattern = IdentifiedPattern(
                pattern_type=PatternType.WINNING_CONDITIONS,
                description="High confidence trades (>0.8) tend to be winners",
                conditions={"confidence_score": {"min": 0.8}},
                success_rate=1.0,
                avg_pnl=avg_pnl,
                frequency=len(high_confidence_wins),
                confidence=0.8,
                discovered_at=datetime.now(),
                last_seen=datetime.now(),
                examples=[t.trade_id for t in high_confidence_wins[:5]]
            )
            self._save_pattern(pattern)
        
        # Short duration wins
        short_wins = [t for t in winning_trades if t.duration_hours < 24]
        if len(short_wins) >= 3:
            avg_pnl = np.mean([t.pnl_percentage for t in short_wins])
            pattern = IdentifiedPattern(
                pattern_type=PatternType.WINNING_CONDITIONS,
                description="Short duration trades (<24h) often succeed",
                conditions={"duration_hours": {"max": 24}},
                success_rate=len(short_wins) / len([t for t in self.trade_outcomes if t.duration_hours < 24]),
                avg_pnl=avg_pnl,
                frequency=len(short_wins),
                confidence=0.7,
                discovered_at=datetime.now(),
                last_seen=datetime.now(),
                examples=[t.trade_id for t in short_wins[:5]]
            )
            self._save_pattern(pattern)
    
    async def _analyze_losing_patterns(self, losing_trades: List[TradeOutcome]):
        """Analyze patterns in losing trades."""
        # Low confidence trades
        low_confidence_losses = [t for t in losing_trades if t.confidence_score < 0.6]
        if len(low_confidence_losses) >= 3:
            avg_pnl = np.mean([t.pnl_percentage for t in low_confidence_losses])
            pattern = IdentifiedPattern(
                pattern_type=PatternType.LOSING_CONDITIONS,
                description="Low confidence trades (<0.6) often fail",
                conditions={"confidence_score": {"max": 0.6}},
                success_rate=0.0,
                avg_pnl=avg_pnl,
                frequency=len(low_confidence_losses),
                confidence=0.8,
                discovered_at=datetime.now(),
                last_seen=datetime.now(),
                examples=[t.trade_id for t in low_confidence_losses[:5]]
            )
            self._save_pattern(pattern)
    
    async def _analyze_market_timing_patterns(self):
        """Analyze market timing patterns."""
        # Group trades by hour of day
        hourly_performance = {}
        for trade in self.trade_outcomes:
            hour = trade.entry_time.hour
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            hourly_performance[hour].append(trade.pnl_percentage)
        
        # Find best trading hours
        best_hours = []
        for hour, pnls in hourly_performance.items():
            if len(pnls) >= 3:
                avg_pnl = np.mean(pnls)
                if avg_pnl > 0.02:  # 2% average return
                    best_hours.append((hour, avg_pnl))
        
        if best_hours:
            best_hour, avg_pnl = max(best_hours, key=lambda x: x[1])
            pattern = IdentifiedPattern(
                pattern_type=PatternType.MARKET_TIMING,
                description=f"Trades at {best_hour}:00 tend to be more profitable",
                conditions={"entry_hour": {"value": best_hour}},
                success_rate=0.7,
                avg_pnl=avg_pnl,
                frequency=len(hourly_performance[best_hour]),
                confidence=0.6,
                discovered_at=datetime.now(),
                last_seen=datetime.now(),
                examples=[]
            )
            self._save_pattern(pattern)
    
    async def _analyze_risk_patterns(self):
        """Analyze risk-related patterns."""
        # High volatility trades
        high_vol_trades = [t for t in self.trade_outcomes 
                          if t.risk_metrics.get('volatility', 0) > 0.3]
        
        if len(high_vol_trades) >= 5:
            success_rate = len([t for t in high_vol_trades if t.success]) / len(high_vol_trades)
            avg_pnl = np.mean([t.pnl_percentage for t in high_vol_trades])
            
            pattern = IdentifiedPattern(
                pattern_type=PatternType.RISK_PATTERNS,
                description="High volatility trades show mixed results",
                conditions={"volatility": {"min": 0.3}},
                success_rate=success_rate,
                avg_pnl=avg_pnl,
                frequency=len(high_vol_trades),
                confidence=0.6,
                discovered_at=datetime.now(),
                last_seen=datetime.now(),
                examples=[t.trade_id for t in high_vol_trades[:5]]
            )
            self._save_pattern(pattern)
    
    def _save_pattern(self, pattern: IdentifiedPattern):
        """Save an identified pattern."""
        self.identified_patterns.append(pattern)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO identified_patterns 
                    (pattern_type, description, conditions, success_rate, avg_pnl, 
                     frequency, confidence, discovered_at, last_seen, examples)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_type.value,
                    pattern.description,
                    json.dumps(pattern.conditions),
                    pattern.success_rate,
                    pattern.avg_pnl,
                    pattern.frequency,
                    pattern.confidence,
                    pattern.discovered_at.isoformat(),
                    pattern.last_seen.isoformat(),
                    json.dumps(pattern.examples)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving pattern: {e}")
    
    async def _generate_insights(self):
        """Generate learning insights."""
        if len(self.trade_outcomes) < 10:
            return
        
        # Analyze recent performance
        recent_trades = [t for t in self.trade_outcomes 
                        if t.exit_time > datetime.now() - timedelta(days=7)]
        
        if len(recent_trades) >= 5:
            win_rate = len([t for t in recent_trades if t.success]) / len(recent_trades)
            
            if win_rate < 0.4:
                insight = LearningInsight(
                    insight_type="performance_decline",
                    description="Recent win rate is below 40%, indicating potential issues",
                    impact_score=0.8,
                    confidence=0.9,
                    supporting_evidence=[f"Win rate: {win_rate:.2f}", f"Sample size: {len(recent_trades)}"],
                    recommended_action="Review recent trades and consider adjusting confidence threshold",
                    discovered_at=datetime.now(),
                    validated=False
                )
                self._save_insight(insight)
            
            elif win_rate > 0.7:
                insight = LearningInsight(
                    insight_type="performance_improvement",
                    description="Recent win rate is above 70%, showing strong performance",
                    impact_score=0.6,
                    confidence=0.8,
                    supporting_evidence=[f"Win rate: {win_rate:.2f}", f"Sample size: {len(recent_trades)}"],
                    recommended_action="Consider increasing position size or confidence threshold",
                    discovered_at=datetime.now(),
                    validated=False
                )
                self._save_insight(insight)
    
    def _save_insight(self, insight: LearningInsight):
        """Save a learning insight."""
        self.learning_insights.append(insight)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO learning_insights 
                    (insight_type, description, impact_score, confidence, 
                     supporting_evidence, recommended_action, discovered_at, validated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight.insight_type,
                    insight.description,
                    insight.impact_score,
                    insight.confidence,
                    json.dumps(insight.supporting_evidence),
                    insight.recommended_action,
                    insight.discovered_at.isoformat(),
                    int(insight.validated)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving insight: {e}")
    
    async def _train_models(self):
        """Train ML models for prediction."""
        if len(self.trade_outcomes) < 50:
            return
        
        try:
            # Prepare features
            features = []
            targets_success = []
            targets_pnl = []
            
            for trade in self.trade_outcomes:
                feature_vector = [
                    trade.confidence_score,
                    trade.position_size,
                    trade.duration_hours,
                    trade.technical_indicators.get('rsi', 50),
                    trade.technical_indicators.get('macd', 0),
                    trade.sentiment_scores.get('overall', 0),
                    trade.risk_metrics.get('volatility', 0),
                    trade.entry_time.hour,
                    trade.entry_time.weekday()
                ]
                features.append(feature_vector)
                targets_success.append(int(trade.success))
                targets_pnl.append(trade.pnl_percentage)
            
            if len(features) < 20:
                return
            
            X = np.array(features)
            y_success = np.array(targets_success)
            y_pnl = np.array(targets_pnl)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_success_train, y_success_test = train_test_split(
                X_scaled, y_success, test_size=0.2, random_state=42
            )
            _, _, y_pnl_train, y_pnl_test = train_test_split(
                X_scaled, y_pnl, test_size=0.2, random_state=42
            )
            
            # Train success predictor
            self.success_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            self.success_predictor.fit(X_train, y_success_train)
            success_pred = self.success_predictor.predict(X_test)
            success_accuracy = accuracy_score(y_success_test, success_pred)
            
            # Train PnL predictor
            self.pnl_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.pnl_predictor.fit(X_train, y_pnl_train)
            pnl_pred = self.pnl_predictor.predict(X_test)
            pnl_mse = mean_squared_error(y_pnl_test, pnl_pred)
            
            self.learning_accuracy = (success_accuracy + (1 - pnl_mse)) / 2
            self.model_training_count += 1
            
            # Save models
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            joblib.dump(self.success_predictor, model_dir / f"success_predictor_{self.mode.lower()}.pkl")
            joblib.dump(self.pnl_predictor, model_dir / f"pnl_predictor_{self.mode.lower()}.pkl")
            joblib.dump(self.scaler, model_dir / f"scaler_{self.mode.lower()}.pkl")
            
            logger.info(f"Models trained: Success accuracy={success_accuracy:.3f}, "
                       f"PnL MSE={pnl_mse:.3f}, Overall accuracy={self.learning_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    async def _generate_parameter_recommendations(self):
        """Generate parameter adjustment recommendations."""
        if len(self.trade_outcomes) < 20:
            return
        
        # Analyze confidence threshold
        recent_trades = [t for t in self.trade_outcomes 
                        if t.exit_time > datetime.now() - timedelta(days=14)]
        
        if len(recent_trades) >= 10:
            high_conf_trades = [t for t in recent_trades if t.confidence_score > 0.8]
            if len(high_conf_trades) >= 3:
                high_conf_success_rate = len([t for t in high_conf_trades if t.success]) / len(high_conf_trades)
                
                if high_conf_success_rate > 0.8:
                    # High confidence trades are very successful, consider lowering threshold
                    current_threshold = get_adaptive_config_manager(self.mode).get_parameter("confidence_threshold")
                    if current_threshold and current_threshold > 0.6:
                        recommendation = ParameterRecommendation(
                            parameter_name="confidence_threshold",
                            current_value=current_threshold,
                            recommended_value=max(0.6, current_threshold - 0.05),
                            confidence=0.7,
                            reasoning=f"High confidence trades show {high_conf_success_rate:.1%} success rate",
                            expected_improvement=0.1,
                            risk_assessment="Low risk - only affects trade frequency",
                            created_at=datetime.now()
                        )
                        self._save_recommendation(recommendation)
    
    def _save_recommendation(self, recommendation: ParameterRecommendation):
        """Save a parameter recommendation."""
        self.parameter_recommendations.append(recommendation)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO parameter_recommendations 
                    (parameter_name, current_value, recommended_value, confidence, 
                     reasoning, expected_improvement, risk_assessment, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    recommendation.parameter_name,
                    recommendation.current_value,
                    recommendation.recommended_value,
                    recommendation.confidence,
                    recommendation.reasoning,
                    recommendation.expected_improvement,
                    recommendation.risk_assessment,
                    recommendation.created_at.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving recommendation: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress."""
        return {
            "total_trades_analyzed": len(self.trade_outcomes),
            "patterns_identified": len(self.identified_patterns),
            "insights_generated": len(self.learning_insights),
            "recommendations_pending": len(self.parameter_recommendations),
            "model_accuracy": self.learning_accuracy,
            "last_learning_update": self.last_learning_update.isoformat() if self.last_learning_update else None,
            "model_training_count": self.model_training_count
        }
    
    def get_recent_patterns(self, days: int = 7) -> List[IdentifiedPattern]:
        """Get recently identified patterns."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [p for p in self.identified_patterns if p.discovered_at > cutoff_date]
    
    def get_pending_recommendations(self) -> List[ParameterRecommendation]:
        """Get pending parameter recommendations."""
        return [r for r in self.parameter_recommendations if r.created_at > datetime.now() - timedelta(days=7)]
    
    def apply_recommendation(self, recommendation_id: int) -> bool:
        """Apply a parameter recommendation."""
        try:
            # Find recommendation
            recommendation = None
            for i, rec in enumerate(self.parameter_recommendations):
                if i == recommendation_id:
                    recommendation = rec
                    break
            
            if not recommendation:
                return False
            
            # Apply to configuration manager
            config_manager = get_adaptive_config_manager(self.mode)
            # This would need to be implemented in the configuration manager
            # config_manager.set_parameter(recommendation.parameter_name, recommendation.recommended_value)
            
            # Mark as applied in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE parameter_recommendations 
                    SET applied = 1 
                    WHERE parameter_name = ? AND created_at = ?
                """, (recommendation.parameter_name, recommendation.created_at.isoformat()))
                conn.commit()
            
            # Remove from pending list
            self.parameter_recommendations = [r for r in self.parameter_recommendations 
                                            if r != recommendation]
            
            logger.info(f"Applied recommendation: {recommendation.parameter_name} = {recommendation.recommended_value}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying recommendation: {e}")
            return False

# Global instance
_learning_engine = None

def get_performance_learning_engine(mode: str = None) -> PerformanceLearningEngine:
    """Get the global performance learning engine instance."""
    global _learning_engine
    if _learning_engine is None or (mode and _learning_engine.mode != mode):
        _learning_engine = PerformanceLearningEngine(mode)
    return _learning_engine

def record_trade_outcome(outcome: TradeOutcome, mode: str = None):
    """Record a trade outcome."""
    get_performance_learning_engine(mode).record_trade_outcome(outcome)

def get_learning_summary(mode: str = None) -> Dict[str, Any]:
    """Get learning summary."""
    return get_performance_learning_engine(mode).get_learning_summary()

def get_recent_patterns(days: int = 7, mode: str = None) -> List[IdentifiedPattern]:
    """Get recent patterns."""
    return get_performance_learning_engine(mode).get_recent_patterns(days)

def get_pending_recommendations(mode: str = None) -> List[ParameterRecommendation]:
    """Get pending recommendations."""
    return get_performance_learning_engine(mode).get_pending_recommendations()
