"""
Autonomous Trading System with Self-Learning

This module implements a fully autonomous trading system that can learn
from its own trading decisions and continuously improve its performance
through self-reflection, adaptive learning, and meta-cognitive processes.
"""

import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
from collections import deque
import random

from src.ai.advanced_models import get_advanced_ai_models, get_ensemble_prediction, EnsemblePrediction
from src.ai.reinforcement_learning import get_rl_trainer, TradingEnvironment
from src.ai.meta_learning import get_meta_learning_trainer
from src.trading.positions import get_position_manager, get_open_positions, get_portfolio_summary
from src.trading.execution import execute_buy_order, execute_sell_order, OrderType
from src.config.mode_manager import get_current_mode

logger = logging.getLogger(__name__)

@dataclass
class TradingDecision:
    """Represents a trading decision made by the autonomous system."""
    timestamp: datetime
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    reasoning: str
    market_features: Dict[str, Any]
    risk_assessment: Dict[str, float]
    expected_outcome: Dict[str, Any]
    model_contributions: Dict[str, float]

@dataclass
class TradingOutcome:
    """Represents the outcome of a trading decision."""
    decision: TradingDecision
    actual_outcome: Dict[str, Any]
    performance_metrics: Dict[str, float]
    learning_signal: float
    success: bool
    timestamp: datetime

@dataclass
class SelfReflection:
    """Represents a self-reflection on trading performance."""
    reflection_id: str
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    decisions_analyzed: int
    performance_summary: Dict[str, float]
    insights: List[str]
    improvements: List[str]
    action_plan: List[str]

@dataclass
class AdaptiveLearningState:
    """Represents the current state of adaptive learning."""
    learning_rate: float
    exploration_rate: float
    confidence_threshold: float
    risk_tolerance: float
    model_weights: Dict[str, float]
    performance_history: List[float]
    adaptation_frequency: int

class AutonomousTradingSystem:
    """Fully autonomous trading system with self-learning capabilities."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        
        # Core components
        self.advanced_models = get_advanced_ai_models(mode)
        self.rl_trainer = get_rl_trainer(mode)
        self.meta_trainer = get_meta_learning_trainer(mode)
        self.position_manager = get_position_manager()
        
        # Self-learning state
        self.learning_state = AdaptiveLearningState(
            learning_rate=0.01,
            exploration_rate=0.1,
            confidence_threshold=0.7,
            risk_tolerance=0.05,
            model_weights={},
            performance_history=[],
            adaptation_frequency=100
        )
        
        # Decision tracking
        self.decisions_history = deque(maxlen=10000)
        self.outcomes_history = deque(maxlen=10000)
        self.reflections_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_confidence": 0.0,
            "risk_adjusted_return": 0.0
        }
        
        # Self-reflection configuration
        self.reflection_interval = timedelta(hours=6)  # Reflect every 6 hours
        self.last_reflection = datetime.now() - self.reflection_interval
        
        # Learning configuration
        self.learning_interval = timedelta(hours=1)  # Learn every hour
        self.last_learning = datetime.now() - self.learning_interval
        
        # System state
        self.is_autonomous = False
        self.learning_enabled = True
        
        logger.info(f"Autonomous Trading System initialized for {mode} mode")
    
    def start_autonomous_trading(self) -> bool:
        """Start autonomous trading operations."""
        try:
            logger.info("Starting autonomous trading system...")
            
            # Validate system components
            if not self._validate_system():
                logger.error("System validation failed")
                return False
            
            # Initialize learning state
            self._initialize_learning_state()
            
            # Start autonomous mode
            self.is_autonomous = True
            self.learning_enabled = True
            
            logger.info("Autonomous trading system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting autonomous trading: {e}")
            return False
    
    def stop_autonomous_trading(self) -> bool:
        """Stop autonomous trading operations."""
        try:
            logger.info("Stopping autonomous trading system...")
            
            # Perform final reflection
            self._perform_self_reflection()
            
            # Save learning state
            self._save_learning_state()
            
            # Stop autonomous mode
            self.is_autonomous = False
            
            logger.info("Autonomous trading system stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping autonomous trading: {e}")
            return False
    
    def make_trading_decision(self, symbol: str, market_features: Dict[str, Any]) -> TradingDecision:
        """Make an autonomous trading decision."""
        try:
            # Get ensemble prediction
            ensemble_pred = get_ensemble_prediction(symbol, market_features, self.mode)
            
            # Apply adaptive learning adjustments
            adjusted_pred = self._apply_adaptive_adjustments(ensemble_pred, market_features)
            
            # Create trading decision
            decision = TradingDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action=adjusted_pred.action,
                confidence=adjusted_pred.confidence,
                reasoning=adjusted_pred.reasoning,
                market_features=market_features,
                risk_assessment=adjusted_pred.risk_assessment,
                expected_outcome=self._predict_outcome(adjusted_pred, market_features),
                model_contributions=adjusted_pred.model_contributions
            )
            
            # Store decision
            self.decisions_history.append(decision)
            self.performance_metrics["total_decisions"] += 1
            
            # Update average confidence
            self._update_confidence_metrics(decision.confidence)
            
            logger.info(f"Trading decision made: {symbol} {decision.action} (confidence: {decision.confidence:.2f})")
            return decision
            
        except Exception as e:
            logger.error(f"Error making trading decision: {e}")
            # Return safe decision
            return TradingDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reasoning=f"Error: {e}",
                market_features=market_features,
                risk_assessment={},
                expected_outcome={},
                model_contributions={}
            )
    
    def execute_decision(self, decision: TradingDecision) -> bool:
        """Execute a trading decision."""
        try:
            # Check if decision meets confidence threshold
            if decision.confidence < self.learning_state.confidence_threshold:
                logger.info(f"Decision confidence too low: {decision.confidence:.2f} < {self.learning_state.confidence_threshold:.2f}")
                return False
            
            # Check risk tolerance
            if not self._check_risk_tolerance(decision):
                logger.info("Decision exceeds risk tolerance")
                return False
            
            # Execute the decision
            if decision.action == "BUY":
                result = execute_buy_order(
                    symbol=decision.symbol,
                    quantity=self._calculate_position_size(decision),
                    price=self._get_current_price(decision.symbol),
                    order_type=OrderType.MARKET
                )
            elif decision.action == "SELL":
                # Find existing position
                open_positions = get_open_positions(self.mode)
                position = next((p for p in open_positions if p.symbol == decision.symbol), None)
                
                if position:
                    result = execute_sell_order(
                        position=position,
                        exit_price=self._get_current_price(decision.symbol),
                        order_type=OrderType.MARKET
                    )
                else:
                    logger.warning(f"No position found for {decision.symbol}")
                    return False
            else:  # HOLD
                logger.info(f"Holding position for {decision.symbol}")
                return True
            
            if result.success:
                logger.info(f"Successfully executed {decision.action} for {decision.symbol}")
                return True
            else:
                logger.warning(f"Failed to execute {decision.action} for {decision.symbol}: {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing decision: {e}")
            return False
    
    def learn_from_outcome(self, decision: TradingDecision, outcome: TradingOutcome):
        """Learn from the outcome of a trading decision."""
        try:
            # Store outcome
            self.outcomes_history.append(outcome)
            
            # Update performance metrics
            self._update_performance_metrics(outcome)
            
            # Generate learning signal
            learning_signal = self._generate_learning_signal(decision, outcome)
            
            # Update learning state
            self._update_learning_state(learning_signal)
            
            # Check if it's time for reflection
            if datetime.now() - self.last_reflection >= self.reflection_interval:
                self._perform_self_reflection()
            
            # Check if it's time for learning
            if datetime.now() - self.last_learning >= self.learning_interval:
                self._perform_adaptive_learning()
            
            logger.info(f"Learned from outcome: {outcome.success}, signal: {learning_signal:.3f}")
            
        except Exception as e:
            logger.error(f"Error learning from outcome: {e}")
    
    def _validate_system(self) -> bool:
        """Validate system components."""
        try:
            # Check advanced models
            if not self.advanced_models.validate_advanced_models():
                logger.error("Advanced models validation failed")
                return False
            
            # Check position manager
            if not self.position_manager:
                logger.error("Position manager not available")
                return False
            
            logger.info("System validation passed")
            return True
            
        except Exception as e:
            logger.error(f"System validation error: {e}")
            return False
    
    def _initialize_learning_state(self):
        """Initialize the learning state."""
        try:
            # Load existing learning state if available
            state_file = f"data/learning_state_{self.mode}.json"
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    self.learning_state.learning_rate = state_data.get("learning_rate", 0.01)
                    self.learning_state.exploration_rate = state_data.get("exploration_rate", 0.1)
                    self.learning_state.confidence_threshold = state_data.get("confidence_threshold", 0.7)
                    self.learning_state.risk_tolerance = state_data.get("risk_tolerance", 0.05)
                    self.learning_state.model_weights = state_data.get("model_weights", {})
                    self.learning_state.performance_history = state_data.get("performance_history", [])
                
                logger.info("Loaded existing learning state")
            else:
                logger.info("Initialized new learning state")
                
        except Exception as e:
            logger.error(f"Error initializing learning state: {e}")
    
    def _apply_adaptive_adjustments(self, prediction: EnsemblePrediction, 
                                  market_features: Dict[str, Any]) -> EnsemblePrediction:
        """Apply adaptive learning adjustments to predictions."""
        try:
            # Adjust confidence based on recent performance
            performance_factor = self._get_performance_factor()
            adjusted_confidence = prediction.confidence * performance_factor
            
            # Adjust for market conditions
            market_adjustment = self._get_market_adjustment(market_features)
            adjusted_confidence *= market_adjustment
            
            # Apply exploration
            if random.random() < self.learning_state.exploration_rate:
                # Randomly adjust action for exploration
                actions = ["BUY", "SELL", "HOLD"]
                if prediction.action in actions:
                    actions.remove(prediction.action)
                    prediction.action = random.choice(actions)
                    adjusted_confidence *= 0.5  # Reduce confidence for exploration
            
            # Clamp confidence
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            # Create adjusted prediction
            adjusted_pred = EnsemblePrediction(
                symbol=prediction.symbol,
                action=prediction.action,
                confidence=adjusted_confidence,
                reasoning=prediction.reasoning + f" [Adjusted: {performance_factor:.2f}x]",
                model_contributions=prediction.model_contributions,
                risk_assessment=prediction.risk_assessment
            )
            
            return adjusted_pred
            
        except Exception as e:
            logger.error(f"Error applying adaptive adjustments: {e}")
            return prediction
    
    def _predict_outcome(self, prediction: EnsemblePrediction, 
                        market_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the expected outcome of a trading decision."""
        try:
            # Simple outcome prediction based on market features
            market_trend = market_features.get("market_trend", 0.0)
            volatility = market_features.get("volatility", 0.02)
            sentiment = market_features.get("sentiment", 0.0)
            
            # Calculate expected return
            if prediction.action == "BUY":
                expected_return = market_trend * 0.1 + sentiment * 0.05
            elif prediction.action == "SELL":
                expected_return = -market_trend * 0.1 - sentiment * 0.05
            else:  # HOLD
                expected_return = 0.0
            
            # Calculate risk
            expected_risk = volatility * 2.0
            
            # Calculate success probability
            success_prob = 0.5 + (prediction.confidence - 0.5) * 0.5
            
            return {
                "expected_return": expected_return,
                "expected_risk": expected_risk,
                "success_probability": success_prob,
                "time_horizon": "short_term"
            }
            
        except Exception as e:
            logger.error(f"Error predicting outcome: {e}")
            return {
                "expected_return": 0.0,
                "expected_risk": 0.02,
                "success_probability": 0.5,
                "time_horizon": "short_term"
            }
    
    def _check_risk_tolerance(self, decision: TradingDecision) -> bool:
        """Check if decision is within risk tolerance."""
        try:
            # Get current portfolio risk
            portfolio_summary = get_portfolio_summary(self.mode)
            current_risk = portfolio_summary.get("total_unrealized_pnl_percent", 0.0) / 100.0
            
            # Check decision risk
            decision_risk = decision.risk_assessment.get("overall_risk", 0.5)
            
            # Combined risk
            total_risk = current_risk + decision_risk
            
            return total_risk <= self.learning_state.risk_tolerance
            
        except Exception as e:
            logger.error(f"Error checking risk tolerance: {e}")
            return False
    
    def _calculate_position_size(self, decision: TradingDecision) -> int:
        """Calculate position size for a trading decision."""
        try:
            # Get current portfolio value
            portfolio_summary = get_portfolio_summary(self.mode)
            portfolio_value = portfolio_summary.get("total_invested", 10000.0)
            
            # Calculate position size based on confidence and risk
            base_size = 100  # Base position size
            confidence_factor = decision.confidence
            risk_factor = 1.0 - decision.risk_assessment.get("overall_risk", 0.5)
            
            position_size = int(base_size * confidence_factor * risk_factor)
            
            # Ensure minimum size
            position_size = max(1, position_size)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 10  # Default size
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        # This would integrate with real market data in production
        # For now, return simulated price
        base_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0,
            "AMZN": 3000.0,
            "TSLA": 800.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        variation = np.random.normal(0, 0.02)
        return base_price * (1 + variation)
    
    def _update_confidence_metrics(self, confidence: float):
        """Update confidence-related metrics."""
        try:
            # Update average confidence
            total_decisions = self.performance_metrics["total_decisions"]
            current_avg = self.performance_metrics["avg_confidence"]
            
            new_avg = (current_avg * (total_decisions - 1) + confidence) / total_decisions
            self.performance_metrics["avg_confidence"] = new_avg
            
        except Exception as e:
            logger.error(f"Error updating confidence metrics: {e}")
    
    def _update_performance_metrics(self, outcome: TradingOutcome):
        """Update performance metrics based on outcome."""
        try:
            if outcome.success:
                self.performance_metrics["successful_decisions"] += 1
            
            # Update win rate
            total_decisions = self.performance_metrics["total_decisions"]
            successful_decisions = self.performance_metrics["successful_decisions"]
            self.performance_metrics["win_rate"] = successful_decisions / total_decisions if total_decisions > 0 else 0.0
            
            # Update total P&L
            pnl = outcome.performance_metrics.get("pnl", 0.0)
            self.performance_metrics["total_pnl"] += pnl
            
            # Update performance history
            self.learning_state.performance_history.append(pnl)
            if len(self.learning_state.performance_history) > 1000:
                self.learning_state.performance_history.pop(0)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _generate_learning_signal(self, decision: TradingDecision, outcome: TradingOutcome) -> float:
        """Generate a learning signal from decision and outcome."""
        try:
            # Base signal from outcome success
            base_signal = 1.0 if outcome.success else -1.0
            
            # Adjust for confidence
            confidence_factor = decision.confidence
            signal = base_signal * confidence_factor
            
            # Adjust for risk
            risk_factor = 1.0 - decision.risk_assessment.get("overall_risk", 0.5)
            signal *= risk_factor
            
            # Adjust for performance
            performance_factor = outcome.performance_metrics.get("pnl", 0.0)
            if performance_factor > 0:
                signal *= 1.2
            else:
                signal *= 0.8
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating learning signal: {e}")
            return 0.0
    
    def _update_learning_state(self, learning_signal: float):
        """Update learning state based on learning signal."""
        try:
            # Update learning rate based on performance
            if learning_signal > 0:
                self.learning_state.learning_rate *= 1.01  # Increase learning rate
            else:
                self.learning_state.learning_rate *= 0.99  # Decrease learning rate
            
            # Clamp learning rate
            self.learning_state.learning_rate = max(0.001, min(0.1, self.learning_state.learning_rate))
            
            # Update exploration rate
            if len(self.learning_state.performance_history) > 10:
                recent_performance = np.mean(self.learning_state.performance_history[-10:])
                if recent_performance > 0:
                    self.learning_state.exploration_rate *= 0.99  # Reduce exploration
                else:
                    self.learning_state.exploration_rate *= 1.01  # Increase exploration
            
            # Clamp exploration rate
            self.learning_state.exploration_rate = max(0.01, min(0.5, self.learning_state.exploration_rate))
            
            # Update confidence threshold
            win_rate = self.performance_metrics["win_rate"]
            if win_rate > 0.6:
                self.learning_state.confidence_threshold *= 0.99  # Lower threshold
            elif win_rate < 0.4:
                self.learning_state.confidence_threshold *= 1.01  # Raise threshold
            
            # Clamp confidence threshold
            self.learning_state.confidence_threshold = max(0.5, min(0.9, self.learning_state.confidence_threshold))
            
        except Exception as e:
            logger.error(f"Error updating learning state: {e}")
    
    def _perform_self_reflection(self):
        """Perform self-reflection on trading performance."""
        try:
            logger.info("Performing self-reflection...")
            
            # Analyze recent decisions and outcomes
            recent_decisions = list(self.decisions_history)[-100:] if self.decisions_history else []
            recent_outcomes = list(self.outcomes_history)[-100:] if self.outcomes_history else []
            
            # Calculate performance summary
            performance_summary = self._calculate_performance_summary(recent_outcomes)
            
            # Generate insights
            insights = self._generate_insights(recent_decisions, recent_outcomes)
            
            # Generate improvements
            improvements = self._generate_improvements(performance_summary, insights)
            
            # Create action plan
            action_plan = self._create_action_plan(improvements)
            
            # Create reflection
            reflection = SelfReflection(
                reflection_id=f"reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                period_start=self.last_reflection,
                period_end=datetime.now(),
                decisions_analyzed=len(recent_decisions),
                performance_summary=performance_summary,
                insights=insights,
                improvements=improvements,
                action_plan=action_plan
            )
            
            # Store reflection
            self.reflections_history.append(reflection)
            self.last_reflection = datetime.now()
            
            # Log reflection
            logger.info(f"Self-reflection completed: {len(insights)} insights, {len(improvements)} improvements")
            
        except Exception as e:
            logger.error(f"Error performing self-reflection: {e}")
    
    def _perform_adaptive_learning(self):
        """Perform adaptive learning based on recent performance."""
        try:
            logger.info("Performing adaptive learning...")
            
            # Update model weights based on performance
            self._update_model_weights()
            
            # Retrain models if needed
            if self._should_retrain_models():
                self._retrain_models()
            
            # Update learning parameters
            self._update_learning_parameters()
            
            self.last_learning = datetime.now()
            
            logger.info("Adaptive learning completed")
            
        except Exception as e:
            logger.error(f"Error performing adaptive learning: {e}")
    
    def _get_performance_factor(self) -> float:
        """Get performance factor for confidence adjustment."""
        try:
            if len(self.learning_state.performance_history) < 10:
                return 1.0
            
            recent_performance = np.mean(self.learning_state.performance_history[-10:])
            
            if recent_performance > 0:
                return 1.0 + min(0.2, recent_performance * 10)
            else:
                return 1.0 - min(0.2, abs(recent_performance) * 10)
                
        except Exception as e:
            logger.error(f"Error getting performance factor: {e}")
            return 1.0
    
    def _get_market_adjustment(self, market_features: Dict[str, Any]) -> float:
        """Get market adjustment factor for confidence."""
        try:
            # Adjust based on market volatility
            volatility = market_features.get("volatility", 0.02)
            if volatility > 0.05:  # High volatility
                return 0.8
            elif volatility < 0.01:  # Low volatility
                return 1.2
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error getting market adjustment: {e}")
            return 1.0
    
    def _calculate_performance_summary(self, outcomes: List[TradingOutcome]) -> Dict[str, float]:
        """Calculate performance summary from outcomes."""
        try:
            if not outcomes:
                return {}
            
            total_pnl = sum(o.performance_metrics.get("pnl", 0.0) for o in outcomes)
            successful_outcomes = sum(1 for o in outcomes if o.success)
            win_rate = successful_outcomes / len(outcomes) if outcomes else 0.0
            
            return {
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "avg_pnl": total_pnl / len(outcomes) if outcomes else 0.0,
                "successful_outcomes": successful_outcomes,
                "total_outcomes": len(outcomes)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            return {}
    
    def _generate_insights(self, decisions: List[TradingDecision], 
                          outcomes: List[TradingOutcome]) -> List[str]:
        """Generate insights from decisions and outcomes."""
        insights = []
        
        try:
            # Analyze confidence vs performance
            if decisions and outcomes:
                high_confidence_decisions = [d for d in decisions if d.confidence > 0.8]
                high_confidence_outcomes = [o for o in outcomes if o.decision.confidence > 0.8]
                
                if high_confidence_outcomes:
                    high_conf_success_rate = sum(1 for o in high_confidence_outcomes if o.success) / len(high_confidence_outcomes)
                    if high_conf_success_rate > 0.7:
                        insights.append("High confidence decisions are performing well")
                    elif high_conf_success_rate < 0.5:
                        insights.append("High confidence decisions are underperforming")
            
            # Analyze action performance
            action_performance = {}
            for outcome in outcomes:
                action = outcome.decision.action
                if action not in action_performance:
                    action_performance[action] = []
                action_performance[action].append(outcome.success)
            
            for action, successes in action_performance.items():
                success_rate = sum(successes) / len(successes)
                if success_rate > 0.6:
                    insights.append(f"{action} actions are performing well ({success_rate:.1%})")
                elif success_rate < 0.4:
                    insights.append(f"{action} actions are underperforming ({success_rate:.1%})")
            
            # Analyze risk vs reward
            if outcomes:
                high_risk_outcomes = [o for o in outcomes if o.decision.risk_assessment.get("overall_risk", 0) > 0.7]
                if high_risk_outcomes:
                    high_risk_success_rate = sum(1 for o in high_risk_outcomes if o.success) / len(high_risk_outcomes)
                    if high_risk_success_rate < 0.3:
                        insights.append("High risk decisions are not paying off")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append(f"Error generating insights: {e}")
        
        return insights
    
    def _generate_improvements(self, performance_summary: Dict[str, float], 
                             insights: List[str]) -> List[str]:
        """Generate improvement suggestions."""
        improvements = []
        
        try:
            # Based on performance summary
            win_rate = performance_summary.get("win_rate", 0.0)
            if win_rate < 0.5:
                improvements.append("Improve decision accuracy by increasing confidence threshold")
            
            avg_pnl = performance_summary.get("avg_pnl", 0.0)
            if avg_pnl < 0:
                improvements.append("Focus on risk management to reduce losses")
            
            # Based on insights
            for insight in insights:
                if "underperforming" in insight.lower():
                    improvements.append("Review and adjust model weights")
                elif "high risk" in insight.lower():
                    improvements.append("Implement stricter risk controls")
                elif "confidence" in insight.lower():
                    improvements.append("Calibrate confidence thresholds")
            
        except Exception as e:
            logger.error(f"Error generating improvements: {e}")
            improvements.append(f"Error generating improvements: {e}")
        
        return improvements
    
    def _create_action_plan(self, improvements: List[str]) -> List[str]:
        """Create action plan from improvements."""
        action_plan = []
        
        try:
            for improvement in improvements:
                if "confidence threshold" in improvement:
                    action_plan.append("Adjust confidence threshold by 0.05")
                elif "risk controls" in improvement:
                    action_plan.append("Reduce risk tolerance by 0.01")
                elif "model weights" in improvement:
                    action_plan.append("Retrain models with recent data")
                elif "risk management" in improvement:
                    action_plan.append("Implement position sizing limits")
            
        except Exception as e:
            logger.error(f"Error creating action plan: {e}")
            action_plan.append(f"Error creating action plan: {e}")
        
        return action_plan
    
    def _update_model_weights(self):
        """Update model weights based on performance."""
        try:
            # Get model performances
            model_performances = self.advanced_models.get_model_performances()
            
            if model_performances:
                # Calculate new weights based on performance
                total_score = 0.0
                scores = {}
                
                for model_type, performance in model_performances.items():
                    score = performance.accuracy * 0.4 + performance.sharpe_ratio * 0.3 + performance.win_rate * 0.3
                    scores[model_type] = score
                    total_score += score
                
                # Normalize weights
                if total_score > 0:
                    for model_type in self.learning_state.model_weights:
                        if model_type in scores:
                            self.learning_state.model_weights[model_type] = scores[model_type] / total_score
                
                # Update ensemble weights
                self.advanced_models.ensemble_weights.update(self.learning_state.model_weights)
                
        except Exception as e:
            logger.error(f"Error updating model weights: {e}")
    
    def _should_retrain_models(self) -> bool:
        """Check if models should be retrained."""
        try:
            # Retrain if performance has degraded
            if len(self.learning_state.performance_history) > 50:
                recent_performance = np.mean(self.learning_state.performance_history[-20:])
                older_performance = np.mean(self.learning_state.performance_history[-50:-20])
                
                if recent_performance < older_performance * 0.8:  # 20% degradation
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if models should retrain: {e}")
            return False
    
    def _retrain_models(self):
        """Retrain models with recent data."""
        try:
            logger.info("Retraining models...")
            
            # Retrain with fewer episodes for quick adaptation
            self.advanced_models.train_all_models(episodes=100, meta_epochs=20)
            
            logger.info("Models retrained successfully")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def _update_learning_parameters(self):
        """Update learning parameters based on performance."""
        try:
            # Update learning rate based on recent performance
            if len(self.learning_state.performance_history) > 20:
                recent_performance = np.mean(self.learning_state.performance_history[-20:])
                
                if recent_performance > 0:
                    self.learning_state.learning_rate *= 1.05
                else:
                    self.learning_state.learning_rate *= 0.95
                
                # Clamp learning rate
                self.learning_state.learning_rate = max(0.001, min(0.1, self.learning_state.learning_rate))
            
        except Exception as e:
            logger.error(f"Error updating learning parameters: {e}")
    
    def _save_learning_state(self):
        """Save learning state to file."""
        try:
            state_data = {
                "learning_rate": self.learning_state.learning_rate,
                "exploration_rate": self.learning_state.exploration_rate,
                "confidence_threshold": self.learning_state.confidence_threshold,
                "risk_tolerance": self.learning_state.risk_tolerance,
                "model_weights": self.learning_state.model_weights,
                "performance_history": self.learning_state.performance_history[-100:],  # Keep last 100
                "performance_metrics": self.performance_metrics
            }
            
            state_file = f"data/learning_state_{self.mode}.json"
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.info("Learning state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")
    
    def get_autonomous_status(self) -> Dict[str, Any]:
        """Get autonomous trading system status."""
        return {
            "is_autonomous": self.is_autonomous,
            "learning_enabled": self.learning_enabled,
            "learning_state": {
                "learning_rate": self.learning_state.learning_rate,
                "exploration_rate": self.learning_state.exploration_rate,
                "confidence_threshold": self.learning_state.confidence_threshold,
                "risk_tolerance": self.learning_state.risk_tolerance
            },
            "performance_metrics": self.performance_metrics,
            "decisions_count": len(self.decisions_history),
            "outcomes_count": len(self.outcomes_history),
            "reflections_count": len(self.reflections_history),
            "last_reflection": self.last_reflection.isoformat(),
            "last_learning": self.last_learning.isoformat()
        }
    
    def get_recent_decisions(self, limit: int = 10) -> List[TradingDecision]:
        """Get recent trading decisions."""
        return list(self.decisions_history)[-limit:] if self.decisions_history else []
    
    def get_recent_outcomes(self, limit: int = 10) -> List[TradingOutcome]:
        """Get recent trading outcomes."""
        return list(self.outcomes_history)[-limit:] if self.outcomes_history else []
    
    def get_recent_reflections(self, limit: int = 5) -> List[SelfReflection]:
        """Get recent self-reflections."""
        return list(self.reflections_history)[-limit:] if self.reflections_history else []

# Global autonomous trading system instance
_autonomous_trading_system: Optional[AutonomousTradingSystem] = None

def get_autonomous_trading_system(mode: str = None) -> AutonomousTradingSystem:
    """Get the global autonomous trading system instance."""
    global _autonomous_trading_system
    if _autonomous_trading_system is None:
        if mode is None:
            mode = get_current_mode()
        _autonomous_trading_system = AutonomousTradingSystem(mode)
    return _autonomous_trading_system

def start_autonomous_trading(mode: str = None) -> bool:
    """Start autonomous trading."""
    return get_autonomous_trading_system(mode).start_autonomous_trading()

def stop_autonomous_trading(mode: str = None) -> bool:
    """Stop autonomous trading."""
    return get_autonomous_trading_system(mode).stop_autonomous_trading()

def make_trading_decision(symbol: str, market_features: Dict[str, Any], mode: str = None) -> TradingDecision:
    """Make an autonomous trading decision."""
    return get_autonomous_trading_system(mode).make_trading_decision(symbol, market_features)

def execute_autonomous_decision(decision: TradingDecision, mode: str = None) -> bool:
    """Execute an autonomous trading decision."""
    return get_autonomous_trading_system(mode).execute_decision(decision)

def learn_from_outcome(decision: TradingDecision, outcome: TradingOutcome, mode: str = None):
    """Learn from trading outcome."""
    get_autonomous_trading_system(mode).learn_from_outcome(decision, outcome)

def get_autonomous_status(mode: str = None) -> Dict[str, Any]:
    """Get autonomous trading status."""
    return get_autonomous_trading_system(mode).get_autonomous_status()
