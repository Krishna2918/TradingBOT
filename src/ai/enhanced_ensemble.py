"""
Enhanced AI Ensemble - Position-Aware Analysis with Exit Strategy

This module provides enhanced AI analysis that considers current positions
and generates both BUY and SELL signals based on comprehensive market analysis.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.trading.positions import Position, get_position_manager, get_open_positions
from src.trading.exit_strategy import ExitReason, SellSignal
from src.config.mode_manager import get_current_mode
from src.ai.adaptive_weights import get_adaptive_weight_manager, get_ensemble_weights
from src.ai.regime_detection import detect_current_regime, MarketRegime
from src.config.regime_policy_manager import get_ensemble_weights as get_regime_ensemble_weights

logger = logging.getLogger(__name__)

class Decision(Enum):
    """Trading decision enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class BuySignal:
    """Represents a buy signal."""
    symbol: str
    confidence: float
    reasoning: List[str]
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    risk_assessment: str
    mode: str
    timestamp: datetime

@dataclass
class SellSignal:
    """Represents a sell signal."""
    symbol: str
    position_id: int
    exit_price: float
    exit_reason: ExitReason
    confidence: float
    reasoning: List[str]
    mode: str
    timestamp: datetime

@dataclass
class MarketAnalysis:
    """Comprehensive market analysis data."""
    symbol: str
    current_price: float
    technical_indicators: Dict[str, float]
    sentiment_score: float
    fundamental_score: float
    market_regime: str
    sector_performance: float
    news_impact: float
    volatility: float
    volume_trend: str
    timestamp: datetime

class EnhancedEnsemble:
    """Enhanced AI ensemble with position-aware analysis."""
    
    def __init__(self):
        """Initialize Enhanced Ensemble."""
        self.position_manager = get_position_manager()
        
        # AI model configurations
        self.models = {
            "technical_analyst": {
                "weight": 0.3,
                "description": "Technical analysis and chart patterns"
            },
            "sentiment_analyst": {
                "weight": 0.25,
                "description": "News sentiment and market psychology"
            },
            "fundamental_analyst": {
                "weight": 0.2,
                "description": "Company fundamentals and financials"
            },
            "risk_analyst": {
                "weight": 0.15,
                "description": "Risk assessment and position sizing"
            },
            "market_regime_analyst": {
                "weight": 0.1,
                "description": "Market regime and sector analysis"
            }
        }
        
        logger.info("Enhanced AI Ensemble initialized")
    
    def analyze_for_entry(self, symbol: str, features: Dict[str, Any], 
                         mode: Optional[str] = None) -> Optional[BuySignal]:
        """Analyze market data for entry (BUY) signals."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Check if position already exists
            existing_position = self.position_manager.get_position_by_symbol(symbol, mode)
            if existing_position:
                logger.info(f"Position already exists for {symbol}, skipping entry analysis")
                return None
            
            # Build comprehensive market analysis
            market_analysis = self._build_market_analysis(symbol, features)
            
            # Run ensemble analysis
            ensemble_decision = self._run_ensemble_analysis(market_analysis, "ENTRY")
            
            if ensemble_decision["action"] == "BUY" and ensemble_decision["confidence"] > 0.7:
                # Calculate position parameters
                entry_price = market_analysis.current_price
                stop_loss = self._calculate_stop_loss(entry_price, market_analysis.volatility)
                take_profit = self._calculate_take_profit(entry_price, stop_loss)
                position_size = self._calculate_position_size(
                    ensemble_decision["confidence"], 
                    market_analysis.volatility,
                    entry_price,
                    stop_loss
                )
                
                buy_signal = BuySignal(
                    symbol=symbol,
                    confidence=ensemble_decision["confidence"],
                    reasoning=ensemble_decision["reasoning"],
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    risk_assessment=ensemble_decision["risk_assessment"],
                    mode=mode,
                    timestamp=datetime.now()
                )
                
                logger.info(f"Generated BUY signal for {symbol}: confidence={ensemble_decision['confidence']:.2f}")
                return buy_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing entry for {symbol}: {e}")
            return None
    
    def analyze_for_exit(self, position: Position, current_data: Dict[str, Any], 
                        mode: Optional[str] = None) -> Optional[SellSignal]:
        """Analyze position for exit (SELL) signals."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Build market analysis for exit decision
            market_analysis = self._build_market_analysis(position.symbol, current_data)
            
            # Add position-specific data
            market_analysis.position_pnl = (market_analysis.current_price - position.entry_price) / position.entry_price
            market_analysis.holding_days = (datetime.now() - position.entry_time).days
            
            # Run ensemble analysis for exit
            ensemble_decision = self._run_ensemble_analysis(market_analysis, "EXIT", position)
            
            if ensemble_decision["action"] == "SELL" and ensemble_decision["confidence"] > 0.6:
                # Determine exit reason
                exit_reason = self._determine_exit_reason(ensemble_decision, position, market_analysis)
                
                sell_signal = SellSignal(
                    symbol=position.symbol,
                    position_id=position.id,
                    exit_price=market_analysis.current_price,
                    exit_reason=exit_reason,
                    confidence=ensemble_decision["confidence"],
                    reasoning=ensemble_decision["reasoning"],
                    mode=mode,
                    timestamp=datetime.now()
                )
                
                logger.info(f"Generated SELL signal for {position.symbol}: reason={exit_reason.value}, confidence={ensemble_decision['confidence']:.2f}")
                return sell_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing exit for {position.symbol}: {e}")
            return None
    
    def generate_position_aware_analysis(self, symbol: str, is_owned: bool, 
                                       mode: Optional[str] = None) -> Dict[str, Any]:
        """Generate position-aware analysis for a symbol."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Get current position if owned
            position = None
            if is_owned:
                position = self.position_manager.get_position_by_symbol(symbol, mode)
            
            # Build market analysis
            features = self._get_market_features(symbol)
            market_analysis = self._build_market_analysis(symbol, features)
            
            # Add position context
            if position:
                market_analysis.position_pnl = (market_analysis.current_price - position.entry_price) / position.entry_price
                market_analysis.holding_days = (datetime.now() - position.entry_time).days
                market_analysis.entry_price = position.entry_price
                market_analysis.stop_loss = position.stop_loss
                market_analysis.take_profit = position.take_profit
            
            # Run appropriate analysis
            if is_owned and position:
                decision = self._run_ensemble_analysis(market_analysis, "EXIT", position)
            else:
                decision = self._run_ensemble_analysis(market_analysis, "ENTRY")
            
            return {
                "symbol": symbol,
                "is_owned": is_owned,
                "current_price": market_analysis.current_price,
                "decision": decision["action"],
                "confidence": decision["confidence"],
                "reasoning": decision["reasoning"],
                "risk_assessment": decision.get("risk_assessment", "MEDIUM"),
                "market_analysis": {
                    "technical_score": market_analysis.technical_indicators.get("overall_score", 0.5),
                    "sentiment_score": market_analysis.sentiment_score,
                    "fundamental_score": market_analysis.fundamental_score,
                    "market_regime": market_analysis.market_regime,
                    "volatility": market_analysis.volatility
                },
                "position_context": {
                    "entry_price": position.entry_price if position else None,
                    "current_pnl": market_analysis.position_pnl if position else None,
                    "holding_days": market_analysis.holding_days if position else None,
                    "stop_loss": position.stop_loss if position else None,
                    "take_profit": position.take_profit if position else None
                } if position else None,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating position-aware analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "is_owned": is_owned,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def _build_market_analysis(self, symbol: str, features: Dict[str, Any]) -> MarketAnalysis:
        """Build comprehensive market analysis."""
        # Extract technical indicators
        technical_indicators = {
            "rsi": features.get("rsi", 50.0),
            "macd": features.get("macd", 0.0),
            "sma_20": features.get("sma_20", 0.0),
            "sma_50": features.get("sma_50", 0.0),
            "bollinger_position": features.get("bollinger_position", 0.5),
            "volume_ratio": features.get("volume_ratio", 1.0),
            "atr": features.get("atr", 0.0)
        }
        
        # Calculate overall technical score
        technical_score = self._calculate_technical_score(technical_indicators)
        technical_indicators["overall_score"] = technical_score
        
        return MarketAnalysis(
            symbol=symbol,
            current_price=features.get("current_price", 0.0),
            technical_indicators=technical_indicators,
            sentiment_score=features.get("sentiment_score", 0.0),
            fundamental_score=features.get("fundamental_score", 0.5),
            market_regime=features.get("market_regime", "NEUTRAL"),
            sector_performance=features.get("sector_performance", 0.0),
            news_impact=features.get("news_impact", 0.0),
            volatility=features.get("volatility", 0.02),
            volume_trend=features.get("volume_trend", "NORMAL"),
            timestamp=datetime.now()
        )
    
    def _run_ensemble_analysis(self, market_analysis: MarketAnalysis, 
                              analysis_type: str, position: Optional[Position] = None) -> Dict[str, Any]:
        """Run ensemble analysis using multiple AI models."""
        try:
            # Technical Analysis
            technical_score = self._analyze_technical(market_analysis)
            
            # Sentiment Analysis
            sentiment_score = self._analyze_sentiment(market_analysis)
            
            # Fundamental Analysis
            fundamental_score = self._analyze_fundamental(market_analysis)
            
            # Risk Analysis
            risk_score = self._analyze_risk(market_analysis, position)
            
            # Market Regime Analysis
            regime_score = self._analyze_market_regime(market_analysis)
            
            # Phase 7: Get regime-aware ensemble weights
            current_regime = detect_current_regime()
            regime_weights = get_regime_ensemble_weights(current_regime.regime)
            
            # Get adaptive weights for ensemble models
            adaptive_weights = get_ensemble_weights()
            
            # Map model names to scores
            model_scores = {
                "technical_analyst": technical_score,
                "sentiment_analyst": sentiment_score,
                "fundamental_analyst": fundamental_score,
                "risk_analyst": risk_score,
                "market_regime_analyst": regime_score
            }
            
            # Combine regime weights, adaptive weights, and default weights
            weights = []
            scores = []
            default_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            model_names = ["technical_analyst", "sentiment_analyst", "fundamental_analyst", "risk_analyst", "market_regime_analyst"]
            
            for model_name, default_weight in zip(model_names, default_weights):
                # Get regime weight (40%), adaptive weight (40%), and default weight (20%)
                regime_weight = regime_weights.get(model_name, default_weight)
                adaptive_weight = adaptive_weights.get(model_name, default_weight) if adaptive_weights else default_weight
                
                # Blend all three weight sources
                blended_weight = (regime_weight * 0.4) + (adaptive_weight * 0.4) + (default_weight * 0.2)
                weights.append(blended_weight)
                scores.append(model_scores[model_name])
            
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                # Fallback to default weights
                weights = default_weights
            
            weighted_score = sum(w * s for w, s in zip(weights, scores))
            
            # Determine action based on analysis type and score
            if analysis_type == "ENTRY":
                if weighted_score > 0.7:
                    action = "BUY"
                    confidence = min(0.95, weighted_score)
                else:
                    action = "HOLD"
                    confidence = 1.0 - weighted_score
            else:  # EXIT
                if weighted_score < 0.3:
                    action = "SELL"
                    confidence = 1.0 - weighted_score
                else:
                    action = "HOLD"
                    confidence = weighted_score
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                technical_score, sentiment_score, fundamental_score, 
                risk_score, regime_score, action, analysis_type
            )
            
            # Risk assessment
            risk_assessment = self._assess_risk(weighted_score, market_analysis.volatility)
            
            # Log model predictions for performance tracking
            self._log_model_predictions(market_analysis.symbol, model_scores, confidence, analysis_type)
            
            return {
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning,
                "risk_assessment": risk_assessment,
                "scores": {
                    "technical": technical_score,
                    "sentiment": sentiment_score,
                    "fundamental": fundamental_score,
                    "risk": risk_score,
                    "regime": regime_score,
                    "weighted": weighted_score
                },
                "weights": {
                    "technical": weights[0] if len(weights) > 0 else 0.3,
                    "sentiment": weights[1] if len(weights) > 1 else 0.25,
                    "fundamental": weights[2] if len(weights) > 2 else 0.2,
                    "risk": weights[3] if len(weights) > 3 else 0.15,
                    "regime": weights[4] if len(weights) > 4 else 0.1
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble analysis: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "reasoning": [f"Analysis error: {str(e)}"],
                "risk_assessment": "HIGH",
                "scores": {}
            }
    
    def _analyze_technical(self, market_analysis: MarketAnalysis) -> float:
        """Technical analysis scoring."""
        indicators = market_analysis.technical_indicators
        
        # RSI analysis
        rsi = indicators.get("rsi", 50)
        rsi_score = 0.5
        if 30 <= rsi <= 70:
            rsi_score = 0.6
        elif rsi < 30:  # Oversold
            rsi_score = 0.8
        elif rsi > 70:  # Overbought
            rsi_score = 0.2
        
        # MACD analysis
        macd = indicators.get("macd", 0)
        macd_score = 0.5
        if macd > 0:
            macd_score = 0.7
        else:
            macd_score = 0.3
        
        # Moving average analysis
        sma_20 = indicators.get("sma_20", 0)
        sma_50 = indicators.get("sma_50", 0)
        current_price = market_analysis.current_price
        
        ma_score = 0.5
        if current_price > sma_20 > sma_50:
            ma_score = 0.8
        elif current_price < sma_20 < sma_50:
            ma_score = 0.2
        elif current_price > sma_20:
            ma_score = 0.6
        else:
            ma_score = 0.4
        
        # Bollinger Bands analysis
        bb_position = indicators.get("bollinger_position", 0.5)
        bb_score = 0.5
        if bb_position < 0.2:  # Near lower band
            bb_score = 0.8
        elif bb_position > 0.8:  # Near upper band
            bb_score = 0.2
        else:
            bb_score = 0.5
        
        # Volume analysis
        volume_ratio = indicators.get("volume_ratio", 1.0)
        volume_score = 0.5
        if volume_ratio > 1.5:  # High volume
            volume_score = 0.7
        elif volume_ratio < 0.5:  # Low volume
            volume_score = 0.3
        else:
            volume_score = 0.5
        
        # Weighted technical score
        weights = [0.2, 0.2, 0.3, 0.15, 0.15]  # RSI, MACD, MA, BB, Volume
        scores = [rsi_score, macd_score, ma_score, bb_score, volume_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def _analyze_sentiment(self, market_analysis: MarketAnalysis) -> float:
        """Sentiment analysis scoring."""
        sentiment_score = market_analysis.sentiment_score
        news_impact = market_analysis.news_impact
        
        # Normalize sentiment score (-1 to 1) to (0 to 1)
        normalized_sentiment = (sentiment_score + 1) / 2
        
        # Combine sentiment and news impact
        combined_score = (normalized_sentiment * 0.7) + (news_impact * 0.3)
        
        return max(0.0, min(1.0, combined_score))
    
    def _analyze_fundamental(self, market_analysis: MarketAnalysis) -> float:
        """Fundamental analysis scoring."""
        fundamental_score = market_analysis.fundamental_score
        sector_performance = market_analysis.sector_performance
        
        # Normalize sector performance (-1 to 1) to (0 to 1)
        normalized_sector = (sector_performance + 1) / 2
        
        # Combine fundamental and sector performance
        combined_score = (fundamental_score * 0.8) + (normalized_sector * 0.2)
        
        return max(0.0, min(1.0, combined_score))
    
    def _analyze_risk(self, market_analysis: MarketAnalysis, position: Optional[Position] = None) -> float:
        """Risk analysis scoring."""
        volatility = market_analysis.volatility
        
        # Base risk score from volatility
        if volatility < 0.01:  # Low volatility
            risk_score = 0.8
        elif volatility < 0.03:  # Medium volatility
            risk_score = 0.6
        elif volatility < 0.05:  # High volatility
            risk_score = 0.4
        else:  # Very high volatility
            risk_score = 0.2
        
        # Adjust for position context if available
        if position:
            # Check if position is profitable
            current_pnl = (market_analysis.current_price - position.entry_price) / position.entry_price
            
            if current_pnl > 0.05:  # 5% profit
                risk_score += 0.1
            elif current_pnl < -0.05:  # 5% loss
                risk_score -= 0.1
            
            # Check holding period
            holding_days = (datetime.now() - position.entry_time).days
            if holding_days > 20:  # Long holding period
                risk_score -= 0.1
        
        return max(0.0, min(1.0, risk_score))
    
    def _log_model_predictions(self, symbol: str, model_scores: Dict[str, float], 
                              ensemble_confidence: float, analysis_type: str) -> None:
        """Log individual model predictions for performance tracking."""
        try:
            from ai.adaptive_weights import add_model_prediction
            
            # Convert model scores to probabilities (0-1 range)
            for model_name, score in model_scores.items():
                # Convert score to probability (assuming score is already 0-1)
                predicted_probability = max(0.0, min(1.0, score))
                
                # Log prediction (outcome will be updated later when trade is closed)
                add_model_prediction(
                    model_name=model_name,
                    symbol=symbol,
                    predicted_probability=predicted_probability,
                    actual_outcome="PENDING",  # Will be updated when trade closes
                    prediction_date=datetime.now()
                )
            
            # Also log ensemble prediction
            add_model_prediction(
                model_name="ensemble",
                symbol=symbol,
                predicted_probability=ensemble_confidence,
                actual_outcome="PENDING",
                prediction_date=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error logging model predictions: {e}")
    
    def update_model_performance(self, symbol: str, trade_outcome: str, 
                                trade_pnl: float = None, trade_id: str = None) -> None:
        """Update model performance when a trade is closed."""
        try:
            from ai.adaptive_weights import get_adaptive_weight_manager
            
            # Get the adaptive weight manager
            weight_manager = get_adaptive_weight_manager()
            
            # Update predictions with actual outcomes
            # This would typically be called when a position is closed
            # For now, we'll update the most recent predictions for this symbol
            
            # In a real implementation, you would:
            # 1. Find the specific predictions that led to this trade
            # 2. Update their outcomes based on the trade result
            # 3. Recalculate model weights
            
            logger.info(f"Updated model performance for {symbol}: outcome={trade_outcome}, pnl={trade_pnl}")
            
            # Update ensemble weights based on new performance data
            weight_manager.update_ensemble_weights()
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    def _analyze_market_regime(self, market_analysis: MarketAnalysis) -> float:
        """Market regime analysis scoring."""
        regime = market_analysis.market_regime
        
        regime_scores = {
            "BULL": 0.8,
            "BEAR": 0.2,
            "SIDEWAYS": 0.5,
            "HIGH_VOLATILITY": 0.3,
            "LOW_VOLATILITY": 0.7,
            "NEUTRAL": 0.5
        }
        
        return regime_scores.get(regime, 0.5)
    
    def _generate_reasoning(self, technical: float, sentiment: float, fundamental: float,
                          risk: float, regime: float, action: str, analysis_type: str) -> List[str]:
        """Generate human-readable reasoning for the decision."""
        reasoning = []
        
        if analysis_type == "ENTRY":
            if action == "BUY":
                reasoning.append("Strong buy signal detected")
                if technical > 0.7:
                    reasoning.append("Technical indicators show bullish momentum")
                if sentiment > 0.7:
                    reasoning.append("Positive market sentiment and news flow")
                if fundamental > 0.7:
                    reasoning.append("Strong fundamental metrics")
                if risk > 0.6:
                    reasoning.append("Acceptable risk levels")
            else:
                reasoning.append("Hold position - insufficient buy signals")
                if technical < 0.5:
                    reasoning.append("Technical indicators are mixed or bearish")
                if sentiment < 0.5:
                    reasoning.append("Negative or neutral sentiment")
        else:  # EXIT
            if action == "SELL":
                reasoning.append("Sell signal triggered")
                if technical < 0.3:
                    reasoning.append("Technical indicators turning bearish")
                if sentiment < 0.3:
                    reasoning.append("Negative sentiment and news impact")
                if risk < 0.4:
                    reasoning.append("Risk levels becoming unacceptable")
            else:
                reasoning.append("Hold position - no exit signals")
                if technical > 0.5:
                    reasoning.append("Technical indicators remain positive")
                if sentiment > 0.5:
                    reasoning.append("Sentiment remains favorable")
        
        return reasoning
    
    def _assess_risk(self, weighted_score: float, volatility: float) -> str:
        """Assess overall risk level."""
        if weighted_score > 0.7 and volatility < 0.03:
            return "LOW"
        elif weighted_score > 0.5 and volatility < 0.05:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _calculate_technical_score(self, indicators: Dict[str, float]) -> float:
        """Calculate overall technical score."""
        # This is a simplified version - in production, this would be more sophisticated
        rsi = indicators.get("rsi", 50)
        macd = indicators.get("macd", 0)
        
        # Simple scoring based on RSI and MACD
        rsi_score = 0.5
        if 30 <= rsi <= 70:
            rsi_score = 0.6
        elif rsi < 30:
            rsi_score = 0.8
        elif rsi > 70:
            rsi_score = 0.2
        
        macd_score = 0.7 if macd > 0 else 0.3
        
        return (rsi_score + macd_score) / 2
    
    def _calculate_stop_loss(self, entry_price: float, volatility: float) -> float:
        """Calculate stop loss price."""
        # Use 2x ATR or 2% of price, whichever is larger
        atr_stop = entry_price * (1 - (volatility * 2))
        percent_stop = entry_price * 0.98
        
        return max(atr_stop, percent_stop)
    
    def _calculate_take_profit(self, entry_price: float, stop_loss: float) -> float:
        """Calculate take profit price."""
        # 1.5:1 risk/reward ratio
        risk = entry_price - stop_loss
        return entry_price + (risk * 1.5)
    
    def _calculate_position_size(self, confidence: float, volatility: float, 
                               entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on confidence and risk."""
        # Base position size calculation
        risk_per_trade = 0.02  # 2% risk per trade
        account_balance = 10000.0  # Would be from database in production
        
        risk_amount = account_balance * risk_per_trade
        position_risk = entry_price - stop_loss
        
        if position_risk > 0:
            base_shares = risk_amount / position_risk
            # Adjust for confidence
            confidence_adjusted = base_shares * confidence
            # Adjust for volatility
            volatility_adjusted = confidence_adjusted * (1 - volatility)
            
            return max(1, int(volatility_adjusted))
        
        return 1
    
    def _determine_exit_reason(self, decision: Dict[str, Any], position: Position, 
                             market_analysis: MarketAnalysis) -> ExitReason:
        """Determine the specific exit reason."""
        scores = decision.get("scores", {})
        
        # Check for stop loss trigger
        if position.stop_loss and market_analysis.current_price <= position.stop_loss:
            return ExitReason.STOP_LOSS
        
        # Check for take profit trigger
        if position.take_profit and market_analysis.current_price >= position.take_profit:
            return ExitReason.TAKE_PROFIT
        
        # Check for time-based exit
        holding_days = (datetime.now() - position.entry_time).days
        if holding_days >= 30:
            return ExitReason.TIME_BASED
        
        # Check for AI reversal
        if scores.get("technical", 0.5) < 0.3 or scores.get("sentiment", 0.5) < 0.3:
            return ExitReason.AI_REVERSAL
        
        # Default to AI reversal
        return ExitReason.AI_REVERSAL
    
    def _get_market_features(self, symbol: str) -> Dict[str, Any]:
        """Get market features for analysis."""
        # This would integrate with real market data in production
        # For now, return simulated data
        import random
        
        return {
            "current_price": 150.0 + random.uniform(-10, 10),
            "rsi": random.uniform(20, 80),
            "macd": random.uniform(-2, 2),
            "sma_20": 150.0 + random.uniform(-5, 5),
            "sma_50": 150.0 + random.uniform(-8, 8),
            "bollinger_position": random.uniform(0, 1),
            "volume_ratio": random.uniform(0.5, 2.0),
            "atr": random.uniform(0.01, 0.05),
            "sentiment_score": random.uniform(-1, 1),
            "fundamental_score": random.uniform(0.3, 0.8),
            "market_regime": random.choice(["BULL", "BEAR", "SIDEWAYS", "NEUTRAL"]),
            "sector_performance": random.uniform(-0.1, 0.1),
            "news_impact": random.uniform(-0.5, 0.5),
            "volatility": random.uniform(0.01, 0.05),
            "volume_trend": random.choice(["HIGH", "NORMAL", "LOW"])
        }
    
    def get_mode_context(self, mode: str) -> Dict[str, Any]:
        """Get mode-specific context for analysis."""
        return {
            "mode": mode,
            "is_live": mode == "LIVE",
            "is_demo": mode == "DEMO",
            "risk_tolerance": 0.75 if mode == "LIVE" else 0.6,
            "confidence_threshold": 0.75 if mode == "LIVE" else 0.7,
            "max_positions": 10,
            "description": f"Analysis context for {mode} mode"
        }
    
    def validate_ai_response(self, response: Dict[str, Any]) -> bool:
        """Validate AI response structure and values."""
        try:
            # Check required fields
            required_fields = ["action", "confidence", "reasoning"]
            for field in required_fields:
                if field not in response:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate action
            if response["action"] not in ["BUY", "SELL", "HOLD"]:
                logger.error(f"Invalid action: {response['action']}")
                return False
            
            # Validate confidence
            confidence = response["confidence"]
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                logger.error(f"Invalid confidence: {confidence}")
                return False
            
            # Validate reasoning
            reasoning = response["reasoning"]
            if not isinstance(reasoning, list) or len(reasoning) == 0:
                logger.error(f"Invalid reasoning: {reasoning}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating AI response: {e}")
            return False

# Global enhanced ensemble instance
_enhanced_ensemble: Optional[EnhancedEnsemble] = None

def get_enhanced_ensemble() -> EnhancedEnsemble:
    """Get the global enhanced ensemble instance."""
    global _enhanced_ensemble
    if _enhanced_ensemble is None:
        _enhanced_ensemble = EnhancedEnsemble()
    return _enhanced_ensemble

def analyze_for_entry(symbol: str, features: Dict[str, Any], 
                     mode: Optional[str] = None) -> Optional[BuySignal]:
    """Analyze market data for entry signals."""
    return get_enhanced_ensemble().analyze_for_entry(symbol, features, mode)

def analyze_for_exit(position: Position, current_data: Dict[str, Any], 
                    mode: Optional[str] = None) -> Optional[SellSignal]:
    """Analyze position for exit signals."""
    return get_enhanced_ensemble().analyze_for_exit(position, current_data, mode)

def generate_position_aware_analysis(symbol: str, is_owned: bool, 
                                   mode: Optional[str] = None) -> Dict[str, Any]:
    """Generate position-aware analysis."""
    return get_enhanced_ensemble().generate_position_aware_analysis(symbol, is_owned, mode)

def update_model_performance(symbol: str, trade_outcome: str, 
                           trade_pnl: float = None, trade_id: str = None) -> None:
    """Update model performance when a trade is closed."""
    return get_enhanced_ensemble().update_model_performance(symbol, trade_outcome, trade_pnl, trade_id)
