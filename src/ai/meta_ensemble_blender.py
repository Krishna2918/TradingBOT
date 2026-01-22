"""
Meta-Ensemble Blender - Deterministic Fusion Logic
Production-ready decision fusion with hard risk clamps
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class PredictionSignal:
    """Individual prediction signal"""
    symbol: str
    direction: float  # -1 to 1 (negative = sell, positive = buy)
    confidence: float  # 0 to 1
    source: str  # 'lstm', 'gru', 'ppo', 'dqn'
    timestamp: str

@dataclass
class BlendedDecision:
    """Final blended trading decision"""
    symbol: str
    action: TradeAction
    score: float  # 0 to 1
    confidence: float  # 0 to 1
    position_size_pre_risk: float  # 0 to 1
    position_size_post_risk: float  # 0 to 1
    reasoning: List[str]
    risk_adjustments: List[str]
    timestamp: str

class MetaEnsembleBlender:
    """
    Meta-Ensemble Blender - Deterministic Fusion Logic
    
    Implements the exact fusion formula:
    base = 0.4*p_short + 0.3*p_mid + 0.3*sigmoid(PPO_size_hint)
    adjust = - penalty_if_vol_spike - penalty_if_news_neg
    score = clamp(base + adjust, 0, 1)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Base weights (can be adjusted by GPT-5)
        self.base_weights = {
            'short_term': 0.4,  # LSTM
            'mid_term': 0.3,    # GRU/Transformer
            'rl_signal': 0.3    # PPO/DQN
        }
        
        # Risk adjustment factors
        self.risk_factors = {
            'vol_spike_penalty': 0.15,  # Penalty for volatility spikes
            'news_negative_penalty': 0.10,  # Penalty for negative news
            'liquidity_penalty': 0.05,  # Penalty for low liquidity
            'correlation_penalty': 0.08  # Penalty for correlation breakdown
        }
        
        # Decision thresholds
        self.decision_thresholds = {
            'buy_threshold': 0.67,  # score >= 0.67 -> Long bias
            'sell_threshold': 0.33,  # score <= 0.33 -> Short bias
            'hold_range': (0.33, 0.67)  # else -> Hold
        }
        
        # Position sizing parameters
        self.position_sizing = {
            'base_multiplier': 2.0,  # 2*|score-0.5|
            'max_position': 0.15,  # 15% max position
            'min_position': 0.01   # 1% min position
        }
        
        # Risk monitoring
        self.market_state = {
            'volatility_spike': False,
            'news_sentiment': 0.0,  # -1 to 1
            'liquidity_score': 1.0,  # 0 to 1
            'correlation_breakdown': False
        }
        
        logger.info("Meta-Ensemble Blender initialized")
        logger.info(f"Base weights: {self.base_weights}")
        logger.info(f"Decision thresholds: {self.decision_thresholds}")
    
    def update_market_state(self, market_state: Dict[str, Any]):
        """Update market state for risk adjustments"""
        self.market_state.update(market_state)
        
        logger.info(f" Market state updated: {self.market_state}")
    
    def blend_predictions(self, predictions: List[PredictionSignal]) -> Dict[str, BlendedDecision]:
        """
        Blend predictions using deterministic fusion logic
        
        Args:
            predictions: List of prediction signals from different models
            
        Returns:
            Dictionary of blended decisions by symbol
        """
        
        # Group predictions by symbol
        symbol_predictions = {}
        for pred in predictions:
            if pred.symbol not in symbol_predictions:
                symbol_predictions[pred.symbol] = []
            symbol_predictions[pred.symbol].append(pred)
        
        blended_decisions = {}
        
        for symbol, symbol_preds in symbol_predictions.items():
            try:
                decision = self._blend_symbol_predictions(symbol, symbol_preds)
                blended_decisions[symbol] = decision
                
                logger.info(f" {symbol}: {decision.action.value} (score: {decision.score:.3f}, confidence: {decision.confidence:.3f})")
                
            except Exception as e:
                logger.error(f" Failed to blend predictions for {symbol}: {e}")
                # Create conservative hold decision
                blended_decisions[symbol] = self._create_hold_decision(symbol)
        
        return blended_decisions
    
    def _blend_symbol_predictions(self, symbol: str, predictions: List[PredictionSignal]) -> BlendedDecision:
        """Blend predictions for a single symbol"""
        
        # Separate predictions by source
        short_term_preds = [p for p in predictions if p.source in ['lstm', 'short_term']]
        mid_term_preds = [p for p in predictions if p.source in ['gru', 'transformer', 'mid_term']]
        rl_preds = [p for p in predictions if p.source in ['ppo', 'dqn', 'rl']]
        
        # Calculate base score components
        short_score = self._calculate_component_score(short_term_preds)
        mid_score = self._calculate_component_score(mid_term_preds)
        rl_score = self._calculate_component_score(rl_preds)
        
        # Apply sigmoid to RL signal
        rl_sigmoid = self._sigmoid(rl_score)
        
        # Calculate base score
        base_score = (
            self.base_weights['short_term'] * short_score +
            self.base_weights['mid_term'] * mid_score +
            self.base_weights['rl_signal'] * rl_sigmoid
        )
        
        # Calculate risk adjustments
        risk_adjustments = self._calculate_risk_adjustments(symbol)
        total_adjustment = sum(risk_adjustments.values())
        
        # Final score with risk adjustments
        final_score = np.clip(base_score - total_adjustment, 0.0, 1.0)
        
        # Determine action
        action = self._determine_action(final_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(predictions, final_score)
        
        # Calculate position size
        position_size_pre = self._calculate_position_size(final_score)
        position_size_post = self._apply_risk_clamps(symbol, position_size_pre)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            short_score, mid_score, rl_score, risk_adjustments, final_score
        )
        
        return BlendedDecision(
            symbol=symbol,
            action=action,
            score=final_score,
            confidence=confidence,
            position_size_pre_risk=position_size_pre,
            position_size_post_risk=position_size_post,
            reasoning=reasoning,
            risk_adjustments=list(risk_adjustments.keys()),
            timestamp=pd.Timestamp.now().isoformat()
        )
    
    def _calculate_component_score(self, predictions: List[PredictionSignal]) -> float:
        """Calculate weighted average score for a component"""
        if not predictions:
            return 0.5  # Neutral if no predictions
        
        # Weight by confidence
        total_weight = sum(p.confidence for p in predictions)
        if total_weight == 0:
            return 0.5
        
        weighted_score = sum(p.direction * p.confidence for p in predictions) / total_weight
        
        # Convert from [-1, 1] to [0, 1]
        return (weighted_score + 1) / 2
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for RL signal"""
        return 1 / (1 + np.exp(-x))
    
    def _calculate_risk_adjustments(self, symbol: str) -> Dict[str, float]:
        """Calculate risk adjustments"""
        adjustments = {}
        
        # Volatility spike penalty
        if self.market_state['volatility_spike']:
            adjustments['vol_spike'] = self.risk_factors['vol_spike_penalty']
        
        # News sentiment penalty
        if self.market_state['news_sentiment'] < -0.3:  # Negative news
            adjustments['news_negative'] = self.risk_factors['news_negative_penalty']
        
        # Liquidity penalty
        if self.market_state['liquidity_score'] < 0.5:  # Low liquidity
            adjustments['low_liquidity'] = self.risk_factors['liquidity_penalty']
        
        # Correlation breakdown penalty
        if self.market_state['correlation_breakdown']:
            adjustments['correlation_breakdown'] = self.risk_factors['correlation_penalty']
        
        return adjustments
    
    def _determine_action(self, score: float) -> TradeAction:
        """Determine trading action based on score"""
        if score >= self.decision_thresholds['buy_threshold']:
            return TradeAction.BUY
        elif score <= self.decision_thresholds['sell_threshold']:
            return TradeAction.SELL
        else:
            return TradeAction.HOLD
    
    def _calculate_confidence(self, predictions: List[PredictionSignal], final_score: float) -> float:
        """Calculate overall confidence"""
        if not predictions:
            return 0.0
        
        # Average confidence weighted by source importance
        source_weights = {'lstm': 0.4, 'gru': 0.3, 'ppo': 0.2, 'dqn': 0.1}
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for pred in predictions:
            weight = source_weights.get(pred.source, 0.1)
            weighted_confidence += pred.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_confidence = weighted_confidence / total_weight
        
        # Adjust confidence based on score extremity
        score_extremity = abs(final_score - 0.5) * 2  # 0 to 1
        confidence_multiplier = 0.5 + 0.5 * score_extremity
        
        return min(base_confidence * confidence_multiplier, 1.0)
    
    def _calculate_position_size(self, score: float) -> float:
        """Calculate position size based on score"""
        # Position size = bucket_limit * (2 * |score - 0.5|)
        score_deviation = abs(score - 0.5)
        position_size = self.position_sizing['base_multiplier'] * score_deviation
        
        # Apply min/max bounds
        position_size = np.clip(
            position_size,
            self.position_sizing['min_position'],
            self.position_sizing['max_position']
        )
        
        return position_size
    
    def _apply_risk_clamps(self, symbol: str, position_size: float) -> float:
        """Apply risk clamps based on symbol characteristics"""
        
        # This would integrate with your risk management system
        # For now, apply basic clamps
        
        # Apply strictest clamp across all applicable categories
        applicable_clamps = []
        
        # Penny stock clamp (if symbol is penny stock)
        if self._is_penny_stock(symbol):
            applicable_clamps.append(0.004)  # 0.4% max for penny stocks
        
        # Core stock clamp
        if self._is_core_stock(symbol):
            applicable_clamps.append(0.015)  # 1.5% max for core stocks
        
        # F&O clamp (if symbol has options)
        if self._has_options(symbol):
            applicable_clamps.append(0.007)  # 0.7% max for F&O
        
        # Apply minimum (strictest) clamp if any apply
        if applicable_clamps:
            strictest_clamp = min(applicable_clamps)
            position_size = min(position_size, strictest_clamp)
            logger.debug(f"Applied strictest clamp {strictest_clamp:.4f} for {symbol} (from {applicable_clamps})")
        
        return position_size
    
    def _generate_reasoning(self, short_score: float, mid_score: float, rl_score: float,
                          risk_adjustments: Dict[str, float], final_score: float) -> List[str]:
        """Generate human-readable reasoning"""
        reasoning = []
        
        # Base component reasoning
        if short_score > 0.6:
            reasoning.append(f"Strong short-term signal ({short_score:.2f})")
        elif short_score < 0.4:
            reasoning.append(f"Weak short-term signal ({short_score:.2f})")
        
        if mid_score > 0.6:
            reasoning.append(f"Positive mid-term trend ({mid_score:.2f})")
        elif mid_score < 0.4:
            reasoning.append(f"Negative mid-term trend ({mid_score:.2f})")
        
        if rl_score > 0.6:
            reasoning.append(f"RL agent bullish ({rl_score:.2f})")
        elif rl_score < 0.4:
            reasoning.append(f"RL agent bearish ({rl_score:.2f})")
        
        # Risk adjustment reasoning
        for risk_type, adjustment in risk_adjustments.items():
            if adjustment > 0:
                reasoning.append(f"Risk penalty: {risk_type} (-{adjustment:.2f})")
        
        # Final score reasoning
        if final_score > 0.7:
            reasoning.append(f"Strong buy signal ({final_score:.2f})")
        elif final_score < 0.3:
            reasoning.append(f"Strong sell signal ({final_score:.2f})")
        else:
            reasoning.append(f"Neutral signal ({final_score:.2f})")
        
        return reasoning
    
    def _create_hold_decision(self, symbol: str) -> BlendedDecision:
        """Create conservative hold decision"""
        return BlendedDecision(
            symbol=symbol,
            action=TradeAction.HOLD,
            score=0.5,
            confidence=0.0,
            position_size_pre_risk=0.0,
            position_size_post_risk=0.0,
            reasoning=["Conservative hold - insufficient data"],
            risk_adjustments=[],
            timestamp=pd.Timestamp.now().isoformat()
        )
    
    # Helper methods for symbol classification
    def _is_penny_stock(self, symbol: str) -> bool:
        """Check if symbol is a penny stock"""
        # This would integrate with your symbol classification
        penny_suffixes = ['.V', '.TSXV']
        return any(symbol.endswith(suffix) for suffix in penny_suffixes)
    
    def _is_core_stock(self, symbol: str) -> bool:
        """Check if symbol is a core stock"""
        # This would integrate with your symbol classification
        core_suffixes = ['.TO', '.TSX']
        return any(symbol.endswith(suffix) for suffix in core_suffixes)
    
    def _has_options(self, symbol: str) -> bool:
        """Check if symbol has options available"""
        # This would integrate with your options data
        # For now, assume major TSX stocks have options
        major_stocks = ['TD.TO', 'RY.TO', 'SHOP.TO', 'CNR.TO', 'ENB.TO']
        return symbol in major_stocks
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update base weights (called by GPT-5)"""
        self.base_weights.update(new_weights)
        logger.info(f" Updated base weights: {self.base_weights}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current blender status"""
        return {
            'base_weights': self.base_weights.copy(),
            'decision_thresholds': self.decision_thresholds.copy(),
            'market_state': self.market_state.copy(),
            'position_sizing': self.position_sizing.copy()
        }


# Example usage
if __name__ == "__main__":
    config = {}
    blender = MetaEnsembleBlender(config)
    
    # Sample predictions
    predictions = [
        PredictionSignal('TD.TO', 0.7, 0.8, 'lstm', '2025-10-05T10:00:00'),
        PredictionSignal('TD.TO', 0.6, 0.7, 'gru', '2025-10-05T10:00:00'),
        PredictionSignal('TD.TO', 0.8, 0.9, 'ppo', '2025-10-05T10:00:00'),
        PredictionSignal('RY.TO', -0.3, 0.6, 'lstm', '2025-10-05T10:00:00'),
        PredictionSignal('RY.TO', -0.2, 0.5, 'gru', '2025-10-05T10:00:00'),
    ]
    
    # Update market state
    market_state = {
        'volatility_spike': False,
        'news_sentiment': 0.2,
        'liquidity_score': 0.8,
        'correlation_breakdown': False
    }
    blender.update_market_state(market_state)
    
    # Blend predictions
    decisions = blender.blend_predictions(predictions)
    
    # Print results
    for symbol, decision in decisions.items():
        print(f"\n{symbol}:")
        print(f"  Action: {decision.action.value}")
        print(f"  Score: {decision.score:.3f}")
        print(f"  Confidence: {decision.confidence:.3f}")
        print(f"  Position Size: {decision.position_size_post_risk:.3f}")
        print(f"  Reasoning: {', '.join(decision.reasoning)}")
