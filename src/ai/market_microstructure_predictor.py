"""Market Microstructure Prediction for Trading Optimization"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    """Order book snapshot data."""
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]  # (price, quantity)
    last_price: float
    volume: float

@dataclass
class MicrostructureMetrics:
    """Market microstructure metrics."""
    spread: float
    mid_price: float
    bid_ask_imbalance: float
    order_flow_imbalance: float
    price_impact: float
    market_depth: float
    volatility: float
    liquidity_score: float
    timestamp: datetime

@dataclass
class MicrostructurePrediction:
    """Market microstructure prediction."""
    predicted_spread: float
    predicted_liquidity: float
    predicted_price_impact: float
    predicted_volatility: float
    confidence: float
    time_horizon: int  # minutes
    recommendations: Dict[str, Any]
    timestamp: datetime

class OrderBookAnalyzer:
    """Analyzes order book data for microstructure insights."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.order_book_history = deque(maxlen=max_history)
        self.metrics_history = deque(maxlen=max_history)
        
    def add_order_book_snapshot(self, snapshot: OrderBookSnapshot):
        """Add order book snapshot for analysis."""
        self.order_book_history.append(snapshot)
        
        # Calculate metrics
        metrics = self._calculate_microstructure_metrics(snapshot)
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_microstructure_metrics(self, snapshot: OrderBookSnapshot) -> MicrostructureMetrics:
        """Calculate microstructure metrics from order book snapshot."""
        
        # Calculate spread
        if snapshot.bids and snapshot.asks:
            best_bid = max(snapshot.bids, key=lambda x: x[0])[0]
            best_ask = min(snapshot.asks, key=lambda x: x[0])[0]
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2.0
        else:
            spread = 0.0
            mid_price = snapshot.last_price
        
        # Calculate bid-ask imbalance
        if snapshot.bids and snapshot.asks:
            total_bid_volume = sum(qty for _, qty in snapshot.bids)
            total_ask_volume = sum(qty for _, qty in snapshot.asks)
            if total_bid_volume + total_ask_volume > 0:
                bid_ask_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            else:
                bid_ask_imbalance = 0.0
        else:
            bid_ask_imbalance = 0.0
        
        # Calculate order flow imbalance (simplified)
        order_flow_imbalance = bid_ask_imbalance  # Simplified for now
        
        # Calculate price impact (simplified)
        price_impact = spread / mid_price if mid_price > 0 else 0.0
        
        # Calculate market depth
        if snapshot.bids and snapshot.asks:
            # Average depth within 1% of mid price
            depth_threshold = mid_price * 0.01
            bid_depth = sum(qty for price, qty in snapshot.bids if abs(price - mid_price) <= depth_threshold)
            ask_depth = sum(qty for price, qty in snapshot.asks if abs(price - mid_price) <= depth_threshold)
            market_depth = (bid_depth + ask_depth) / 2.0
        else:
            market_depth = 0.0
        
        # Calculate volatility (from recent price changes)
        volatility = self._calculate_volatility()
        
        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(spread, market_depth, volatility)
        
        return MicrostructureMetrics(
            spread=spread,
            mid_price=mid_price,
            bid_ask_imbalance=bid_ask_imbalance,
            order_flow_imbalance=order_flow_imbalance,
            price_impact=price_impact,
            market_depth=market_depth,
            volatility=volatility,
            liquidity_score=liquidity_score,
            timestamp=snapshot.timestamp
        )
    
    def _calculate_volatility(self) -> float:
        """Calculate volatility from recent price changes."""
        if len(self.order_book_history) < 2:
            return 0.0
        
        recent_prices = [snapshot.last_price for snapshot in list(self.order_book_history)[-20:]]
        if len(recent_prices) < 2:
            return 0.0
        
        price_changes = np.diff(recent_prices)
        return np.std(price_changes) if len(price_changes) > 0 else 0.0
    
    def _calculate_liquidity_score(self, spread: float, depth: float, volatility: float) -> float:
        """Calculate liquidity score (higher is better)."""
        if spread <= 0 or volatility <= 0:
            return 0.0
        
        # Higher depth and lower spread/volatility = better liquidity
        depth_score = min(depth / 1000000, 1.0)  # Normalize depth
        spread_score = max(0, 1.0 - spread / 0.01)  # Penalize wide spreads
        volatility_score = max(0, 1.0 - volatility / 0.05)  # Penalize high volatility
        
        return (depth_score + spread_score + volatility_score) / 3.0

class LiquidityPredictor:
    """Predicts market liquidity conditions."""
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.liquidity_history = deque(maxlen=lookback_window)
        self.feature_weights = {
            'spread': -0.3,
            'depth': 0.4,
            'volatility': -0.2,
            'volume': 0.3,
            'time_of_day': 0.1
        }
        
    def add_liquidity_sample(self, metrics: MicrostructureMetrics, volume: float):
        """Add liquidity sample for prediction."""
        sample = {
            'spread': metrics.spread,
            'depth': metrics.market_depth,
            'volatility': metrics.volatility,
            'volume': volume,
            'time_of_day': metrics.timestamp.hour,
            'liquidity_score': metrics.liquidity_score,
            'timestamp': metrics.timestamp
        }
        self.liquidity_history.append(sample)
    
    def predict_liquidity(self, time_horizon_minutes: int = 5) -> Dict[str, Any]:
        """Predict liquidity for given time horizon."""
        if len(self.liquidity_history) < 10:
            return {
                'predicted_liquidity': 0.5,
                'confidence': 0.0,
                'factors': {},
                'recommendations': ['Insufficient data for prediction']
            }
        
        # Get recent samples
        recent_samples = list(self.liquidity_history)[-20:]
        
        # Calculate feature values
        avg_spread = np.mean([s['spread'] for s in recent_samples])
        avg_depth = np.mean([s['depth'] for s in recent_samples])
        avg_volatility = np.mean([s['volatility'] for s in recent_samples])
        avg_volume = np.mean([s['volume'] for s in recent_samples])
        current_hour = datetime.now().hour
        
        # Calculate liquidity score
        liquidity_score = (
            self.feature_weights['spread'] * (1.0 - min(avg_spread / 0.01, 1.0)) +
            self.feature_weights['depth'] * min(avg_depth / 1000000, 1.0) +
            self.feature_weights['volatility'] * (1.0 - min(avg_volatility / 0.05, 1.0)) +
            self.feature_weights['volume'] * min(avg_volume / 10000000, 1.0) +
            self.feature_weights['time_of_day'] * (1.0 - abs(current_hour - 14) / 14.0)  # Peak at 2 PM
        )
        
        # Normalize to 0-1 range
        predicted_liquidity = max(0.0, min(1.0, (liquidity_score + 1.0) / 2.0))
        
        # Calculate confidence based on data consistency
        liquidity_scores = [s['liquidity_score'] for s in recent_samples]
        confidence = 1.0 - np.std(liquidity_scores) if len(liquidity_scores) > 1 else 0.5
        
        # Generate recommendations
        recommendations = []
        if predicted_liquidity > 0.7:
            recommendations.append("High liquidity expected - good for large orders")
        elif predicted_liquidity > 0.4:
            recommendations.append("Moderate liquidity - consider order splitting")
        else:
            recommendations.append("Low liquidity expected - use limit orders")
        
        if avg_spread > 0.005:
            recommendations.append("Wide spreads expected - consider market making")
        
        if avg_volatility > 0.03:
            recommendations.append("High volatility - reduce position sizes")
        
        return {
            'predicted_liquidity': predicted_liquidity,
            'confidence': confidence,
            'factors': {
                'spread_impact': avg_spread,
                'depth_impact': avg_depth,
                'volatility_impact': avg_volatility,
                'volume_impact': avg_volume,
                'time_impact': current_hour
            },
            'recommendations': recommendations
        }

class PriceImpactPredictor:
    """Predicts price impact of trades."""
    
    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self.trade_history = deque(maxlen=max_history)
        self.impact_model = None
        
    def add_trade(self, trade_size: float, price_impact: float, market_conditions: Dict[str, Any]):
        """Add trade data for impact prediction."""
        trade_data = {
            'size': trade_size,
            'price_impact': price_impact,
            'market_conditions': market_conditions,
            'timestamp': datetime.now()
        }
        self.trade_history.append(trade_data)
    
    def predict_price_impact(self, trade_size: float, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Predict price impact for a given trade size."""
        if len(self.trade_history) < 10:
            # Simple linear model as fallback
            base_impact = trade_size * 0.0001  # 0.01% per 1000 shares
            return {
                'predicted_impact': base_impact,
                'confidence': 0.3,
                'method': 'linear_fallback',
                'recommendations': ['Insufficient data - use conservative estimates']
            }
        
        # Get similar trades
        similar_trades = self._find_similar_trades(trade_size, market_conditions)
        
        if similar_trades:
            # Use historical similar trades
            impacts = [trade['price_impact'] for trade in similar_trades]
            predicted_impact = np.mean(impacts)
            confidence = 1.0 - np.std(impacts) / (np.mean(impacts) + 1e-8)
        else:
            # Use size-based estimation
            sizes = [trade['size'] for trade in self.trade_history]
            impacts = [trade['price_impact'] for trade in self.trade_history]
            
            # Simple linear regression
            if len(sizes) > 1:
                correlation = np.corrcoef(sizes, impacts)[0, 1]
                if not np.isnan(correlation) and correlation > 0:
                    slope = correlation * np.std(impacts) / np.std(sizes)
                    predicted_impact = slope * trade_size
                else:
                    predicted_impact = trade_size * 0.0001
            else:
                predicted_impact = trade_size * 0.0001
            
            confidence = 0.5
        
        # Generate recommendations
        recommendations = []
        if predicted_impact > 0.01:  # > 1%
            recommendations.append("High price impact expected - consider order splitting")
        elif predicted_impact > 0.005:  # > 0.5%
            recommendations.append("Moderate price impact - use TWAP/VWAP strategies")
        else:
            recommendations.append("Low price impact - market orders acceptable")
        
        return {
            'predicted_impact': predicted_impact,
            'confidence': confidence,
            'method': 'historical_analysis',
            'recommendations': recommendations
        }
    
    def _find_similar_trades(self, trade_size: float, market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find trades with similar size and market conditions."""
        similar_trades = []
        size_tolerance = trade_size * 0.2  # 20% size tolerance
        
        for trade in self.trade_history:
            # Check size similarity
            if abs(trade['size'] - trade_size) <= size_tolerance:
                # Check market condition similarity
                if self._conditions_similar(trade['market_conditions'], market_conditions):
                    similar_trades.append(trade)
        
        return similar_trades
    
    def _conditions_similar(self, conditions1: Dict[str, Any], conditions2: Dict[str, Any]) -> bool:
        """Check if market conditions are similar."""
        # Simple similarity check
        regime_match = conditions1.get('regime') == conditions2.get('regime')
        volatility_match = abs(conditions1.get('volatility', 0) - conditions2.get('volatility', 0)) < 0.01
        
        return regime_match and volatility_match

class MarketMicrostructurePredictor:
    """Main predictor for market microstructure."""
    
    def __init__(self):
        self.order_book_analyzer = OrderBookAnalyzer()
        self.liquidity_predictor = LiquidityPredictor()
        self.price_impact_predictor = PriceImpactPredictor()
        self.prediction_history = deque(maxlen=1000)
        
    def add_market_data(self, order_book_data: pd.DataFrame, trade_data: pd.DataFrame = None):
        """Add market data for analysis."""
        for _, row in order_book_data.iterrows():
            # Create order book snapshot
            snapshot = OrderBookSnapshot(
                timestamp=row.get('timestamp', datetime.now()),
                bids=self._parse_order_book_levels(row.get('bids', [])),
                asks=self._parse_order_book_levels(row.get('asks', [])),
                last_price=row.get('last_price', 0.0),
                volume=row.get('volume', 0.0)
            )
            
            # Analyze order book
            metrics = self.order_book_analyzer.add_order_book_snapshot(snapshot)
            
            # Add to liquidity predictor
            self.liquidity_predictor.add_liquidity_sample(metrics, snapshot.volume)
        
        # Process trade data if available
        if trade_data is not None:
            for _, row in trade_data.iterrows():
                trade_size = row.get('size', 0.0)
                price_impact = row.get('price_impact', 0.0)
                market_conditions = {
                    'regime': row.get('regime', 'unknown'),
                    'volatility': row.get('volatility', 0.0)
                }
                
                self.price_impact_predictor.add_trade(trade_size, price_impact, market_conditions)
    
    def _parse_order_book_levels(self, levels_data: List) -> List[Tuple[float, float]]:
        """Parse order book levels from data."""
        if not levels_data:
            return []
        
        levels = []
        for level in levels_data:
            if isinstance(level, dict) and 'price' in level and 'quantity' in level:
                levels.append((level['price'], level['quantity']))
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                levels.append((float(level[0]), float(level[1])))
        
        return levels
    
    def predict_microstructure(self, time_horizon_minutes: int = 5, 
                             trade_size: float = 1000.0,
                             market_conditions: Dict[str, Any] = None) -> MicrostructurePrediction:
        """Predict market microstructure conditions."""
        
        if market_conditions is None:
            market_conditions = {
                'regime': 'unknown',
                'volatility': 0.02,
                'liquidity': 'medium'
            }
        
        # Predict liquidity
        liquidity_prediction = self.liquidity_predictor.predict_liquidity(time_horizon_minutes)
        
        # Predict price impact
        impact_prediction = self.price_impact_predictor.predict_price_impact(trade_size, market_conditions)
        
        # Predict spread (simplified)
        if self.order_book_analyzer.metrics_history:
            recent_spreads = [m.spread for m in list(self.order_book_analyzer.metrics_history)[-10:]]
            predicted_spread = np.mean(recent_spreads) if recent_spreads else 0.001
        else:
            predicted_spread = 0.001
        
        # Predict volatility
        if self.order_book_analyzer.metrics_history:
            recent_volatilities = [m.volatility for m in list(self.order_book_analyzer.metrics_history)[-10:]]
            predicted_volatility = np.mean(recent_volatilities) if recent_volatilities else 0.02
        else:
            predicted_volatility = 0.02
        
        # Calculate overall confidence
        confidence = (liquidity_prediction['confidence'] + impact_prediction['confidence']) / 2.0
        
        # Generate recommendations
        recommendations = {
            'liquidity': liquidity_prediction['recommendations'],
            'price_impact': impact_prediction['recommendations'],
            'execution_strategy': self._recommend_execution_strategy(
                liquidity_prediction['predicted_liquidity'],
                impact_prediction['predicted_impact'],
                predicted_volatility
            )
        }
        
        prediction = MicrostructurePrediction(
            predicted_spread=predicted_spread,
            predicted_liquidity=liquidity_prediction['predicted_liquidity'],
            predicted_price_impact=impact_prediction['predicted_impact'],
            predicted_volatility=predicted_volatility,
            confidence=confidence,
            time_horizon=time_horizon_minutes,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
        # Store prediction
        self.prediction_history.append(prediction)
        
        return prediction
    
    def _recommend_execution_strategy(self, liquidity: float, price_impact: float, volatility: float) -> List[str]:
        """Recommend execution strategy based on predictions."""
        recommendations = []
        
        if liquidity > 0.7 and price_impact < 0.005:
            recommendations.append("Market orders recommended - high liquidity, low impact")
        elif liquidity > 0.4:
            recommendations.append("TWAP strategy recommended - moderate liquidity")
        else:
            recommendations.append("VWAP or limit orders recommended - low liquidity")
        
        if volatility > 0.03:
            recommendations.append("Reduce position sizes due to high volatility")
        
        if price_impact > 0.01:
            recommendations.append("Consider order splitting to reduce impact")
        
        return recommendations
    
    def get_microstructure_statistics(self) -> Dict[str, Any]:
        """Get microstructure analysis statistics."""
        stats = {
            'total_order_book_snapshots': len(self.order_book_analyzer.order_book_history),
            'total_metrics_calculated': len(self.order_book_analyzer.metrics_history),
            'total_predictions_made': len(self.prediction_history),
            'liquidity_samples': len(self.liquidity_predictor.liquidity_history),
            'trade_samples': len(self.price_impact_predictor.trade_history)
        }
        
        # Recent performance metrics
        if self.order_book_analyzer.metrics_history:
            recent_metrics = list(self.order_book_analyzer.metrics_history)[-10:]
            stats['recent_avg_spread'] = np.mean([m.spread for m in recent_metrics])
            stats['recent_avg_liquidity'] = np.mean([m.liquidity_score for m in recent_metrics])
            stats['recent_avg_volatility'] = np.mean([m.volatility for m in recent_metrics])
        
        return stats

# Global instance
_microstructure_predictor = None

def get_microstructure_predictor() -> MarketMicrostructurePredictor:
    """Get the global microstructure predictor instance."""
    global _microstructure_predictor
    if _microstructure_predictor is None:
        _microstructure_predictor = MarketMicrostructurePredictor()
    return _microstructure_predictor