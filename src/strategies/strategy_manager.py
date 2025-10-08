"""
Strategy Manager
Coordinates all 5 trading strategies and manages position allocation
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import yaml

from .momentum_scalping import MomentumScalpingStrategy, MomentumSignal
from .news_volatility import NewsVolatilityStrategy, NewsSignal
from .gamma_oi_squeeze import GammaOISqueezeStrategy, GammaOISignal
from .arbitrage import ArbitrageStrategy, ArbitrageSignal
from .ai_ml_patterns import AIMLPatternStrategy, MLSignal

logger = logging.getLogger(__name__)

class StrategyManager:
    """Manages all trading strategies and coordinates their execution"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize all strategies
        self.strategies = {}
        self._initialize_strategies()
        
        # Strategy allocation
        self.strategy_allocations = {
            'momentum_scalping': 0.25,
            'news_volatility': 0.20,
            'gamma_oi_squeeze': 0.15,
            'arbitrage': 0.20,
            'ai_ml_patterns': 0.20
        }
        
        # Active positions across all strategies
        self.all_positions = {}
        self.signal_history = []
        self.performance_metrics = {}
    
    def _initialize_strategies(self):
        """Initialize all trading strategies"""
        try:
            strategy_configs = self.config.get('strategies', {})
            
            # Initialize each strategy
            self.strategies['momentum_scalping'] = MomentumScalpingStrategy(
                strategy_configs.get('momentum_scalping', {})
            )
            
            self.strategies['news_volatility'] = NewsVolatilityStrategy(
                strategy_configs.get('news_volatility', {})
            )
            
            self.strategies['gamma_oi_squeeze'] = GammaOISqueezeStrategy(
                strategy_configs.get('gamma_oi_squeeze', {})
            )
            
            self.strategies['arbitrage'] = ArbitrageStrategy(
                strategy_configs.get('arbitrage', {})
            )
            
            self.strategies['ai_ml_patterns'] = AIMLPatternStrategy(
                strategy_configs.get('ai_ml_patterns', {})
            )
            
            logger.info(f"Initialized {len(self.strategies)} trading strategies")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
    
    def analyze_market_conditions(self, market_data: Dict, news_data: List[Dict] = None, options_data: Dict = None) -> Dict[str, List]:
        """Analyze market conditions using all strategies"""
        try:
            all_signals = {}
            
            # Get signals from each strategy
            if 'momentum_scalping' in self.strategies:
                momentum_signals = self.strategies['momentum_scalping'].analyze_market_data(market_data)
                all_signals['momentum_scalping'] = momentum_signals
            
            if 'news_volatility' in self.strategies:
                news_signals = self.strategies['news_volatility'].analyze_news_and_volatility(news_data or [], market_data)
                all_signals['news_volatility'] = news_signals
            
            if 'gamma_oi_squeeze' in self.strategies:
                gamma_signals = self.strategies['gamma_oi_squeeze'].analyze_options_data(options_data or {}, market_data)
                all_signals['gamma_oi_squeeze'] = gamma_signals
            
            if 'arbitrage' in self.strategies:
                arbitrage_signals = self.strategies['arbitrage'].analyze_arbitrage_opportunities(market_data)
                all_signals['arbitrage'] = arbitrage_signals
            
            if 'ai_ml_patterns' in self.strategies:
                ml_signals = self.strategies['ai_ml_patterns'].analyze_ml_patterns(market_data, news_data)
                all_signals['ai_ml_patterns'] = ml_signals
            
            # Store signals in history
            self.signal_history.append({
                'timestamp': datetime.now(),
                'signals': all_signals,
                'market_data': market_data
            })
            
            # Keep only last 1000 signal sets
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            total_signals = sum(len(signals) for signals in all_signals.values())
            logger.info(f"Generated {total_signals} total signals across all strategies")
            
            return all_signals
            
        except Exception as e:
            logger.error(f"Failed to analyze market conditions: {e}")
            return {}
    
    def get_consolidated_signals(self, all_signals: Dict[str, List]) -> List[Dict]:
        """Consolidate signals from all strategies into unified format"""
        try:
            consolidated = []
            
            for strategy_name, signals in all_signals.items():
                for signal in signals:
                    # Convert signal to unified format
                    unified_signal = {
                        'strategy': strategy_name,
                        'symbol': getattr(signal, 'symbol', getattr(signal, 'symbol_pair', 'UNKNOWN')),
                        'side': signal.side,
                        'strength': signal.strength,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'confidence': signal.confidence,
                        'timestamp': signal.timestamp,
                        'allocation': self.strategy_allocations.get(strategy_name, 0.0)
                    }
                    
                    # Add strategy-specific data
                    if hasattr(signal, 'news_source'):
                        unified_signal['news_source'] = signal.news_source
                        unified_signal['sentiment_score'] = signal.sentiment_score
                    elif hasattr(signal, 'arbitrage_type'):
                        unified_signal['arbitrage_type'] = signal.arbitrage_type
                        unified_signal['spread_amount'] = signal.spread_amount
                    elif hasattr(signal, 'pattern_type'):
                        unified_signal['pattern_type'] = signal.pattern_type
                        unified_signal['feature_importance'] = signal.feature_importance
                    elif hasattr(signal, 'gamma_exposure'):
                        unified_signal['gamma_exposure'] = signal.gamma_exposure
                        unified_signal['oi_change'] = signal.oi_change
                    
                    consolidated.append(unified_signal)
            
            # Sort by confidence and strength
            consolidated.sort(key=lambda x: (x['confidence'] * x['strength']), reverse=True)
            
            return consolidated
            
        except Exception as e:
            logger.error(f"Failed to consolidate signals: {e}")
            return []
    
    def filter_signals(self, signals: List[Dict], max_signals: int = 10) -> List[Dict]:
        """Filter and rank signals based on quality and allocation"""
        try:
            # Filter by minimum confidence
            min_confidence = 0.6
            filtered_signals = [s for s in signals if s['confidence'] >= min_confidence]
            
            # Remove duplicate symbols (keep highest confidence)
            symbol_signals = {}
            for signal in filtered_signals:
                symbol = signal['symbol']
                if symbol not in symbol_signals or signal['confidence'] > symbol_signals[symbol]['confidence']:
                    symbol_signals[symbol] = signal
            
            # Convert back to list and sort
            final_signals = list(symbol_signals.values())
            final_signals.sort(key=lambda x: (x['confidence'] * x['strength']), reverse=True)
            
            # Limit to max signals
            return final_signals[:max_signals]
            
        except Exception as e:
            logger.error(f"Failed to filter signals: {e}")
            return []
    
    def calculate_position_sizes(self, signals: List[Dict], available_capital: float) -> List[Dict]:
        """Calculate position sizes for signals based on allocation and risk"""
        try:
            sized_signals = []
            
            for signal in signals:
                # Get strategy allocation
                strategy_allocation = signal['allocation']
                
                # Calculate base position size
                base_size = available_capital * strategy_allocation
                
                # Adjust based on confidence and strength
                confidence_multiplier = signal['confidence']
                strength_multiplier = signal['strength']
                
                position_size = base_size * confidence_multiplier * strength_multiplier
                
                # Apply risk management
                risk_per_trade = 0.02  # 2% risk per trade
                max_position_size = available_capital * risk_per_trade
                
                final_size = min(position_size, max_position_size)
                
                signal['position_size'] = final_size
                signal['risk_amount'] = abs(signal['entry_price'] - signal['stop_loss']) * (final_size / signal['entry_price'])
                
                sized_signals.append(signal)
            
            return sized_signals
            
        except Exception as e:
            logger.error(f"Failed to calculate position sizes: {e}")
            return []
    
    def update_positions(self, symbol: str, current_price: float, timestamp: datetime):
        """Update positions and check exit conditions"""
        try:
            positions_to_exit = []
            
            for strategy_name, strategy in self.strategies.items():
                if hasattr(strategy, 'should_exit_position'):
                    if strategy.should_exit_position(symbol, current_price, timestamp):
                        positions_to_exit.append({
                            'strategy': strategy_name,
                            'symbol': symbol,
                            'exit_price': current_price,
                            'exit_time': timestamp
                        })
            
            return positions_to_exit
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
            return []
    
    def get_strategy_performance(self) -> Dict:
        """Get performance metrics for all strategies"""
        try:
            performance = {}
            
            for strategy_name, strategy in self.strategies.items():
                status = strategy.get_strategy_status()
                performance[strategy_name] = {
                    'enabled': status.get('enabled', False),
                    'allocation': status.get('allocation', 0.0),
                    'active_positions': status.get('active_positions', 0),
                    'signals_generated': status.get('signals_generated', 0),
                    'max_leverage': status.get('max_leverage', 1.0)
                }
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return {}
    
    def get_total_positions(self) -> int:
        """Get total number of active positions across all strategies"""
        try:
            total = 0
            for strategy in self.strategies.values():
                if hasattr(strategy, 'active_positions'):
                    total += len(strategy.active_positions)
            return total
            
        except Exception as e:
            logger.error(f"Failed to get total positions: {e}")
            return 0
    
    def get_strategy_summary(self) -> Dict:
        """Get comprehensive strategy summary"""
        try:
            return {
                'total_strategies': len(self.strategies),
                'enabled_strategies': sum(1 for s in self.strategies.values() if s.enabled),
                'total_positions': self.get_total_positions(),
                'total_signals': len(self.signal_history),
                'strategy_allocations': self.strategy_allocations,
                'performance': self.get_strategy_performance(),
                'last_analysis': self.signal_history[-1]['timestamp'] if self.signal_history else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get strategy summary: {e}")
            return {}

# Test the strategy manager
if __name__ == "__main__":
    manager = StrategyManager("config/strategy_config.yaml")
    
    # Test with mock data
    mock_market_data = {
        'RY.TO': {'close': 100.0, 'volume': 1000000, 'high': 102.0, 'low': 99.0},
        'TSX': {'close': 21000.0, 'volume': 5000000, 'high': 21100.0, 'low': 20900.0}
    }
    
    mock_news_data = [
        {'title': 'Royal Bank reports strong earnings', 'source': 'financialpost', 'timestamp': datetime.now()}
    ]
    
    # Analyze market conditions
    all_signals = manager.analyze_market_conditions(mock_market_data, mock_news_data)
    print(f"Generated signals: {sum(len(signals) for signals in all_signals.values())}")
    
    # Get consolidated signals
    consolidated = manager.get_consolidated_signals(all_signals)
    print(f"Consolidated signals: {len(consolidated)}")
    
    # Get strategy summary
    summary = manager.get_strategy_summary()
    print(f"Strategy summary: {summary}")
