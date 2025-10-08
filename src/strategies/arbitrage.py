"""
Arbitrage Strategy
Trades based on price discrepancies and statistical arbitrage
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ArbitrageSignal:
    """Arbitrage trading signal"""
    symbol_pair: str
    side: str  # 'BUY' or 'SELL'
    strength: float
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    arbitrage_type: str  # 'spread', 'statistical', 'cross_currency'
    spread_amount: float
    expected_duration: int  # minutes
    timestamp: datetime

class ArbitrageStrategy:
    """Arbitrage Strategy for Canadian Markets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.allocation = config.get('allocation', 0.20)
        self.max_leverage = config.get('max_leverage', 1.0)
        
        # Strategy parameters
        self.tsx_spy_spread = config.get('signals', {}).get('tsx_spy_spread', 0.1)
        self.futures_cash_basis = config.get('signals', {}).get('futures_cash_basis', 0.05)
        self.cross_currency_arb = config.get('signals', {}).get('cross_currency_arb', 0.02)
        self.stop_loss = config.get('risk', {}).get('stop_loss', 0.1)
        self.take_profit = config.get('risk', {}).get('take_profit', 0.2)
        self.max_holding_time = config.get('risk', {}).get('max_holding_time', 10)
        
        # Canadian market instruments
        self.instruments = config.get('instruments', [
            "TSX", "TSX60", "SPY", "QQQ"
        ])
        
        # Arbitrage pairs
        self.arbitrage_pairs = [
            ("TSX", "SPY"),  # TSX vs S&P 500
            ("TSX60", "SPY"),  # TSX 60 vs S&P 500
            ("TSX", "TSX60"),  # TSX vs TSX 60
        ]
        
        self.active_positions = {}
        self.signal_history = []
        self.price_history = {}
        self.correlation_matrix = {}
    
    def analyze_arbitrage_opportunities(self, market_data: Dict) -> List[ArbitrageSignal]:
        """Analyze market data for arbitrage opportunities"""
        try:
            signals = []
            
            # Update price history
            self._update_price_history(market_data)
            
            # Check different types of arbitrage
            signals.extend(self._check_spread_arbitrage(market_data))
            signals.extend(self._check_statistical_arbitrage(market_data))
            signals.extend(self._check_cross_currency_arbitrage(market_data))
            
            logger.info(f"Generated {len(signals)} arbitrage signals")
            return signals
            
        except Exception as e:
            logger.error(f"Failed to analyze arbitrage opportunities: {e}")
            return []
    
    def _update_price_history(self, market_data: Dict):
        """Update price history for correlation analysis"""
        try:
            current_time = datetime.now()
            
            for symbol, data in market_data.items():
                if symbol in self.instruments:
                    price = data.get('close', 0)
                    if price > 0:
                        if symbol not in self.price_history:
                            self.price_history[symbol] = []
                        
                        self.price_history[symbol].append({
                            'price': price,
                            'timestamp': current_time
                        })
                        
                        # Keep only last 100 data points
                        if len(self.price_history[symbol]) > 100:
                            self.price_history[symbol] = self.price_history[symbol][-100:]
            
        except Exception as e:
            logger.error(f"Failed to update price history: {e}")
    
    def _check_spread_arbitrage(self, market_data: Dict) -> List[ArbitrageSignal]:
        """Check for spread arbitrage opportunities"""
        try:
            signals = []
            
            for pair in self.arbitrage_pairs:
                symbol1, symbol2 = pair
                
                if symbol1 in market_data and symbol2 in market_data:
                    price1 = market_data[symbol1].get('close', 0)
                    price2 = market_data[symbol2].get('close', 0)
                    
                    if price1 > 0 and price2 > 0:
                        # Calculate spread
                        spread = abs(price1 - price2)
                        spread_percent = (spread / min(price1, price2)) * 100
                        
                        # Check if spread exceeds threshold
                        if spread_percent > self.tsx_spy_spread:
                            signal = self._create_spread_signal(pair, price1, price2, spread_percent)
                            if signal:
                                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to check spread arbitrage: {e}")
            return []
    
    def _check_statistical_arbitrage(self, market_data: Dict) -> List[ArbitrageSignal]:
        """Check for statistical arbitrage opportunities"""
        try:
            signals = []
            
            # Calculate correlations if we have enough data
            if len(self.price_history.get('TSX', [])) > 20:
                correlations = self._calculate_correlations()
                
                for pair, correlation in correlations.items():
                    if abs(correlation) > 0.8:  # High correlation
                        signal = self._create_statistical_signal(pair, correlation, market_data)
                        if signal:
                            signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to check statistical arbitrage: {e}")
            return []
    
    def _check_cross_currency_arbitrage(self, market_data: Dict) -> List[ArbitrageSignal]:
        """Check for cross-currency arbitrage opportunities"""
        try:
            signals = []
            
            # Mock cross-currency arbitrage check
            # In real implementation, this would check USD/CAD rates and cross-listed stocks
            
            # Example: Check if Canadian stocks are mispriced relative to USD equivalents
            cad_usd_rate = 1.35  # Mock exchange rate
            
            for symbol in ['TSX', 'TSX60']:
                if symbol in market_data:
                    cad_price = market_data[symbol].get('close', 0)
                    usd_equivalent = cad_price / cad_usd_rate
                    
                    # Check against US equivalent (mock)
                    usd_price = usd_equivalent * 1.02  # 2% premium
                    
                    if abs(usd_price - usd_equivalent) / usd_equivalent > self.cross_currency_arb:
                        signal = self._create_cross_currency_signal(symbol, cad_price, usd_price)
                        if signal:
                            signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to check cross-currency arbitrage: {e}")
            return []
    
    def _calculate_correlations(self) -> Dict[Tuple[str, str], float]:
        """Calculate correlations between instruments"""
        try:
            correlations = {}
            
            symbols = list(self.price_history.keys())
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    
                    prices1 = [p['price'] for p in self.price_history[symbol1]]
                    prices2 = [p['price'] for p in self.price_history[symbol2]]
                    
                    if len(prices1) == len(prices2) and len(prices1) > 10:
                        correlation = np.corrcoef(prices1, prices2)[0, 1]
                        correlations[(symbol1, symbol2)] = correlation
            
            return correlations
            
        except Exception as e:
            logger.error(f"Failed to calculate correlations: {e}")
            return {}
    
    def _create_spread_signal(self, pair: Tuple[str, str], price1: float, price2: float, spread_percent: float) -> Optional[ArbitrageSignal]:
        """Create spread arbitrage signal"""
        try:
            symbol1, symbol2 = pair
            symbol_pair = f"{symbol1}/{symbol2}"
            
            # Determine which side to trade
            if price1 > price2:
                side = 'SELL'
                entry_price = price1
                exit_price = price2
            else:
                side = 'BUY'
                entry_price = price2
                exit_price = price1
            
            strength = min(spread_percent / 2.0, 1.0)  # Normalize to 0-1
            confidence = min(strength * 1.5, 1.0)
            
            return ArbitrageSignal(
                symbol_pair=symbol_pair,
                side=side,
                strength=strength,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=self._calculate_stop_loss(entry_price, side),
                take_profit=self._calculate_take_profit(entry_price, side),
                confidence=confidence,
                arbitrage_type='spread',
                spread_amount=abs(price1 - price2),
                expected_duration=5,  # 5 minutes
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to create spread signal: {e}")
            return None
    
    def _create_statistical_signal(self, pair: Tuple[str, str], correlation: float, market_data: Dict) -> Optional[ArbitrageSignal]:
        """Create statistical arbitrage signal"""
        try:
            symbol1, symbol2 = pair
            symbol_pair = f"{symbol1}/{symbol2}"
            
            # Mock statistical arbitrage logic
            # In real implementation, this would use cointegration tests
            
            price1 = market_data[symbol1].get('close', 0)
            price2 = market_data[symbol2].get('close', 0)
            
            if price1 > 0 and price2 > 0:
                # Simple mean reversion signal
                ratio = price1 / price2
                historical_ratio = 1.0  # Mock historical ratio
                
                if ratio > historical_ratio * 1.02:  # 2% deviation
                    side = 'SELL'
                    entry_price = price1
                else:
                    side = 'BUY'
                    entry_price = price2
                
                strength = min(abs(ratio - historical_ratio) * 10, 1.0)
                confidence = min(strength * correlation, 1.0)
                
                return ArbitrageSignal(
                    symbol_pair=symbol_pair,
                    side=side,
                    strength=strength,
                    entry_price=entry_price,
                    exit_price=entry_price,  # Will be updated
                    stop_loss=self._calculate_stop_loss(entry_price, side),
                    take_profit=self._calculate_take_profit(entry_price, side),
                    confidence=confidence,
                    arbitrage_type='statistical',
                    spread_amount=abs(ratio - historical_ratio),
                    expected_duration=15,  # 15 minutes
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create statistical signal: {e}")
            return None
    
    def _create_cross_currency_signal(self, symbol: str, cad_price: float, usd_price: float) -> Optional[ArbitrageSignal]:
        """Create cross-currency arbitrage signal"""
        try:
            symbol_pair = f"{symbol}/USD"
            
            # Determine trade direction
            if cad_price > usd_price:
                side = 'SELL'
                entry_price = cad_price
            else:
                side = 'BUY'
                entry_price = usd_price
            
            spread_percent = abs(cad_price - usd_price) / min(cad_price, usd_price) * 100
            strength = min(spread_percent / 5.0, 1.0)
            confidence = min(strength * 1.2, 1.0)
            
            return ArbitrageSignal(
                symbol_pair=symbol_pair,
                side=side,
                strength=strength,
                entry_price=entry_price,
                exit_price=entry_price,  # Will be updated
                stop_loss=self._calculate_stop_loss(entry_price, side),
                take_profit=self._calculate_take_profit(entry_price, side),
                confidence=confidence,
                arbitrage_type='cross_currency',
                spread_amount=abs(cad_price - usd_price),
                expected_duration=10,  # 10 minutes
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to create cross-currency signal: {e}")
            return None
    
    def _calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        if side == 'BUY':
            return entry_price * (1 - self.stop_loss / 100)
        else:
            return entry_price * (1 + self.stop_loss / 100)
    
    def _calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        if side == 'BUY':
            return entry_price * (1 + self.take_profit / 100)
        else:
            return entry_price * (1 - self.take_profit / 100)
    
    def should_exit_position(self, symbol_pair: str, current_price: float, entry_time: datetime) -> bool:
        """Check if position should be exited"""
        try:
            # Check holding time
            holding_time = datetime.now() - entry_time
            if holding_time.total_seconds() > (self.max_holding_time * 60):
                logger.info(f"Exiting {symbol_pair} due to max holding time")
                return True
            
            # Check stop loss and take profit
            if symbol_pair in self.active_positions:
                position = self.active_positions[symbol_pair]
                side = position['side']
                
                if side == 'BUY':
                    if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                        return True
                else:
                    if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check exit conditions for {symbol_pair}: {e}")
            return False
    
    def get_strategy_status(self) -> Dict:
        """Get current strategy status"""
        return {
            'enabled': self.enabled,
            'allocation': self.allocation,
            'max_leverage': self.max_leverage,
            'active_positions': len(self.active_positions),
            'signals_generated': len(self.signal_history),
            'instruments': self.instruments,
            'arbitrage_pairs': self.arbitrage_pairs,
            'parameters': {
                'tsx_spy_spread': self.tsx_spy_spread,
                'futures_cash_basis': self.futures_cash_basis,
                'cross_currency_arb': self.cross_currency_arb,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'max_holding_time': self.max_holding_time
            }
        }

# Test the strategy
if __name__ == "__main__":
    config = {
        'enabled': True,
        'allocation': 0.20,
        'max_leverage': 1.0,
        'instruments': ['TSX', 'SPY'],
        'signals': {
            'tsx_spy_spread': 0.1,
            'futures_cash_basis': 0.05,
            'cross_currency_arb': 0.02
        },
        'risk': {
            'stop_loss': 0.1,
            'take_profit': 0.2,
            'max_holding_time': 10
        }
    }
    
    strategy = ArbitrageStrategy(config)
    
    # Test with mock market data
    mock_market_data = {
        'TSX': {'close': 21000.0},
        'SPY': {'close': 450.0},
        'TSX60': {'close': 1200.0}
    }
    
    signals = strategy.analyze_arbitrage_opportunities(mock_market_data)
    print(f"Generated {len(signals)} arbitrage signals")
    
    status = strategy.get_strategy_status()
    print(f"Strategy status: {status}")
