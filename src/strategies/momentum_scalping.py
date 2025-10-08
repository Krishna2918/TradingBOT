"""
Momentum Scalping 2.0 Strategy
Advanced momentum-based scalping for Canadian markets
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MomentumSignal:
    """Momentum trading signal"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    strength: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: datetime

class MomentumScalpingStrategy:
    """Momentum Scalping 2.0 Strategy for Canadian Markets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.allocation = config.get('allocation', 0.25)
        self.max_leverage = config.get('max_leverage', 2.0)
        
        # Strategy parameters
        self.vwap_deviation = config.get('signals', {}).get('vwap_deviation', 0.5)
        self.volume_spike = config.get('signals', {}).get('volume_spike', 2.0)
        self.rsi_threshold = config.get('signals', {}).get('rsi_threshold', 70)
        self.stop_loss = config.get('risk', {}).get('stop_loss', 0.3)
        self.take_profit = config.get('risk', {}).get('take_profit', 0.6)
        self.max_holding_time = config.get('risk', {}).get('max_holding_time', 15)
        
        # Canadian market instruments
        self.instruments = config.get('instruments', [
            "RY.TO", "TD.TO", "SHOP.TO", "CNR.TO", "CP.TO"
        ])
        
        self.active_positions = {}
        self.signal_history = []
    
    def analyze_market_data(self, market_data: Dict) -> List[MomentumSignal]:
        """Analyze market data and generate momentum signals"""
        try:
            signals = []
            
            for symbol in self.instruments:
                if symbol in market_data:
                    signal = self._analyze_symbol(symbol, market_data[symbol])
                    if signal:
                        signals.append(signal)
            
            logger.info(f"Generated {len(signals)} momentum signals")
            return signals
            
        except Exception as e:
            logger.error(f"Failed to analyze market data: {e}")
            return []
    
    def _analyze_symbol(self, symbol: str, data: Dict) -> Optional[MomentumSignal]:
        """Analyze individual symbol for momentum signals"""
        try:
            # Extract price data
            current_price = data.get('close', 0)
            volume = data.get('volume', 0)
            high = data.get('high', 0)
            low = data.get('low', 0)
            
            if current_price <= 0:
                return None
            
            # Calculate technical indicators
            vwap_deviation = self._calculate_vwap_deviation(data)
            volume_ratio = self._calculate_volume_ratio(symbol, volume)
            rsi = self._calculate_rsi(data)
            
            # Generate signal based on momentum criteria
            signal_strength = 0.0
            side = None
            
            # VWAP deviation signal
            if vwap_deviation > self.vwap_deviation:
                signal_strength += 0.3
                side = 'BUY' if vwap_deviation > 0 else 'SELL'
            
            # Volume spike signal
            if volume_ratio > self.volume_spike:
                signal_strength += 0.3
            
            # RSI momentum signal
            if rsi > self.rsi_threshold:
                signal_strength += 0.2
                if side is None:
                    side = 'SELL'  # Overbought
            elif rsi < (100 - self.rsi_threshold):
                signal_strength += 0.2
                if side is None:
                    side = 'BUY'  # Oversold
            
            # Only generate signal if strength is sufficient
            if signal_strength >= 0.6 and side:
                return MomentumSignal(
                    symbol=symbol,
                    side=side,
                    strength=signal_strength,
                    entry_price=current_price,
                    stop_loss=self._calculate_stop_loss(current_price, side),
                    take_profit=self._calculate_take_profit(current_price, side),
                    confidence=min(signal_strength, 1.0),
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze symbol {symbol}: {e}")
            return None
    
    def _calculate_vwap_deviation(self, data: Dict) -> float:
        """Calculate VWAP deviation percentage"""
        try:
            # Simplified VWAP calculation
            current_price = data.get('close', 0)
            vwap = (data.get('high', 0) + data.get('low', 0) + data.get('close', 0)) / 3
            
            if vwap > 0:
                return ((current_price - vwap) / vwap) * 100
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate VWAP deviation: {e}")
            return 0.0
    
    def _calculate_volume_ratio(self, symbol: str, current_volume: int) -> float:
        """Calculate volume ratio vs average"""
        try:
            # Mock average volume calculation
            # In real implementation, this would use historical data
            avg_volume = 1000000  # Mock average volume
            
            if avg_volume > 0:
                return current_volume / avg_volume
            return 1.0
            
        except Exception as e:
            logger.error(f"Failed to calculate volume ratio: {e}")
            return 1.0
    
    def _calculate_rsi(self, data: Dict) -> float:
        """Calculate RSI indicator"""
        try:
            # Simplified RSI calculation
            # In real implementation, this would use historical price data
            current_price = data.get('close', 0)
            high = data.get('high', 0)
            low = data.get('low', 0)
            
            # Mock RSI calculation
            price_range = high - low
            if price_range > 0:
                rsi = 50 + ((current_price - low) / price_range - 0.5) * 100
                return max(0, min(100, rsi))
            
            return 50.0
            
        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return 50.0
    
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
    
    def should_exit_position(self, symbol: str, current_price: float, entry_time: datetime) -> bool:
        """Check if position should be exited"""
        try:
            # Check holding time
            holding_time = datetime.now() - entry_time
            if holding_time.total_seconds() > (self.max_holding_time * 60):
                logger.info(f"Exiting {symbol} due to max holding time")
                return True
            
            # Check stop loss and take profit
            if symbol in self.active_positions:
                position = self.active_positions[symbol]
                side = position['side']
                
                if side == 'BUY':
                    if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                        return True
                else:
                    if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check exit conditions for {symbol}: {e}")
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
            'parameters': {
                'vwap_deviation': self.vwap_deviation,
                'volume_spike': self.volume_spike,
                'rsi_threshold': self.rsi_threshold,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'max_holding_time': self.max_holding_time
            }
        }

# Test the strategy
if __name__ == "__main__":
    config = {
        'enabled': True,
        'allocation': 0.25,
        'max_leverage': 2.0,
        'instruments': ['RY.TO', 'TD.TO', 'SHOP.TO'],
        'signals': {
            'vwap_deviation': 0.5,
            'volume_spike': 2.0,
            'rsi_threshold': 70
        },
        'risk': {
            'stop_loss': 0.3,
            'take_profit': 0.6,
            'max_holding_time': 15
        }
    }
    
    strategy = MomentumScalpingStrategy(config)
    
    # Test with mock data
    mock_data = {
        'RY.TO': {
            'open': 100.0,
            'high': 102.0,
            'low': 99.0,
            'close': 101.5,
            'volume': 2000000
        }
    }
    
    signals = strategy.analyze_market_data(mock_data)
    print(f"Generated {len(signals)} signals")
    
    status = strategy.get_strategy_status()
    print(f"Strategy status: {status}")
