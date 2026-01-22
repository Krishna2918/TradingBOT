"""
Gamma/OI Squeeze Strategy
Trades based on options gamma exposure and open interest changes
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class GammaOISignal:
    """Gamma/OI squeeze trading signal"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    gamma_exposure: float
    oi_change: float
    put_call_ratio: float
    timestamp: datetime

class GammaOISqueezeStrategy:
    """Gamma/OI Squeeze Strategy for Canadian Markets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.allocation = config.get('allocation', 0.15)
        self.max_leverage = config.get('max_leverage', 3.0)
        
        # Strategy parameters
        self.oi_increase = config.get('signals', {}).get('oi_increase', 30)
        self.gamma_exposure = config.get('signals', {}).get('gamma_exposure', 0.8)
        self.put_call_ratio = config.get('signals', {}).get('put_call_ratio', 0.7)
        self.stop_loss = config.get('risk', {}).get('stop_loss', 0.8)
        self.take_profit = config.get('risk', {}).get('take_profit', 1.5)
        self.max_holding_time = config.get('risk', {}).get('max_holding_time', 90)
        
        # Canadian market instruments
        self.instruments = config.get('instruments', [
            "TSX", "TSX60", "RY.TO", "TD.TO"
        ])
        
        self.active_positions = {}
        self.signal_history = []
        self.oi_history = {}
    
    def analyze_options_data(self, options_data: Dict, market_data: Dict) -> List[GammaOISignal]:
        """Analyze options data for gamma/OI squeeze signals"""
        try:
            signals = []
            
            for symbol in self.instruments:
                if symbol in options_data and symbol in market_data:
                    signal = self._analyze_symbol_options(symbol, options_data[symbol], market_data[symbol])
                    if signal:
                        signals.append(signal)
            
            logger.info(f"Generated {len(signals)} gamma/OI squeeze signals")
            return signals
            
        except Exception as e:
            logger.error(f"Failed to analyze options data: {e}")
            return []
    
    def _analyze_symbol_options(self, symbol: str, options_data: Dict, market_data: Dict) -> Optional[GammaOISignal]:
        """Analyze options data for individual symbol"""
        try:
            current_price = market_data.get('close', 0)
            if current_price <= 0:
                return None
            
            # Extract options metrics
            oi_change = options_data.get('oi_change_percent', 0)
            gamma_exposure = options_data.get('gamma_exposure', 0)
            put_call_ratio = options_data.get('put_call_ratio', 1.0)
            total_oi = options_data.get('total_oi', 0)
            
            # Check OI increase threshold
            if oi_change < self.oi_increase:
                return None
            
            # Check gamma exposure threshold
            if gamma_exposure < self.gamma_exposure:
                return None
            
            # Determine trade direction based on put/call ratio
            if put_call_ratio < self.put_call_ratio:
                side = 'BUY'  # Bullish squeeze
                strength = (1.0 - put_call_ratio) * 2  # Convert to 0-1 scale
            else:
                side = 'SELL'  # Bearish squeeze
                strength = (put_call_ratio - 1.0) * 2  # Convert to 0-1 scale
            
            # Only proceed if strength is sufficient
            if strength < 0.3:
                return None
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(oi_change, gamma_exposure, put_call_ratio, total_oi)
            
            return GammaOISignal(
                symbol=symbol,
                side=side,
                strength=min(strength, 1.0),
                entry_price=current_price,
                stop_loss=self._calculate_stop_loss(current_price, side),
                take_profit=self._calculate_take_profit(current_price, side),
                confidence=confidence,
                gamma_exposure=gamma_exposure,
                oi_change=oi_change,
                put_call_ratio=put_call_ratio,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze options for {symbol}: {e}")
            return None
    
    def _calculate_confidence(self, oi_change: float, gamma_exposure: float, put_call_ratio: float, total_oi: int) -> float:
        """Calculate signal confidence based on multiple factors"""
        try:
            confidence = 0.0
            
            # OI change factor (0-0.4)
            oi_factor = min(oi_change / 100, 1.0) * 0.4
            confidence += oi_factor
            
            # Gamma exposure factor (0-0.3)
            gamma_factor = min(gamma_exposure, 1.0) * 0.3
            confidence += gamma_factor
            
            # Put/call ratio factor (0-0.2)
            pcr_factor = 0.2 if abs(put_call_ratio - 1.0) > 0.3 else 0.1
            confidence += pcr_factor
            
            # Volume factor (0-0.1)
            volume_factor = 0.1 if total_oi > 10000 else 0.05
            confidence += volume_factor
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.5
    
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
                'oi_increase': self.oi_increase,
                'gamma_exposure': self.gamma_exposure,
                'put_call_ratio': self.put_call_ratio,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'max_holding_time': self.max_holding_time
            }
        }

# Test the strategy
if __name__ == "__main__":
    config = {
        'enabled': True,
        'allocation': 0.15,
        'max_leverage': 3.0,
        'instruments': ['TSX', 'RY.TO'],
        'signals': {
            'oi_increase': 30,
            'gamma_exposure': 0.8,
            'put_call_ratio': 0.7
        },
        'risk': {
            'stop_loss': 0.8,
            'take_profit': 1.5,
            'max_holding_time': 90
        }
    }
    
    strategy = GammaOISqueezeStrategy(config)
    
    # Test with mock options data
    mock_options_data = {
        'TSX': {
            'oi_change_percent': 35,
            'gamma_exposure': 0.85,
            'put_call_ratio': 0.65,
            'total_oi': 15000
        }
    }
    
    mock_market_data = {
        'TSX': {
            'close': 21000.0
        }
    }
    
    signals = strategy.analyze_options_data(mock_options_data, mock_market_data)
    print(f"Generated {len(signals)} gamma/OI signals")
    
    status = strategy.get_strategy_status()
    print(f"Strategy status: {status}")
