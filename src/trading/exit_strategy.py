"""
Exit Strategy Engine - Dual Mode Exit Signal Generation

This module manages exit strategies for both LIVE and DEMO modes,
providing comprehensive exit signal generation based on multiple criteria.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from src.trading.positions import Position, get_position_manager, get_open_positions
from src.config.mode_manager import get_current_mode

logger = logging.getLogger(__name__)

class ExitReason(Enum):
    """Exit reason enumeration."""
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TIME_BASED = "TIME_BASED"
    AI_REVERSAL = "AI_REVERSAL"
    MANUAL = "MANUAL"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"

@dataclass
class SellSignal:
    """Represents a sell signal."""
    symbol: str
    position_id: int
    exit_price: float
    exit_reason: ExitReason
    confidence: float
    reasoning: str
    mode: str
    timestamp: datetime
    
    def __post_init__(self):
        """Validate sell signal data."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.exit_price <= 0:
            raise ValueError("Exit price must be positive")
        if self.mode not in ['LIVE', 'DEMO']:
            raise ValueError("Invalid mode")

class ExitStrategyEngine:
    """Manages exit strategies for trading positions."""
    
    def __init__(self):
        """Initialize Exit Strategy Engine."""
        self.position_manager = get_position_manager()
        
        # Exit strategy parameters
        self.max_holding_days = 30  # Maximum holding period
        self.stop_loss_multiplier = 2.0  # ATR multiplier for stop loss
        self.take_profit_multiplier = 3.0  # ATR multiplier for take profit
        self.min_confidence_threshold = 0.7  # Minimum confidence for AI reversal
        
        logger.info("Exit Strategy Engine initialized")
    
    def generate_sell_signals(self, positions: Optional[List[Position]] = None, 
                            mode: Optional[str] = None) -> List[SellSignal]:
        """Generate sell signals for all open positions."""
        if mode is None:
            mode = get_current_mode()
        
        if positions is None:
            positions = self.position_manager.get_open_positions(mode)
        
        sell_signals = []
        
        for position in positions:
            # Check each exit condition
            signals = self._check_exit_conditions(position, mode)
            sell_signals.extend(signals)
        
        logger.info(f"Generated {len(sell_signals)} sell signals for {mode} mode")
        return sell_signals
    
    def _check_exit_conditions(self, position: Position, mode: str) -> List[SellSignal]:
        """Check all exit conditions for a position."""
        signals = []
        
        # Check stop loss
        stop_loss_signal = self.check_stop_loss(position, mode)
        if stop_loss_signal:
            signals.append(stop_loss_signal)
        
        # Check take profit
        take_profit_signal = self.check_take_profit(position, mode)
        if take_profit_signal:
            signals.append(take_profit_signal)
        
        # Check time-based exit
        time_exit_signal = self.check_time_based_exit(position, mode)
        if time_exit_signal:
            signals.append(time_exit_signal)
        
        # Check AI reversal signal
        ai_reversal_signal = self.check_ai_reversal_signal(position, mode)
        if ai_reversal_signal:
            signals.append(ai_reversal_signal)
        
        return signals
    
    def check_stop_loss(self, position: Position, mode: str, 
                       current_price: Optional[float] = None) -> Optional[SellSignal]:
        """Check if stop loss should be triggered."""
        if not position.stop_loss:
            return None
        
        # For now, use entry price as current price (would be real-time in production)
        if current_price is None:
            current_price = position.entry_price
        
        # Check if current price is below stop loss
        if current_price <= position.stop_loss:
            return SellSignal(
                symbol=position.symbol,
                position_id=position.id,
                exit_price=position.stop_loss,
                exit_reason=ExitReason.STOP_LOSS,
                confidence=1.0,  # Stop loss is always 100% confident
                reasoning=f"Stop loss triggered at ${position.stop_loss:.2f} (current: ${current_price:.2f})",
                mode=mode,
                timestamp=datetime.now()
            )
        
        return None
    
    def check_take_profit(self, position: Position, mode: str,
                         current_price: Optional[float] = None) -> Optional[SellSignal]:
        """Check if take profit should be triggered."""
        if not position.take_profit:
            return None
        
        # For now, use entry price as current price (would be real-time in production)
        if current_price is None:
            current_price = position.entry_price
        
        # Check if current price is above take profit
        if current_price >= position.take_profit:
            return SellSignal(
                symbol=position.symbol,
                position_id=position.id,
                exit_price=position.take_profit,
                exit_reason=ExitReason.TAKE_PROFIT,
                confidence=1.0,  # Take profit is always 100% confident
                reasoning=f"Take profit triggered at ${position.take_profit:.2f} (current: ${current_price:.2f})",
                mode=mode,
                timestamp=datetime.now()
            )
        
        return None
    
    def check_time_based_exit(self, position: Position, mode: str) -> Optional[SellSignal]:
        """Check if position should be closed based on time."""
        current_time = datetime.now()
        holding_days = (current_time - position.entry_time).days
        
        # Check if position has been held too long
        if holding_days >= self.max_holding_days:
            # Calculate exit price (for now, use entry price)
            exit_price = position.entry_price
            
            return SellSignal(
                symbol=position.symbol,
                position_id=position.id,
                exit_price=exit_price,
                exit_reason=ExitReason.TIME_BASED,
                confidence=0.8,  # High confidence for time-based exits
                reasoning=f"Position held for {holding_days} days (max: {self.max_holding_days})",
                mode=mode,
                timestamp=current_time
            )
        
        return None
    
    def check_ai_reversal_signal(self, position: Position, mode: str) -> Optional[SellSignal]:
        """Check if AI suggests a reversal (sell signal)."""
        # This would integrate with the AI ensemble in production
        # For now, we'll simulate based on position performance
        
        # Calculate current P&L percentage
        current_price = position.entry_price  # Would be real-time in production
        pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        
        # Simulate AI reversal logic
        # If position is losing more than 5%, AI might suggest exit
        if pnl_percent < -5.0:
            confidence = min(0.9, abs(pnl_percent) / 10.0)  # Higher confidence for larger losses
            
            if confidence >= self.min_confidence_threshold:
                return SellSignal(
                    symbol=position.symbol,
                    position_id=position.id,
                    exit_price=current_price,
                    exit_reason=ExitReason.AI_REVERSAL,
                    confidence=confidence,
                    reasoning=f"AI reversal signal: P&L {pnl_percent:.2f}% (confidence: {confidence:.2f})",
                    mode=mode,
                    timestamp=datetime.now()
                )
        
        return None
    
    def calculate_exit_price(self, position: Position, exit_reason: ExitReason,
                           current_price: Optional[float] = None) -> float:
        """Calculate appropriate exit price based on exit reason."""
        if current_price is None:
            current_price = position.entry_price
        
        if exit_reason == ExitReason.STOP_LOSS:
            return position.stop_loss or current_price
        elif exit_reason == ExitReason.TAKE_PROFIT:
            return position.take_profit or current_price
        else:
            return current_price
    
    def get_exit_strategy_summary(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of exit strategy configuration."""
        if mode is None:
            mode = get_current_mode()
        
        open_positions = self.position_manager.get_open_positions(mode)
        
        # Analyze positions by exit strategy
        stop_loss_positions = [p for p in open_positions if p.stop_loss]
        take_profit_positions = [p for p in open_positions if p.take_profit]
        time_risk_positions = [
            p for p in open_positions 
            if (datetime.now() - p.entry_time).days >= self.max_holding_days * 0.8
        ]
        
        summary = {
            "mode": mode,
            "total_open_positions": len(open_positions),
            "positions_with_stop_loss": len(stop_loss_positions),
            "positions_with_take_profit": len(take_profit_positions),
            "positions_at_time_risk": len(time_risk_positions),
            "max_holding_days": self.max_holding_days,
            "stop_loss_multiplier": self.stop_loss_multiplier,
            "take_profit_multiplier": self.take_profit_multiplier,
            "min_confidence_threshold": self.min_confidence_threshold,
            "exit_strategies": {
                "stop_loss": {
                    "enabled": True,
                    "positions": len(stop_loss_positions),
                    "description": "Price-based stop loss protection"
                },
                "take_profit": {
                    "enabled": True,
                    "positions": len(take_profit_positions),
                    "description": "Price-based profit taking"
                },
                "time_based": {
                    "enabled": True,
                    "positions_at_risk": len(time_risk_positions),
                    "description": f"Maximum holding period: {self.max_holding_days} days"
                },
                "ai_reversal": {
                    "enabled": True,
                    "min_confidence": self.min_confidence_threshold,
                    "description": "AI-driven reversal signals"
                }
            }
        }
        
        return summary
    
    def update_exit_parameters(self, max_holding_days: Optional[int] = None,
                             stop_loss_multiplier: Optional[float] = None,
                             take_profit_multiplier: Optional[float] = None,
                             min_confidence_threshold: Optional[float] = None) -> bool:
        """Update exit strategy parameters."""
        try:
            if max_holding_days is not None:
                if max_holding_days <= 0:
                    raise ValueError("Max holding days must be positive")
                self.max_holding_days = max_holding_days
            
            if stop_loss_multiplier is not None:
                if stop_loss_multiplier <= 0:
                    raise ValueError("Stop loss multiplier must be positive")
                self.stop_loss_multiplier = stop_loss_multiplier
            
            if take_profit_multiplier is not None:
                if take_profit_multiplier <= 0:
                    raise ValueError("Take profit multiplier must be positive")
                self.take_profit_multiplier = take_profit_multiplier
            
            if min_confidence_threshold is not None:
                if not (0.0 <= min_confidence_threshold <= 1.0):
                    raise ValueError("Confidence threshold must be between 0.0 and 1.0")
                self.min_confidence_threshold = min_confidence_threshold
            
            logger.info("Exit strategy parameters updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update exit parameters: {e}")
            return False
    
    def validate_exit_conditions(self, mode: Optional[str] = None) -> bool:
        """Validate exit strategy configuration."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Validate parameters
            if self.max_holding_days <= 0:
                logger.error("Invalid max holding days")
                return False
            
            if self.stop_loss_multiplier <= 0:
                logger.error("Invalid stop loss multiplier")
                return False
            
            if self.take_profit_multiplier <= 0:
                logger.error("Invalid take profit multiplier")
                return False
            
            if not (0.0 <= self.min_confidence_threshold <= 1.0):
                logger.error("Invalid confidence threshold")
                return False
            
            # Validate positions
            open_positions = self.position_manager.get_open_positions(mode)
            
            for position in open_positions:
                # Check if stop loss is below entry price
                if position.stop_loss and position.stop_loss >= position.entry_price:
                    logger.warning(f"Stop loss above entry price for {position.symbol}")
                
                # Check if take profit is above entry price
                if position.take_profit and position.take_profit <= position.entry_price:
                    logger.warning(f"Take profit below entry price for {position.symbol}")
            
            logger.info(f"Exit conditions validation passed for {mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Exit conditions validation error: {e}")
            return False

# Global exit strategy engine instance
_exit_strategy_engine: Optional[ExitStrategyEngine] = None

def get_exit_strategy_engine() -> ExitStrategyEngine:
    """Get the global exit strategy engine instance."""
    global _exit_strategy_engine
    if _exit_strategy_engine is None:
        _exit_strategy_engine = ExitStrategyEngine()
    return _exit_strategy_engine

def generate_sell_signals(positions: Optional[List[Position]] = None, 
                        mode: Optional[str] = None) -> List[SellSignal]:
    """Generate sell signals for positions."""
    return get_exit_strategy_engine().generate_sell_signals(positions, mode)

def check_stop_loss(position: Position, mode: str, 
                   current_price: Optional[float] = None) -> Optional[SellSignal]:
    """Check stop loss condition."""
    return get_exit_strategy_engine().check_stop_loss(position, mode, current_price)

def check_take_profit(position: Position, mode: str,
                     current_price: Optional[float] = None) -> Optional[SellSignal]:
    """Check take profit condition."""
    return get_exit_strategy_engine().check_take_profit(position, mode, current_price)

def check_time_based_exit(position: Position, mode: str) -> Optional[SellSignal]:
    """Check time-based exit condition."""
    return get_exit_strategy_engine().check_time_based_exit(position, mode)

def check_ai_reversal_signal(position: Position, mode: str) -> Optional[SellSignal]:
    """Check AI reversal signal."""
    return get_exit_strategy_engine().check_ai_reversal_signal(position, mode)

def get_exit_strategy_summary(mode: Optional[str] = None) -> Dict[str, Any]:
    """Get exit strategy summary."""
    return get_exit_strategy_engine().get_exit_strategy_summary(mode)
