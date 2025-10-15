"""
Risk Management System - Dual Mode Risk Control

This module manages risk parameters for both LIVE and DEMO modes,
providing comprehensive risk management with position sizing and portfolio limits.
"""

import logging
import math
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from .positions import get_position_manager, get_open_positions
from src.config.mode_manager import get_mode_manager, get_current_mode, get_risk_limits
from src.config.database import execute_query, execute_update

# Phase 4: Import confidence calibration
from src.adaptive.confidence_calibration import get_confidence_calibrator

# Phase 7: Import regime detection and policy management
from src.ai.regime_detection import detect_current_regime
from src.config.regime_policy_manager import get_kelly_adjustments, get_risk_management as get_regime_risk_management

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio."""
    position_size: float
    risk_amount: float
    risk_percent: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    max_position_value: float
    portfolio_risk: float
    raw_confidence: float = 0.0
    calibrated_confidence: float = 0.0
    drawdown_scale: float = 1.0
    daily_drawdown: float = 0.0
    kelly_fraction: float = 0.0
    
    def __ge__(self, other):
        """Compare position_size for >= operator."""
        if isinstance(other, (int, float)):
            return self.position_size >= other
        elif isinstance(other, RiskMetrics):
            return self.position_size >= other.position_size
        return NotImplemented
    
    def __le__(self, other):
        """Compare position_size for <= operator."""
        if isinstance(other, (int, float)):
            return self.position_size <= other
        elif isinstance(other, RiskMetrics):
            return self.position_size <= other.position_size
        return NotImplemented
    
    def __gt__(self, other):
        """Compare position_size for > operator."""
        if isinstance(other, (int, float)):
            return self.position_size > other
        elif isinstance(other, RiskMetrics):
            return self.position_size > other.position_size
        return NotImplemented
    
    def __lt__(self, other):
        """Compare position_size for < operator."""
        if isinstance(other, (int, float)):
            return self.position_size < other
        elif isinstance(other, RiskMetrics):
            return self.position_size < other.position_size
        return NotImplemented

class RiskManager:
    """Manages risk parameters and position sizing."""
    
    def __init__(self, mode: Optional[str] = None):
        """Initialize Risk Manager."""
        self.mode_manager = get_mode_manager()
        self.position_manager = get_position_manager()
        self.mode = mode  # Store mode parameter for compatibility
        
        # Risk parameters (can be overridden by mode config)
        self.default_risk_per_trade = 0.02  # 2% risk per trade
        self.max_portfolio_risk = 0.20  # 20% max portfolio risk
        self.max_position_size = 0.10  # 10% max position size
        self.max_daily_drawdown = 0.05  # 5% max daily drawdown
        self.min_confidence = 0.70  # 70% minimum confidence
        
        # Phase 6: Drawdown-aware parameters
        self.min_drawdown_scale = 0.3  # Minimum Kelly scale factor
        self.drawdown_window_hours = 24  # Drawdown calculation window
        
        logger.info("Risk Manager initialized")
    
    def calculate_position_size(self, signal_confidence: float = None, account_balance: float = None, 
                              volatility: float = None, entry_price: float = None, stop_loss: float = None,
                              mode: Optional[str] = None, model_name: str = None,
                              symbol: str = None, trade_date: datetime = None, 
                              price: float = None, confidence: float = None, signal_strength: float = None) -> RiskMetrics:
        """Calculate optimal position size based on risk parameters."""
        if mode is None:
            mode = get_current_mode()
        
        # Handle parameter mapping for compatibility
        if price is not None:
            entry_price = price
        if confidence is not None:
            signal_confidence = confidence
        if signal_strength is not None:
            # Map signal_strength to volatility (simplified mapping)
            volatility = signal_strength * 0.2  # Convert to volatility range
        
        # Set defaults if not provided
        if signal_confidence is None:
            signal_confidence = 0.7
        if account_balance is None:
            account_balance = 10000.0  # Default account balance
        if volatility is None:
            volatility = 0.2  # Default volatility
        if entry_price is None:
            entry_price = 100.0  # Default entry price
        if stop_loss is None:
            stop_loss = entry_price * 0.95  # Default 5% stop loss
        
        # Get mode-specific risk limits
        risk_limits = get_risk_limits()
        
        # Phase 4: Apply confidence calibration if model info is provided
        calibrated_confidence = signal_confidence
        if model_name and symbol and trade_date:
            try:
                calibrator = get_confidence_calibrator()
                calibrated_confidence = calibrator.calibrate_confidence(
                    model_name, signal_confidence, trade_date
                )
                logger.debug(f"Confidence calibrated for {model_name}: "
                           f"{signal_confidence:.3f} -> {calibrated_confidence:.3f}")
            except Exception as e:
                logger.warning(f"Failed to calibrate confidence for {model_name}: {e}")
                calibrated_confidence = signal_confidence
        
        # Phase 6: Calculate drawdown-aware Kelly sizing
        drawdown_info = self.calculate_daily_drawdown(mode)
        drawdown_scale = self.calculate_drawdown_scale(mode)
        
        # Phase 7: Get regime-aware Kelly adjustments
        current_regime = detect_current_regime()
        kelly_adjustments = get_kelly_adjustments(current_regime.regime)
        regime_kelly_multiplier = kelly_adjustments.get('kelly_multiplier', 1.0)
        regime_max_risk = kelly_adjustments.get('max_position_risk', 0.02)
        regime_confidence_threshold = kelly_adjustments.get('confidence_threshold', 0.7)
        
        # Check if confidence meets regime threshold
        if calibrated_confidence < regime_confidence_threshold:
            logger.debug(f"Confidence {calibrated_confidence:.3f} below regime threshold {regime_confidence_threshold:.3f}")
            # Return minimal position size
            return RiskMetrics(
                position_size=0,
                risk_amount=0,
                risk_percent=0,
                stop_loss=stop_loss,
                take_profit=entry_price * 1.02,  # Default 2% take profit
                risk_reward_ratio=1.0,
                max_position_value=0,
                portfolio_risk=0,
                raw_confidence=signal_confidence,
                calibrated_confidence=calibrated_confidence,
                drawdown_scale=drawdown_scale,
                daily_drawdown=drawdown_info.get('drawdown_percent', 0.0),
                kelly_fraction=0.0
            )
        
        # Calculate Kelly fraction (simplified version using confidence as win probability)
        # In a real implementation, you would use historical win/loss data
        win_probability = calibrated_confidence
        avg_win = 0.02  # 2% average win (simplified)
        avg_loss = 0.015  # 1.5% average loss (simplified)
        kelly_fraction = self.calculate_kelly_fraction(win_probability, avg_win, avg_loss)
        
        # Apply regime and drawdown scaling to Kelly fraction
        scaled_kelly = kelly_fraction * drawdown_scale * regime_kelly_multiplier
        
        # Calculate risk amount using Kelly sizing
        kelly_risk_amount = account_balance * scaled_kelly
        
        # Fallback to traditional risk per trade if Kelly is too small
        risk_per_trade = risk_limits.get('max_position_size', self.default_risk_per_trade)
        traditional_risk_amount = account_balance * risk_per_trade
        
        # Use regime-specific max risk limit
        max_risk_amount = account_balance * regime_max_risk
        risk_amount = min(max(kelly_risk_amount, traditional_risk_amount), max_risk_amount)
        
        # Adjust risk based on calibrated confidence
        confidence_multiplier = min(1.0, calibrated_confidence / 0.8)  # Scale up to 80% confidence
        adjusted_risk_amount = risk_amount * confidence_multiplier
        
        # Calculate position size based on stop loss distance
        if stop_loss and stop_loss < entry_price:
            stop_distance = entry_price - stop_loss
            position_size = adjusted_risk_amount / stop_distance
        else:
            # Fallback: use volatility-based sizing
            volatility_risk = entry_price * volatility * 0.02  # 2% of price * volatility
            position_size = adjusted_risk_amount / volatility_risk
        
        # Apply maximum position size limit
        max_position_value = account_balance * risk_limits.get('max_position_size', self.max_position_size)
        max_shares = max_position_value / entry_price
        position_size = min(position_size, max_shares)
        
        # Ensure minimum of 1 share if affordable
        if position_size >= 1.0:
            position_size = math.floor(position_size)
        else:
            position_size = 0  # Position too small
        
        # Calculate take profit based on risk/reward ratio
        risk_reward_ratio = 1.5  # 1.5:1 risk/reward
        if stop_loss:
            risk_distance = entry_price - stop_loss
            take_profit = entry_price + (risk_distance * risk_reward_ratio)
        else:
            take_profit = entry_price * 1.03  # 3% default take profit
        
        # Calculate actual risk metrics
        actual_risk_amount = position_size * (entry_price - stop_loss) if stop_loss else 0
        actual_risk_percent = (actual_risk_amount / account_balance) * 100 if account_balance > 0 else 0
        portfolio_risk = self._calculate_portfolio_risk(mode)
        
        risk_metrics = RiskMetrics(
            position_size=position_size,
            risk_amount=actual_risk_amount,
            risk_percent=actual_risk_percent,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            max_position_value=max_position_value,
            portfolio_risk=portfolio_risk,
            raw_confidence=signal_confidence,
            calibrated_confidence=calibrated_confidence,
            drawdown_scale=drawdown_scale,
            daily_drawdown=drawdown_info["daily_drawdown"],
            kelly_fraction=kelly_fraction
        )
        
        logger.info(f"Position size calculated: {position_size} shares, risk: ${actual_risk_amount:.2f} ({actual_risk_percent:.2f}%)")
        return risk_metrics
    
    def check_portfolio_limits(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Check if portfolio is within risk limits."""
        if mode is None:
            mode = get_current_mode()
        
        risk_limits = get_risk_limits()
        open_positions = get_open_positions(mode)
        
        # Calculate current portfolio metrics
        total_positions = len(open_positions)
        max_positions = risk_limits.get('max_positions', 10)
        
        # Calculate total invested amount
        total_invested = sum(pos.entry_price * pos.quantity for pos in open_positions)
        
        # Calculate total unrealized P&L
        total_unrealized_pnl = sum(pos.pnl for pos in open_positions)
        
        # Calculate portfolio risk
        portfolio_risk = self._calculate_portfolio_risk(mode)
        max_portfolio_risk = risk_limits.get('max_portfolio_risk', self.max_portfolio_risk)
        
        # Check limits
        limits_status = {
            "mode": mode,
            "total_positions": total_positions,
            "max_positions": max_positions,
            "positions_within_limit": total_positions <= max_positions,
            "total_invested": total_invested,
            "total_unrealized_pnl": total_unrealized_pnl,
            "portfolio_risk": portfolio_risk,
            "max_portfolio_risk": max_portfolio_risk,
            "portfolio_risk_within_limit": portfolio_risk <= max_portfolio_risk,
            "can_open_new_position": total_positions < max_positions and portfolio_risk <= max_portfolio_risk,
            "warnings": []
        }
        
        # Add warnings for limit violations
        if total_positions >= max_positions:
            limits_status["warnings"].append(f"Maximum positions limit reached ({total_positions}/{max_positions})")
        
        if portfolio_risk > max_portfolio_risk:
            limits_status["warnings"].append(f"Portfolio risk limit exceeded ({portfolio_risk:.2%}/{max_portfolio_risk:.2%})")
        
        return limits_status
    
    def calculate_stop_loss(self, entry_price: float, atr: float, 
                          volatility_multiplier: float = 2.0) -> float:
        """Calculate stop loss based on ATR and volatility."""
        if atr <= 0:
            # Fallback: use 2% of entry price
            return entry_price * 0.98
        
        # Calculate stop loss: entry_price - (ATR * multiplier)
        stop_loss = entry_price - (atr * volatility_multiplier)
        
        # Ensure stop loss is not too close (minimum 1% below entry)
        min_stop_loss = entry_price * 0.99
        stop_loss = max(stop_loss, min_stop_loss)
        
        return round(stop_loss, 2)
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                            risk_reward_ratio: float = 1.5) -> float:
        """Calculate take profit based on risk/reward ratio."""
        if stop_loss <= 0 or stop_loss >= entry_price:
            # Fallback: use 3% above entry price
            return entry_price * 1.03
        
        # Calculate risk distance
        risk_distance = entry_price - stop_loss
        
        # Calculate take profit: entry_price + (risk_distance * ratio)
        take_profit = entry_price + (risk_distance * risk_reward_ratio)
        
        return round(take_profit, 2)
    
    def get_max_position_size(self, symbol: str, volatility: float, 
                            account_balance: float, mode: Optional[str] = None) -> float:
        """Get maximum position size for a symbol."""
        if mode is None:
            mode = get_current_mode()
        
        risk_limits = get_risk_limits()
        max_position_percent = risk_limits.get('max_position_size', self.max_position_size)
        
        # Calculate maximum position value
        max_position_value = account_balance * max_position_percent
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_adjustment = max(0.5, 1.0 - volatility)  # Reduce size for high volatility
        adjusted_max_value = max_position_value * volatility_adjustment
        
        return adjusted_max_value
    
    def check_daily_drawdown(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Check daily drawdown limits."""
        if mode is None:
            mode = get_current_mode()
        
        risk_limits = get_risk_limits()
        max_daily_drawdown = risk_limits.get('max_daily_drawdown', self.max_daily_drawdown)
        
        # Get today's trades and P&L
        today = datetime.now().date()
        query = """
            SELECT SUM(pnl) as daily_pnl, COUNT(*) as trade_count
            FROM trade_results 
            WHERE DATE(created_at) = ? AND mode = ?
        """
        
        result = execute_query(query, (today, mode), mode)
        daily_pnl = result[0]['daily_pnl'] if result[0]['daily_pnl'] else 0
        trade_count = result[0]['trade_count']
        
        # Get account balance (would be from account balance table in production)
        # For now, assume starting balance of $10,000
        account_balance = 10000.0
        daily_drawdown_percent = abs(daily_pnl) / account_balance if account_balance > 0 else 0
        
        drawdown_status = {
            "mode": mode,
            "date": today,
            "daily_pnl": daily_pnl,
            "daily_drawdown_percent": daily_drawdown_percent,
            "max_daily_drawdown": max_daily_drawdown,
            "within_drawdown_limit": daily_drawdown_percent <= max_daily_drawdown,
            "trade_count": trade_count,
            "can_trade": daily_drawdown_percent <= max_daily_drawdown
        }
        
        if daily_drawdown_percent > max_daily_drawdown:
            logger.warning(f"Daily drawdown limit exceeded: {daily_drawdown_percent:.2%}/{max_daily_drawdown:.2%}")
        
        return drawdown_status
    
    def get_mode_specific_limits(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get risk limits for the specified mode."""
        if mode is None:
            mode = get_current_mode()
        
        risk_limits = get_risk_limits()
        
        return {
            "mode": mode,
            "max_position_size": risk_limits.get('max_position_size', self.max_position_size),
            "max_daily_drawdown": risk_limits.get('max_daily_drawdown', self.max_daily_drawdown),
            "max_portfolio_risk": risk_limits.get('max_portfolio_risk', self.max_portfolio_risk),
            "min_confidence": risk_limits.get('min_confidence', self.min_confidence),
            "max_positions": risk_limits.get('max_positions', 10),
            "risk_per_trade": self.default_risk_per_trade
        }
    
    def _calculate_portfolio_risk(self, mode: str) -> float:
        """Calculate current portfolio risk percentage."""
        open_positions = get_open_positions(mode)
        
        if not open_positions:
            return 0.0
        
        # Calculate total risk exposure
        total_risk = 0.0
        for position in open_positions:
            if position.stop_loss:
                position_risk = (position.entry_price - position.stop_loss) * position.quantity
                total_risk += position_risk
        
        # Assume account balance of $10,000 (would be from database in production)
        account_balance = 10000.0
        portfolio_risk = total_risk / account_balance if account_balance > 0 else 0
        
        return portfolio_risk
    
    def validate_risk_parameters(self, mode: Optional[str] = None) -> bool:
        """Validate risk management parameters."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            risk_limits = get_risk_limits()
            
            # Validate risk limits
            required_limits = ['max_position_size', 'max_daily_drawdown', 'max_portfolio_risk', 'min_confidence', 'max_positions']
            for limit in required_limits:
                if limit not in risk_limits:
                    logger.error(f"Missing risk limit: {limit}")
                    return False
                
                value = risk_limits[limit]
                if not isinstance(value, (int, float)):
                    logger.error(f"Invalid risk limit type for {limit}")
                    return False
                
                if limit in ['max_position_size', 'max_daily_drawdown', 'max_portfolio_risk', 'min_confidence']:
                    if not (0 < value <= 1):
                        logger.error(f"Risk limit {limit} must be between 0 and 1")
                        return False
                
                if limit == 'max_positions':
                    if not (1 <= value <= 100):
                        logger.error(f"Max positions must be between 1 and 100")
                        return False
            
            # Validate portfolio limits
            portfolio_limits = self.check_portfolio_limits(mode)
            if not portfolio_limits['can_open_new_position']:
                logger.warning("Portfolio limits reached - new positions may be restricted")
            
            # Validate daily drawdown
            drawdown_status = self.check_daily_drawdown(mode)
            if not drawdown_status['can_trade']:
                logger.warning("Daily drawdown limit reached - trading may be restricted")
            
            logger.info(f"Risk parameters validation passed for {mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Risk parameters validation error: {e}")
            return False
    
    def log_risk_event(self, event_type: str, symbol: str, description: str, 
                      severity: str = "INFO", mode: Optional[str] = None) -> None:
        """Log a risk management event."""
        if mode is None:
            mode = get_current_mode()
        
        query = """
            INSERT INTO risk_events (event_type, symbol, description, severity, mode)
            VALUES (?, ?, ?, ?, ?)
        """
        
        execute_update(query, (event_type, symbol, description, severity, mode), mode)
        logger.info(f"Risk event logged: {event_type} - {symbol} - {description}")
    
    def get_risk_summary(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive risk management summary."""
        if mode is None:
            mode = get_current_mode()
        
        # Get all risk metrics
        portfolio_limits = self.check_portfolio_limits(mode)
        daily_drawdown = self.check_daily_drawdown(mode)
        mode_limits = self.get_mode_specific_limits(mode)
        
        # Get recent risk events
        query = """
            SELECT * FROM risk_events 
            WHERE mode = ? 
            ORDER BY timestamp DESC 
            LIMIT 10
        """
        
        recent_events = execute_query(query, (mode,), mode)
        
        summary = {
            "mode": mode,
            "timestamp": datetime.now(),
            "portfolio_limits": portfolio_limits,
            "daily_drawdown": daily_drawdown,
            "mode_limits": mode_limits,
            "recent_risk_events": [
                {
                    "event_type": event['event_type'],
                    "symbol": event['symbol'],
                    "description": event['description'],
                    "severity": event['severity'],
                    "timestamp": event['timestamp']
                }
                for event in recent_events
            ],
            "risk_status": {
                "can_trade": portfolio_limits['can_open_new_position'] and daily_drawdown['can_trade'],
                "portfolio_healthy": portfolio_limits['portfolio_risk_within_limit'],
                "daily_limits_ok": daily_drawdown['within_drawdown_limit'],
                "overall_status": "HEALTHY" if (
                    portfolio_limits['can_open_new_position'] and 
                    daily_drawdown['can_trade'] and
                    portfolio_limits['portfolio_risk_within_limit']
                ) else "WARNING"
            }
        }
        
        return summary
    
    def calculate_daily_drawdown(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Calculate current daily drawdown."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Get portfolio value at start of day
            start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Query for portfolio value at start of day
            query = """
                SELECT portfolio_value FROM portfolio_snapshots 
                WHERE mode = ? AND timestamp >= ? 
                ORDER BY timestamp ASC 
                LIMIT 1
            """
            
            result = execute_query(query, (mode, start_of_day.isoformat()), mode)
            
            if result:
                start_value = result[0]['portfolio_value']
            else:
                # Fallback: use current portfolio value if no snapshot available
                portfolio_summary = self.position_manager.get_portfolio_summary(mode)
                start_value = portfolio_summary.get('total_value', 10000.0)
            
            # Get current portfolio value
            current_summary = self.position_manager.get_portfolio_summary(mode)
            current_value = current_summary.get('total_value', 10000.0)
            
            # Calculate drawdown
            if start_value > 0:
                daily_drawdown = (start_value - current_value) / start_value
            else:
                daily_drawdown = 0.0
            
            # Check if within limits
            within_limit = daily_drawdown <= self.max_daily_drawdown
            can_trade = within_limit and daily_drawdown < (self.max_daily_drawdown * 0.8)  # 80% of limit
            
            return {
                "daily_drawdown": daily_drawdown,
                "start_value": start_value,
                "current_value": current_value,
                "max_daily_drawdown": self.max_daily_drawdown,
                "within_drawdown_limit": within_limit,
                "can_trade": can_trade,
                "drawdown_percent": daily_drawdown * 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating daily drawdown: {e}")
            return {
                "daily_drawdown": 0.0,
                "start_value": 10000.0,
                "current_value": 10000.0,
                "max_daily_drawdown": self.max_daily_drawdown,
                "within_drawdown_limit": True,
                "can_trade": True,
                "drawdown_percent": 0.0
            }
    
    def calculate_drawdown_scale(self, mode: Optional[str] = None) -> float:
        """Calculate Kelly scale factor based on current drawdown."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            drawdown_info = self.calculate_daily_drawdown(mode)
            daily_drawdown = drawdown_info["daily_drawdown"]
            
            # Calculate scale factor: max(0.3, 1 - daily_drawdown / daily_limit)
            if self.max_daily_drawdown > 0:
                scale_factor = max(self.min_drawdown_scale, 
                                 1.0 - (daily_drawdown / self.max_daily_drawdown))
            else:
                scale_factor = 1.0
            
            logger.debug(f"Drawdown scale calculated: {scale_factor:.3f} "
                        f"(drawdown: {daily_drawdown:.3f}, limit: {self.max_daily_drawdown:.3f})")
            
            return scale_factor
            
        except Exception as e:
            logger.error(f"Error calculating drawdown scale: {e}")
            return 1.0  # Default to no scaling
    
    def calculate_kelly_fraction(self, win_probability: float, avg_win: float, 
                               avg_loss: float) -> float:
        """Calculate Kelly fraction for position sizing."""
        try:
            if avg_loss <= 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_probability, q = 1-p
            b = avg_win / avg_loss
            p = win_probability
            q = 1.0 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Ensure Kelly fraction is non-negative and capped
            kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            logger.debug(f"Kelly fraction calculated: {kelly_fraction:.3f} "
                        f"(win_prob: {win_probability:.3f}, avg_win: {avg_win:.3f}, avg_loss: {avg_loss:.3f})")
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.0

# Global risk manager instance
_risk_manager: Optional[RiskManager] = None

def get_risk_manager() -> RiskManager:
    """Get the global risk manager instance."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager

def calculate_position_size(signal_confidence: float, account_balance: float, 
                          volatility: float, entry_price: float, stop_loss: float,
                          mode: Optional[str] = None, model_name: str = None,
                          symbol: str = None, trade_date: datetime = None) -> RiskMetrics:
    """Calculate optimal position size."""
    return get_risk_manager().calculate_position_size(signal_confidence, account_balance, 
                                                    volatility, entry_price, stop_loss, mode,
                                                    model_name, symbol, trade_date)

def check_portfolio_limits(mode: Optional[str] = None) -> Dict[str, Any]:
    """Check portfolio risk limits."""
    return get_risk_manager().check_portfolio_limits(mode)

def calculate_stop_loss(entry_price: float, atr: float, 
                       volatility_multiplier: float = 2.0) -> float:
    """Calculate stop loss based on ATR."""
    return get_risk_manager().calculate_stop_loss(entry_price, atr, volatility_multiplier)

def calculate_take_profit(entry_price: float, stop_loss: float, 
                         risk_reward_ratio: float = 1.5) -> float:
    """Calculate take profit based on risk/reward ratio."""
    return get_risk_manager().calculate_take_profit(entry_price, stop_loss, risk_reward_ratio)

def check_daily_drawdown(mode: Optional[str] = None) -> Dict[str, Any]:
    """Check daily drawdown limits."""
    return get_risk_manager().check_daily_drawdown(mode)

def get_risk_summary(mode: Optional[str] = None) -> Dict[str, Any]:
    """Get risk management summary."""
    return get_risk_manager().get_risk_summary(mode)

def calculate_daily_drawdown(mode: Optional[str] = None) -> Dict[str, Any]:
    """Calculate current daily drawdown."""
    return get_risk_manager().calculate_daily_drawdown(mode)

def calculate_drawdown_scale(mode: Optional[str] = None) -> float:
    """Calculate Kelly scale factor based on current drawdown."""
    return get_risk_manager().calculate_drawdown_scale(mode)

def calculate_kelly_fraction(win_probability: float, avg_win: float, avg_loss: float) -> float:
    """Calculate Kelly fraction for position sizing."""
    return get_risk_manager().calculate_kelly_fraction(win_probability, avg_win, avg_loss)