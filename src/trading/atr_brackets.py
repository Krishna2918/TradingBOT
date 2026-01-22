"""
ATR Brackets - Dynamic Stop Loss and Take Profit Management
=========================================================

This module implements ATR-based stop loss and take profit brackets with
dynamic R-multiple calculation based on realized trade outcomes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from src.config.database import execute_query, execute_update, get_connection
from src.config.mode_manager import get_current_mode

# Phase 7: Import regime detection and policy management
from src.ai.regime_detection import detect_current_regime
from src.config.regime_policy_manager import get_atr_brackets as get_regime_atr_brackets

logger = logging.getLogger(__name__)

class BracketType(Enum):
    """Bracket type enumeration."""
    ATR_BASED = "ATR_BASED"
    PERCENTAGE_BASED = "PERCENTAGE_BASED"
    FIXED_DOLLAR = "FIXED_DOLLAR"

@dataclass
class BracketParameters:
    """Parameters for stop loss and take profit brackets."""
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    atr_multiplier: float
    r_multiple: float
    bracket_type: BracketType
    volatility_percent: float
    risk_amount: float
    reward_amount: float
    created_at: datetime
    updated_at: datetime
    
    def __getitem__(self, key: str):
        """Support dictionary-style access for compatibility."""
        return getattr(self, key)

@dataclass
class TradeOutcome:
    """Trade outcome for R-multiple calculation."""
    symbol: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    exit_reason: str
    r_multiple: float
    pnl: float
    trade_date: datetime
    exit_date: datetime

class ATRBracketManager:
    """ATR bracket manager (alias for ATRBracketCalculator)."""
    pass

class ATRBracketCalculator(ATRBracketManager):
    """ATR bracket calculator class for compatibility."""
    """Manages ATR-based stop loss and take profit brackets."""
    
    def __init__(self):
        """Initialize ATR Bracket Manager."""
        self.min_stop_loss_percent = 0.02  # 2% minimum stop loss
        self.max_stop_loss_percent = 0.10  # 10% maximum stop loss
        self.default_atr_multiplier = 2.0  # Default ATR multiplier
        self.min_r_multiple = 1.0  # Minimum R-multiple
        self.max_r_multiple = 3.0  # Maximum R-multiple
        self.default_r_multiple = 1.5  # Default R-multiple
        self.lookback_days = 30  # Days to look back for R-multiple calculation
        
        logger.info("ATR Bracket Manager initialized")
    
    def calculate_brackets(self, symbol: str, entry_price: float, atr: float = None,
                          volatility: float = None, account_balance: float = 10000.0,
                          direction: str = "LONG", market_data: pd.DataFrame = None) -> BracketParameters:
        """Calculate brackets (alias for calculate_atr_brackets)."""
        if atr is None and volatility is not None:
            # Estimate ATR from volatility
            atr = entry_price * volatility * 0.1
        elif atr is None:
            # Default ATR
            atr = entry_price * 0.02
        
        # If market_data is provided, calculate ATR from it
        if market_data is not None and not market_data.empty:
            try:
                # Calculate ATR from market data
                high = market_data['high'].values
                low = market_data['low'].values
                close = market_data['close'].values
                
                # Simple ATR calculation (14-period)
                tr = np.maximum(high[1:] - low[1:], 
                              np.maximum(np.abs(high[1:] - close[:-1]), 
                                        np.abs(low[1:] - close[:-1])))
                atr = np.mean(tr) if len(tr) > 0 else entry_price * 0.02
            except Exception:
                atr = entry_price * 0.02
        
        return self.calculate_atr_brackets(symbol, entry_price, atr)
    
    def calculate_atr_brackets(self, symbol: str, entry_price: float, atr: float,
                             volatility_multiplier: float = None, 
                             mode: Optional[str] = None) -> BracketParameters:
        """
        Calculate ATR-based stop loss and take profit brackets.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for the position
            atr: Average True Range value
            volatility_multiplier: ATR multiplier (default: 2.0)
            mode: Trading mode (LIVE/DEMO)
            
        Returns:
            BracketParameters with calculated stop loss and take profit
        """
        if mode is None:
            mode = get_current_mode()
        
        # Phase 7: Get regime-aware ATR bracket parameters
        current_regime = detect_current_regime()
        regime_brackets = get_regime_atr_brackets(current_regime.regime)
        
        # Use regime-specific parameters
        regime_atr_multiplier = regime_brackets.get('atr_multiplier', self.default_atr_multiplier)
        regime_min_stop_loss = regime_brackets.get('min_stop_loss_percent', self.min_stop_loss_percent)
        regime_max_stop_loss = regime_brackets.get('max_stop_loss_percent', self.max_stop_loss_percent)
        regime_r_multiple = regime_brackets.get('r_multiple', self.default_r_multiple)
        
        if volatility_multiplier is None:
            volatility_multiplier = regime_atr_multiplier
        
        try:
            # Calculate ATR-based stop loss
            atr_stop_distance = atr * volatility_multiplier
            atr_stop_loss = entry_price - atr_stop_distance
            
            # Ensure stop loss is within regime-specific bounds
            min_stop_loss = entry_price * (1 - regime_max_stop_loss)
            max_stop_loss = entry_price * (1 - regime_min_stop_loss)
            stop_loss = max(min_stop_loss, min(atr_stop_loss, max_stop_loss))
            
            # Calculate actual stop distance and volatility percentage
            actual_stop_distance = entry_price - stop_loss
            volatility_percent = (actual_stop_distance / entry_price) * 100
            
            # Calculate R-multiple based on historical performance
            r_multiple = self._calculate_optimal_r_multiple(symbol, mode)
            
            # Use regime-specific R-multiple if available, otherwise use calculated optimal
            final_r_multiple = regime_r_multiple if regime_r_multiple != self.default_r_multiple else r_multiple
            
            # Calculate take profit based on R-multiple
            take_profit = entry_price + (actual_stop_distance * final_r_multiple)
            
            # Calculate risk and reward amounts
            risk_amount = actual_stop_distance
            reward_amount = take_profit - entry_price
            
            bracket_params = BracketParameters(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr=atr,
                atr_multiplier=volatility_multiplier,
                r_multiple=final_r_multiple,
                bracket_type=BracketType.ATR_BASED,
                volatility_percent=volatility_percent,
                risk_amount=risk_amount,
                reward_amount=reward_amount,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Log bracket parameters
            self._log_bracket_parameters(bracket_params, mode)
            
            logger.info(f"ATR brackets calculated for {symbol}: "
                       f"SL={stop_loss:.2f}, TP={take_profit:.2f}, R={r_multiple:.2f}")
            
            return bracket_params
            
        except Exception as e:
            logger.error(f"Error calculating ATR brackets for {symbol}: {e}")
            # Return default brackets as fallback
            return self._create_default_brackets(symbol, entry_price, atr)
    
    def _calculate_optimal_r_multiple(self, symbol: str, mode: str) -> float:
        """
        Calculate optimal R-multiple based on historical trade outcomes.
        
        Args:
            symbol: Trading symbol
            mode: Trading mode
            
        Returns:
            Optimal R-multiple value
        """
        try:
            # Get historical trade outcomes for this symbol
            trade_outcomes = self._get_historical_trade_outcomes(symbol, mode)
            
            if not trade_outcomes:
                # No historical data, use default R-multiple
                return self.default_r_multiple
            
            # Calculate R-multiples for winning trades
            winning_r_multiples = []
            for outcome in trade_outcomes:
                if outcome.r_multiple > 0:  # Winning trade
                    winning_r_multiples.append(outcome.r_multiple)
            
            if not winning_r_multiples:
                # No winning trades, use default
                return self.default_r_multiple
            
            # Calculate median R-multiple of winners
            median_r_multiple = np.median(winning_r_multiples)
            
            # Apply bounds
            optimal_r_multiple = max(self.min_r_multiple, 
                                   min(median_r_multiple, self.max_r_multiple))
            
            logger.debug(f"Optimal R-multiple for {symbol}: {optimal_r_multiple:.2f} "
                        f"(based on {len(winning_r_multiples)} winning trades)")
            
            return optimal_r_multiple
            
        except Exception as e:
            logger.error(f"Error calculating optimal R-multiple for {symbol}: {e}")
            return self.default_r_multiple
    
    def _get_historical_trade_outcomes(self, symbol: str, mode: str, 
                                     limit: int = 50) -> List[TradeOutcome]:
        """
        Get historical trade outcomes for R-multiple calculation.
        
        Args:
            symbol: Trading symbol
            mode: Trading mode
            limit: Maximum number of trades to retrieve
            
        Returns:
            List of TradeOutcome objects
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            
            query = """
                SELECT p.symbol, p.entry_price, p.exit_price, p.stop_loss, p.take_profit,
                       p.exit_reason, p.pnl, p.entry_time, p.exit_time
                FROM positions p
                WHERE p.symbol = ? AND p.mode = ? AND p.status = 'CLOSED'
                  AND p.exit_time >= ?
                ORDER BY p.exit_time DESC
                LIMIT ?
            """
            
            results = execute_query(query, (symbol, mode, cutoff_date.isoformat(), limit), mode)
            
            trade_outcomes = []
            for row in results:
                if row['exit_price'] and row['stop_loss']:
                    # Calculate R-multiple
                    risk_distance = row['entry_price'] - row['stop_loss']
                    if risk_distance > 0:
                        r_multiple = (row['exit_price'] - row['entry_price']) / risk_distance
                    else:
                        r_multiple = 0.0
                    
                    trade_outcome = TradeOutcome(
                        symbol=row['symbol'],
                        entry_price=row['entry_price'],
                        exit_price=row['exit_price'],
                        stop_loss=row['stop_loss'],
                        take_profit=row['take_profit'],
                        exit_reason=row['exit_reason'],
                        r_multiple=r_multiple,
                        pnl=row['pnl'],
                        trade_date=datetime.fromisoformat(row['entry_time']),
                        exit_date=datetime.fromisoformat(row['exit_time'])
                    )
                    trade_outcomes.append(trade_outcome)
            
            return trade_outcomes
            
        except Exception as e:
            logger.error(f"Error getting historical trade outcomes for {symbol}: {e}")
            return []
    
    def _log_bracket_parameters(self, bracket_params: BracketParameters, mode: str) -> None:
        """Log bracket parameters to database."""
        try:
            query = """
                INSERT INTO bracket_parameters 
                (symbol, entry_price, stop_loss, take_profit, atr, atr_multiplier,
                 r_multiple, bracket_type, volatility_percent, risk_amount, reward_amount,
                 mode, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            execute_update(
                query,
                (bracket_params.symbol, bracket_params.entry_price, bracket_params.stop_loss,
                 bracket_params.take_profit, bracket_params.atr, bracket_params.atr_multiplier,
                 bracket_params.r_multiple, bracket_params.bracket_type.value,
                 bracket_params.volatility_percent, bracket_params.risk_amount,
                 bracket_params.reward_amount, mode, 
                 bracket_params.created_at.isoformat(), bracket_params.updated_at.isoformat()),
                mode
            )
            
        except Exception as e:
            logger.error(f"Error logging bracket parameters: {e}")
    
    def _create_default_brackets(self, symbol: str, entry_price: float, atr: float) -> BracketParameters:
        """Create default brackets as fallback."""
        stop_loss = entry_price * (1 - self.min_stop_loss_percent)
        take_profit = entry_price * (1 + (self.min_stop_loss_percent * self.default_r_multiple))
        
        return BracketParameters(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            atr_multiplier=self.default_atr_multiplier,
            r_multiple=self.default_r_multiple,
            bracket_type=BracketType.PERCENTAGE_BASED,
            volatility_percent=self.min_stop_loss_percent * 100,
            risk_amount=entry_price - stop_loss,
            reward_amount=take_profit - entry_price,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def update_bracket_parameters(self, symbol: str, new_r_multiple: float = None,
                                new_atr_multiplier: float = None, mode: Optional[str] = None) -> bool:
        """
        Update bracket parameters based on new market conditions or performance.
        
        Args:
            symbol: Trading symbol
            new_r_multiple: New R-multiple value
            new_atr_multiplier: New ATR multiplier
            mode: Trading mode
            
        Returns:
            True if update was successful
        """
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Get current bracket parameters
            query = """
                SELECT * FROM bracket_parameters 
                WHERE symbol = ? AND mode = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            
            results = execute_query(query, (symbol, mode), mode)
            
            if not results:
                logger.warning(f"No bracket parameters found for {symbol}")
                return False
            
            current_params = results[0]
            
            # Update parameters
            updated_params = current_params.copy()
            if new_r_multiple is not None:
                updated_params['r_multiple'] = max(self.min_r_multiple, 
                                                 min(new_r_multiple, self.max_r_multiple))
            if new_atr_multiplier is not None:
                updated_params['atr_multiplier'] = new_atr_multiplier
            
            # Recalculate take profit if R-multiple changed
            if new_r_multiple is not None:
                risk_distance = updated_params['entry_price'] - updated_params['stop_loss']
                updated_params['take_profit'] = updated_params['entry_price'] + (risk_distance * updated_params['r_multiple'])
                updated_params['reward_amount'] = updated_params['take_profit'] - updated_params['entry_price']
            
            # Update in database
            update_query = """
                UPDATE bracket_parameters 
                SET r_multiple = ?, atr_multiplier = ?, take_profit = ?, 
                    reward_amount = ?, updated_at = ?
                WHERE id = ?
            """
            
            execute_update(
                update_query,
                (updated_params['r_multiple'], updated_params['atr_multiplier'],
                 updated_params['take_profit'], updated_params['reward_amount'],
                 datetime.now().isoformat(), current_params['id']),
                mode
            )
            
            logger.info(f"Updated bracket parameters for {symbol}: "
                       f"R={updated_params['r_multiple']:.2f}, "
                       f"ATR_mult={updated_params['atr_multiplier']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating bracket parameters for {symbol}: {e}")
            return False
    
    def get_bracket_history(self, symbol: str, mode: Optional[str] = None, 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get bracket parameter history for a symbol.
        
        Args:
            symbol: Trading symbol
            mode: Trading mode
            limit: Maximum number of records to retrieve
            
        Returns:
            List of bracket parameter records
        """
        if mode is None:
            mode = get_current_mode()
        
        try:
            query = """
                SELECT * FROM bracket_parameters 
                WHERE symbol = ? AND mode = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """
            
            return execute_query(query, (symbol, mode, limit), mode)
            
        except Exception as e:
            logger.error(f"Error getting bracket history for {symbol}: {e}")
            return []
    
    def calculate_volatility_adjusted_brackets(self, symbol: str, entry_price: float, 
                                            current_atr: float, historical_atr: float,
                                            mode: Optional[str] = None) -> BracketParameters:
        """
        Calculate brackets adjusted for current vs historical volatility.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            current_atr: Current ATR value
            historical_atr: Historical average ATR
            mode: Trading mode
            
        Returns:
            Volatility-adjusted bracket parameters
        """
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Calculate volatility ratio
            if historical_atr > 0:
                volatility_ratio = current_atr / historical_atr
            else:
                volatility_ratio = 1.0
            
            # Adjust ATR multiplier based on volatility
            if volatility_ratio > 1.2:  # High volatility
                adjusted_multiplier = self.default_atr_multiplier * 1.2
            elif volatility_ratio < 0.8:  # Low volatility
                adjusted_multiplier = self.default_atr_multiplier * 0.8
            else:
                adjusted_multiplier = self.default_atr_multiplier
            
            # Calculate brackets with adjusted multiplier
            return self.calculate_atr_brackets(symbol, entry_price, current_atr, 
                                             adjusted_multiplier, mode)
            
        except Exception as e:
            logger.error(f"Error calculating volatility-adjusted brackets for {symbol}: {e}")
            return self._create_default_brackets(symbol, entry_price, current_atr)

# Global ATR bracket manager instance
_atr_bracket_manager: Optional[ATRBracketManager] = None

def get_atr_bracket_manager() -> ATRBracketManager:
    """Get the global ATR bracket manager instance."""
    global _atr_bracket_manager
    if _atr_bracket_manager is None:
        _atr_bracket_manager = ATRBracketManager()
    return _atr_bracket_manager

# Convenience functions
def calculate_atr_brackets(symbol: str, entry_price: float, atr: float,
                         volatility_multiplier: float = None, 
                         mode: Optional[str] = None) -> BracketParameters:
    """Calculate ATR-based stop loss and take profit brackets."""
    return get_atr_bracket_manager().calculate_atr_brackets(
        symbol, entry_price, atr, volatility_multiplier, mode
    )

def update_bracket_parameters(symbol: str, new_r_multiple: float = None,
                            new_atr_multiplier: float = None, 
                            mode: Optional[str] = None) -> bool:
    """Update bracket parameters based on new conditions."""
    return get_atr_bracket_manager().update_bracket_parameters(
        symbol, new_r_multiple, new_atr_multiplier, mode
    )

def get_bracket_history(symbol: str, mode: Optional[str] = None, 
                       limit: int = 10) -> List[Dict[str, Any]]:
    """Get bracket parameter history for a symbol."""
    return get_atr_bracket_manager().get_bracket_history(symbol, mode, limit)

def log_bracket_parameters(bracket_params: BracketParameters, mode: Optional[str] = None) -> None:
    """Log bracket parameters to database."""
    return get_atr_bracket_manager()._log_bracket_parameters(bracket_params, mode)

def calculate_volatility_adjusted_brackets(symbol: str, entry_price: float, 
                                         current_atr: float, historical_atr: float,
                                         mode: Optional[str] = None) -> BracketParameters:
    """Calculate volatility-adjusted brackets."""
    return get_atr_bracket_manager().calculate_volatility_adjusted_brackets(
        symbol, entry_price, current_atr, historical_atr, mode
    )
