"""
Intraday Trading Engine

Handles day trading with:
- Pattern Day Trader (PDT) rule enforcement
- Intraday position tracking
- Auto-close before market close
- Day trading risk management
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
import pytz

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Position type classification"""
    SWING = "swing"  # Held overnight
    INTRADAY = "intraday"  # Opened and closed same day


@dataclass
class IntradayPosition:
    """Represents an intraday trading position"""
    symbol: str
    side: str  # LONG or SHORT
    shares: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    position_id: str = ""
    max_profit: float = 0.0  # Track max profit for trailing stop
    max_loss: float = 0.0  # Track max loss
    
    def __post_init__(self):
        if not self.position_id:
            self.position_id = f"intra_{self.symbol}_{self.entry_time.strftime('%Y%m%d_%H%M%S')}"


@dataclass
class IntradayTrade:
    """Represents a completed intraday trade"""
    symbol: str
    side: str
    shares: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    commission: float
    trade_id: str = ""
    
    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = f"trade_{self.symbol}_{self.entry_time.strftime('%Y%m%d_%H%M%S')}"


@dataclass
class DayTradingMetrics:
    """Day trading performance metrics"""
    date: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    total_commission: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    day_trades_used: int = 0  # For PDT rule


class IntradayTradingEngine:
    """
    Intraday Trading Engine with PDT rule compliance.
    
    Key Features:
    - Pattern Day Trader (PDT) rule enforcement (max 3 day trades per 5 days for accounts < $25k)
    - Intraday position tracking
    - Auto-close positions before market close
    - Trailing stops and profit targets
    - Day trading statistics
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # PDT rule settings
        self.account_value = self.config.get('account_value', 10000.0)
        self.pdt_threshold = self.config.get('pdt_threshold', 25000.0)
        self.max_day_trades_per_5_days = self.config.get('max_day_trades', 3)
        self.pdt_enabled = self.account_value < self.pdt_threshold
        
        # Trading hours (Eastern Time)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        self.auto_close_time = time(15, 45)  # Close all positions 15 min before close
        self.timezone = pytz.timezone('America/New_York')
        
        # Risk management
        self.max_intraday_positions = self.config.get('max_intraday_positions', 5)
        self.max_position_size_pct = self.config.get('max_position_size_pct', 0.20)  # 20% max per position
        self.default_stop_loss_pct = self.config.get('default_stop_loss_pct', 0.02)  # 2% stop loss
        self.default_take_profit_pct = self.config.get('default_take_profit_pct', 0.04)  # 4% take profit
        self.trailing_stop_enabled = self.config.get('trailing_stop_enabled', True)
        self.trailing_stop_pct = self.config.get('trailing_stop_pct', 0.015)  # 1.5% trailing
        
        # Position tracking
        self.open_positions: Dict[str, IntradayPosition] = {}
        self.closed_trades: List[IntradayTrade] = []
        self.daily_metrics: Dict[str, DayTradingMetrics] = {}
        
        # Day trade tracking (for PDT rule)
        self.day_trades_history: List[Tuple[datetime, str]] = []  # (date, symbol)
        
        logger.info(f"Intraday Trading Engine initialized (PDT: {self.pdt_enabled}, Account: ${self.account_value:,.2f})")
    
    def can_open_day_trade(self) -> Tuple[bool, str]:
        """
        Check if a new day trade can be opened (PDT rule compliance).
        
        Returns:
            (can_trade, reason)
        """
        if not self.pdt_enabled:
            return True, "Account value above PDT threshold"
        
        # Count day trades in last 5 business days
        five_days_ago = datetime.now() - timedelta(days=5)
        recent_day_trades = [dt for dt, _ in self.day_trades_history if dt >= five_days_ago]
        
        if len(recent_day_trades) >= self.max_day_trades_per_5_days:
            return False, f"PDT limit reached: {len(recent_day_trades)}/{self.max_day_trades_per_5_days} day trades in 5 days"
        
        return True, f"Day trades available: {self.max_day_trades_per_5_days - len(recent_day_trades)}"
    
    def open_intraday_position(self, symbol: str, side: str, shares: float, entry_price: float,
                               stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict:
        """
        Open a new intraday position.
        
        Args:
            symbol: Stock symbol
            side: 'LONG' or 'SHORT'
            shares: Number of shares
            entry_price: Entry price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
        
        Returns:
            Result dictionary
        """
        try:
            # Check if market is open
            if not self.is_market_open():
                return {'success': False, 'reason': 'Market is closed'}
            
            # Check PDT rule
            can_trade, reason = self.can_open_day_trade()
            if not can_trade:
                return {'success': False, 'reason': reason}
            
            # Check max positions
            if len(self.open_positions) >= self.max_intraday_positions:
                return {'success': False, 'reason': f'Max intraday positions ({self.max_intraday_positions}) reached'}
            
            # Check position size
            position_value = shares * entry_price
            max_position_value = self.account_value * self.max_position_size_pct
            if position_value > max_position_value:
                return {'success': False, 'reason': f'Position size ${position_value:,.2f} exceeds max ${max_position_value:,.2f}'}
            
            # Set default stop loss and take profit if not provided
            if stop_loss is None:
                if side == 'LONG':
                    stop_loss = entry_price * (1 - self.default_stop_loss_pct)
                else:
                    stop_loss = entry_price * (1 + self.default_stop_loss_pct)
            
            if take_profit is None:
                if side == 'LONG':
                    take_profit = entry_price * (1 + self.default_take_profit_pct)
                else:
                    take_profit = entry_price * (1 - self.default_take_profit_pct)
            
            # Create position
            position = IntradayPosition(
                symbol=symbol,
                side=side,
                shares=shares,
                entry_price=entry_price,
                entry_time=datetime.now(self.timezone),
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=entry_price
            )
            
            self.open_positions[position.position_id] = position
            
            logger.info(f"Opened intraday {side} position: {symbol} {shares} @ ${entry_price:.2f} (SL: ${stop_loss:.2f}, TP: ${take_profit:.2f})")
            
            return {
                'success': True,
                'position_id': position.position_id,
                'symbol': symbol,
                'side': side,
                'shares': shares,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except Exception as e:
            logger.error(f"Error opening intraday position: {e}")
            return {'success': False, 'reason': str(e)}
    
    def close_intraday_position(self, position_id: str, exit_price: float, reason: str = "") -> Dict:
        """
        Close an intraday position.
        
        Args:
            position_id: Position ID to close
            exit_price: Exit price
            reason: Reason for closing (e.g., "Take Profit", "Stop Loss")
        
        Returns:
            Result dictionary
        """
        try:
            if position_id not in self.open_positions:
                return {'success': False, 'reason': 'Position not found'}
            
            position = self.open_positions[position_id]
            
            # Calculate P&L
            if position.side == 'LONG':
                pnl = (exit_price - position.entry_price) * position.shares
            else:  # SHORT
                pnl = (position.entry_price - exit_price) * position.shares
            
            pnl_pct = (pnl / (position.entry_price * position.shares)) * 100
            
            # Assume $1 commission per trade
            commission = 2.0  # Entry + exit
            net_pnl = pnl - commission
            
            # Create closed trade record
            trade = IntradayTrade(
                symbol=position.symbol,
                side=position.side,
                shares=position.shares,
                entry_price=position.entry_price,
                exit_price=exit_price,
                entry_time=position.entry_time,
                exit_time=datetime.now(self.timezone),
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                commission=commission
            )
            
            self.closed_trades.append(trade)
            
            # Check if this was a day trade
            if trade.entry_time.date() == trade.exit_time.date():
                self.day_trades_history.append((trade.entry_time, position.symbol))
                logger.info(f"Day trade executed: {position.symbol} (PDT count: {len(self.day_trades_history)})")
            
            # Update daily metrics
            self._update_daily_metrics(trade)
            
            # Remove from open positions
            del self.open_positions[position_id]
            
            logger.info(f"Closed intraday position: {position.symbol} {position.side} @ ${exit_price:.2f} | P&L: ${net_pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}")
            
            return {
                'success': True,
                'trade_id': trade.trade_id,
                'symbol': position.symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'pnl': net_pnl,
                'pnl_pct': pnl_pct,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error closing intraday position: {e}")
            return {'success': False, 'reason': str(e)}
    
    def update_positions(self, market_data: Dict[str, float]):
        """
        Update all open positions with current market prices.
        Checks for stop loss, take profit, and trailing stops.
        
        Args:
            market_data: Dictionary of {symbol: current_price}
        """
        try:
            positions_to_close = []
            
            for position_id, position in self.open_positions.items():
                if position.symbol not in market_data:
                    continue
                
                current_price = market_data[position.symbol]
                position.current_price = current_price
                
                # Calculate unrealized P&L
                if position.side == 'LONG':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.shares
                else:  # SHORT
                    position.unrealized_pnl = (position.entry_price - current_price) * position.shares
                
                # Track max profit/loss for trailing stops
                position.max_profit = max(position.max_profit, position.unrealized_pnl)
                position.max_loss = min(position.max_loss, position.unrealized_pnl)
                
                # Check stop loss
                if position.side == 'LONG' and current_price <= position.stop_loss:
                    positions_to_close.append((position_id, current_price, "Stop Loss Hit"))
                elif position.side == 'SHORT' and current_price >= position.stop_loss:
                    positions_to_close.append((position_id, current_price, "Stop Loss Hit"))
                
                # Check take profit
                elif position.side == 'LONG' and current_price >= position.take_profit:
                    positions_to_close.append((position_id, current_price, "Take Profit Hit"))
                elif position.side == 'SHORT' and current_price <= position.take_profit:
                    positions_to_close.append((position_id, current_price, "Take Profit Hit"))
                
                # Check trailing stop
                elif self.trailing_stop_enabled and position.max_profit > 0:
                    trailing_threshold = position.max_profit * (1 - self.trailing_stop_pct)
                    if position.unrealized_pnl < trailing_threshold:
                        positions_to_close.append((position_id, current_price, "Trailing Stop Hit"))
            
            # Close positions that hit triggers
            for position_id, exit_price, reason in positions_to_close:
                self.close_intraday_position(position_id, exit_price, reason)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def auto_close_all_positions(self, market_data: Dict[str, float]):
        """
        Close all open intraday positions (called before market close).
        
        Args:
            market_data: Dictionary of {symbol: current_price}
        """
        try:
            logger.info(f"Auto-closing {len(self.open_positions)} intraday positions before market close")
            
            for position_id in list(self.open_positions.keys()):
                position = self.open_positions[position_id]
                exit_price = market_data.get(position.symbol, position.current_price)
                self.close_intraday_position(position_id, exit_price, "Auto-close before market close")
            
        except Exception as e:
            logger.error(f"Error auto-closing positions: {e}")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now(self.timezone)
        current_time = now.time()
        
        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if within trading hours
        return self.market_open <= current_time <= self.market_close
    
    def should_auto_close(self) -> bool:
        """Check if it's time to auto-close positions"""
        now = datetime.now(self.timezone)
        return now.time() >= self.auto_close_time
    
    def _update_daily_metrics(self, trade: IntradayTrade):
        """Update daily trading metrics"""
        date_key = trade.exit_time.strftime('%Y-%m-%d')
        
        if date_key not in self.daily_metrics:
            self.daily_metrics[date_key] = DayTradingMetrics(date=trade.exit_time.date())
        
        metrics = self.daily_metrics[date_key]
        metrics.total_trades += 1
        
        if trade.pnl > 0:
            metrics.winning_trades += 1
            metrics.gross_profit += trade.pnl
            metrics.largest_win = max(metrics.largest_win, trade.pnl)
        else:
            metrics.losing_trades += 1
            metrics.gross_loss += abs(trade.pnl)
            metrics.largest_loss = min(metrics.largest_loss, trade.pnl)
        
        metrics.net_pnl += trade.pnl
        metrics.total_commission += trade.commission
        
        # Calculate derived metrics
        if metrics.total_trades > 0:
            metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100
        
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        
        if metrics.winning_trades > 0:
            metrics.average_win = metrics.gross_profit / metrics.winning_trades
        
        if metrics.losing_trades > 0:
            metrics.average_loss = metrics.gross_loss / metrics.losing_trades
        
        # Count day trades
        if trade.entry_time.date() == trade.exit_time.date():
            metrics.day_trades_used += 1
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open intraday positions"""
        return [{
            'position_id': pos.position_id,
            'symbol': pos.symbol,
            'side': pos.side,
            'shares': pos.shares,
            'entry_price': pos.entry_price,
            'current_price': pos.current_price,
            'unrealized_pnl': pos.unrealized_pnl,
            'stop_loss': pos.stop_loss,
            'take_profit': pos.take_profit,
            'entry_time': pos.entry_time.isoformat()
        } for pos in self.open_positions.values()]
    
    def get_daily_metrics(self, date: Optional[str] = None) -> Dict:
        """Get daily trading metrics"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        if date not in self.daily_metrics:
            return {}
        
        metrics = self.daily_metrics[date]
        return {
            'date': date,
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'win_rate': metrics.win_rate,
            'gross_profit': metrics.gross_profit,
            'gross_loss': metrics.gross_loss,
            'net_pnl': metrics.net_pnl,
            'total_commission': metrics.total_commission,
            'profit_factor': metrics.profit_factor,
            'average_win': metrics.average_win,
            'average_loss': metrics.average_loss,
            'largest_win': metrics.largest_win,
            'largest_loss': metrics.largest_loss,
            'day_trades_used': metrics.day_trades_used
        }
    
    def get_pdt_status(self) -> Dict:
        """Get Pattern Day Trader status"""
        five_days_ago = datetime.now() - timedelta(days=5)
        recent_day_trades = [dt for dt, _ in self.day_trades_history if dt >= five_days_ago]
        
        return {
            'pdt_restricted': self.pdt_enabled,
            'account_value': self.account_value,
            'pdt_threshold': self.pdt_threshold,
            'day_trades_used': len(recent_day_trades),
            'day_trades_remaining': max(0, self.max_day_trades_per_5_days - len(recent_day_trades)),
            'max_day_trades': self.max_day_trades_per_5_days
        }
    
    def get_statistics(self) -> Dict:
        """Get overall intraday trading statistics"""
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for t in self.closed_trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in self.closed_trades)
        
        return {
            'total_intraday_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'open_positions': len(self.open_positions),
            'pdt_status': self.get_pdt_status(),
            'market_open': self.is_market_open(),
            'auto_close_active': self.should_auto_close()
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize engine
    engine = IntradayTradingEngine({
        'account_value': 10000.0,  # Under PDT threshold
        'max_intraday_positions': 3,
        'trailing_stop_enabled': True
    })
    
    # Check PDT status
    print("\n1. PDT Status:")
    print(engine.get_pdt_status())
    
    # Open position
    print("\n2. Opening position:")
    result = engine.open_intraday_position('AAPL', 'LONG', 10, 150.0)
    print(result)
    
    # Update with market data
    print("\n3. Updating positions:")
    engine.update_positions({'AAPL': 152.0})
    print(engine.get_open_positions())
    
    # Close position
    print("\n4. Closing position:")
    if engine.open_positions:
        pos_id = list(engine.open_positions.keys())[0]
        result = engine.close_intraday_position(pos_id, 152.0, "Manual close")
        print(result)
    
    # Get statistics
    print("\n5. Statistics:")
    print(engine.get_statistics())

