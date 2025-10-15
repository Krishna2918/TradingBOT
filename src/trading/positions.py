"""
Position Management System - Dual Mode Position Tracking

This module manages position tracking for both LIVE and DEMO modes,
providing complete position lifecycle management with real-time P&L calculation.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

from src.config.mode_manager import get_mode_manager, get_current_mode
from src.config.database import get_connection, execute_query, execute_update

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a trading position."""
    id: Optional[int]
    symbol: str
    entry_price: float
    quantity: int
    entry_time: datetime
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    status: str  # 'OPEN', 'CLOSED', 'PENDING'
    stop_loss: Optional[float]
    take_profit: Optional[float]
    pnl: float
    pnl_percent: float
    mode: str
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self):
        """Validate position data after initialization."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if self.status not in ['OPEN', 'CLOSED', 'PENDING']:
            raise ValueError("Invalid status")
        if self.mode not in ['LIVE', 'DEMO']:
            raise ValueError("Invalid mode")

class PositionManager:
    """Manages trading positions for dual mode operation."""
    
    def __init__(self, mode: Optional[str] = None):
        """Initialize Position Manager."""
        self.mode_manager = get_mode_manager()
        self.mode = mode  # Store mode parameter for compatibility
        logger.info("Position Manager initialized")
    
    def track_position(self, symbol: str, entry_price: float, quantity: int, 
                      stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                      mode: Optional[str] = None) -> Position:
        """Track a new position."""
        if mode is None:
            mode = get_current_mode()
        
        # Validate inputs
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty")
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        # Calculate initial P&L (should be 0 for new position)
        pnl = 0.0
        pnl_percent = 0.0
        
        # Insert position into database
        query = """
            INSERT INTO positions 
            (symbol, entry_price, quantity, entry_time, status, stop_loss, take_profit, pnl, pnl_percent, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        entry_time = datetime.now()
        params = (symbol.upper(), entry_price, quantity, entry_time, 'OPEN', 
                 stop_loss, take_profit, pnl, pnl_percent, mode)
        
        with get_connection(mode) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            position_id = cursor.lastrowid
            conn.commit()
        
        # Create Position object
        position = Position(
            id=position_id,
            symbol=symbol.upper(),
            entry_price=entry_price,
            quantity=quantity,
            entry_time=entry_time,
            exit_price=None,
            exit_time=None,
            status='OPEN',
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl=pnl,
            pnl_percent=pnl_percent,
            mode=mode,
            created_at=entry_time,
            updated_at=entry_time
        )
        
        logger.info(f"Position tracked: {symbol} {quantity} shares @ ${entry_price:.2f} in {mode} mode")
        return position
    
    def get_open_positions(self, mode: Optional[str] = None) -> List[Position]:
        """Get all open positions for the specified mode."""
        if mode is None:
            mode = get_current_mode()
        
        query = """
            SELECT * FROM positions 
            WHERE status = 'OPEN' AND mode = ?
            ORDER BY entry_time DESC
        """
        
        rows = execute_query(query, (mode,), mode)
        positions = []
        
        for row in rows:
            position = Position(
                id=row['id'],
                symbol=row['symbol'],
                entry_price=row['entry_price'],
                quantity=row['quantity'],
                entry_time=datetime.fromisoformat(row['entry_time']),
                exit_price=row['exit_price'],
                exit_time=datetime.fromisoformat(row['exit_time']) if row['exit_time'] else None,
                status=row['status'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                pnl=row['pnl'],
                pnl_percent=row['pnl_percent'],
                mode=row['mode'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            )
            positions.append(position)
        
        logger.info(f"Retrieved {len(positions)} open positions for {mode} mode")
        return positions
    
    def get_position_by_id(self, position_id: int, mode: Optional[str] = None) -> Optional[Position]:
        """Get a specific position by ID."""
        if mode is None:
            mode = get_current_mode()
        
        query = "SELECT * FROM positions WHERE id = ? AND mode = ?"
        rows = execute_query(query, (position_id, mode), mode)
        
        if not rows:
            return None
        
        row = rows[0]
        return Position(
            id=row['id'],
            symbol=row['symbol'],
            entry_price=row['entry_price'],
            quantity=row['quantity'],
            entry_time=datetime.fromisoformat(row['entry_time']),
            exit_price=row['exit_price'],
            exit_time=datetime.fromisoformat(row['exit_time']) if row['exit_time'] else None,
            status=row['status'],
            stop_loss=row['stop_loss'],
            take_profit=row['take_profit'],
            pnl=row['pnl'],
            pnl_percent=row['pnl_percent'],
            mode=row['mode'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )
    
    def get_position_by_symbol(self, symbol: str, mode: Optional[str] = None) -> Optional[Position]:
        """Get open position for a specific symbol."""
        if mode is None:
            mode = get_current_mode()
        
        query = """
            SELECT * FROM positions 
            WHERE symbol = ? AND status = 'OPEN' AND mode = ?
            ORDER BY entry_time DESC
            LIMIT 1
        """
        
        rows = execute_query(query, (symbol.upper(), mode), mode)
        
        if not rows:
            return None
        
        row = rows[0]
        return Position(
            id=row['id'],
            symbol=row['symbol'],
            entry_price=row['entry_price'],
            quantity=row['quantity'],
            entry_time=datetime.fromisoformat(row['entry_time']),
            exit_price=row['exit_price'],
            exit_time=datetime.fromisoformat(row['exit_time']) if row['exit_time'] else None,
            status=row['status'],
            stop_loss=row['stop_loss'],
            take_profit=row['take_profit'],
            pnl=row['pnl'],
            pnl_percent=row['pnl_percent'],
            mode=row['mode'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )
    
    def calculate_pnl(self, symbol: str, current_price: float, mode: Optional[str] = None) -> float:
        """Calculate P&L for a position based on current price."""
        if mode is None:
            mode = get_current_mode()
        
        position = self.get_position_by_symbol(symbol, mode)
        if not position:
            return 0.0
        
        # Calculate P&L: (current_price - entry_price) * quantity
        pnl = (current_price - position.entry_price) * position.quantity
        return round(pnl, 2)
    
    def calculate_pnl_percent(self, symbol: str, current_price: float, mode: Optional[str] = None) -> float:
        """Calculate P&L percentage for a position."""
        if mode is None:
            mode = get_current_mode()
        
        position = self.get_position_by_symbol(symbol, mode)
        if not position:
            return 0.0
        
        # Calculate P&L percentage: ((current_price - entry_price) / entry_price) * 100
        pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        return round(pnl_percent, 2)
    
    def update_position_pnl(self, symbol: str, current_price: float, mode: Optional[str] = None) -> bool:
        """Update P&L for a position based on current price."""
        if mode is None:
            mode = get_current_mode()
        
        position = self.get_position_by_symbol(symbol, mode)
        if not position:
            logger.warning(f"No open position found for {symbol} in {mode} mode")
            return False
        
        # Calculate new P&L
        pnl = self.calculate_pnl(symbol, current_price, mode)
        pnl_percent = self.calculate_pnl_percent(symbol, current_price, mode)
        
        # Update database
        query = """
            UPDATE positions 
            SET pnl = ?, pnl_percent = ?, updated_at = ?
            WHERE id = ? AND mode = ?
        """
        
        updated_rows = execute_update(query, (pnl, pnl_percent, datetime.now(), position.id, mode), mode)
        
        if updated_rows > 0:
            logger.info(f"Updated P&L for {symbol}: ${pnl:.2f} ({pnl_percent:.2f}%)")
            return True
        else:
            logger.error(f"Failed to update P&L for {symbol}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, mode: Optional[str] = None) -> Optional[Position]:
        """Close a position and calculate final P&L."""
        if mode is None:
            mode = get_current_mode()
        
        position = self.get_position_by_symbol(symbol, mode)
        if not position:
            logger.warning(f"No open position found for {symbol} in {mode} mode")
            return None
        
        # Calculate final P&L
        pnl = self.calculate_pnl(symbol, exit_price, mode)
        pnl_percent = self.calculate_pnl_percent(symbol, exit_price, mode)
        
        # Update position in database
        query = """
            UPDATE positions 
            SET exit_price = ?, exit_time = ?, status = 'CLOSED', pnl = ?, pnl_percent = ?, updated_at = ?
            WHERE id = ? AND mode = ?
        """
        
        exit_time = datetime.now()
        updated_rows = execute_update(query, (exit_price, exit_time, pnl, pnl_percent, exit_time, position.id, mode), mode)
        
        if updated_rows > 0:
            # Create updated position object
            closed_position = Position(
                id=position.id,
                symbol=position.symbol,
                entry_price=position.entry_price,
                quantity=position.quantity,
                entry_time=position.entry_time,
                exit_price=exit_price,
                exit_time=exit_time,
                status='CLOSED',
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                pnl=pnl,
                pnl_percent=pnl_percent,
                mode=position.mode,
                created_at=position.created_at,
                updated_at=exit_time
            )
            
            logger.info(f"Position closed: {symbol} @ ${exit_price:.2f}, P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")
            return closed_position
        else:
            logger.error(f"Failed to close position for {symbol}")
            return None
    
    def get_position_metrics(self, symbol: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive metrics for a position."""
        if mode is None:
            mode = get_current_mode()
        
        position = self.get_position_by_symbol(symbol, mode)
        if not position:
            return {}
        
        # Calculate additional metrics
        current_time = datetime.now()
        holding_days = (current_time - position.entry_time).days
        holding_hours = (current_time - position.entry_time).total_seconds() / 3600
        
        # Calculate unrealized P&L (assuming current price = entry price for now)
        # This would be updated with real-time prices in production
        unrealized_pnl = position.pnl
        unrealized_pnl_percent = position.pnl_percent
        
        metrics = {
            "symbol": position.symbol,
            "entry_price": position.entry_price,
            "quantity": position.quantity,
            "entry_time": position.entry_time,
            "holding_days": holding_days,
            "holding_hours": round(holding_hours, 2),
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_percent": unrealized_pnl_percent,
            "status": position.status,
            "mode": position.mode
        }
        
        return metrics
    
    def get_all_positions(self, mode: Optional[str] = None) -> List[Position]:
        """Get all positions (open and closed) for the specified mode."""
        if mode is None:
            mode = get_current_mode()
        
        query = """
            SELECT * FROM positions 
            WHERE mode = ?
            ORDER BY entry_time DESC
        """
        
        rows = execute_query(query, (mode,), mode)
        positions = []
        
        for row in rows:
            position = Position(
                id=row['id'],
                symbol=row['symbol'],
                entry_price=row['entry_price'],
                quantity=row['quantity'],
                entry_time=datetime.fromisoformat(row['entry_time']),
                exit_price=row['exit_price'],
                exit_time=datetime.fromisoformat(row['exit_time']) if row['exit_time'] else None,
                status=row['status'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                pnl=row['pnl'],
                pnl_percent=row['pnl_percent'],
                mode=row['mode'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            )
            positions.append(position)
        
        return positions
    
    def get_portfolio_summary(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get portfolio summary for the specified mode."""
        if mode is None:
            mode = get_current_mode()
        
        open_positions = self.get_open_positions(mode)
        all_positions = self.get_all_positions(mode)
        
        # Calculate summary metrics
        total_open_positions = len(open_positions)
        total_positions = len(all_positions)
        
        # Calculate total unrealized P&L
        total_unrealized_pnl = sum(pos.pnl for pos in open_positions)
        total_unrealized_pnl_percent = sum(pos.pnl_percent for pos in open_positions)
        
        # Calculate total realized P&L (closed positions)
        closed_positions = [pos for pos in all_positions if pos.status == 'CLOSED']
        total_realized_pnl = sum(pos.pnl for pos in closed_positions)
        
        # Calculate total invested amount
        total_invested = sum(pos.entry_price * pos.quantity for pos in open_positions)
        
        summary = {
            "mode": mode,
            "total_open_positions": total_open_positions,
            "total_positions": total_positions,
            "total_invested": round(total_invested, 2),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "total_unrealized_pnl_percent": round(total_unrealized_pnl_percent, 2),
            "total_realized_pnl": round(total_realized_pnl, 2),
            "total_pnl": round(total_unrealized_pnl + total_realized_pnl, 2),
            "positions": [
                {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "pnl": pos.pnl,
                    "pnl_percent": pos.pnl_percent,
                    "status": pos.status
                }
                for pos in open_positions
            ]
        }
        
        return summary
    
    def validate_position_data(self, mode: Optional[str] = None) -> bool:
        """Validate position data integrity."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Check for invalid data
            query = """
                SELECT COUNT(*) as invalid_count FROM positions 
                WHERE mode = ? AND (quantity <= 0 OR entry_price <= 0 OR symbol = '')
            """
            
            result = execute_query(query, (mode,), mode)
            invalid_count = result[0]['invalid_count']
            
            if invalid_count > 0:
                logger.error(f"Found {invalid_count} invalid positions in {mode} mode")
                return False
            
            # Check for orphaned positions (positions without corresponding orders)
            query = """
                SELECT COUNT(*) as orphaned_count FROM positions p
                LEFT JOIN orders o ON p.id = o.position_id
                WHERE p.mode = ? AND o.id IS NULL
            """
            
            result = execute_query(query, (mode,), mode)
            orphaned_count = result[0]['orphaned_count']
            
            if orphaned_count > 0:
                logger.warning(f"Found {orphaned_count} orphaned positions in {mode} mode")
            
            logger.info(f"Position data validation passed for {mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Position data validation error: {e}")
            return False

# Global position manager instance
_position_manager: Optional[PositionManager] = None

def get_position_manager() -> PositionManager:
    """Get the global position manager instance."""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager()
    return _position_manager

def track_position(symbol: str, entry_price: float, quantity: int, 
                  stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                  mode: Optional[str] = None) -> Position:
    """Track a new position."""
    return get_position_manager().track_position(symbol, entry_price, quantity, stop_loss, take_profit, mode)

def get_open_positions(mode: Optional[str] = None) -> List[Position]:
    """Get all open positions."""
    return get_position_manager().get_open_positions(mode)

def get_position_by_symbol(symbol: str, mode: Optional[str] = None) -> Optional[Position]:
    """Get position by symbol."""
    return get_position_manager().get_position_by_symbol(symbol, mode)

def calculate_pnl(symbol: str, current_price: float, mode: Optional[str] = None) -> float:
    """Calculate P&L for a position."""
    return get_position_manager().calculate_pnl(symbol, current_price, mode)

def close_position(symbol: str, exit_price: float, mode: Optional[str] = None) -> Optional[Position]:
    """Close a position."""
    return get_position_manager().close_position(symbol, exit_price, mode)

def get_portfolio_summary(mode: Optional[str] = None) -> Dict[str, Any]:
    """Get portfolio summary."""
    return get_position_manager().get_portfolio_summary(mode)
