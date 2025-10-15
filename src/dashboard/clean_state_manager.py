"""
Clean state management with single source of truth
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path

@dataclass
class Trade:
    timestamp: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: int
    price: float
    reasoning: str
    confidence: float
    pnl: Optional[float] = None

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    session_id: str

@dataclass
class TradingSession:
    session_id: str
    mode: str  # DEMO or LIVE
    starting_capital: float
    current_capital: float
    positions: List[Position] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    ai_decisions_today: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = False

class CleanStateManager:
    def __init__(self):
        self.current_session: Optional[TradingSession] = None
        self.state_file = Path("data/clean_trading_state.json")
        self.load()  # Load existing session if available
        
    def start_new_session(self, capital: float, mode: str) -> str:
        """Start a fresh session, clearing all old data"""
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        self.current_session = TradingSession(
            session_id=session_id,
            mode=mode,
            starting_capital=capital,
            current_capital=capital,
            is_active=True
        )
        self.save()
        return session_id
    
    def get_current_state(self) -> Dict:
        """Get current session as dict for dashboard display"""
        if not self.current_session:
            return self._empty_state()
        return asdict(self.current_session)
    
    def add_trade(self, trade: Trade):
        """Add trade and update positions"""
        if not self.current_session:
            return
        
        self.current_session.trades.append(trade)
        self.current_session.ai_decisions_today += 1
        
        # Update positions
        if trade.action == "BUY":
            self._update_position_buy(trade)
        elif trade.action == "SELL":
            self._update_position_sell(trade)
        
        self.save()
    
    def update_position_prices(self, symbol: str, new_price: float):
        """Update current price for a position"""
        if not self.current_session:
            return
            
        for pos in self.current_session.positions:
            if pos.symbol == symbol:
                pos.current_price = new_price
                pos.pnl = (new_price - pos.avg_price) * pos.quantity
                pos.pnl_pct = ((new_price - pos.avg_price) / pos.avg_price) * 100 if pos.avg_price > 0 else 0
        
        self.save()
    
    def load(self):
        """Load existing session from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Convert dict back to TradingSession
                    self.current_session = TradingSession(**data)
            except Exception as e:
                print(f"Error loading session: {e}")
                self.current_session = None
    
    def save(self):
        """Persist state to disk"""
        if self.current_session:
            self.state_file.parent.mkdir(exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(asdict(self.current_session), f, indent=2)
    
    def _update_position_buy(self, trade: Trade):
        # Find existing position or create new
        existing = next((p for p in self.current_session.positions if p.symbol == trade.symbol), None)
        
        if existing:
            # Average down
            total_cost = existing.avg_price * existing.quantity + trade.price * trade.quantity
            existing.quantity += trade.quantity
            existing.avg_price = total_cost / existing.quantity
            existing.current_price = trade.price
        else:
            # New position
            self.current_session.positions.append(Position(
                symbol=trade.symbol,
                quantity=trade.quantity,
                avg_price=trade.price,
                current_price=trade.price,
                pnl=0,
                pnl_pct=0,
                session_id=self.current_session.session_id
            ))
        
        # Deduct cost from capital
        cost = trade.price * trade.quantity
        self.current_session.current_capital -= cost
    
    def _update_position_sell(self, trade: Trade):
        existing = next((p for p in self.current_session.positions if p.symbol == trade.symbol), None)
        if not existing:
            return
        
        # Calculate realized P&L
        realized_pnl = (trade.price - existing.avg_price) * trade.quantity
        trade.pnl = realized_pnl
        
        # Update position
        existing.quantity -= trade.quantity
        if existing.quantity <= 0:
            self.current_session.positions.remove(existing)
        
        # Add proceeds to capital
        proceeds = trade.price * trade.quantity
        self.current_session.current_capital += proceeds
    
    def _empty_state(self) -> Dict:
        return {
            "session_id": None,
            "mode": "DEMO",
            "starting_capital": 0,
            "current_capital": 0,
            "positions": [],
            "trades": [],
            "ai_decisions_today": 0,
            "is_active": False
        }

# Global instance - SINGLE SOURCE OF TRUTH
state_manager = CleanStateManager()
