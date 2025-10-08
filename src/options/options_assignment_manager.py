"""
Options Assignment and Expiration Manager

Handles:
- Options expiration monitoring
- Assignment risk for short options
- Automatic expiration processing
- ITM/OTM detection
- Exercise decisions for long options
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
import pytz

logger = logging.getLogger(__name__)


class OptionStatus(Enum):
    """Option position status"""
    ACTIVE = "active"
    EXPIRING_TODAY = "expiring_today"
    EXPIRED = "expired"
    ASSIGNED = "assigned"
    EXERCISED = "exercised"


class ExpirationAction(Enum):
    """Actions to take at expiration"""
    EXERCISE = "exercise"  # Exercise long ITM option
    LET_EXPIRE = "let_expire"  # Let OTM option expire worthless
    CLOSE_POSITION = "close_position"  # Close before expiration
    AWAIT_ASSIGNMENT = "await_assignment"  # Short ITM option - may be assigned


@dataclass
class OptionPosition:
    """Represents an option position"""
    symbol: str
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiry: datetime
    quantity: int  # Positive = long, Negative = short
    entry_price: float
    current_price: float
    underlying_price: float
    status: OptionStatus = OptionStatus.ACTIVE
    days_to_expiry: int = 0
    is_itm: bool = False
    intrinsic_value: float = 0.0
    time_value: float = 0.0


@dataclass
class AssignmentEvent:
    """Represents an options assignment event"""
    event_id: str
    symbol: str
    option_type: str
    strike: float
    expiry: datetime
    quantity: int
    assignment_price: float
    stock_quantity: int  # Shares assigned
    timestamp: datetime
    reason: str


@dataclass
class ExerciseEvent:
    """Represents an options exercise event"""
    event_id: str
    symbol: str
    option_type: str
    strike: float
    expiry: datetime
    quantity: int
    exercise_price: float
    stock_quantity: int  # Shares received/delivered
    profit_loss: float
    timestamp: datetime


class OptionsAssignmentManager:
    """
    Manages options expiration and assignment.
    
    Features:
    - Monitors expiration dates
    - Detects ITM/OTM options
    - Handles assignment for short options
    - Auto-exercises long ITM options
    - Closes positions before expiration (optional)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Expiration settings
        self.auto_close_before_expiration = self.config.get('auto_close_before_expiration', True)
        self.days_before_expiration_to_close = self.config.get('days_before_expiration_to_close', 0)  # Close on expiration day
        self.auto_exercise_itm_threshold = self.config.get('auto_exercise_itm_threshold', 0.01)  # $0.01 ITM
        
        # Risk management
        self.max_assignment_risk = self.config.get('max_assignment_risk', 50000.0)  # $50k max
        self.alert_on_assignment_risk = self.config.get('alert_on_assignment_risk', True)
        
        # Market hours (Eastern Time)
        self.timezone = pytz.timezone('America/New_York')
        self.expiration_time = time(16, 0)  # Options expire at 4:00 PM ET
        
        # Position tracking
        self.positions: Dict[str, OptionPosition] = {}
        self.assignment_history: List[AssignmentEvent] = []
        self.exercise_history: List[ExerciseEvent] = []
        self.expiration_alerts: List[Dict] = []
        
        logger.info("Options Assignment Manager initialized")
    
    def add_position(self, symbol: str, option_type: str, strike: float, expiry: datetime,
                    quantity: int, entry_price: float) -> str:
        """
        Add an option position to track.
        
        Args:
            symbol: Option symbol
            option_type: 'CALL' or 'PUT'
            strike: Strike price
            expiry: Expiration date
            quantity: Number of contracts (positive = long, negative = short)
            entry_price: Entry price per contract
        
        Returns:
            Position ID
        """
        try:
            position = OptionPosition(
                symbol=symbol,
                option_type=option_type,
                strike=strike,
                expiry=expiry,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                underlying_price=strike
            )
            
            self.positions[symbol] = position
            
            logger.info(f"Added option position: {symbol} {quantity} contracts @ ${entry_price:.2f}")
            return symbol
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            raise
    
    def update_position(self, symbol: str, current_price: float, underlying_price: float):
        """
        Update position with current market data.
        
        Args:
            symbol: Option symbol
            current_price: Current option price
            underlying_price: Current underlying price
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"Position {symbol} not found")
                return
            
            position = self.positions[symbol]
            position.current_price = current_price
            position.underlying_price = underlying_price
            
            # Calculate days to expiry
            now = datetime.now(self.timezone)
            position.days_to_expiry = (position.expiry - now).days
            
            # Determine if ITM
            if position.option_type == 'CALL':
                position.is_itm = underlying_price > position.strike
                position.intrinsic_value = max(0, underlying_price - position.strike)
            else:  # PUT
                position.is_itm = underlying_price < position.strike
                position.intrinsic_value = max(0, position.strike - underlying_price)
            
            # Calculate time value
            position.time_value = max(0, current_price - position.intrinsic_value)
            
            # Update status
            if position.days_to_expiry == 0:
                position.status = OptionStatus.EXPIRING_TODAY
            elif position.days_to_expiry < 0:
                position.status = OptionStatus.EXPIRED
            
            logger.debug(f"Updated position: {symbol} | ITM: {position.is_itm} | DTE: {position.days_to_expiry}")
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def check_expiration_alerts(self) -> List[Dict]:
        """
        Check all positions for expiration alerts.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        try:
            for symbol, position in self.positions.items():
                if position.status == OptionStatus.EXPIRED:
                    continue
                
                # Alert for expiring today
                if position.days_to_expiry == 0:
                    action = self._determine_expiration_action(position)
                    
                    alert = {
                        'symbol': symbol,
                        'type': 'EXPIRING_TODAY',
                        'option_type': position.option_type,
                        'strike': position.strike,
                        'quantity': position.quantity,
                        'is_itm': position.is_itm,
                        'intrinsic_value': position.intrinsic_value,
                        'recommended_action': action.value,
                        'assignment_risk': self._calculate_assignment_risk(position)
                    }
                    
                    alerts.append(alert)
                
                # Alert for expiring soon
                elif position.days_to_expiry <= 7:
                    alerts.append({
                        'symbol': symbol,
                        'type': 'EXPIRING_SOON',
                        'days_to_expiry': position.days_to_expiry,
                        'is_itm': position.is_itm,
                        'quantity': position.quantity
                    })
            
            self.expiration_alerts = alerts
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking expiration alerts: {e}")
            return []
    
    def process_expiration(self, symbol: str) -> Dict:
        """
        Process expiration for a position.
        
        Args:
            symbol: Option symbol
        
        Returns:
            Result dictionary with action taken
        """
        try:
            if symbol not in self.positions:
                return {'success': False, 'reason': 'Position not found'}
            
            position = self.positions[symbol]
            
            # Determine action
            action = self._determine_expiration_action(position)
            
            result = {'success': True, 'action': action.value}
            
            if action == ExpirationAction.EXERCISE:
                # Exercise long ITM option
                exercise_result = self._exercise_option(position)
                result.update(exercise_result)
            
            elif action == ExpirationAction.AWAIT_ASSIGNMENT:
                # Short ITM option - may be assigned
                assignment_result = self._process_assignment(position)
                result.update(assignment_result)
            
            elif action == ExpirationAction.LET_EXPIRE:
                # Let OTM option expire worthless
                result['reason'] = 'Option expired worthless (OTM)'
                position.status = OptionStatus.EXPIRED
                logger.info(f"Option expired worthless: {symbol}")
            
            elif action == ExpirationAction.CLOSE_POSITION:
                # Close position before expiration
                result['reason'] = 'Position should be closed before expiration'
                logger.warning(f"Position not closed before expiration: {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing expiration: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _determine_expiration_action(self, position: OptionPosition) -> ExpirationAction:
        """Determine what action to take at expiration"""
        # Long position
        if position.quantity > 0:
            if position.is_itm and position.intrinsic_value >= self.auto_exercise_itm_threshold:
                return ExpirationAction.EXERCISE
            else:
                return ExpirationAction.LET_EXPIRE
        
        # Short position
        else:
            if position.is_itm:
                return ExpirationAction.AWAIT_ASSIGNMENT
            else:
                return ExpirationAction.LET_EXPIRE
    
    def _exercise_option(self, position: OptionPosition) -> Dict:
        """Exercise a long option"""
        try:
            # Calculate stock quantity and value
            stock_quantity = abs(position.quantity) * 100  # 1 contract = 100 shares
            
            if position.option_type == 'CALL':
                # Buy stock at strike price
                exercise_cost = position.strike * stock_quantity
                current_value = position.underlying_price * stock_quantity
                profit_loss = current_value - exercise_cost - (position.entry_price * abs(position.quantity) * 100)
            else:  # PUT
                # Sell stock at strike price
                exercise_value = position.strike * stock_quantity
                current_cost = position.underlying_price * stock_quantity
                profit_loss = exercise_value - current_cost - (position.entry_price * abs(position.quantity) * 100)
            
            # Create exercise event
            event = ExerciseEvent(
                event_id=f"ex_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=position.symbol,
                option_type=position.option_type,
                strike=position.strike,
                expiry=position.expiry,
                quantity=abs(position.quantity),
                exercise_price=position.strike,
                stock_quantity=stock_quantity,
                profit_loss=profit_loss,
                timestamp=datetime.now()
            )
            
            self.exercise_history.append(event)
            position.status = OptionStatus.EXERCISED
            
            logger.info(f"Exercised option: {position.symbol} | P&L: ${profit_loss:.2f} | Shares: {stock_quantity}")
            
            return {
                'exercised': True,
                'stock_quantity': stock_quantity,
                'exercise_price': position.strike,
                'profit_loss': profit_loss
            }
            
        except Exception as e:
            logger.error(f"Error exercising option: {e}")
            return {'exercised': False, 'reason': str(e)}
    
    def _process_assignment(self, position: OptionPosition) -> Dict:
        """Process assignment for a short option"""
        try:
            # Calculate stock quantity and value
            stock_quantity = abs(position.quantity) * 100
            
            if position.option_type == 'CALL':
                # Obligated to sell stock at strike price
                assignment_proceeds = position.strike * stock_quantity
                current_value = position.underlying_price * stock_quantity
                profit_loss = (position.entry_price * abs(position.quantity) * 100) + assignment_proceeds - current_value
            else:  # PUT
                # Obligated to buy stock at strike price
                assignment_cost = position.strike * stock_quantity
                current_value = position.underlying_price * stock_quantity
                profit_loss = (position.entry_price * abs(position.quantity) * 100) + current_value - assignment_cost
            
            # Create assignment event
            event = AssignmentEvent(
                event_id=f"as_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=position.symbol,
                option_type=position.option_type,
                strike=position.strike,
                expiry=position.expiry,
                quantity=abs(position.quantity),
                assignment_price=position.strike,
                stock_quantity=stock_quantity,
                timestamp=datetime.now(),
                reason='ITM at expiration'
            )
            
            self.assignment_history.append(event)
            position.status = OptionStatus.ASSIGNED
            
            logger.warning(f"Option assigned: {position.symbol} | Shares: {stock_quantity} | P&L: ${profit_loss:.2f}")
            
            return {
                'assigned': True,
                'stock_quantity': stock_quantity,
                'assignment_price': position.strike,
                'profit_loss': profit_loss
            }
            
        except Exception as e:
            logger.error(f"Error processing assignment: {e}")
            return {'assigned': False, 'reason': str(e)}
    
    def _calculate_assignment_risk(self, position: OptionPosition) -> float:
        """Calculate potential assignment risk for short position"""
        if position.quantity >= 0:  # Long position - no assignment risk
            return 0.0
        
        if not position.is_itm:  # OTM - low assignment risk
            return 0.0
        
        # ITM short position - calculate capital requirement
        stock_quantity = abs(position.quantity) * 100
        
        if position.option_type == 'CALL':
            # Would need to deliver shares at strike price
            risk = position.underlying_price * stock_quantity
        else:  # PUT
            # Would need to buy shares at strike price
            risk = position.strike * stock_quantity
        
        return risk
    
    def get_expiring_positions(self, days: int = 7) -> List[OptionPosition]:
        """Get positions expiring within specified days"""
        expiring = []
        for position in self.positions.values():
            if 0 <= position.days_to_expiry <= days:
                expiring.append(position)
        return sorted(expiring, key=lambda x: x.days_to_expiry)
    
    def get_assignment_risk_summary(self) -> Dict:
        """Get summary of assignment risk"""
        total_risk = 0.0
        at_risk_positions = []
        
        for symbol, position in self.positions.items():
            if position.quantity < 0 and position.is_itm:
                risk = self._calculate_assignment_risk(position)
                total_risk += risk
                at_risk_positions.append({
                    'symbol': symbol,
                    'option_type': position.option_type,
                    'strike': position.strike,
                    'quantity': position.quantity,
                    'days_to_expiry': position.days_to_expiry,
                    'risk': risk
                })
        
        return {
            'total_assignment_risk': total_risk,
            'at_risk_positions': at_risk_positions,
            'risk_alert': total_risk > self.max_assignment_risk
        }
    
    def get_statistics(self) -> Dict:
        """Get assignment and expiration statistics"""
        return {
            'total_positions': len(self.positions),
            'active_positions': sum(1 for p in self.positions.values() if p.status == OptionStatus.ACTIVE),
            'expiring_today': sum(1 for p in self.positions.values() if p.days_to_expiry == 0),
            'expiring_this_week': sum(1 for p in self.positions.values() if 0 <= p.days_to_expiry <= 7),
            'total_assignments': len(self.assignment_history),
            'total_exercises': len(self.exercise_history),
            'current_alerts': len(self.expiration_alerts),
            'assignment_risk': self.get_assignment_risk_summary()
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize manager
    manager = OptionsAssignmentManager()
    
    # Add positions
    print("\n1. Adding positions:")
    expiry = datetime.now() + timedelta(days=0)  # Expires today
    
    # Long call ITM
    manager.add_position('AAPL_CALL_150', 'CALL', 150.0, expiry, 1, 3.50)
    manager.update_position('AAPL_CALL_150', 5.00, 155.0)  # Underlying at $155
    
    # Short put ITM
    manager.add_position('AAPL_PUT_160', 'PUT', 160.0, expiry, -1, 4.00)
    manager.update_position('AAPL_PUT_160', 6.00, 155.0)  # Underlying at $155
    
    # Check alerts
    print("\n2. Checking expiration alerts:")
    alerts = manager.check_expiration_alerts()
    for alert in alerts:
        print(f"   {alert}")
    
    # Process expirations
    print("\n3. Processing expirations:")
    result1 = manager.process_expiration('AAPL_CALL_150')
    print(f"   CALL: {result1}")
    
    result2 = manager.process_expiration('AAPL_PUT_160')
    print(f"   PUT: {result2}")
    
    # Statistics
    print("\n4. Statistics:")
    stats = manager.get_statistics()
    print(f"   {stats}")

