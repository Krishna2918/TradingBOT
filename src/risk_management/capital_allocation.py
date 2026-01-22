"""
Capital Allocation and Risk Management Module
Handles dynamic capital allocation based on performance and risk metrics
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger(__name__)

@dataclass
class CapitalState:
    """Current capital allocation state"""
    total_capital: float
    active_capital: float
    safety_reserve: float
    allocated_capital: float
    available_capital: float
    daily_pnl: float
    max_drawdown: float
    consecutive_losses: int
    last_loss_time: Optional[datetime]
    etf_allocation: float
    total_profits: float
    etf_invested: float

class CapitalAllocator:
    """Manages dynamic capital allocation based on risk metrics"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['risk']
        
        self.capital_state = CapitalState(
            total_capital=self.config['capital']['total_capital'],
            active_capital=self.config['capital']['active_capital'],
            safety_reserve=self.config['capital']['safety_reserve'],
            allocated_capital=0.0,
            available_capital=self.config['capital']['active_capital'],
            daily_pnl=0.0,
            max_drawdown=0.0,
            consecutive_losses=0,
            last_loss_time=None,
            etf_allocation=self.config['capital'].get('etf_allocation', 0.20),
            total_profits=0.0,
            etf_invested=0.0
        )
        
        self.limits = self.config['limits']
        self.cool_down_active = False
        self.cool_down_start = None
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L and check limits"""
        self.capital_state.daily_pnl = pnl
        
        # Check daily loss limit
        if pnl < -(self.capital_state.total_capital * self.limits['daily_loss_limit']):
            logger.warning(f"Daily loss limit exceeded: ${pnl:,.2f} CAD")
            self.activate_kill_switch("daily_loss_limit")
    
    def update_drawdown(self, drawdown: float):
        """Update maximum drawdown and check limits"""
        if drawdown > self.capital_state.max_drawdown:
            self.capital_state.max_drawdown = drawdown
            
        if drawdown > self.limits['max_drawdown']:
            logger.warning(f"Maximum drawdown exceeded: {drawdown}")
            self.activate_kill_switch("max_drawdown")
    
    def record_loss(self):
        """Record a loss and check consecutive loss limits"""
        current_time = datetime.now()
        self.capital_state.consecutive_losses += 1
        self.capital_state.last_loss_time = current_time
        
        # Check consecutive loss limit
        if self.capital_state.consecutive_losses >= self.limits['consecutive_loss_limit']:
            logger.warning(f"Consecutive loss limit reached: {self.capital_state.consecutive_losses}")
            self.activate_cool_down()
    
    def record_win(self, profit_amount: float = 0.0):
        """Record a win and reset consecutive losses"""
        self.capital_state.consecutive_losses = 0
        self.capital_state.last_loss_time = None
        
        # Update total profits
        if profit_amount > 0:
            self.capital_state.total_profits += profit_amount
            
            # Check if we should allocate to ETFs
            self.check_etf_allocation(profit_amount)
        
        # Exit cool-down mode if active
        if self.cool_down_active:
            self.exit_cool_down()
    
    def activate_cool_down(self):
        """Activate cool-down mode"""
        self.cool_down_active = True
        self.cool_down_start = datetime.now()
        logger.info("Cool-down mode activated")
    
    def exit_cool_down(self):
        """Exit cool-down mode"""
        self.cool_down_active = False
        self.cool_down_start = None
        logger.info("Cool-down mode deactivated")
    
    def is_cool_down_active(self) -> bool:
        """Check if cool-down mode is active"""
        if not self.cool_down_active:
            return False
        
        # Check if cool-down period has elapsed
        if self.cool_down_start:
            elapsed = datetime.now() - self.cool_down_start
            if elapsed.total_seconds() >= (self.limits['cool_down_minutes'] * 60):
                self.exit_cool_down()
                return False
        
        return True
    
    def calculate_position_size(self, base_size: float, strategy: str) -> float:
        """Calculate adjusted position size based on risk state"""
        if self.is_cool_down_active():
            return base_size * 0.25  # 25% during cool-down
        
        # Adjust based on consecutive losses
        if self.capital_state.consecutive_losses >= 2:
            return base_size * 0.5   # 50% after 2+ losses
        
        # Adjust based on drawdown
        if self.capital_state.max_drawdown > 0.1:  # 10%
            return base_size * 0.8   # 80% during high drawdown
        
        return base_size
    
    def get_capital_state(self) -> CapitalState:
        """Get current capital state"""
        return self.capital_state
    
    def check_etf_allocation(self, profit_amount: float):
        """Check if profits should be allocated to ETFs"""
        try:
            profit_config = self.config.get('profit_allocation', {})
            etf_percentage = profit_config.get('etf_percentage', 0.20)
            min_threshold = profit_config.get('min_profit_threshold', 1000)
            
            # Only allocate if profit exceeds minimum threshold
            if profit_amount >= min_threshold:
                etf_amount = profit_amount * etf_percentage
                self.capital_state.etf_invested += etf_amount
                
                logger.info(f"Allocating ${etf_amount:,.2f} CAD ({etf_percentage:.1%}) to ETFs from ${profit_amount:,.2f} CAD profit")
                
                # Trigger ETF purchase (implementation in execution module)
                self.trigger_etf_purchase(etf_amount)
            
        except Exception as e:
            logger.error(f"Failed to check ETF allocation: {e}")
    
    def trigger_etf_purchase(self, amount: float):
        """Trigger ETF purchase with specified amount"""
        try:
            profit_config = self.config.get('profit_allocation', {})
            etf_targets = profit_config.get('etf_targets', [])
            
            if etf_targets:
                # Distribute amount equally among ETF targets
                amount_per_etf = amount / len(etf_targets)
                
                for etf_symbol in etf_targets:
                    logger.info(f"Triggering purchase of ${amount_per_etf:,.2f} CAD of {etf_symbol}")
                    # TODO: Implement actual ETF purchase in execution module
                    
        except Exception as e:
            logger.error(f"Failed to trigger ETF purchase: {e}")
    
    def get_etf_allocation_summary(self) -> Dict:
        """Get summary of ETF allocation"""
        return {
            'total_profits': self.capital_state.total_profits,
            'etf_invested': self.capital_state.etf_invested,
            'etf_percentage': self.capital_state.etf_allocation,
            'etf_targets': self.config.get('profit_allocation', {}).get('etf_targets', []),
            'rebalance_frequency': self.config.get('profit_allocation', {}).get('rebalance_frequency', 'weekly')
        }
    
    def activate_kill_switch(self, reason: str):
        """Activate kill switch for specified reason"""
        logger.critical(f"Kill switch activated: {reason}")
        # Implementation will be added in Week 3
        pass

# Test the capital allocator
if __name__ == "__main__":
    # Test with mock config
    allocator = CapitalAllocator("config/risk_config.yaml")
    
    # Test consecutive losses
    for i in range(4):
        allocator.record_loss()
        print(f"Loss {i+1}: Cool-down active: {allocator.is_cool_down_active()}")
    
    # Test cool-down exit
    allocator.record_win()
    print(f"After win: Cool-down active: {allocator.is_cool_down_active()}")

