"""
ETF Allocation Module
Handles automatic ETF purchases from trading profits
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ETFAllocation:
    """ETF allocation record"""
    symbol: str
    amount: float
    shares: float
    price: float
    timestamp: datetime
    status: str  # pending, filled, failed

class ETFAllocator:
    """Manages automatic ETF allocation from trading profits"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['risk']
        
        self.profit_config = self.config.get('profit_allocation', {})
        self.etf_targets = self.profit_config.get('etf_targets', [])
        self.etf_percentage = self.profit_config.get('etf_percentage', 0.20)
        self.min_threshold = self.profit_config.get('min_profit_threshold', 1000)
        self.rebalance_frequency = self.profit_config.get('rebalance_frequency', 'weekly')
        
        self.allocations = []  # List of ETFAllocation records
        self.total_allocated = 0.0
    
    def allocate_profits_to_etfs(self, profit_amount: float) -> List[ETFAllocation]:
        """Allocate profits to ETFs based on configuration"""
        try:
            if profit_amount < self.min_threshold:
                logger.info(f"Profit ${profit_amount:,.2f} CAD below threshold ${self.min_threshold:,.2f} CAD")
                return []
            
            etf_amount = profit_amount * self.etf_percentage
            logger.info(f"Allocating ${etf_amount:,.2f} CAD ({self.etf_percentage:.1%}) to ETFs from ${profit_amount:,.2f} CAD profit")
            
            allocations = []
            if self.etf_targets:
                amount_per_etf = etf_amount / len(self.etf_targets)
                
                for etf_symbol in self.etf_targets:
                    allocation = self.create_etf_allocation(etf_symbol, amount_per_etf)
                    if allocation:
                        allocations.append(allocation)
                        self.allocations.append(allocation)
                        self.total_allocated += amount_per_etf
            
            logger.info(f"Created {len(allocations)} ETF allocations")
            return allocations
            
        except Exception as e:
            logger.error(f"Failed to allocate profits to ETFs: {e}")
            return []
    
    def create_etf_allocation(self, symbol: str, amount: float) -> Optional[ETFAllocation]:
        """Create an ETF allocation record"""
        try:
            # Get current price (mock implementation)
            current_price = self.get_etf_price(symbol)
            if current_price <= 0:
                logger.error(f"Invalid price for {symbol}: ${current_price}")
                return None
            
            shares = amount / current_price
            
            allocation = ETFAllocation(
                symbol=symbol,
                amount=amount,
                shares=shares,
                price=current_price,
                timestamp=datetime.now(),
                status="pending"
            )
            
            logger.info(f"Created ETF allocation: {shares:.4f} shares of {symbol} at ${current_price:.2f}")
            return allocation
            
        except Exception as e:
            logger.error(f"Failed to create ETF allocation for {symbol}: {e}")
            return None
    
    def get_etf_price(self, symbol: str) -> float:
        """Get current ETF price (mock implementation)"""
        # Mock prices for testing
        mock_prices = {
            "VTI": 250.00,   # Vanguard Total Stock Market ETF
            "VEA": 45.50,    # Vanguard Developed Markets ETF
            "VWO": 42.75,    # Vanguard Emerging Markets ETF
            "BND": 75.25,    # Vanguard Total Bond Market ETF
            "VXUS": 55.80,   # Vanguard Total International Stock ETF
        }
        
        return mock_prices.get(symbol, 100.0)  # Default price
    
    def execute_etf_purchases(self, allocations: List[ETFAllocation]) -> List[ETFAllocation]:
        """Execute ETF purchases (mock implementation)"""
        try:
            executed_allocations = []
            
            for allocation in allocations:
                # Mock execution - in real implementation, this would call broker API
                allocation.status = "filled"
                executed_allocations.append(allocation)
                
                logger.info(f"Executed ETF purchase: {allocation.shares:.4f} shares of {allocation.symbol} for ${allocation.amount:,.2f} CAD")
            
            return executed_allocations
            
        except Exception as e:
            logger.error(f"Failed to execute ETF purchases: {e}")
            return []
    
    def get_allocation_summary(self) -> Dict:
        """Get summary of ETF allocations"""
        try:
            total_allocated = sum(a.amount for a in self.allocations if a.status == "filled")
            total_shares = sum(a.shares for a in self.allocations if a.status == "filled")
            
            by_symbol = {}
            for allocation in self.allocations:
                if allocation.status == "filled":
                    if allocation.symbol not in by_symbol:
                        by_symbol[allocation.symbol] = {
                            'shares': 0,
                            'amount': 0,
                            'avg_price': 0
                        }
                    
                    by_symbol[allocation.symbol]['shares'] += allocation.shares
                    by_symbol[allocation.symbol]['amount'] += allocation.amount
                    by_symbol[allocation.symbol]['avg_price'] = (
                        by_symbol[allocation.symbol]['amount'] / 
                        by_symbol[allocation.symbol]['shares']
                    )
            
            return {
                'total_allocated': total_allocated,
                'total_shares': total_shares,
                'allocations_count': len(self.allocations),
                'by_symbol': by_symbol,
                'etf_targets': self.etf_targets,
                'etf_percentage': self.etf_percentage,
                'rebalance_frequency': self.rebalance_frequency
            }
            
        except Exception as e:
            logger.error(f"Failed to get allocation summary: {e}")
            return {}
    
    def rebalance_etf_portfolio(self) -> List[ETFAllocation]:
        """Rebalance ETF portfolio based on target allocation"""
        try:
            logger.info("Starting ETF portfolio rebalancing...")
            
            # Get current portfolio value
            current_value = sum(a.amount for a in self.allocations if a.status == "filled")
            
            if current_value == 0:
                logger.info("No ETF holdings to rebalance")
                return []
            
            # Calculate target allocation for each ETF
            target_per_etf = current_value / len(self.etf_targets)
            
            rebalance_allocations = []
            for etf_symbol in self.etf_targets:
                current_etf_value = sum(
                    a.amount for a in self.allocations 
                    if a.symbol == etf_symbol and a.status == "filled"
                )
                
                difference = target_per_etf - current_etf_value
                
                if abs(difference) > 100:  # Only rebalance if difference > $100
                    if difference > 0:
                        # Buy more
                        allocation = self.create_etf_allocation(etf_symbol, difference)
                        if allocation:
                            rebalance_allocations.append(allocation)
                            logger.info(f"Rebalancing: Buy ${difference:,.2f} CAD of {etf_symbol}")
                    else:
                        # Sell excess (not implemented in this version)
                        logger.info(f"Rebalancing: Would sell ${abs(difference):,.2f} CAD of {etf_symbol}")
            
            return rebalance_allocations
            
        except Exception as e:
            logger.error(f"Failed to rebalance ETF portfolio: {e}")
            return []

# Test the ETF allocator
if __name__ == "__main__":
    allocator = ETFAllocator("config/risk_config.yaml")
    
    # Test profit allocation
    allocations = allocator.allocate_profits_to_etfs(5000.0)  # $5K profit
    print(f"Created {len(allocations)} ETF allocations")
    
    # Test execution
    executed = allocator.execute_etf_purchases(allocations)
    print(f"Executed {len(executed)} ETF purchases")
    
    # Test summary
    summary = allocator.get_allocation_summary()
    print(f"Allocation Summary: {summary}")
    
    # Test rebalancing
    rebalance_allocations = allocator.rebalance_etf_portfolio()
    print(f"Rebalancing created {len(rebalance_allocations)} new allocations")
