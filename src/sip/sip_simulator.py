"""
SIP (Systematic Investment Plan) Simulator

Implements automated ETF investing strategy:
- 1% of daily simulated profit → VFV.TO (Vanguard S&P 500)
- Dollar-cost averaging
- Automatic rebalancing
- Long-term tracking
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class SIPTransaction:
    """SIP transaction record"""
    transaction_id: str
    date: datetime
    etf_symbol: str
    amount_cad: float
    share_price: float
    shares_purchased: float
    total_shares: float
    total_invested: float
    current_value: float
    profit_source: str  # 'daily_profit', 'manual_contribution'
    timestamp: datetime = field(default_factory=datetime.now)

class SIPSimulator:
    """
    SIP Simulator for Automated ETF Investing
    
    Strategy:
    - Allocate 1% of daily simulated profits to VFV.TO ETF
    - Use dollar-cost averaging (DCA) approach
    - Track long-term performance
    - Automatic rebalancing if needed
    
    Features:
    - Fractional share support
    - Transaction history
    - Performance tracking
    - Tax reporting data
    """
    
    def __init__(
        self,
        primary_etf: str = "VFV.TO",  # Vanguard S&P 500 Index ETF
        profit_allocation_pct: float = 0.01,  # 1% of profits
        min_investment_amount: float = 25.0,  # Minimum $25 CAD per transaction
        data_file: str = "data/sip_transactions.json"
    ):
        self.primary_etf = primary_etf
        self.profit_allocation_pct = profit_allocation_pct
        self.min_investment_amount = min_investment_amount
        self.data_file = Path(data_file)
        
        # Portfolio state
        self.total_shares = 0.0
        self.total_invested = 0.0
        self.avg_purchase_price = 0.0
        self.transactions: List[SIPTransaction] = []
        
        # Create data directory
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing transactions
        self._load_transactions()
        
        logger.info(f" SIP Simulator initialized: {primary_etf} @ {profit_allocation_pct:.1%} of profits")
    
    def calculate_investment_amount(self, daily_profit: float) -> float:
        """
        Calculate investment amount from daily profit
        
        Args:
            daily_profit: Daily trading profit in CAD
        
        Returns:
            Amount to invest in CAD
        """
        
        if daily_profit <= 0:
            return 0.0
        
        # Calculate 1% of profit
        investment_amount = daily_profit * self.profit_allocation_pct
        
        # Check minimum threshold
        if investment_amount < self.min_investment_amount:
            logger.debug(
                f"Investment amount ${investment_amount:.2f} below minimum ${self.min_investment_amount:.2f}, skipping"
            )
            return 0.0
        
        return investment_amount
    
    def execute_sip_investment(
        self,
        investment_amount: float,
        etf_price: float,
        profit_source: str = "daily_profit"
    ) -> Optional[SIPTransaction]:
        """
        Execute SIP investment
        
        Args:
            investment_amount: Amount to invest in CAD
            etf_price: Current ETF price
            profit_source: Source of profit
        
        Returns:
            SIPTransaction if successful
        """
        
        if investment_amount < self.min_investment_amount:
            logger.debug(f"Investment amount ${investment_amount:.2f} too small, skipping")
            return None
        
        if etf_price <= 0:
            logger.error(f"Invalid ETF price: ${etf_price:.2f}")
            return None
        
        try:
            # Calculate shares to purchase (fractional allowed)
            shares_purchased = investment_amount / etf_price
            
            # Update portfolio
            self.total_shares += shares_purchased
            self.total_invested += investment_amount
            
            # Calculate average purchase price
            if self.total_shares > 0:
                self.avg_purchase_price = self.total_invested / self.total_shares
            
            # Calculate current value
            current_value = self.total_shares * etf_price
            
            # Create transaction record
            transaction = SIPTransaction(
                transaction_id=f"SIP-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                date=datetime.now(),
                etf_symbol=self.primary_etf,
                amount_cad=investment_amount,
                share_price=etf_price,
                shares_purchased=shares_purchased,
                total_shares=self.total_shares,
                total_invested=self.total_invested,
                current_value=current_value,
                profit_source=profit_source
            )
            
            # Store transaction
            self.transactions.append(transaction)
            self._save_transactions()
            
            logger.info(
                f" SIP Investment: ${investment_amount:.2f} → {shares_purchased:.4f} shares of {self.primary_etf} @ ${etf_price:.2f}"
            )
            logger.info(
                f" Total: {self.total_shares:.4f} shares, ${self.total_invested:.2f} invested, "
                f"${current_value:.2f} current value"
            )
            
            return transaction
            
        except Exception as e:
            logger.error(f" SIP investment failed: {e}")
            return None
    
    def process_daily_profit(
        self,
        daily_profit: float,
        etf_price: float
    ) -> Optional[SIPTransaction]:
        """
        Process daily profit and execute SIP if applicable
        
        Args:
            daily_profit: Daily trading profit in CAD
            etf_price: Current ETF price
        
        Returns:
            SIPTransaction if investment executed
        """
        
        logger.info(f" Processing daily profit: ${daily_profit:.2f}")
        
        # Calculate investment amount
        investment_amount = self.calculate_investment_amount(daily_profit)
        
        if investment_amount <= 0:
            logger.debug(f"No SIP investment for profit ${daily_profit:.2f}")
            return None
        
        # Execute investment
        return self.execute_sip_investment(
            investment_amount=investment_amount,
            etf_price=etf_price,
            profit_source="daily_profit"
        )
    
    def get_portfolio_value(self, current_etf_price: float) -> Dict:
        """
        Get current portfolio value and metrics
        
        Args:
            current_etf_price: Current ETF price
        
        Returns:
            Portfolio metrics
        """
        
        current_value = self.total_shares * current_etf_price
        unrealized_pnl = current_value - self.total_invested
        return_pct = (unrealized_pnl / self.total_invested * 100) if self.total_invested > 0 else 0.0
        
        return {
            'etf_symbol': self.primary_etf,
            'total_shares': self.total_shares,
            'total_invested': self.total_invested,
            'current_price': current_etf_price,
            'current_value': current_value,
            'avg_purchase_price': self.avg_purchase_price,
            'unrealized_pnl': unrealized_pnl,
            'return_pct': return_pct,
            'num_transactions': len(self.transactions)
        }
    
    def get_performance_metrics(self, current_etf_price: float) -> Dict:
        """
        Get detailed performance metrics
        
        Args:
            current_etf_price: Current ETF price
        
        Returns:
            Performance metrics
        """
        
        if not self.transactions:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'best_transaction': None,
                'worst_transaction': None,
                'avg_transaction_size': 0.0
            }
        
        # Calculate returns for each transaction
        transaction_returns = []
        for txn in self.transactions:
            txn_return = ((current_etf_price - txn.share_price) / txn.share_price) * 100
            transaction_returns.append({
                'transaction_id': txn.transaction_id,
                'date': txn.date,
                'return_pct': txn_return,
                'amount': txn.amount_cad
            })
        
        # Sort by return
        sorted_returns = sorted(transaction_returns, key=lambda x: x['return_pct'], reverse=True)
        
        # Calculate metrics
        total_return = ((current_etf_price - self.avg_purchase_price) / self.avg_purchase_price) * 100
        
        # Annualized return (if > 1 year of data)
        first_transaction = min(self.transactions, key=lambda x: x.date)
        days_held = (datetime.now() - first_transaction.date).days
        years_held = days_held / 365.25
        
        if years_held >= 1:
            annualized_return = ((1 + (total_return / 100)) ** (1 / years_held) - 1) * 100
        else:
            annualized_return = 0.0
        
        avg_transaction = sum(txn.amount_cad for txn in self.transactions) / len(self.transactions)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'best_transaction': sorted_returns[0] if sorted_returns else None,
            'worst_transaction': sorted_returns[-1] if sorted_returns else None,
            'avg_transaction_size': avg_transaction,
            'total_transactions': len(self.transactions),
            'days_active': days_held,
            'years_active': years_held
        }
    
    def get_monthly_summary(self, year: int, month: int) -> Dict:
        """
        Get monthly SIP summary
        
        Args:
            year: Year
            month: Month (1-12)
        
        Returns:
            Monthly summary
        """
        
        # Filter transactions for the month
        monthly_txns = [
            txn for txn in self.transactions
            if txn.date.year == year and txn.date.month == month
        ]
        
        if not monthly_txns:
            return {
                'year': year,
                'month': month,
                'num_investments': 0,
                'total_invested': 0.0,
                'shares_purchased': 0.0
            }
        
        total_invested = sum(txn.amount_cad for txn in monthly_txns)
        shares_purchased = sum(txn.shares_purchased for txn in monthly_txns)
        avg_price = total_invested / shares_purchased if shares_purchased > 0 else 0.0
        
        return {
            'year': year,
            'month': month,
            'num_investments': len(monthly_txns),
            'total_invested': total_invested,
            'shares_purchased': shares_purchased,
            'avg_purchase_price': avg_price,
            'transactions': [
                {
                    'date': txn.date.isoformat(),
                    'amount': txn.amount_cad,
                    'shares': txn.shares_purchased,
                    'price': txn.share_price
                }
                for txn in monthly_txns
            ]
        }
    
    def get_tax_report(self, year: int) -> Dict:
        """
        Generate tax report for the year
        
        Args:
            year: Tax year
        
        Returns:
            Tax report data
        """
        
        # Filter transactions for the year
        yearly_txns = [
            txn for txn in self.transactions
            if txn.date.year == year
        ]
        
        total_invested = sum(txn.amount_cad for txn in yearly_txns)
        
        return {
            'tax_year': year,
            'etf_symbol': self.primary_etf,
            'total_contributions': total_invested,
            'num_transactions': len(yearly_txns),
            'monthly_breakdown': [
                self.get_monthly_summary(year, month)
                for month in range(1, 13)
            ]
        }
    
    def _save_transactions(self):
        """Save transactions to file"""
        try:
            data = {
                'etf_symbol': self.primary_etf,
                'total_shares': self.total_shares,
                'total_invested': self.total_invested,
                'avg_purchase_price': self.avg_purchase_price,
                'transactions': [
                    {
                        'transaction_id': txn.transaction_id,
                        'date': txn.date.isoformat(),
                        'etf_symbol': txn.etf_symbol,
                        'amount_cad': txn.amount_cad,
                        'share_price': txn.share_price,
                        'shares_purchased': txn.shares_purchased,
                        'total_shares': txn.total_shares,
                        'total_invested': txn.total_invested,
                        'current_value': txn.current_value,
                        'profit_source': txn.profit_source,
                        'timestamp': txn.timestamp.isoformat()
                    }
                    for txn in self.transactions
                ]
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f" Saved {len(self.transactions)} SIP transactions")
            
        except Exception as e:
            logger.error(f" Failed to save transactions: {e}")
    
    def _load_transactions(self):
        """Load transactions from file"""
        if not self.data_file.exists():
            logger.info(" No existing SIP transactions found")
            return
        
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            self.primary_etf = data.get('etf_symbol', self.primary_etf)
            self.total_shares = data.get('total_shares', 0.0)
            self.total_invested = data.get('total_invested', 0.0)
            self.avg_purchase_price = data.get('avg_purchase_price', 0.0)
            
            # Load transactions
            for txn_data in data.get('transactions', []):
                txn = SIPTransaction(
                    transaction_id=txn_data['transaction_id'],
                    date=datetime.fromisoformat(txn_data['date']),
                    etf_symbol=txn_data['etf_symbol'],
                    amount_cad=txn_data['amount_cad'],
                    share_price=txn_data['share_price'],
                    shares_purchased=txn_data['shares_purchased'],
                    total_shares=txn_data['total_shares'],
                    total_invested=txn_data['total_invested'],
                    current_value=txn_data['current_value'],
                    profit_source=txn_data['profit_source'],
                    timestamp=datetime.fromisoformat(txn_data['timestamp'])
                )
                self.transactions.append(txn)
            
            logger.info(f" Loaded {len(self.transactions)} SIP transactions")
            
        except Exception as e:
            logger.error(f" Failed to load transactions: {e}")

# Global simulator instance
_simulator_instance = None

def get_sip_simulator() -> SIPSimulator:
    """Get global SIP simulator instance"""
    global _simulator_instance
    if _simulator_instance is None:
        _simulator_instance = SIPSimulator()
    return _simulator_instance

