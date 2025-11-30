"""
ETF Allocation Module
Handles automatic ETF purchases from trading profits

Complete implementation with broker integration for real order execution.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


class AllocationStatus(Enum):
    """ETF allocation status"""
    PENDING = "pending"
    EXECUTING = "executing"
    FILLED = "filled"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ETFAllocation:
    """ETF allocation record"""
    symbol: str
    amount: float
    shares: float
    price: float
    timestamp: datetime
    status: str  # pending, filled, failed
    order_id: Optional[str] = None
    execution_price: Optional[float] = None
    execution_timestamp: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class AllocationResult:
    """Result of an ETF allocation execution"""
    success: bool
    allocation: ETFAllocation
    message: str
    order_response: Optional[Dict[str, Any]] = None


class ETFAllocator:
    """
    Manages automatic ETF allocation from trading profits.

    Features:
    - Real broker integration for order execution
    - Real-time price fetching via yfinance
    - Automatic profit allocation to ETFs
    - Portfolio rebalancing
    - Transaction logging
    """

    def __init__(self, config_path: str = "config/risk_config.yaml",
                 broker: Optional[Any] = None,
                 mode: str = "DEMO"):
        """
        Initialize ETF Allocator.

        Args:
            config_path: Path to risk configuration YAML
            broker: Optional broker instance for order execution
            mode: Trading mode - 'DEMO' or 'LIVE'
        """
        self.mode = mode
        self.broker = broker

        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.config = config.get('risk', {})
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            self.config = {}
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using defaults")
            self.config = {}

        self.profit_config = self.config.get('profit_allocation', {})

        # Default Canadian ETF targets
        self.etf_targets = self.profit_config.get('etf_targets', [
            'VFV.TO',   # Vanguard S&P 500 Index ETF (CAD)
            'XIC.TO',   # iShares Core S&P/TSX Capped Composite Index ETF
            'ZAG.TO',   # BMO Aggregate Bond Index ETF
        ])

        self.etf_percentage = self.profit_config.get('etf_percentage', 0.20)  # 20% of profits
        self.min_threshold = self.profit_config.get('min_profit_threshold', 100)  # $100 CAD min
        self.rebalance_frequency = self.profit_config.get('rebalance_frequency', 'weekly')
        self.max_retries = self.profit_config.get('max_retries', 3)

        # State tracking
        self.allocations: List[ETFAllocation] = []
        self.total_allocated = 0.0
        self.pending_allocations: List[ETFAllocation] = []

        # Price cache
        self.price_cache: Dict[str, tuple] = {}  # symbol -> (price, timestamp)
        self.cache_ttl_seconds = 60  # 1 minute cache

        logger.info(f"ETF Allocator initialized in {mode} mode with targets: {self.etf_targets}")

    def set_broker(self, broker: Any):
        """Set or update the broker instance"""
        self.broker = broker
        logger.info("Broker instance updated for ETF Allocator")

    def get_etf_price(self, symbol: str, use_cache: bool = True) -> float:
        """
        Get current ETF price using yfinance with caching.

        Args:
            symbol: ETF symbol (e.g., 'VFV.TO')
            use_cache: Whether to use cached prices

        Returns:
            Current price or 0.0 if failed
        """
        # Check cache first
        if use_cache and symbol in self.price_cache:
            cached_price, cached_time = self.price_cache[symbol]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl_seconds):
                return cached_price

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')

            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                self.price_cache[symbol] = (price, datetime.now())
                return price

            # Fallback to info
            info = ticker.info
            if 'regularMarketPrice' in info:
                price = float(info['regularMarketPrice'])
                self.price_cache[symbol] = (price, datetime.now())
                return price

        except Exception as e:
            logger.warning(f"Failed to get price for {symbol} via yfinance: {e}")

        # Fallback prices for Canadian ETFs
        fallback_prices = {
            "VFV.TO": 115.00,   # Vanguard S&P 500 Index ETF
            "XIC.TO": 35.00,    # iShares Core S&P/TSX Capped Composite
            "ZAG.TO": 14.50,    # BMO Aggregate Bond Index ETF
            "XIU.TO": 35.00,    # iShares S&P/TSX 60 Index ETF
            "VCN.TO": 42.00,    # Vanguard FTSE Canada All Cap Index ETF
            "XBB.TO": 28.00,    # iShares Core Canadian Universe Bond Index
        }

        price = fallback_prices.get(symbol, 50.0)
        logger.warning(f"Using fallback price for {symbol}: ${price:.2f}")
        return price

    def allocate_profits_to_etfs(self, profit_amount: float) -> List[ETFAllocation]:
        """
        Allocate profits to ETFs based on configuration.

        Args:
            profit_amount: Total profit amount in CAD

        Returns:
            List of created allocations
        """
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
                        self.pending_allocations.append(allocation)

            logger.info(f"Created {len(allocations)} ETF allocations")
            return allocations

        except Exception as e:
            logger.error(f"Failed to allocate profits to ETFs: {e}")
            return []

    def create_etf_allocation(self, symbol: str, amount: float) -> Optional[ETFAllocation]:
        """
        Create an ETF allocation record.

        Args:
            symbol: ETF symbol
            amount: Amount to allocate in CAD

        Returns:
            ETFAllocation or None if failed
        """
        try:
            current_price = self.get_etf_price(symbol)
            if current_price <= 0:
                logger.error(f"Invalid price for {symbol}: ${current_price}")
                return None

            # Calculate shares (whole shares only for most brokers)
            shares = int(amount / current_price)
            if shares < 1:
                logger.warning(f"Amount ${amount:.2f} insufficient for 1 share of {symbol} at ${current_price:.2f}")
                return None

            # Adjust amount to actual purchase amount
            actual_amount = shares * current_price

            allocation = ETFAllocation(
                symbol=symbol,
                amount=actual_amount,
                shares=float(shares),
                price=current_price,
                timestamp=datetime.now(),
                status="pending"
            )

            logger.info(f"Created ETF allocation: {shares} shares of {symbol} at ${current_price:.2f} = ${actual_amount:.2f}")
            return allocation

        except Exception as e:
            logger.error(f"Failed to create ETF allocation for {symbol}: {e}")
            return None

    def execute_etf_purchases(self, allocations: List[ETFAllocation]) -> List[AllocationResult]:
        """
        Execute ETF purchases through the broker.

        Args:
            allocations: List of allocations to execute

        Returns:
            List of AllocationResult with execution details
        """
        results = []

        for allocation in allocations:
            result = self._execute_single_purchase(allocation)
            results.append(result)

            if result.success:
                self.total_allocated += allocation.amount
                # Remove from pending
                if allocation in self.pending_allocations:
                    self.pending_allocations.remove(allocation)

        # Log summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"ETF purchase execution complete: {successful}/{len(results)} successful")

        return results

    def _execute_single_purchase(self, allocation: ETFAllocation) -> AllocationResult:
        """Execute a single ETF purchase"""
        try:
            allocation.status = "executing"

            # Demo mode - simulate execution
            if self.mode == "DEMO" or self.broker is None:
                allocation.status = "filled"
                allocation.execution_price = allocation.price
                allocation.execution_timestamp = datetime.now()
                allocation.order_id = f"DEMO-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                logger.info(f"[DEMO] Executed ETF purchase: {allocation.shares:.0f} shares of {allocation.symbol}")

                return AllocationResult(
                    success=True,
                    allocation=allocation,
                    message=f"Demo execution successful for {allocation.symbol}",
                    order_response={'mode': 'demo', 'filled': True}
                )

            # Live mode - use broker API
            order_response = self.broker.place_order(
                symbol=allocation.symbol,
                quantity=allocation.shares,
                action="Buy",
                order_type="Market",
                time_in_force="Day"
            )

            if order_response and 'error' not in order_response:
                allocation.status = "filled"
                allocation.order_id = str(order_response.get('orderId', ''))
                allocation.execution_price = order_response.get('avgExecPrice', allocation.price)
                allocation.execution_timestamp = datetime.now()

                logger.info(f"[LIVE] Executed ETF purchase: {allocation.shares:.0f} shares of {allocation.symbol}")

                return AllocationResult(
                    success=True,
                    allocation=allocation,
                    message=f"Live execution successful for {allocation.symbol}",
                    order_response=order_response
                )
            else:
                error_msg = order_response.get('message', 'Unknown error') if order_response else 'No response'
                allocation.status = "failed"
                allocation.error_message = error_msg
                allocation.retry_count += 1

                logger.error(f"Failed to execute ETF purchase for {allocation.symbol}: {error_msg}")

                return AllocationResult(
                    success=False,
                    allocation=allocation,
                    message=f"Execution failed: {error_msg}",
                    order_response=order_response
                )

        except Exception as e:
            allocation.status = "failed"
            allocation.error_message = str(e)
            allocation.retry_count += 1

            logger.error(f"Exception executing ETF purchase for {allocation.symbol}: {e}")

            return AllocationResult(
                success=False,
                allocation=allocation,
                message=f"Exception: {str(e)}",
                order_response=None
            )

    def retry_failed_allocations(self) -> List[AllocationResult]:
        """Retry failed allocations that haven't exceeded max retries"""
        failed_allocations = [
            a for a in self.allocations
            if a.status == "failed" and a.retry_count < self.max_retries
        ]

        if not failed_allocations:
            logger.info("No failed allocations to retry")
            return []

        logger.info(f"Retrying {len(failed_allocations)} failed allocations")
        return self.execute_etf_purchases(failed_allocations)

    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of ETF allocations"""
        try:
            filled = [a for a in self.allocations if a.status == "filled"]
            pending = [a for a in self.allocations if a.status == "pending"]
            failed = [a for a in self.allocations if a.status == "failed"]

            total_allocated = sum(a.amount for a in filled)
            total_shares = sum(a.shares for a in filled)

            # Group by symbol
            by_symbol = {}
            for allocation in filled:
                if allocation.symbol not in by_symbol:
                    by_symbol[allocation.symbol] = {
                        'shares': 0,
                        'amount': 0,
                        'avg_price': 0,
                        'current_price': self.get_etf_price(allocation.symbol),
                        'transactions': 0
                    }

                by_symbol[allocation.symbol]['shares'] += allocation.shares
                by_symbol[allocation.symbol]['amount'] += allocation.amount
                by_symbol[allocation.symbol]['transactions'] += 1
                by_symbol[allocation.symbol]['avg_price'] = (
                    by_symbol[allocation.symbol]['amount'] /
                    by_symbol[allocation.symbol]['shares']
                )

            # Calculate current value and P&L
            current_value = sum(
                data['shares'] * data['current_price']
                for data in by_symbol.values()
            )
            unrealized_pnl = current_value - total_allocated

            return {
                'mode': self.mode,
                'total_allocated': total_allocated,
                'total_shares': total_shares,
                'current_value': current_value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': (unrealized_pnl / total_allocated * 100) if total_allocated > 0 else 0,
                'allocations_count': {
                    'total': len(self.allocations),
                    'filled': len(filled),
                    'pending': len(pending),
                    'failed': len(failed)
                },
                'by_symbol': by_symbol,
                'etf_targets': self.etf_targets,
                'etf_percentage': self.etf_percentage,
                'rebalance_frequency': self.rebalance_frequency,
                'summary_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get allocation summary: {e}")
            return {'error': str(e)}

    def rebalance_etf_portfolio(self) -> List[ETFAllocation]:
        """Rebalance ETF portfolio based on target allocation"""
        try:
            logger.info("Starting ETF portfolio rebalancing...")

            filled = [a for a in self.allocations if a.status == "filled"]
            current_value = sum(a.shares * self.get_etf_price(a.symbol) for a in filled)

            if current_value == 0:
                logger.info("No ETF holdings to rebalance")
                return []

            # Calculate target allocation for each ETF (equal weight)
            target_per_etf = current_value / len(self.etf_targets)

            rebalance_allocations = []
            for etf_symbol in self.etf_targets:
                # Current value for this ETF
                current_etf_shares = sum(
                    a.shares for a in filled if a.symbol == etf_symbol
                )
                current_price = self.get_etf_price(etf_symbol)
                current_etf_value = current_etf_shares * current_price

                difference = target_per_etf - current_etf_value

                # Only rebalance if difference > $100 (to avoid excessive trading)
                if abs(difference) > 100:
                    if difference > 0:
                        # Need to buy more
                        allocation = self.create_etf_allocation(etf_symbol, difference)
                        if allocation:
                            rebalance_allocations.append(allocation)
                            self.allocations.append(allocation)
                            logger.info(f"Rebalancing: Buy ${difference:,.2f} CAD of {etf_symbol}")
                    else:
                        # Would need to sell (not implemented - log only)
                        logger.info(f"Rebalancing: Would sell ${abs(difference):,.2f} CAD of {etf_symbol} (sell not implemented)")

            return rebalance_allocations

        except Exception as e:
            logger.error(f"Failed to rebalance ETF portfolio: {e}")
            return []

    def get_pending_allocations(self) -> List[ETFAllocation]:
        """Get all pending allocations"""
        return [a for a in self.allocations if a.status == "pending"]

    def cancel_pending_allocation(self, allocation: ETFAllocation) -> bool:
        """Cancel a pending allocation"""
        if allocation.status == "pending":
            allocation.status = "cancelled"
            if allocation in self.pending_allocations:
                self.pending_allocations.remove(allocation)
            logger.info(f"Cancelled pending allocation for {allocation.symbol}")
            return True
        return False


# Test the ETF allocator
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing ETF Allocator...")

    # Initialize in demo mode
    allocator = ETFAllocator(mode="DEMO")

    # Test price fetching
    print("\n1. Testing price fetching:")
    for symbol in allocator.etf_targets:
        price = allocator.get_etf_price(symbol)
        print(f"   {symbol}: ${price:.2f}")

    # Test profit allocation
    print("\n2. Testing profit allocation ($5,000):")
    allocations = allocator.allocate_profits_to_etfs(5000.0)
    print(f"   Created {len(allocations)} allocations")

    # Test execution
    print("\n3. Testing execution:")
    results = allocator.execute_etf_purchases(allocations)
    for result in results:
        status = "SUCCESS" if result.success else "FAILED"
        print(f"   [{status}] {result.allocation.symbol}: {result.message}")

    # Test summary
    print("\n4. Allocation Summary:")
    summary = allocator.get_allocation_summary()
    print(f"   Total Allocated: ${summary.get('total_allocated', 0):,.2f}")
    print(f"   Current Value: ${summary.get('current_value', 0):,.2f}")
    print(f"   Unrealized P&L: ${summary.get('unrealized_pnl', 0):,.2f}")

    # Test rebalancing
    print("\n5. Testing rebalancing:")
    rebalance = allocator.rebalance_etf_portfolio()
    print(f"   Rebalance allocations: {len(rebalance)}")

    print("\nETF Allocator test complete!")
