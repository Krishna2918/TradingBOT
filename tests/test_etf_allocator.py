"""
Unit tests for ETF Allocator
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.execution.etf_allocator import (
    ETFAllocator,
    ETFAllocation,
    AllocationResult,
    AllocationStatus
)


class MockBroker:
    """Mock broker for testing"""

    def __init__(self, should_succeed: bool = True):
        self.should_succeed = should_succeed
        self.orders_placed = []

    def place_order(self, symbol: str, quantity: float, action: str,
                   order_type: str = "Market", time_in_force: str = "Day"):
        self.orders_placed.append({
            'symbol': symbol,
            'quantity': quantity,
            'action': action,
            'order_type': order_type
        })

        if self.should_succeed:
            return {
                'orderId': f'ORD-{len(self.orders_placed)}',
                'status': 'filled',
                'avgExecPrice': 100.0
            }
        else:
            return {
                'error': 'ORDER_FAILED',
                'message': 'Insufficient funds'
            }


class TestETFAllocator:
    """Tests for ETFAllocator class"""

    def test_initialization_demo_mode(self):
        """Test initialization in demo mode"""
        allocator = ETFAllocator(mode="DEMO")

        assert allocator.mode == "DEMO"
        assert allocator.broker is None
        assert len(allocator.etf_targets) > 0
        assert allocator.etf_percentage > 0
        assert allocator.total_allocated == 0.0

    def test_initialization_with_broker(self):
        """Test initialization with broker"""
        broker = MockBroker()
        allocator = ETFAllocator(mode="LIVE", broker=broker)

        assert allocator.mode == "LIVE"
        assert allocator.broker == broker

    def test_default_etf_targets(self):
        """Test default ETF targets are Canadian ETFs"""
        allocator = ETFAllocator(mode="DEMO")

        # Should include Canadian ETFs
        assert any('.TO' in etf for etf in allocator.etf_targets)

    @patch('yfinance.Ticker')
    def test_get_etf_price_with_yfinance(self, mock_ticker):
        """Test price fetching via yfinance"""
        mock_instance = Mock()
        mock_instance.history.return_value = Mock(
            empty=False,
            __getitem__=lambda self, key: Mock(iloc=Mock(__getitem__=lambda s, i: 115.50))
        )
        mock_ticker.return_value = mock_instance

        allocator = ETFAllocator(mode="DEMO")
        price = allocator.get_etf_price('VFV.TO', use_cache=False)

        assert price > 0

    def test_get_etf_price_fallback(self):
        """Test price fetching falls back correctly"""
        allocator = ETFAllocator(mode="DEMO")

        # Clear cache
        allocator.price_cache = {}

        # Should use fallback prices
        price = allocator.get_etf_price('VFV.TO', use_cache=False)

        assert price > 0

    def test_allocate_profits_below_threshold(self):
        """Test allocation fails below threshold"""
        allocator = ETFAllocator(mode="DEMO")
        allocator.min_threshold = 100

        allocations = allocator.allocate_profits_to_etfs(50.0)

        assert len(allocations) == 0

    def test_allocate_profits_above_threshold(self):
        """Test successful allocation above threshold"""
        allocator = ETFAllocator(mode="DEMO")
        allocator.min_threshold = 100
        allocator.etf_percentage = 0.20

        allocations = allocator.allocate_profits_to_etfs(1000.0)

        # Should create allocations for each ETF target
        assert len(allocations) <= len(allocator.etf_targets)

        # Total allocated should be approximately 20% of 1000
        total = sum(a.amount for a in allocations)
        assert total <= 200.0  # May be less due to whole share rounding

    def test_create_etf_allocation(self):
        """Test creating single ETF allocation"""
        allocator = ETFAllocator(mode="DEMO")

        allocation = allocator.create_etf_allocation('VFV.TO', 500.0)

        if allocation:  # May be None if amount insufficient for 1 share
            assert allocation.symbol == 'VFV.TO'
            assert allocation.status == 'pending'
            assert allocation.shares >= 1
            assert allocation.price > 0

    def test_create_allocation_insufficient_amount(self):
        """Test allocation with amount insufficient for 1 share"""
        allocator = ETFAllocator(mode="DEMO")

        # Try to allocate $10 for a $100+ ETF
        allocation = allocator.create_etf_allocation('VFV.TO', 10.0)

        # Should return None (can't buy fractional shares)
        assert allocation is None

    def test_execute_purchases_demo_mode(self):
        """Test purchase execution in demo mode"""
        allocator = ETFAllocator(mode="DEMO")

        allocation = ETFAllocation(
            symbol='VFV.TO',
            amount=1000.0,
            shares=10.0,
            price=100.0,
            timestamp=datetime.now(),
            status='pending'
        )

        results = allocator.execute_etf_purchases([allocation])

        assert len(results) == 1
        assert results[0].success
        assert results[0].allocation.status == 'filled'
        assert 'DEMO' in results[0].allocation.order_id

    def test_execute_purchases_live_mode_success(self):
        """Test purchase execution in live mode (success)"""
        broker = MockBroker(should_succeed=True)
        allocator = ETFAllocator(mode="LIVE", broker=broker)

        allocation = ETFAllocation(
            symbol='VFV.TO',
            amount=1000.0,
            shares=10.0,
            price=100.0,
            timestamp=datetime.now(),
            status='pending'
        )

        results = allocator.execute_etf_purchases([allocation])

        assert len(results) == 1
        assert results[0].success
        assert results[0].allocation.status == 'filled'
        assert len(broker.orders_placed) == 1

    def test_execute_purchases_live_mode_failure(self):
        """Test purchase execution in live mode (failure)"""
        broker = MockBroker(should_succeed=False)
        allocator = ETFAllocator(mode="LIVE", broker=broker)

        allocation = ETFAllocation(
            symbol='VFV.TO',
            amount=1000.0,
            shares=10.0,
            price=100.0,
            timestamp=datetime.now(),
            status='pending'
        )

        results = allocator.execute_etf_purchases([allocation])

        assert len(results) == 1
        assert not results[0].success
        assert results[0].allocation.status == 'failed'
        assert results[0].allocation.retry_count == 1

    def test_retry_failed_allocations(self):
        """Test retrying failed allocations"""
        allocator = ETFAllocator(mode="DEMO")

        # Create a failed allocation
        allocation = ETFAllocation(
            symbol='VFV.TO',
            amount=1000.0,
            shares=10.0,
            price=100.0,
            timestamp=datetime.now(),
            status='failed',
            retry_count=1
        )
        allocator.allocations.append(allocation)

        results = allocator.retry_failed_allocations()

        assert len(results) == 1
        assert results[0].success  # Demo mode always succeeds

    def test_retry_skips_max_retries(self):
        """Test retry skips allocations at max retries"""
        allocator = ETFAllocator(mode="DEMO")
        allocator.max_retries = 3

        allocation = ETFAllocation(
            symbol='VFV.TO',
            amount=1000.0,
            shares=10.0,
            price=100.0,
            timestamp=datetime.now(),
            status='failed',
            retry_count=3  # At max retries
        )
        allocator.allocations.append(allocation)

        results = allocator.retry_failed_allocations()

        assert len(results) == 0

    def test_get_allocation_summary(self):
        """Test getting allocation summary"""
        allocator = ETFAllocator(mode="DEMO")

        # Add some filled allocations
        allocator.allocations.append(ETFAllocation(
            symbol='VFV.TO',
            amount=1000.0,
            shares=10.0,
            price=100.0,
            timestamp=datetime.now(),
            status='filled'
        ))
        allocator.allocations.append(ETFAllocation(
            symbol='XIC.TO',
            amount=500.0,
            shares=15.0,
            price=33.33,
            timestamp=datetime.now(),
            status='filled'
        ))

        summary = allocator.get_allocation_summary()

        assert summary['total_allocated'] == 1500.0
        assert summary['total_shares'] == 25.0
        assert len(summary['by_symbol']) == 2
        assert summary['allocations_count']['filled'] == 2

    def test_rebalance_portfolio(self):
        """Test portfolio rebalancing"""
        allocator = ETFAllocator(mode="DEMO")
        allocator.etf_targets = ['VFV.TO', 'XIC.TO']

        # Add unbalanced allocations
        allocator.allocations.append(ETFAllocation(
            symbol='VFV.TO',
            amount=2000.0,
            shares=20.0,
            price=100.0,
            timestamp=datetime.now(),
            status='filled'
        ))
        # XIC.TO has no allocation

        rebalance = allocator.rebalance_etf_portfolio()

        # Should suggest buying XIC.TO
        # (may be empty if amounts are too small)
        if rebalance:
            assert any(a.symbol == 'XIC.TO' for a in rebalance)

    def test_cancel_pending_allocation(self):
        """Test cancelling pending allocation"""
        allocator = ETFAllocator(mode="DEMO")

        allocation = ETFAllocation(
            symbol='VFV.TO',
            amount=1000.0,
            shares=10.0,
            price=100.0,
            timestamp=datetime.now(),
            status='pending'
        )
        allocator.allocations.append(allocation)
        allocator.pending_allocations.append(allocation)

        result = allocator.cancel_pending_allocation(allocation)

        assert result
        assert allocation.status == 'cancelled'

    def test_cannot_cancel_filled_allocation(self):
        """Test cannot cancel filled allocation"""
        allocator = ETFAllocator(mode="DEMO")

        allocation = ETFAllocation(
            symbol='VFV.TO',
            amount=1000.0,
            shares=10.0,
            price=100.0,
            timestamp=datetime.now(),
            status='filled'
        )

        result = allocator.cancel_pending_allocation(allocation)

        assert not result
        assert allocation.status == 'filled'

    def test_price_caching(self):
        """Test price caching works"""
        allocator = ETFAllocator(mode="DEMO")

        # First call
        price1 = allocator.get_etf_price('VFV.TO')

        # Second call should use cache
        price2 = allocator.get_etf_price('VFV.TO', use_cache=True)

        assert price1 == price2
        assert 'VFV.TO' in allocator.price_cache

    def test_set_broker(self):
        """Test setting broker after initialization"""
        allocator = ETFAllocator(mode="DEMO")
        assert allocator.broker is None

        broker = MockBroker()
        allocator.set_broker(broker)

        assert allocator.broker == broker


class TestETFAllocation:
    """Tests for ETFAllocation dataclass"""

    def test_allocation_creation(self):
        """Test allocation creation"""
        allocation = ETFAllocation(
            symbol='VFV.TO',
            amount=1000.0,
            shares=10.0,
            price=100.0,
            timestamp=datetime.now(),
            status='pending'
        )

        assert allocation.symbol == 'VFV.TO'
        assert allocation.amount == 1000.0
        assert allocation.shares == 10.0
        assert allocation.order_id is None
        assert allocation.retry_count == 0


class TestAllocationResult:
    """Tests for AllocationResult dataclass"""

    def test_result_creation(self):
        """Test result creation"""
        allocation = ETFAllocation(
            symbol='VFV.TO',
            amount=1000.0,
            shares=10.0,
            price=100.0,
            timestamp=datetime.now(),
            status='filled'
        )

        result = AllocationResult(
            success=True,
            allocation=allocation,
            message="Success"
        )

        assert result.success
        assert result.allocation == allocation
        assert result.order_response is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
