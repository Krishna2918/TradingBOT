"""
Unit tests for ETF Allocation Module
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from execution.etf_allocator import ETFAllocator, ETFAllocation

class TestETFAllocator:
    """Test cases for ETFAllocator class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'risk': {
                'profit_allocation': {
                    'etf_percentage': 0.20,
                    'etf_targets': ['VTI', 'VEA', 'VWO', 'BND', 'VXUS'],
                    'min_profit_threshold': 1000,
                    'rebalance_frequency': 'weekly'
                }
            }
        }
    
    @pytest.fixture
    def allocator(self, mock_config, tmp_path):
        """Create ETFAllocator instance for testing"""
        config_file = tmp_path / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_config, f)
        
        return ETFAllocator(str(config_file))
    
    def test_initial_state(self, allocator):
        """Test initial ETF allocator state"""
        assert allocator.etf_percentage == 0.20
        assert len(allocator.etf_targets) == 5
        assert allocator.min_threshold == 1000
        assert allocator.rebalance_frequency == 'weekly'
        assert len(allocator.allocations) == 0
        assert allocator.total_allocated == 0.0
    
    def test_profit_allocation_below_threshold(self, allocator):
        """Test profit allocation below minimum threshold"""
        profit_amount = 500.0  # Below $1K threshold
        allocations = allocator.allocate_profits_to_etfs(profit_amount)
        
        assert len(allocations) == 0
        assert allocator.total_allocated == 0.0
    
    def test_profit_allocation_above_threshold(self, allocator):
        """Test profit allocation above minimum threshold"""
        profit_amount = 5000.0  # Above $1K threshold
        allocations = allocator.allocate_profits_to_etfs(profit_amount)
        
        assert len(allocations) == 5  # One for each ETF target
        assert allocator.total_allocated == 1000.0  # 20% of $5K
        
        # Check each allocation
        for allocation in allocations:
            assert allocation.amount == 200.0  # $1K / 5 ETFs
            assert allocation.symbol in allocator.etf_targets
            assert allocation.status == "pending"
    
    def test_etf_price_retrieval(self, allocator):
        """Test ETF price retrieval"""
        prices = {
            'VTI': 250.00,
            'VEA': 45.50,
            'VWO': 42.75,
            'BND': 75.25,
            'VXUS': 55.80
        }
        
        for symbol, expected_price in prices.items():
            price = allocator.get_etf_price(symbol)
            assert price == expected_price
    
    def test_etf_allocation_creation(self, allocator):
        """Test ETF allocation creation"""
        allocation = allocator.create_etf_allocation('VTI', 1000.0)
        
        assert allocation is not None
        assert allocation.symbol == 'VTI'
        assert allocation.amount == 1000.0
        assert allocation.shares == 4.0  # $1000 / $250
        assert allocation.price == 250.0
        assert allocation.status == "pending"
    
    def test_etf_purchase_execution(self, allocator):
        """Test ETF purchase execution"""
        # Create test allocations
        allocations = []
        for symbol in ['VTI', 'VEA']:
            allocation = allocator.create_etf_allocation(symbol, 500.0)
            allocations.append(allocation)
        
        # Execute purchases
        executed = allocator.execute_etf_purchases(allocations)
        
        assert len(executed) == 2
        for allocation in executed:
            assert allocation.status == "filled"
    
    def test_allocation_summary(self, allocator):
        """Test allocation summary generation"""
        # Create and execute some allocations
        allocations = allocator.allocate_profits_to_etfs(5000.0)
        executed = allocator.execute_etf_purchases(allocations)
        
        summary = allocator.get_allocation_summary()
        
        assert summary['total_allocated'] == 1000.0
        assert summary['allocations_count'] == 5
        assert len(summary['by_symbol']) == 5
        assert summary['etf_percentage'] == 0.20
        assert summary['rebalance_frequency'] == 'weekly'
    
    def test_portfolio_rebalancing(self, allocator):
        """Test ETF portfolio rebalancing"""
        # Create some initial allocations
        allocations = allocator.allocate_profits_to_etfs(5000.0)
        executed = allocator.execute_etf_purchases(allocations)
        
        # Test rebalancing
        rebalance_allocations = allocator.rebalance_etf_portfolio()
        
        # Should not create new allocations if portfolio is balanced
        assert len(rebalance_allocations) == 0
    
    def test_multiple_profit_allocations(self, allocator):
        """Test multiple profit allocations over time"""
        # First allocation
        allocations1 = allocator.allocate_profits_to_etfs(3000.0)
        executed1 = allocator.execute_etf_purchases(allocations1)
        
        # Second allocation
        allocations2 = allocator.allocate_profits_to_etfs(2000.0)
        executed2 = allocator.execute_etf_purchases(allocations2)
        
        # Check total allocations
        assert len(allocator.allocations) == 10  # 5 + 5
        assert allocator.total_allocated == 1000.0  # 20% of $5K total
        
        # Check summary
        summary = allocator.get_allocation_summary()
        assert summary['total_allocated'] == 1000.0
        assert summary['allocations_count'] == 10

if __name__ == "__main__":
    pytest.main([__file__])
