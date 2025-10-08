"""
Unit tests for Capital Allocation Module
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from risk_management.capital_allocation import CapitalAllocator, CapitalState

class TestCapitalAllocator:
    """Test cases for CapitalAllocator class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'risk': {
                'capital': {
                    'total_capital': 100000,  # CAD
                    'active_capital': 80000,
                    'safety_reserve': 20000,
                    'etf_allocation': 0.20
                },
                'limits': {
                    'daily_loss_limit': 0.08,
                    'max_drawdown': 0.15,
                    'consecutive_loss_limit': 3,
                    'cool_down_minutes': 60
                },
                'profit_allocation': {
                    'etf_percentage': 0.20,
                    'etf_targets': ["VTI", "VEA", "VWO", "BND", "VXUS"],
                    'min_profit_threshold': 1000,
                    'rebalance_frequency': 'weekly'
                }
            }
        }
    
    @pytest.fixture
    def allocator(self, mock_config, tmp_path):
        """Create CapitalAllocator instance for testing"""
        config_file = tmp_path / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_config, f)
        
        return CapitalAllocator(str(config_file))
    
    def test_initial_state(self, allocator):
        """Test initial capital state"""
        state = allocator.get_capital_state()
        assert state.total_capital == 100000
        assert state.active_capital == 80000
        assert state.safety_reserve == 20000
        assert state.consecutive_losses == 0
        assert not allocator.is_cool_down_active()
    
    def test_consecutive_losses(self, allocator):
        """Test consecutive loss tracking"""
        # Record 3 losses
        for i in range(3):
            allocator.record_loss()
        
        assert allocator.capital_state.consecutive_losses == 3
        assert allocator.is_cool_down_active()
    
    def test_cool_down_exit(self, allocator):
        """Test cool-down mode exit after win"""
        # Activate cool-down
        for i in range(3):
            allocator.record_loss()
        
        assert allocator.is_cool_down_active()
        
        # Record win
        allocator.record_win()
        assert not allocator.is_cool_down_active()
        assert allocator.capital_state.consecutive_losses == 0
    
    def test_position_size_calculation(self, allocator):
        """Test position size calculation under different conditions"""
        base_size = 100000
        
        # Normal conditions
        normal_size = allocator.calculate_position_size(base_size, "momentum")
        assert normal_size == base_size
        
        # During cool-down
        for i in range(3):
            allocator.record_loss()
        
        cool_down_size = allocator.calculate_position_size(base_size, "momentum")
        assert cool_down_size == base_size * 0.25
        
        # After 2 losses
        allocator.record_win()  # Exit cool-down
        for i in range(2):
            allocator.record_loss()
        
        reduced_size = allocator.calculate_position_size(base_size, "momentum")
        assert reduced_size == base_size * 0.5
    
    def test_daily_loss_limit(self, allocator):
        """Test daily loss limit enforcement"""
        # Test limit not exceeded
        allocator.update_daily_pnl(-5000)  # -5%
        assert allocator.capital_state.daily_pnl == -5000
        
        # Test limit exceeded - should activate kill switch but not raise
        allocator.update_daily_pnl(-10000)  # -10% exceeds 8% limit
        assert allocator.capital_state.daily_pnl == -10000
    
    def test_drawdown_tracking(self, allocator):
        """Test drawdown tracking and limits"""
        # Test normal drawdown
        allocator.update_drawdown(0.05)  # 5%
        assert allocator.capital_state.max_drawdown == 0.05
        
        # Test limit exceeded - should activate kill switch but not raise
        allocator.update_drawdown(0.20)  # 20% exceeds 15% limit
        assert allocator.capital_state.max_drawdown == 0.20
    
    def test_etf_allocation(self, allocator):
        """Test ETF allocation from profits"""
        # Test profit recording with ETF allocation
        profit_amount = 5000.0  # $5K CAD
        allocator.record_win(profit_amount)
        
        # Check that profits are recorded
        assert allocator.capital_state.total_profits == profit_amount
        
        # Check ETF allocation summary
        etf_summary = allocator.get_etf_allocation_summary()
        assert etf_summary['total_profits'] == profit_amount
        assert etf_summary['etf_percentage'] == 0.20
        assert len(etf_summary['etf_targets']) > 0
    
    def test_etf_allocation_threshold(self, allocator):
        """Test ETF allocation minimum threshold"""
        # Test with profit below threshold
        small_profit = 500.0  # $500 CAD (below $1K threshold)
        allocator.record_win(small_profit)
        
        # Should not trigger ETF allocation
        assert allocator.capital_state.total_profits == small_profit
        assert allocator.capital_state.etf_invested == 0.0
        
        # Test with profit above threshold
        large_profit = 2000.0  # $2K CAD (above $1K threshold)
        allocator.record_win(large_profit)
        
        # Should trigger ETF allocation
        expected_etf_amount = large_profit * 0.20  # 20%
        assert allocator.capital_state.etf_invested == expected_etf_amount

if __name__ == "__main__":
    pytest.main([__file__])

