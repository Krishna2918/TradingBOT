"""
Backtesting Framework Tests

Tests backtesting engine functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.backtesting import BacktestEngine, get_backtest_engine


class TestBacktestEngine:
    """Test Backtest Engine"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = BacktestEngine(initial_capital=100000)
    
    def test_simple_backtest(self):
        """Test simple buy-and-hold backtest"""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=180, freq='D')
        np.random.seed(42)
        
        # Upward trending market
        prices = 100 + np.cumsum(np.random.randn(180) * 2 + 0.1)
        
        data = pd.DataFrame({
            'open': prices + np.random.randn(180) * 0.5,
            'high': prices + np.abs(np.random.randn(180)),
            'low': prices - np.abs(np.random.randn(180)),
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 180)
        }, index=dates)
        
        # Simple buy-and-hold signals (buy at start, sell at end)
        signals = pd.Series([0] * 180, index=dates)
        signals.iloc[0] = 1  # Buy
        signals.iloc[-1] = -1  # Sell
        
        # Run backtest
        result = self.engine.run_backtest(
            strategy_name="BuyAndHold",
            data=data,
            signals=signals
        )
        
        assert result is not None
        assert result.total_trades == 1
        assert result.final_capital != result.initial_capital
        print(f"✅ Backtest: {result.total_return_pct:.2f}% return")
        print(f"   Sharpe: {result.sharpe_ratio:.2f}, Max DD: {result.max_drawdown_pct:.2f}%")
    
    def test_multiple_trades_backtest(self):
        """Test backtest with multiple trades"""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = 100 + np.cumsum(np.random.randn(100) * 1.5)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)
        
        # Multiple buy/sell signals
        signals = pd.Series([0] * 100, index=dates)
        signals.iloc[10] = 1  # Buy
        signals.iloc[20] = -1  # Sell
        signals.iloc[30] = 1  # Buy
        signals.iloc[40] = -1  # Sell
        signals.iloc[50] = 1  # Buy
        signals.iloc[60] = -1  # Sell
        
        result = self.engine.run_backtest(
            strategy_name="MultiTrade",
            data=data,
            signals=signals
        )
        
        assert result.total_trades == 3
        print(f"✅ Multiple trades: {result.total_trades} trades")
        print(f"   Win rate: {result.win_rate:.1f}%")
    
    def test_performance_metrics(self):
        """Test calculation of performance metrics"""
        dates = pd.date_range(start='2024-01-01', periods=252, freq='D')  # 1 year
        np.random.seed(42)
        
        # Create data with volatility
        returns = np.random.randn(252) * 0.02  # 2% daily volatility
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 252)
        }, index=dates)
        
        # Simple strategy
        signals = pd.Series([0] * 252, index=dates)
        signals.iloc[0] = 1
        signals.iloc[-1] = -1
        
        result = self.engine.run_backtest(
            strategy_name="Metrics",
            data=data,
            signals=signals
        )
        
        # Check metrics exist
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'sortino_ratio')
        assert hasattr(result, 'max_drawdown_pct')
        assert hasattr(result, 'win_rate')
        
        print(f"✅ Performance metrics calculated:")
        print(f"   Sharpe: {result.sharpe_ratio:.2f}")
        print(f"   Sortino: {result.sortino_ratio:.2f}")
        print(f"   Max DD: {result.max_drawdown_pct:.2f}%")
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation"""
        # Run a simple backtest first
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = 100 + np.cumsum(np.random.randn(100) * 1.5)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)
        
        signals = pd.Series([0] * 100, index=dates)
        for i in range(10, 90, 10):
            signals.iloc[i] = 1
            signals.iloc[i+5] = -1
        
        result = self.engine.run_backtest(
            strategy_name="MC_Test",
            data=data,
            signals=signals
        )
        
        # Run Monte Carlo
        mc_results = self.engine.monte_carlo_simulation(result, num_simulations=100)
        
        assert 'mean_return' in mc_results
        assert 'percentile_5' in mc_results
        assert 'percentile_95' in mc_results
        assert 'probability_profit' in mc_results
        
        print(f"✅ Monte Carlo simulation (100 runs):")
        print(f"   Mean return: {mc_results['mean_return']:.2f}%")
        print(f"   95% CI: [{mc_results['percentile_5']:.2f}%, {mc_results['percentile_95']:.2f}%]")
        print(f"   Probability of profit: {mc_results['probability_profit']:.1f}%")
    
    def test_stress_testing(self):
        """Test stress testing"""
        # Run a simple backtest
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = 100 + np.cumsum(np.random.randn(100) * 1.5)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)
        
        signals = pd.Series([0] * 100, index=dates)
        signals.iloc[10] = 1
        signals.iloc[90] = -1
        
        result = self.engine.run_backtest(
            strategy_name="Stress_Test",
            data=data,
            signals=signals
        )
        
        # Run stress tests
        stress_results = self.engine.stress_test(result)
        
        assert 'market_crash_20' in stress_results
        assert 'market_crash_30' in stress_results
        assert 'volatility_spike' in stress_results
        
        print(f"✅ Stress tests completed:")
        for scenario, results in stress_results.items():
            print(f"   {scenario}: {results}")
    
    def test_strategy_comparison(self):
        """Test strategy comparison"""
        # Run multiple strategies
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = 100 + np.cumsum(np.random.randn(100) * 1.5)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)
        
        # Strategy 1: Buy and hold
        signals1 = pd.Series([0] * 100, index=dates)
        signals1.iloc[0] = 1
        signals1.iloc[-1] = -1
        
        self.engine.run_backtest("Strategy1", data, signals1)
        
        # Strategy 2: Frequent trading
        signals2 = pd.Series([0] * 100, index=dates)
        for i in range(10, 90, 20):
            signals2.iloc[i] = 1
            signals2.iloc[i+10] = -1
        
        self.engine.run_backtest("Strategy2", data, signals2)
        
        # Compare
        comparison = self.engine.compare_strategies()
        
        assert len(comparison) == 2
        assert 'Return %' in comparison.columns
        assert 'Sharpe' in comparison.columns
        
        print(f"✅ Strategy comparison:")
        print(comparison.to_string())


def run_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 RUNNING BACKTESTING FRAMEWORK TESTS")
    print("=" * 80)
    
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes",
        "-s"
    ])
    
    print("\n" + "=" * 80)
    if exit_code == 0:
        print("✅ ALL BACKTESTING TESTS PASSED!")
        print("\n📊 Features Tested:")
        print("   ✅ Simple backtest (buy-and-hold)")
        print("   ✅ Multiple trades backtest")
        print("   ✅ Performance metrics (Sharpe, Sortino, Max DD)")
        print("   ✅ Monte Carlo simulation")
        print("   ✅ Stress testing")
        print("   ✅ Strategy comparison")
    else:
        print(f"❌ TESTS FAILED (exit code: {exit_code})")
    print("=" * 80 + "\n")
    
    return exit_code


if __name__ == "__main__":
    exit(run_tests())

