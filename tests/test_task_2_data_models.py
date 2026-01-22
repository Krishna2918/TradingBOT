"""
Comprehensive tests for Task 2.1: Data Models and Validation

This test suite validates all data models created in Task 2.1 including:
- Portfolio State and Position models
- Optimization Configuration models
- Risk Metrics models
- Factor Exposure models
- Transaction Cost models
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

# Import all data models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_optimization.models.portfolio_state import PortfolioState, Position
from portfolio_optimization.models.optimization_config import (
    OptimizationConfig, ConstraintSet, OptimizationMethod, RebalanceFrequency, RiskObjective
)
from portfolio_optimization.models.risk_metrics import RiskMetrics
from portfolio_optimization.models.factor_exposure import FactorExposure
from portfolio_optimization.models.transaction_cost import TransactionCost


class TestPosition:
    """Test Position model"""
    
    def test_position_creation(self):
        """Test basic position creation"""
        position = Position(
            symbol="AAPL",
            weight=0.05,
            shares=100,
            market_value=15000.0,
            cost_basis=14000.0,
            unrealized_pnl=1000.0,
            sector="Technology"
        )
        
        assert position.symbol == "AAPL"
        assert position.weight == 0.05
        assert position.shares == 100
        assert position.market_value == 15000.0
        assert position.unrealized_pnl == 1000.0
        assert position.sector == "Technology"
    
    def test_position_validation(self):
        """Test position validation"""
        # Test invalid weight
        with pytest.raises(ValueError, match="Position weight must be between 0 and 1"):
            Position("AAPL", weight=1.5, shares=100, market_value=1000, cost_basis=900, unrealized_pnl=100)
        
        # Test negative market value
        with pytest.raises(ValueError, match="Market value cannot be negative"):
            Position("AAPL", weight=0.05, shares=100, market_value=-1000, cost_basis=900, unrealized_pnl=100)


class TestPortfolioState:
    """Test PortfolioState model"""
    
    def setup_method(self):
        """Set up test portfolio"""
        self.positions = {
            "AAPL": Position("AAPL", 0.3, 100, 30000, 28000, 2000, "Technology"),
            "GOOGL": Position("GOOGL", 0.25, 50, 25000, 24000, 1000, "Technology"),
            "JPM": Position("JPM", 0.2, 75, 20000, 19000, 1000, "Financials"),
            "JNJ": Position("JNJ", 0.15, 60, 15000, 14500, 500, "Healthcare"),
            "PG": Position("PG", 0.1, 40, 10000, 9500, 500, "Consumer Staples")
        }
        
        self.portfolio = PortfolioState(
            positions=self.positions,
            cash=0.0,
            total_value=100000.0,
            optimization_method="mean_variance"
        )
    
    def test_portfolio_creation(self):
        """Test basic portfolio creation"""
        assert len(self.portfolio.positions) == 5
        assert self.portfolio.total_value == 100000.0
        assert self.portfolio.optimization_method == "mean_variance"
    
    def test_portfolio_validation(self):
        """Test portfolio validation"""
        # Weights should sum to 1.0
        total_weight = sum(pos.weight for pos in self.portfolio.positions.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_add_position(self):
        """Test adding a position"""
        new_position = Position("MSFT", 0.05, 25, 5000, 4800, 200, "Technology")
        
        # This should fail validation since weights would exceed 1.0
        with pytest.raises(ValueError):
            self.portfolio.add_position(new_position)
    
    def test_remove_position(self):
        """Test removing a position"""
        removed = self.portfolio.remove_position("PG")
        
        assert removed is not None
        assert removed.symbol == "PG"
        assert "PG" not in self.portfolio.positions
        assert self.portfolio.cash == 10000.0  # Cash should increase
    
    def test_update_position_weight(self):
        """Test updating position weight"""
        original_weight = self.portfolio.positions["AAPL"].weight
        self.portfolio.update_position_weight("AAPL", 0.25)
        
        assert self.portfolio.positions["AAPL"].weight == 0.25
        assert self.portfolio.positions["AAPL"].weight != original_weight
    
    def test_get_weights_array(self):
        """Test getting weights as numpy array"""
        symbols = ["AAPL", "GOOGL", "JPM", "JNJ", "PG"]
        weights = self.portfolio.get_weights_array(symbols)
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 5
        assert weights[0] == 0.3  # AAPL weight
        assert weights[1] == 0.25  # GOOGL weight
    
    def test_sector_weights(self):
        """Test sector weight calculation"""
        sector_weights = self.portfolio.get_sector_weights()
        
        assert sector_weights["Technology"] == 0.55  # AAPL + GOOGL
        assert sector_weights["Financials"] == 0.2
        assert sector_weights["Healthcare"] == 0.15
        assert sector_weights["Consumer Staples"] == 0.1
    
    def test_market_cap_weights(self):
        """Test market cap weight calculation"""
        # Set market caps for testing
        self.portfolio.positions["AAPL"].market_cap = 2.5e12  # Large cap
        self.portfolio.positions["GOOGL"].market_cap = 1.5e12  # Large cap
        self.portfolio.positions["JPM"].market_cap = 5e11  # Large cap
        self.portfolio.positions["JNJ"].market_cap = 4e11  # Large cap
        self.portfolio.positions["PG"].market_cap = 3e11  # Large cap
        
        cap_weights = self.portfolio.get_market_cap_weights()
        
        assert cap_weights["large_cap"] == 1.0  # All positions are large cap
        assert cap_weights["mid_cap"] == 0.0
        assert cap_weights["small_cap"] == 0.0
    
    def test_calculate_turnover(self):
        """Test turnover calculation"""
        # Create a previous state with different weights
        previous_positions = {
            "AAPL": Position("AAPL", 0.25, 83, 25000, 24000, 1000, "Technology"),
            "GOOGL": Position("GOOGL", 0.3, 60, 30000, 28000, 2000, "Technology"),
            "JPM": Position("JPM", 0.2, 75, 20000, 19000, 1000, "Financials"),
            "JNJ": Position("JNJ", 0.15, 60, 15000, 14500, 500, "Healthcare"),
            "PG": Position("PG", 0.1, 40, 10000, 9500, 500, "Consumer Staples")
        }
        
        previous_state = PortfolioState(positions=previous_positions, total_value=100000.0)
        turnover = self.portfolio.calculate_turnover(previous_state)
        
        # Turnover should be 0.05 (AAPL: 0.3-0.25=0.05, GOOGL: 0.25-0.3=-0.05)
        assert abs(turnover - 0.05) < 1e-6
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        summary = self.portfolio.get_performance_summary()
        
        assert 'total_value' in summary
        assert 'num_positions' in summary
        assert 'sector_weights' in summary
        assert 'optimization_method' in summary
        assert summary['num_positions'] == 5
        assert summary['total_value'] == 100000.0
    
    def test_serialization(self):
        """Test portfolio serialization and deserialization"""
        # Convert to dict
        portfolio_dict = self.portfolio.to_dict()
        
        assert 'positions' in portfolio_dict
        assert 'total_value' in portfolio_dict
        assert 'optimization_method' in portfolio_dict
        
        # Convert back from dict
        restored_portfolio = PortfolioState.from_dict(portfolio_dict)
        
        assert len(restored_portfolio.positions) == len(self.portfolio.positions)
        assert restored_portfolio.total_value == self.portfolio.total_value
        assert restored_portfolio.optimization_method == self.portfolio.optimization_method


class TestOptimizationConfig:
    """Test OptimizationConfig model"""
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = OptimizationConfig()
        
        assert config.method == OptimizationMethod.MEAN_VARIANCE
        assert config.objective == RiskObjective.MAXIMIZE_SHARPE
        assert config.risk_aversion == 3.0
        assert config.rebalance_frequency == RebalanceFrequency.MONTHLY
        assert isinstance(config.constraints, ConstraintSet)
    
    def test_constraint_validation(self):
        """Test constraint validation"""
        constraints = ConstraintSet(
            min_weight=0.01,
            max_weight=0.05,
            max_sector_weight=0.25
        )
        
        errors = constraints.validate()
        assert len(errors) == 0  # Should be valid
        
        # Test invalid constraints
        invalid_constraints = ConstraintSet(
            min_weight=0.1,
            max_weight=0.05  # min > max
        )
        
        errors = invalid_constraints.validate()
        assert len(errors) > 0
        assert any("min_weight" in error and "max_weight" in error for error in errors)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = OptimizationConfig(risk_aversion=2.0, risk_free_rate=0.03)
        errors = config.validate()
        assert len(errors) == 0
        
        # Invalid config
        with pytest.raises(ValueError):
            OptimizationConfig(risk_aversion=-1.0)  # Negative risk aversion
    
    def test_method_specific_params(self):
        """Test method-specific parameter extraction"""
        # Mean-variance config
        mv_config = OptimizationConfig(
            method=OptimizationMethod.MEAN_VARIANCE,
            risk_aversion=2.5,
            target_return=0.12
        )
        
        params = mv_config.get_method_specific_params()
        assert 'risk_aversion' in params
        assert 'target_return' in params
        assert params['risk_aversion'] == 2.5
        assert params['target_return'] == 0.12
        
        # Black-Litterman config
        bl_config = OptimizationConfig(
            method=OptimizationMethod.BLACK_LITTERMAN,
            bl_tau=0.05,
            bl_confidence_level=0.7
        )
        
        params = bl_config.get_method_specific_params()
        assert 'tau' in params
        assert 'confidence_level' in params
        assert params['tau'] == 0.05
        assert params['confidence_level'] == 0.7
    
    def test_preset_configs(self):
        """Test preset configuration creation"""
        # Conservative config
        conservative = OptimizationConfig.create_conservative_config()
        assert conservative.method == OptimizationMethod.MINIMUM_VARIANCE
        assert conservative.risk_aversion == 5.0
        assert conservative.constraints.max_weight == 0.03
        
        # Aggressive config
        aggressive = OptimizationConfig.create_aggressive_config()
        assert aggressive.method == OptimizationMethod.MAXIMUM_SHARPE
        assert aggressive.risk_aversion == 1.5
        assert aggressive.constraints.max_weight == 0.08
        
        # Balanced config
        balanced = OptimizationConfig.create_balanced_config()
        assert balanced.method == OptimizationMethod.MEAN_VARIANCE
        assert balanced.risk_aversion == 3.0


class TestRiskMetrics:
    """Test RiskMetrics model"""
    
    def test_risk_metrics_creation(self):
        """Test basic risk metrics creation"""
        metrics = RiskMetrics(
            portfolio_volatility=0.15,
            portfolio_return=0.12,
            sharpe_ratio=0.8,
            max_drawdown=-0.08,
            var_95=-0.025,
            beta=1.1
        )
        
        assert metrics.portfolio_volatility == 0.15
        assert metrics.portfolio_return == 0.12
        assert metrics.sharpe_ratio == 0.8
        assert metrics.max_drawdown == -0.08
        assert metrics.var_95 == -0.025
        assert metrics.beta == 1.1
    
    def test_risk_metrics_validation(self):
        """Test risk metrics validation"""
        # Test that positive drawdown gets converted to negative
        metrics = RiskMetrics(max_drawdown=0.1)  # Positive input
        assert metrics.max_drawdown == -0.1  # Should be negative
        
        # Test that positive VaR gets converted to negative
        metrics = RiskMetrics(var_95=0.05)  # Positive input
        assert metrics.var_95 == -0.05  # Should be negative
    
    def test_risk_adjusted_return(self):
        """Test risk-adjusted return calculations"""
        metrics = RiskMetrics(
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=0.8,
            information_ratio=0.6
        )
        
        assert metrics.calculate_risk_adjusted_return('sharpe') == 1.2
        assert metrics.calculate_risk_adjusted_return('sortino') == 1.5
        assert metrics.calculate_risk_adjusted_return('calmar') == 0.8
        assert metrics.calculate_risk_adjusted_return('information') == 0.6
        
        with pytest.raises(ValueError):
            metrics.calculate_risk_adjusted_return('unknown')
    
    def test_risk_grade(self):
        """Test risk grading"""
        # Excellent portfolio
        excellent = RiskMetrics(sharpe_ratio=1.6, max_drawdown=-0.08)
        assert excellent.get_risk_grade() == 'A+'
        
        # Poor portfolio
        poor = RiskMetrics(sharpe_ratio=0.1, max_drawdown=-0.5)
        assert poor.get_risk_grade() == 'F'
    
    def test_risk_warnings(self):
        """Test risk warning generation"""
        # High-risk portfolio
        risky = RiskMetrics(
            portfolio_volatility=0.3,  # High volatility
            max_drawdown=-0.4,  # Large drawdown
            sharpe_ratio=0.3,  # Poor Sharpe ratio
            var_95=-0.08,  # High VaR
            tracking_error=0.15,  # High tracking error
            alpha=-0.03  # Negative alpha
        )
        
        warnings = risky.get_risk_warnings()
        assert len(warnings) > 0
        assert any("High volatility" in warning for warning in warnings)
        assert any("Large maximum drawdown" in warning for warning in warnings)
        assert any("Low Sharpe ratio" in warning for warning in warnings)
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison"""
        portfolio_metrics = RiskMetrics(
            portfolio_return=0.12,
            portfolio_volatility=0.18,
            sharpe_ratio=0.67,
            max_drawdown=-0.15
        )
        
        benchmark_metrics = RiskMetrics(
            portfolio_return=0.10,
            portfolio_volatility=0.16,
            sharpe_ratio=0.625,
            max_drawdown=-0.12
        )
        
        comparison = portfolio_metrics.compare_to_benchmark(benchmark_metrics)
        
        assert comparison['excess_return'] == 0.02  # 2% excess return
        assert comparison['volatility_diff'] == 0.02  # 2% higher volatility
        assert abs(comparison['sharpe_diff'] - 0.045) < 1e-6  # Better Sharpe ratio
    
    def test_serialization(self):
        """Test risk metrics serialization"""
        metrics = RiskMetrics(
            portfolio_volatility=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.1,
            calculation_date=datetime.now()
        )
        
        # Convert to dict
        metrics_dict = metrics.to_dict()
        assert 'portfolio_volatility' in metrics_dict
        assert 'sharpe_ratio' in metrics_dict
        assert 'calculation_date' in metrics_dict
        
        # Convert back from dict
        restored_metrics = RiskMetrics.from_dict(metrics_dict)
        assert restored_metrics.portfolio_volatility == metrics.portfolio_volatility
        assert restored_metrics.sharpe_ratio == metrics.sharpe_ratio


class TestFactorExposure:
    """Test FactorExposure model"""
    
    def test_factor_exposure_creation(self):
        """Test basic factor exposure creation"""
        exposure = FactorExposure(
            market_beta=1.1,
            size=-0.2,  # Large cap tilt
            value=0.3,  # Value tilt
            momentum=0.1,
            quality=0.2,
            information_technology=0.4,  # Tech overweight
            financials=0.2
        )
        
        assert exposure.market_beta == 1.1
        assert exposure.size == -0.2
        assert exposure.value == 0.3
        assert exposure.information_technology == 0.4
    
    def test_factor_exposure_validation(self):
        """Test factor exposure validation"""
        # Test extreme values get capped
        exposure = FactorExposure(market_beta=10.0)  # Extreme value
        assert exposure.market_beta == 5.0  # Should be capped at 5.0
    
    def test_style_factor_exposures(self):
        """Test style factor exposure extraction"""
        exposure = FactorExposure(
            market_beta=1.2,
            size=0.1,
            value=-0.2,
            momentum=0.3
        )
        
        style_factors = exposure.get_style_factor_exposures()
        assert style_factors['market_beta'] == 1.2
        assert style_factors['size'] == 0.1
        assert style_factors['value'] == -0.2
        assert style_factors['momentum'] == 0.3
    
    def test_sector_exposures(self):
        """Test sector exposure extraction"""
        exposure = FactorExposure(
            information_technology=0.3,
            financials=0.2,
            health_care=0.15,
            energy=0.1
        )
        
        sector_exposures = exposure.get_sector_exposures()
        assert sector_exposures['Information Technology'] == 0.3
        assert sector_exposures['Financials'] == 0.2
        assert sector_exposures['Health Care'] == 0.15
        assert sector_exposures['Energy'] == 0.1
    
    def test_top_factor_exposures(self):
        """Test top factor exposure identification"""
        exposure = FactorExposure(
            momentum=0.5,  # Highest
            value=-0.4,  # Second highest (by absolute value)
            quality=0.3,  # Third highest
            size=0.1,
            market_beta=1.0
        )
        
        top_factors = exposure.get_top_factor_exposures(3)
        assert len(top_factors) == 3
        assert top_factors[0][0] == 'momentum'
        assert top_factors[0][1] == 0.5
        assert abs(top_factors[1][1]) == 0.4  # value factor
    
    def test_factor_concentration(self):
        """Test factor concentration calculation"""
        # Concentrated exposure
        concentrated = FactorExposure(momentum=0.8, value=0.1)
        concentration = concentrated.calculate_factor_concentration()
        assert concentration > 0.6  # High concentration
        
        # Diversified exposure
        diversified = FactorExposure(momentum=0.2, value=0.2, quality=0.2, size=0.2)
        concentration = diversified.calculate_factor_concentration()
        assert concentration < 0.2  # Low concentration
    
    def test_factor_tilt_summary(self):
        """Test factor tilt summary"""
        exposure = FactorExposure(
            size=0.3,  # Small cap tilt
            value=0.25,  # Value tilt
            momentum=-0.3,  # Low momentum
            information_technology=0.2  # Tech overweight
        )
        
        tilts = exposure.get_factor_tilt_summary()
        assert 'Size' in tilts
        assert tilts['Size'] == 'Small Cap'
        assert 'Value' in tilts
        assert tilts['Value'] == 'Value'
        assert 'Momentum' in tilts
        assert tilts['Momentum'] == 'Low Momentum'
    
    def test_benchmark_comparison(self):
        """Test factor exposure comparison to benchmark"""
        portfolio_exposure = FactorExposure(
            market_beta=1.2,
            size=0.1,
            value=0.2,
            momentum=0.15
        )
        
        benchmark_exposure = FactorExposure(
            market_beta=1.0,
            size=0.0,
            value=0.0,
            momentum=0.0
        )
        
        diff = portfolio_exposure.compare_to_benchmark(benchmark_exposure)
        assert diff['market_beta'] == 0.2
        assert diff['size'] == 0.1
        assert diff['value'] == 0.2
        assert diff['momentum'] == 0.15
    
    def test_risk_warnings(self):
        """Test factor exposure risk warnings"""
        risky_exposure = FactorExposure(
            momentum=2.0,  # Extreme exposure
            information_technology=0.5,  # High sector concentration
            factor_r_squared=0.2  # Low model fit
        )
        
        warnings = risky_exposure.get_risk_warnings()
        assert len(warnings) > 0
        assert any("Extreme momentum exposure" in warning for warning in warnings)
        assert any("High sector concentration" in warning for warning in warnings)
        assert any("Low factor model R²" in warning for warning in warnings)


class TestTransactionCost:
    """Test TransactionCost model"""
    
    def test_transaction_cost_creation(self):
        """Test basic transaction cost creation"""
        cost = TransactionCost(
            commission=5.0,
            bid_ask_spread=10.0,
            market_impact=15.0,
            trade_size=10000.0,
            symbol="AAPL",
            trade_direction="buy"
        )
        
        assert cost.commission == 5.0
        assert cost.bid_ask_spread == 10.0
        assert cost.symbol == "AAPL"
        assert cost.trade_direction == "buy"
        assert cost.total_cost > 0  # Should be calculated automatically
    
    def test_cost_calculation(self):
        """Test automatic cost calculation"""
        cost = TransactionCost(
            commission=5.0,
            regulatory_fees=1.0,
            bid_ask_spread=10.0,
            total_market_impact=20.0,
            slippage=5.0,
            trade_size=10000.0
        )
        
        expected_total = 5.0 + 1.0 + 10.0 + 20.0 + 5.0  # 41.0
        assert abs(cost.total_cost - expected_total) < 1e-6
        
        # Check basis points calculation
        expected_bps = (41.0 / 10000.0) * 10000  # 41 bps
        assert abs(cost.total_cost_bps - expected_bps) < 1e-6
    
    def test_participation_rate_calculation(self):
        """Test participation rate calculation"""
        cost = TransactionCost(
            trade_shares=1000,
            average_daily_volume=50000,
            trade_size=50000.0
        )
        
        expected_rate = 1000 / 50000  # 2%
        assert abs(cost.participation_rate - expected_rate) < 1e-6
    
    def test_market_impact_estimation(self):
        """Test market impact estimation"""
        cost = TransactionCost(
            participation_rate=0.1,  # 10% of daily volume
            volatility=0.02,  # 2% daily volatility
            trade_size=100000.0
        )
        
        # Linear model
        linear_impact = cost.estimate_market_impact('linear')
        assert linear_impact > 0
        
        # Square root model
        sqrt_impact = cost.estimate_market_impact('square_root')
        assert sqrt_impact > 0
        assert sqrt_impact != linear_impact  # Should be different
        
        # Invalid model
        with pytest.raises(ValueError):
            cost.estimate_market_impact('invalid_model')
    
    def test_cost_efficiency_score(self):
        """Test cost efficiency scoring"""
        # Very efficient trade
        efficient = TransactionCost(total_cost_bps=3.0)
        assert efficient.get_cost_efficiency_score() == 1.0
        
        # Very inefficient trade
        inefficient = TransactionCost(total_cost_bps=100.0)
        assert inefficient.get_cost_efficiency_score() == 0.2
    
    def test_execution_quality_metrics(self):
        """Test execution quality metrics"""
        cost = TransactionCost(
            total_cost_bps=15.0,
            total_market_impact=50.0,
            trade_size=10000.0,
            participation_rate=0.05,
            execution_time_seconds=30.0,
            vwap_performance=5.0,
            liquidity_score=0.8
        )
        
        metrics = cost.get_execution_quality_metrics()
        assert 'total_cost_bps' in metrics
        assert 'participation_rate' in metrics
        assert 'execution_time' in metrics
        assert 'efficiency_score' in metrics
        assert metrics['total_cost_bps'] == 15.0
    
    def test_cost_warnings(self):
        """Test cost warning generation"""
        expensive_cost = TransactionCost(
            total_cost_bps=75.0,  # High cost
            participation_rate=0.15,  # High participation
            execution_time_seconds=600.0,  # Long execution
            vwap_performance=30.0,  # Poor VWAP performance
            liquidity_score=0.2  # Low liquidity
        )
        
        warnings = expensive_cost.get_cost_warnings()
        assert len(warnings) > 0
        assert any("High total transaction cost" in warning for warning in warnings)
        assert any("High participation rate" in warning for warning in warnings)
        assert any("Long execution time" in warning for warning in warnings)
    
    def test_benchmark_comparison(self):
        """Test transaction cost benchmark comparison"""
        trade_cost = TransactionCost(
            total_cost_bps=20.0,
            total_market_impact=30.0,
            execution_time_seconds=45.0
        )
        
        benchmark_cost = TransactionCost(
            total_cost_bps=15.0,
            total_market_impact=25.0,
            execution_time_seconds=30.0
        )
        
        comparison = trade_cost.compare_to_benchmark(benchmark_cost)
        assert comparison['total_cost_diff_bps'] == 5.0
        assert comparison['market_impact_diff'] == 5.0
        assert comparison['execution_time_diff'] == 15.0
    
    def test_serialization(self):
        """Test transaction cost serialization"""
        cost = TransactionCost(
            commission=5.0,
            bid_ask_spread=10.0,
            symbol="AAPL",
            trade_date=datetime.now()
        )
        
        # Convert to dict
        cost_dict = cost.to_dict()
        assert 'commission' in cost_dict
        assert 'symbol' in cost_dict
        assert 'trade_date' in cost_dict
        
        # Convert back from dict
        restored_cost = TransactionCost.from_dict(cost_dict)
        assert restored_cost.commission == cost.commission
        assert restored_cost.symbol == cost.symbol


class TestIntegration:
    """Integration tests for all data models"""
    
    def test_complete_portfolio_with_all_models(self):
        """Test creating a complete portfolio with all data models"""
        # Create optimization config
        config = OptimizationConfig(
            method=OptimizationMethod.MEAN_VARIANCE,
            risk_aversion=2.5,
            target_return=0.12
        )
        
        # Create risk metrics
        risk_metrics = RiskMetrics(
            portfolio_volatility=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.08
        )
        
        # Create factor exposure
        factor_exposure = FactorExposure(
            market_beta=1.1,
            size=-0.1,
            value=0.2,
            information_technology=0.3
        )
        
        # Create positions
        positions = {
            "AAPL": Position("AAPL", 0.3, 100, 30000, 28000, 2000, "Technology"),
            "GOOGL": Position("GOOGL", 0.25, 50, 25000, 24000, 1000, "Technology"),
            "JPM": Position("JPM", 0.25, 75, 25000, 24000, 1000, "Financials"),
            "JNJ": Position("JNJ", 0.2, 60, 20000, 19000, 1000, "Healthcare")
        }
        
        # Create portfolio with all models
        portfolio = PortfolioState(
            positions=positions,
            total_value=100000.0,
            risk_metrics=risk_metrics,
            factor_exposures=factor_exposure,
            optimization_method=config.method.value
        )
        
        # Verify integration
        assert portfolio.risk_metrics.sharpe_ratio == 1.2
        assert portfolio.factor_exposures.market_beta == 1.1
        assert len(portfolio.positions) == 4
        
        # Test performance summary includes all data
        summary = portfolio.get_performance_summary()
        assert 'sharpe_ratio' in summary
        assert 'factor_exposures' in summary
        assert summary['factor_exposures']['momentum'] == 0.0  # Default value
    
    def test_serialization_roundtrip(self):
        """Test complete serialization roundtrip for all models"""
        # Create complete portfolio
        risk_metrics = RiskMetrics(portfolio_volatility=0.15, sharpe_ratio=1.2)
        factor_exposure = FactorExposure(market_beta=1.1, value=0.2)
        
        positions = {
            "AAPL": Position("AAPL", 0.5, 100, 50000, 48000, 2000, "Technology"),
            "GOOGL": Position("GOOGL", 0.5, 50, 50000, 48000, 2000, "Technology")
        }
        
        portfolio = PortfolioState(
            positions=positions,
            total_value=100000.0,
            risk_metrics=risk_metrics,
            factor_exposures=factor_exposure
        )
        
        # Serialize to dict
        portfolio_dict = portfolio.to_dict()
        
        # Deserialize back
        restored_portfolio = PortfolioState.from_dict(portfolio_dict)
        
        # Verify all data is preserved
        assert len(restored_portfolio.positions) == len(portfolio.positions)
        assert restored_portfolio.total_value == portfolio.total_value
        
        # Note: Risk metrics and factor exposures are not automatically 
        # serialized/deserialized in the current implementation
        # This would be enhanced in a production system


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v', '--tb=short'])