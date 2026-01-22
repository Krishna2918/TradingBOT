"""
Advanced Risk Management and Portfolio Optimization

This module implements advanced risk management techniques including
Value at Risk (VaR), Conditional Value at Risk (CVaR), portfolio optimization,
dynamic hedging, and stress testing.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import scipy.optimize as opt
from scipy import stats
import json
import os

from src.trading.positions import get_position_manager, get_open_positions, get_portfolio_summary
from src.config.mode_manager import get_current_mode

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    cvar_95: float  # Conditional Value at Risk at 95% confidence
    cvar_99: float  # Conditional Value at Risk at 99% confidence
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    portfolio_beta: float
    tracking_error: float
    information_ratio: float
    volatility: float
    skewness: float
    kurtosis: float

@dataclass
class PortfolioOptimization:
    """Portfolio optimization results."""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    efficient_frontier: List[Tuple[float, float]]
    risk_budget: Dict[str, float]
    diversification_ratio: float

@dataclass
class StressTestResult:
    """Stress test results."""
    scenario_name: str
    portfolio_loss: float
    portfolio_loss_percent: float
    worst_performing_assets: List[Tuple[str, float]]
    risk_contributions: Dict[str, float]
    recommendations: List[str]

@dataclass
class DynamicHedge:
    """Dynamic hedging strategy."""
    hedge_ratio: float
    hedge_instruments: List[str]
    hedge_effectiveness: float
    cost_benefit_ratio: float
    rebalancing_frequency: str

class AdvancedRiskManager:
    """Advanced risk management system."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        self.position_manager = get_position_manager()
        
        # Risk parameters
        self.confidence_levels = [0.95, 0.99]
        self.lookback_period = 252  # 1 year of trading days
        self.rebalancing_frequency = "daily"
        
        # Risk limits
        self.max_var_95 = 0.05  # 5% VaR limit
        self.max_var_99 = 0.10  # 10% VaR limit
        self.max_drawdown = 0.15  # 15% max drawdown
        self.max_volatility = 0.25  # 25% max volatility
        
        # Portfolio optimization parameters
        self.optimization_method = "mean_variance"
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.max_weight_per_asset = 0.20  # 20% max weight per asset
        
        # Historical data storage
        self.price_history = {}
        self.returns_history = {}
        self.portfolio_history = []
        
        # Risk models
        self.risk_models = {
            "historical": self._calculate_historical_var,
            "parametric": self._calculate_parametric_var,
            "monte_carlo": self._calculate_monte_carlo_var
        }
        
        logger.info(f"Advanced Risk Manager initialized for {mode} mode")
    
    def calculate_comprehensive_risk_metrics(self, portfolio_data: Optional[Dict[str, Any]] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics for the portfolio."""
        try:
            if portfolio_data is None:
                portfolio_data = get_portfolio_summary(self.mode)
            
            # Get historical returns
            returns = self._get_portfolio_returns()
            
            if len(returns) < 30:  # Need sufficient data
                logger.warning("Insufficient data for risk calculation")
                return self._get_default_risk_metrics()
            
            # Calculate VaR and CVaR
            var_95 = self._calculate_var(returns, 0.95)
            var_99 = self._calculate_var(returns, 0.99)
            cvar_95 = self._calculate_cvar(returns, 0.95)
            cvar_99 = self._calculate_cvar(returns, 0.99)
            
            # Calculate drawdown
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Calculate risk-adjusted returns
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            
            # Calculate portfolio characteristics
            portfolio_beta = self._calculate_portfolio_beta(returns)
            tracking_error = self._calculate_tracking_error(returns)
            information_ratio = self._calculate_information_ratio(returns)
            
            # Calculate distribution characteristics
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                portfolio_beta=portfolio_beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                volatility=volatility,
                skewness=skewness,
                kurtosis=kurtosis
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self._get_default_risk_metrics()
    
    def optimize_portfolio(self, expected_returns: Optional[Dict[str, float]] = None,
                          risk_model: str = "mean_variance") -> PortfolioOptimization:
        """Optimize portfolio using modern portfolio theory."""
        try:
            # Get current positions
            open_positions = get_open_positions(self.mode)
            
            if not open_positions:
                logger.warning("No positions to optimize")
                return self._get_default_optimization()
            
            # Get asset returns
            asset_returns = self._get_asset_returns()
            
            if len(asset_returns) < 30:
                logger.warning("Insufficient data for portfolio optimization")
                return self._get_default_optimization()
            
            # Calculate expected returns if not provided
            if expected_returns is None:
                expected_returns = self._calculate_expected_returns(asset_returns)
            
            # Calculate covariance matrix
            cov_matrix = self._calculate_covariance_matrix(asset_returns)
            
            # Get asset symbols
            symbols = list(expected_returns.keys())
            
            if risk_model == "mean_variance":
                return self._mean_variance_optimization(symbols, expected_returns, cov_matrix)
            elif risk_model == "risk_parity":
                return self._risk_parity_optimization(symbols, cov_matrix)
            elif risk_model == "black_litterman":
                return self._black_litterman_optimization(symbols, expected_returns, cov_matrix)
            else:
                logger.warning(f"Unknown risk model: {risk_model}")
                return self._mean_variance_optimization(symbols, expected_returns, cov_matrix)
                
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return self._get_default_optimization()
    
    def perform_stress_test(self, scenarios: Optional[List[Dict[str, Any]]] = None) -> List[StressTestResult]:
        """Perform stress testing on the portfolio."""
        try:
            if scenarios is None:
                scenarios = self._get_default_stress_scenarios()
            
            results = []
            
            for scenario in scenarios:
                result = self._run_stress_scenario(scenario)
                results.append(result)
            
            logger.info(f"Stress testing completed: {len(results)} scenarios")
            return results
            
        except Exception as e:
            logger.error(f"Error performing stress test: {e}")
            return []
    
    def calculate_dynamic_hedge(self, target_risk: float = 0.10) -> DynamicHedge:
        """Calculate dynamic hedging strategy."""
        try:
            # Get current portfolio
            portfolio_summary = get_portfolio_summary(self.mode)
            open_positions = get_open_positions(self.mode)
            
            if not open_positions:
                return self._get_default_hedge()
            
            # Calculate current portfolio risk
            current_risk = self._calculate_portfolio_risk(open_positions)
            
            # Calculate hedge ratio
            hedge_ratio = min(1.0, current_risk / target_risk)
            
            # Identify hedge instruments
            hedge_instruments = self._identify_hedge_instruments(open_positions)
            
            # Calculate hedge effectiveness
            hedge_effectiveness = self._calculate_hedge_effectiveness(hedge_instruments)
            
            # Calculate cost-benefit ratio
            cost_benefit_ratio = self._calculate_hedge_cost_benefit(hedge_ratio, hedge_effectiveness)
            
            return DynamicHedge(
                hedge_ratio=hedge_ratio,
                hedge_instruments=hedge_instruments,
                hedge_effectiveness=hedge_effectiveness,
                cost_benefit_ratio=cost_benefit_ratio,
                rebalancing_frequency="daily"
            )
            
        except Exception as e:
            logger.error(f"Error calculating dynamic hedge: {e}")
            return self._get_default_hedge()
    
    def monitor_risk_limits(self) -> Dict[str, Any]:
        """Monitor risk limits and generate alerts."""
        try:
            # Calculate current risk metrics
            risk_metrics = self.calculate_comprehensive_risk_metrics()
            
            # Check risk limits
            alerts = []
            violations = []
            
            # VaR limits
            if risk_metrics.var_95 > self.max_var_95:
                violations.append(f"VaR 95% limit exceeded: {risk_metrics.var_95:.3f} > {self.max_var_95:.3f}")
            
            if risk_metrics.var_99 > self.max_var_99:
                violations.append(f"VaR 99% limit exceeded: {risk_metrics.var_99:.3f} > {self.max_var_99:.3f}")
            
            # Drawdown limit
            if risk_metrics.max_drawdown > self.max_drawdown:
                violations.append(f"Max drawdown limit exceeded: {risk_metrics.max_drawdown:.3f} > {self.max_drawdown:.3f}")
            
            # Volatility limit
            if risk_metrics.volatility > self.max_volatility:
                violations.append(f"Volatility limit exceeded: {risk_metrics.volatility:.3f} > {self.max_volatility:.3f}")
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(risk_metrics, violations)
            
            return {
                "risk_metrics": risk_metrics,
                "violations": violations,
                "alerts": alerts,
                "recommendations": recommendations,
                "status": "HEALTHY" if not violations else "AT_RISK"
            }
            
        except Exception as e:
            logger.error(f"Error monitoring risk limits: {e}")
            return {
                "risk_metrics": self._get_default_risk_metrics(),
                "violations": [f"Error: {e}"],
                "alerts": [],
                "recommendations": ["Fix system error"],
                "status": "ERROR"
            }
    
    def _get_portfolio_returns(self) -> np.ndarray:
        """Get historical portfolio returns."""
        try:
            # This would integrate with real portfolio data in production
            # For now, generate simulated returns
            np.random.seed(42)  # For reproducibility
            returns = np.random.normal(0.001, 0.02, self.lookback_period)
            return returns
            
        except Exception as e:
            logger.error(f"Error getting portfolio returns: {e}")
            return np.array([])
    
    def _get_asset_returns(self) -> Dict[str, np.ndarray]:
        """Get historical returns for individual assets."""
        try:
            # This would integrate with real market data in production
            # For now, generate simulated returns
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            asset_returns = {}
            
            np.random.seed(42)
            for symbol in symbols:
                returns = np.random.normal(0.001, 0.02, self.lookback_period)
                asset_returns[symbol] = returns
            
            return asset_returns
            
        except Exception as e:
            logger.error(f"Error getting asset returns: {e}")
            return {}
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        try:
            return np.percentile(returns, (1 - confidence_level) * 100)
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def _calculate_historical_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Historical Value at Risk (VaR)."""
        try:
            if len(returns) == 0:
                return 0.0
            
            # Sort returns in ascending order
            sorted_returns = np.sort(returns)
            
            # Calculate the index for the confidence level
            index = int((1 - confidence_level) * len(sorted_returns))
            
            # Return the VaR (negative of the return at the confidence level)
            return -sorted_returns[index]
        except Exception as e:
            logger.error(f"Error calculating historical VaR: {e}")
            return 0.0
    
    def _calculate_parametric_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Parametric Value at Risk (VaR) using normal distribution assumption."""
        try:
            if len(returns) == 0:
                return 0.0
            
            # Calculate mean and standard deviation
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Calculate z-score for the confidence level
            from scipy.stats import norm
            z_score = norm.ppf(1 - confidence_level)
            
            # Calculate parametric VaR
            var = mean_return + z_score * std_return
            
            return -var
        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {e}")
            return 0.0
    
    def _calculate_monte_carlo_var(self, returns: np.ndarray, confidence_level: float, num_simulations: int = 10000) -> float:
        """Calculate Monte Carlo Value at Risk (VaR)."""
        try:
            if len(returns) == 0:
                return 0.0
            
            # Calculate mean and standard deviation
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Generate random samples from normal distribution
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
            
            # Sort simulated returns
            sorted_returns = np.sort(simulated_returns)
            
            # Calculate the index for the confidence level
            index = int((1 - confidence_level) * len(sorted_returns))
            
            # Return the VaR (negative of the return at the confidence level)
            return -sorted_returns[index]
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return 0.0
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk."""
        try:
            var = self._calculate_var(returns, confidence_level)
            return np.mean(returns[returns <= var])
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            return np.min(drawdown)
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        try:
            excess_returns = returns - self.risk_free_rate / 252
            return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        try:
            excess_returns = returns - self.risk_free_rate / 252
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
            return np.mean(excess_returns) / downside_std * np.sqrt(252)
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        try:
            annual_return = np.mean(returns) * 252
            return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    def _calculate_portfolio_beta(self, returns: np.ndarray) -> float:
        """Calculate portfolio beta."""
        try:
            # This would use market index returns in production
            # For now, return a simulated beta
            return 1.2
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0
    
    def _calculate_tracking_error(self, returns: np.ndarray) -> float:
        """Calculate tracking error."""
        try:
            # This would use benchmark returns in production
            # For now, return a simulated tracking error
            return 0.05
        except Exception as e:
            logger.error(f"Error calculating tracking error: {e}")
            return 0.0
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate information ratio."""
        try:
            # This would use benchmark returns in production
            # For now, return a simulated information ratio
            return 0.5
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            return 0.0
    
    def _calculate_expected_returns(self, asset_returns: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate expected returns for assets."""
        try:
            expected_returns = {}
            for symbol, returns in asset_returns.items():
                expected_returns[symbol] = np.mean(returns) * 252  # Annualized
            return expected_returns
        except Exception as e:
            logger.error(f"Error calculating expected returns: {e}")
            return {}
    
    def _calculate_covariance_matrix(self, asset_returns: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate covariance matrix for assets."""
        try:
            symbols = list(asset_returns.keys())
            returns_matrix = np.array([asset_returns[symbol] for symbol in symbols])
            cov_matrix = np.cov(returns_matrix)
            return cov_matrix
        except Exception as e:
            logger.error(f"Error calculating covariance matrix: {e}")
            return np.eye(len(asset_returns))
    
    def _mean_variance_optimization(self, symbols: List[str], expected_returns: Dict[str, float],
                                   cov_matrix: np.ndarray) -> PortfolioOptimization:
        """Perform mean-variance optimization."""
        try:
            n_assets = len(symbols)
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
            
            # Bounds: 0 <= weight <= max_weight_per_asset
            bounds = [(0, self.max_weight_per_asset) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = dict(zip(symbols, result.x))
                expected_return = sum(optimal_weights[symbol] * expected_returns[symbol] for symbol in symbols)
                expected_volatility = np.sqrt(result.fun)
                sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility
                
                return PortfolioOptimization(
                    optimal_weights=optimal_weights,
                    expected_return=expected_return,
                    expected_volatility=expected_volatility,
                    sharpe_ratio=sharpe_ratio,
                    efficient_frontier=[],
                    risk_budget={},
                    diversification_ratio=0.0
                )
            else:
                logger.warning("Portfolio optimization failed")
                return self._get_default_optimization()
                
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            return self._get_default_optimization()
    
    def _risk_parity_optimization(self, symbols: List[str], cov_matrix: np.ndarray) -> PortfolioOptimization:
        """Perform risk parity optimization."""
        try:
            n_assets = len(symbols)
            
            # Objective function: minimize sum of squared risk contributions
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                risk_contributions = (weights * np.dot(cov_matrix, weights)) / portfolio_variance
                return np.sum((risk_contributions - 1/n_assets) ** 2)
            
            # Constraints
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Bounds
            bounds = [(0, self.max_weight_per_asset) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = dict(zip(symbols, result.x))
                portfolio_variance = np.dot(result.x.T, np.dot(cov_matrix, result.x))
                expected_volatility = np.sqrt(portfolio_variance)
                
                return PortfolioOptimization(
                    optimal_weights=optimal_weights,
                    expected_return=0.0,  # Risk parity doesn't optimize for return
                    expected_volatility=expected_volatility,
                    sharpe_ratio=0.0,
                    efficient_frontier=[],
                    risk_budget={},
                    diversification_ratio=0.0
                )
            else:
                logger.warning("Risk parity optimization failed")
                return self._get_default_optimization()
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return self._get_default_optimization()
    
    def _black_litterman_optimization(self, symbols: List[str], expected_returns: Dict[str, float],
                                     cov_matrix: np.ndarray) -> PortfolioOptimization:
        """Perform Black-Litterman optimization."""
        try:
            # This is a simplified implementation
            # In production, this would include views and confidence levels
            
            # For now, use mean-variance optimization
            return self._mean_variance_optimization(symbols, expected_returns, cov_matrix)
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            return self._get_default_optimization()
    
    def _get_default_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Get default stress test scenarios."""
        return [
            {
                "name": "Market Crash",
                "description": "50% decline in all equity markets",
                "asset_shocks": {"AAPL": -0.5, "MSFT": -0.5, "GOOGL": -0.5, "AMZN": -0.5, "TSLA": -0.5}
            },
            {
                "name": "Tech Sector Crash",
                "description": "30% decline in technology stocks",
                "asset_shocks": {"AAPL": -0.3, "MSFT": -0.3, "GOOGL": -0.3, "AMZN": -0.3, "TSLA": -0.3}
            },
            {
                "name": "Interest Rate Shock",
                "description": "2% increase in interest rates",
                "asset_shocks": {"AAPL": -0.1, "MSFT": -0.1, "GOOGL": -0.1, "AMZN": -0.1, "TSLA": -0.1}
            },
            {
                "name": "Volatility Spike",
                "description": "Doubling of market volatility",
                "asset_shocks": {"AAPL": -0.2, "MSFT": -0.2, "GOOGL": -0.2, "AMZN": -0.2, "TSLA": -0.2}
            }
        ]
    
    def _run_stress_scenario(self, scenario: Dict[str, Any]) -> StressTestResult:
        """Run a single stress test scenario."""
        try:
            # Get current positions
            open_positions = get_open_positions(self.mode)
            
            if not open_positions:
                return StressTestResult(
                    scenario_name=scenario["name"],
                    portfolio_loss=0.0,
                    portfolio_loss_percent=0.0,
                    worst_performing_assets=[],
                    risk_contributions={},
                    recommendations=["No positions to stress test"]
                )
            
            # Calculate portfolio loss
            total_loss = 0.0
            worst_performing_assets = []
            
            for position in open_positions:
                symbol = position.symbol
                shock = scenario["asset_shocks"].get(symbol, 0.0)
                position_loss = position.entry_price * position.quantity * shock
                total_loss += position_loss
                
                if shock < -0.1:  # Significant loss
                    worst_performing_assets.append((symbol, shock))
            
            # Calculate portfolio loss percentage
            portfolio_value = sum(p.entry_price * p.quantity for p in open_positions)
            portfolio_loss_percent = total_loss / portfolio_value if portfolio_value > 0 else 0.0
            
            # Calculate risk contributions
            risk_contributions = {}
            for position in open_positions:
                symbol = position.symbol
                shock = scenario["asset_shocks"].get(symbol, 0.0)
                position_loss = position.entry_price * position.quantity * shock
                risk_contributions[symbol] = position_loss / total_loss if total_loss != 0 else 0.0
            
            # Generate recommendations
            recommendations = self._generate_stress_recommendations(scenario, portfolio_loss_percent, worst_performing_assets)
            
            return StressTestResult(
                scenario_name=scenario["name"],
                portfolio_loss=total_loss,
                portfolio_loss_percent=portfolio_loss_percent,
                worst_performing_assets=worst_performing_assets,
                risk_contributions=risk_contributions,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error running stress scenario: {e}")
            return StressTestResult(
                scenario_name=scenario["name"],
                portfolio_loss=0.0,
                portfolio_loss_percent=0.0,
                worst_performing_assets=[],
                risk_contributions={},
                recommendations=[f"Error: {e}"]
            )
    
    def _generate_stress_recommendations(self, scenario: Dict[str, Any], 
                                       portfolio_loss_percent: float,
                                       worst_performing_assets: List[Tuple[str, float]]) -> List[str]:
        """Generate recommendations based on stress test results."""
        recommendations = []
        
        if portfolio_loss_percent > 0.2:  # 20% loss
            recommendations.append("Consider reducing portfolio exposure")
        
        if len(worst_performing_assets) > 2:
            recommendations.append("Diversify away from worst-performing assets")
        
        if scenario["name"] == "Market Crash":
            recommendations.append("Consider adding defensive assets")
        
        if scenario["name"] == "Tech Sector Crash":
            recommendations.append("Reduce technology sector concentration")
        
        if scenario["name"] == "Interest Rate Shock":
            recommendations.append("Consider interest rate hedging")
        
        if scenario["name"] == "Volatility Spike":
            recommendations.append("Implement volatility hedging strategies")
        
        return recommendations
    
    def _calculate_portfolio_risk(self, positions: List) -> float:
        """Calculate current portfolio risk."""
        try:
            # Simplified risk calculation
            total_value = sum(p.entry_price * p.quantity for p in positions)
            if total_value == 0:
                return 0.0
            
            # Calculate weighted average risk (simplified)
            risk = 0.0
            for position in positions:
                position_weight = (position.entry_price * position.quantity) / total_value
                # Assume 20% risk per position (simplified)
                risk += position_weight * 0.20
            
            return risk
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return 0.0
    
    def _identify_hedge_instruments(self, positions: List) -> List[str]:
        """Identify appropriate hedge instruments."""
        try:
            # Simplified hedge instrument identification
            hedge_instruments = []
            
            for position in positions:
                symbol = position.symbol
                # Add corresponding ETF or index as hedge
                if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
                    hedge_instruments.append("QQQ")  # NASDAQ ETF
            
            return hedge_instruments
            
        except Exception as e:
            logger.error(f"Error identifying hedge instruments: {e}")
            return []
    
    def _calculate_hedge_effectiveness(self, hedge_instruments: List[str]) -> float:
        """Calculate hedge effectiveness."""
        try:
            # Simplified hedge effectiveness calculation
            if not hedge_instruments:
                return 0.0
            
            # Assume 80% effectiveness for ETF hedges
            return 0.8
            
        except Exception as e:
            logger.error(f"Error calculating hedge effectiveness: {e}")
            return 0.0
    
    def _calculate_hedge_cost_benefit(self, hedge_ratio: float, hedge_effectiveness: float) -> float:
        """Calculate hedge cost-benefit ratio."""
        try:
            # Simplified cost-benefit calculation
            benefit = hedge_ratio * hedge_effectiveness
            cost = hedge_ratio * 0.02  # Assume 2% cost
            return benefit / cost if cost > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating hedge cost-benefit: {e}")
            return 0.0
    
    def _generate_risk_recommendations(self, risk_metrics: RiskMetrics, 
                                     violations: List[str]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if violations:
            recommendations.append("Immediate action required to address risk limit violations")
        
        if risk_metrics.var_95 > self.max_var_95 * 0.8:  # Approaching limit
            recommendations.append("Consider reducing position sizes to lower VaR")
        
        if risk_metrics.max_drawdown > self.max_drawdown * 0.8:  # Approaching limit
            recommendations.append("Implement stop-loss orders to limit drawdown")
        
        if risk_metrics.volatility > self.max_volatility * 0.8:  # Approaching limit
            recommendations.append("Diversify portfolio to reduce volatility")
        
        if risk_metrics.sharpe_ratio < 0.5:  # Low risk-adjusted return
            recommendations.append("Review portfolio allocation for better risk-adjusted returns")
        
        if risk_metrics.skewness < -1.0:  # Negative skewness
            recommendations.append("Consider adding assets with positive skewness")
        
        if risk_metrics.kurtosis > 3.0:  # High kurtosis (fat tails)
            recommendations.append("Consider tail risk hedging strategies")
        
        return recommendations
    
    def _get_default_risk_metrics(self) -> RiskMetrics:
        """Get default risk metrics when calculation fails."""
        return RiskMetrics(
            var_95=0.02,
            var_99=0.05,
            cvar_95=0.03,
            cvar_99=0.07,
            max_drawdown=0.05,
            sharpe_ratio=0.5,
            sortino_ratio=0.6,
            calmar_ratio=0.4,
            portfolio_beta=1.0,
            tracking_error=0.05,
            information_ratio=0.3,
            volatility=0.15,
            skewness=0.0,
            kurtosis=3.0
        )
    
    def _get_default_optimization(self) -> PortfolioOptimization:
        """Get default optimization when calculation fails."""
        return PortfolioOptimization(
            optimal_weights={},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            efficient_frontier=[],
            risk_budget={},
            diversification_ratio=0.0
        )
    
    def _get_default_hedge(self) -> DynamicHedge:
        """Get default hedge when calculation fails."""
        return DynamicHedge(
            hedge_ratio=0.0,
            hedge_instruments=[],
            hedge_effectiveness=0.0,
            cost_benefit_ratio=0.0,
            rebalancing_frequency="daily"
        )
    
    def save_risk_data(self, filepath: str):
        """Save risk data to file."""
        try:
            risk_data = {
                "risk_metrics": self.calculate_comprehensive_risk_metrics(),
                "portfolio_optimization": self.optimize_portfolio(),
                "stress_test_results": self.perform_stress_test(),
                "dynamic_hedge": self.calculate_dynamic_hedge(),
                "risk_limits_status": self.monitor_risk_limits()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(risk_data, f, indent=2, default=str)
            
            logger.info(f"Risk data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving risk data: {e}")

# Global advanced risk manager instance
_advanced_risk_manager: Optional[AdvancedRiskManager] = None

def get_advanced_risk_manager(mode: str = None) -> AdvancedRiskManager:
    """Get the global advanced risk manager instance."""
    global _advanced_risk_manager
    if _advanced_risk_manager is None:
        if mode is None:
            mode = get_current_mode()
        _advanced_risk_manager = AdvancedRiskManager(mode)
    return _advanced_risk_manager

def calculate_risk_metrics(portfolio_data: Optional[Dict[str, Any]] = None, mode: str = None) -> RiskMetrics:
    """Calculate comprehensive risk metrics."""
    return get_advanced_risk_manager(mode).calculate_comprehensive_risk_metrics(portfolio_data)

def optimize_portfolio(expected_returns: Optional[Dict[str, float]] = None, 
                      risk_model: str = "mean_variance", mode: str = None) -> PortfolioOptimization:
    """Optimize portfolio."""
    return get_advanced_risk_manager(mode).optimize_portfolio(expected_returns, risk_model)

def perform_stress_test(scenarios: Optional[List[Dict[str, Any]]] = None, mode: str = None) -> List[StressTestResult]:
    """Perform stress testing."""
    return get_advanced_risk_manager(mode).perform_stress_test(scenarios)

def calculate_dynamic_hedge(target_risk: float = 0.10, mode: str = None) -> DynamicHedge:
    """Calculate dynamic hedge."""
    return get_advanced_risk_manager(mode).calculate_dynamic_hedge(target_risk)

def monitor_risk_limits(mode: str = None) -> Dict[str, Any]:
    """Monitor risk limits."""
    return get_advanced_risk_manager(mode).monitor_risk_limits()
