"""
VaR/Beta Tracking System
Implements Value at Risk (VaR) and Beta tracking for portfolio risk management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

@dataclass
class VaRResult:
    """VaR calculation result"""
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    confidence_level: float
    time_horizon: int  # days
    method: str
    timestamp: datetime

@dataclass
class BetaResult:
    """Beta calculation result"""
    beta: float
    alpha: float
    r_squared: float
    correlation: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    time_period: int  # days
    benchmark: str
    timestamp: datetime

class VaRCalculator:
    """Value at Risk calculator with multiple methods"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.returns_history = {}
        self.var_history = []
        
        logger.info("VaR Calculator initialized")
    
    def calculate_historical_var(self, returns: pd.Series, confidence_level: float = 0.95, 
                               time_horizon: int = 1) -> VaRResult:
        """
        Calculate VaR using historical simulation method
        
        Args:
            returns: Historical returns series
            confidence_level: Confidence level (0.95 for 95% VaR)
            time_horizon: Time horizon in days
        
        Returns:
            VaRResult object
        """
        try:
            if len(returns) < 30:
                logger.warning("Insufficient data for VaR calculation")
                return self._create_empty_var_result(confidence_level, time_horizon, "historical")
            
            # Sort returns
            sorted_returns = returns.sort_values()
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(sorted_returns, var_percentile)
            
            # Calculate Expected Shortfall (Conditional VaR)
            tail_returns = sorted_returns[sorted_returns <= var_value]
            expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_value
            
            # Scale for time horizon
            var_scaled = var_value * np.sqrt(time_horizon)
            es_scaled = expected_shortfall * np.sqrt(time_horizon)
            
            result = VaRResult(
                var_95=var_scaled if confidence_level == 0.95 else 0,
                var_99=var_scaled if confidence_level == 0.99 else 0,
                expected_shortfall_95=es_scaled if confidence_level == 0.95 else 0,
                expected_shortfall_99=es_scaled if confidence_level == 0.99 else 0,
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                method="historical",
                timestamp=datetime.now()
            )
            
            # Store in history
            self.var_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating historical VaR: {e}")
            return self._create_empty_var_result(confidence_level, time_horizon, "historical")
    
    def calculate_parametric_var(self, returns: pd.Series, confidence_level: float = 0.95,
                               time_horizon: int = 1) -> VaRResult:
        """
        Calculate VaR using parametric (normal distribution) method
        
        Args:
            returns: Historical returns series
            confidence_level: Confidence level
            time_horizon: Time horizon in days
        
        Returns:
            VaRResult object
        """
        try:
            if len(returns) < 30:
                logger.warning("Insufficient data for parametric VaR calculation")
                return self._create_empty_var_result(confidence_level, time_horizon, "parametric")
            
            # Calculate mean and standard deviation
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Calculate z-score for confidence level
            z_score = stats.norm.ppf(1 - confidence_level)
            
            # Calculate VaR
            var_value = mean_return + z_score * std_return
            
            # Calculate Expected Shortfall
            expected_shortfall = mean_return - std_return * stats.norm.pdf(z_score) / (1 - confidence_level)
            
            # Scale for time horizon
            var_scaled = var_value * np.sqrt(time_horizon)
            es_scaled = expected_shortfall * np.sqrt(time_horizon)
            
            result = VaRResult(
                var_95=var_scaled if confidence_level == 0.95 else 0,
                var_99=var_scaled if confidence_level == 0.99 else 0,
                expected_shortfall_95=es_scaled if confidence_level == 0.95 else 0,
                expected_shortfall_99=es_scaled if confidence_level == 0.99 else 0,
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                method="parametric",
                timestamp=datetime.now()
            )
            
            # Store in history
            self.var_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {e}")
            return self._create_empty_var_result(confidence_level, time_horizon, "parametric")
    
    def calculate_monte_carlo_var(self, returns: pd.Series, confidence_level: float = 0.95,
                                time_horizon: int = 1, num_simulations: int = 10000) -> VaRResult:
        """
        Calculate VaR using Monte Carlo simulation
        
        Args:
            returns: Historical returns series
            confidence_level: Confidence level
            time_horizon: Time horizon in days
            num_simulations: Number of Monte Carlo simulations
        
        Returns:
            VaRResult object
        """
        try:
            if len(returns) < 30:
                logger.warning("Insufficient data for Monte Carlo VaR calculation")
                return self._create_empty_var_result(confidence_level, time_horizon, "monte_carlo")
            
            # Fit distribution to returns
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
            
            # Calculate portfolio returns for time horizon
            portfolio_returns = []
            for _ in range(num_simulations):
                # Simulate returns for time horizon
                horizon_returns = np.random.normal(mean_return, std_return, time_horizon)
                portfolio_return = np.prod(1 + horizon_returns) - 1
                portfolio_returns.append(portfolio_return)
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(portfolio_returns, var_percentile)
            
            # Calculate Expected Shortfall
            tail_returns = [r for r in portfolio_returns if r <= var_value]
            expected_shortfall = np.mean(tail_returns) if tail_returns else var_value
            
            result = VaRResult(
                var_95=var_value if confidence_level == 0.95 else 0,
                var_99=var_value if confidence_level == 0.99 else 0,
                expected_shortfall_95=expected_shortfall if confidence_level == 0.95 else 0,
                expected_shortfall_99=expected_shortfall if confidence_level == 0.99 else 0,
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                method="monte_carlo",
                timestamp=datetime.now()
            )
            
            # Store in history
            self.var_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return self._create_empty_var_result(confidence_level, time_horizon, "monte_carlo")
    
    def calculate_portfolio_var(self, portfolio_weights: Dict[str, float], 
                              returns_data: Dict[str, pd.Series],
                              confidence_level: float = 0.95) -> VaRResult:
        """
        Calculate portfolio VaR using covariance matrix
        
        Args:
            portfolio_weights: Dictionary of symbol -> weight
            returns_data: Dictionary of symbol -> returns series
            confidence_level: Confidence level
        
        Returns:
            VaRResult object
        """
        try:
            # Create returns matrix
            symbols = list(portfolio_weights.keys())
            returns_matrix = pd.DataFrame({symbol: returns_data[symbol] for symbol in symbols})
            
            # Calculate portfolio returns
            weights = np.array([portfolio_weights[symbol] for symbol in symbols])
            portfolio_returns = (returns_matrix * weights).sum(axis=1)
            
            # Calculate portfolio VaR
            return self.calculate_historical_var(portfolio_returns, confidence_level)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return self._create_empty_var_result(confidence_level, 1, "portfolio")
    
    def _create_empty_var_result(self, confidence_level: float, time_horizon: int, 
                               method: str) -> VaRResult:
        """Create empty VaR result for error cases"""
        return VaRResult(
            var_95=0.0,
            var_99=0.0,
            expected_shortfall_95=0.0,
            expected_shortfall_99=0.0,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=method,
            timestamp=datetime.now()
        )
    
    def get_var_statistics(self) -> Dict:
        """Get VaR calculation statistics"""
        try:
            if not self.var_history:
                return {}
            
            # Calculate statistics
            var_95_values = [var.var_95 for var in self.var_history if var.var_95 != 0]
            var_99_values = [var.var_99 for var in self.var_history if var.var_99 != 0]
            
            return {
                'total_calculations': len(self.var_history),
                'var_95_stats': {
                    'mean': np.mean(var_95_values) if var_95_values else 0,
                    'std': np.std(var_95_values) if var_95_values else 0,
                    'min': np.min(var_95_values) if var_95_values else 0,
                    'max': np.max(var_95_values) if var_95_values else 0
                },
                'var_99_stats': {
                    'mean': np.mean(var_99_values) if var_99_values else 0,
                    'std': np.std(var_99_values) if var_99_values else 0,
                    'min': np.min(var_99_values) if var_99_values else 0,
                    'max': np.max(var_99_values) if var_99_values else 0
                },
                'methods_used': list(set(var.method for var in self.var_history))
            }
            
        except Exception as e:
            logger.error(f"Error getting VaR statistics: {e}")
            return {}

class BetaCalculator:
    """Beta calculation for portfolio and individual assets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.beta_history = []
        self.benchmark_data = {}
        
        logger.info("Beta Calculator initialized")
    
    def calculate_beta(self, asset_returns: pd.Series, benchmark_returns: pd.Series,
                      time_period: int = 252) -> BetaResult:
        """
        Calculate beta using linear regression
        
        Args:
            asset_returns: Asset returns series
            benchmark_returns: Benchmark returns series
            time_period: Time period for calculation
        
        Returns:
            BetaResult object
        """
        try:
            # Align returns data
            aligned_data = pd.DataFrame({
                'asset': asset_returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_data) < 30:
                logger.warning("Insufficient data for beta calculation")
                return self._create_empty_beta_result(time_period, "unknown")
            
            # Calculate beta using linear regression
            from sklearn.linear_model import LinearRegression
            
            X = aligned_data['benchmark'].values.reshape(-1, 1)
            y = aligned_data['asset'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            beta = model.coef_[0]
            alpha = model.intercept_
            
            # Calculate R-squared
            y_pred = model.predict(X)
            r_squared = model.score(X, y)
            
            # Calculate correlation
            correlation = np.corrcoef(aligned_data['asset'], aligned_data['benchmark'])[0, 1]
            
            # Calculate standard error
            residuals = y - y_pred
            mse = np.mean(residuals ** 2)
            standard_error = np.sqrt(mse)
            
            # Calculate confidence interval for beta
            n = len(aligned_data)
            t_value = stats.t.ppf(0.975, n - 2)  # 95% confidence
            se_beta = standard_error / np.sqrt(np.sum((X.flatten() - np.mean(X))**2))
            ci_lower = beta - t_value * se_beta
            ci_upper = beta + t_value * se_beta
            
            result = BetaResult(
                beta=beta,
                alpha=alpha,
                r_squared=r_squared,
                correlation=correlation,
                standard_error=standard_error,
                confidence_interval=(ci_lower, ci_upper),
                time_period=time_period,
                benchmark="market_index",
                timestamp=datetime.now()
            )
            
            # Store in history
            self.beta_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return self._create_empty_beta_result(time_period, "unknown")
    
    def calculate_rolling_beta(self, asset_returns: pd.Series, benchmark_returns: pd.Series,
                             window: int = 252) -> pd.Series:
        """
        Calculate rolling beta over time
        
        Args:
            asset_returns: Asset returns series
            benchmark_returns: Benchmark returns series
            window: Rolling window size
        
        Returns:
            Series of rolling beta values
        """
        try:
            # Align returns data
            aligned_data = pd.DataFrame({
                'asset': asset_returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            rolling_beta = []
            rolling_dates = []
            
            for i in range(window, len(aligned_data)):
                window_data = aligned_data.iloc[i-window:i]
                
                # Calculate beta for this window
                beta_result = self.calculate_beta(
                    window_data['asset'], 
                    window_data['benchmark'],
                    window
                )
                
                rolling_beta.append(beta_result.beta)
                rolling_dates.append(aligned_data.index[i])
            
            return pd.Series(rolling_beta, index=rolling_dates)
            
        except Exception as e:
            logger.error(f"Error calculating rolling beta: {e}")
            return pd.Series()
    
    def calculate_portfolio_beta(self, portfolio_weights: Dict[str, float],
                               asset_betas: Dict[str, float]) -> float:
        """
        Calculate portfolio beta as weighted average of individual asset betas
        
        Args:
            portfolio_weights: Dictionary of symbol -> weight
            asset_betas: Dictionary of symbol -> beta
        
        Returns:
            Portfolio beta
        """
        try:
            portfolio_beta = 0.0
            
            for symbol, weight in portfolio_weights.items():
                if symbol in asset_betas:
                    portfolio_beta += weight * asset_betas[symbol]
            
            return portfolio_beta
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 0.0
    
    def _create_empty_beta_result(self, time_period: int, benchmark: str) -> BetaResult:
        """Create empty beta result for error cases"""
        return BetaResult(
            beta=0.0,
            alpha=0.0,
            r_squared=0.0,
            correlation=0.0,
            standard_error=0.0,
            confidence_interval=(0.0, 0.0),
            time_period=time_period,
            benchmark=benchmark,
            timestamp=datetime.now()
        )
    
    def get_beta_statistics(self) -> Dict:
        """Get beta calculation statistics"""
        try:
            if not self.beta_history:
                return {}
            
            # Calculate statistics
            betas = [beta.beta for beta in self.beta_history]
            alphas = [beta.alpha for beta in self.beta_history]
            r_squareds = [beta.r_squared for beta in self.beta_history]
            
            return {
                'total_calculations': len(self.beta_history),
                'beta_stats': {
                    'mean': np.mean(betas),
                    'std': np.std(betas),
                    'min': np.min(betas),
                    'max': np.max(betas)
                },
                'alpha_stats': {
                    'mean': np.mean(alphas),
                    'std': np.std(alphas),
                    'min': np.min(alphas),
                    'max': np.max(alphas)
                },
                'r_squared_stats': {
                    'mean': np.mean(r_squareds),
                    'std': np.std(r_squareds),
                    'min': np.min(r_squareds),
                    'max': np.max(r_squareds)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting beta statistics: {e}")
            return {}

class VaRBetaTracker:
    """Combined VaR and Beta tracking system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.var_calculator = VaRCalculator(config)
        self.beta_calculator = BetaCalculator(config)
        self.portfolio_data = {}
        
        logger.info("VaR/Beta Tracker initialized")
    
    def update_portfolio_data(self, symbol: str, returns: pd.Series, 
                            benchmark_returns: pd.Series = None):
        """Update portfolio data for a symbol"""
        try:
            self.portfolio_data[symbol] = {
                'returns': returns,
                'benchmark_returns': benchmark_returns,
                'last_updated': datetime.now()
            }
            
            logger.info(f"Updated portfolio data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio data: {e}")
    
    def calculate_comprehensive_risk_metrics(self, portfolio_weights: Dict[str, float],
                                           confidence_level: float = 0.95) -> Dict:
        """
        Calculate comprehensive risk metrics for portfolio
        
        Args:
            portfolio_weights: Dictionary of symbol -> weight
            confidence_level: Confidence level for VaR
        
        Returns:
            Dictionary of risk metrics
        """
        try:
            risk_metrics = {}
            
            # Calculate individual asset metrics
            individual_metrics = {}
            asset_betas = {}
            
            for symbol, weight in portfolio_weights.items():
                if symbol in self.portfolio_data:
                    data = self.portfolio_data[symbol]
                    returns = data['returns']
                    benchmark_returns = data['benchmark_returns']
                    
                    # Calculate VaR for individual asset
                    var_result = self.var_calculator.calculate_historical_var(
                        returns, confidence_level
                    )
                    
                    # Calculate beta if benchmark data available
                    beta_result = None
                    if benchmark_returns is not None:
                        beta_result = self.beta_calculator.calculate_beta(
                            returns, benchmark_returns
                        )
                        asset_betas[symbol] = beta_result.beta
                    
                    individual_metrics[symbol] = {
                        'weight': weight,
                        'var_95': var_result.var_95,
                        'var_99': var_result.var_99,
                        'beta': beta_result.beta if beta_result else 0.0,
                        'alpha': beta_result.alpha if beta_result else 0.0,
                        'r_squared': beta_result.r_squared if beta_result else 0.0
                    }
            
            # Calculate portfolio VaR
            portfolio_returns_data = {
                symbol: data['returns'] 
                for symbol, data in self.portfolio_data.items() 
                if symbol in portfolio_weights
            }
            
            portfolio_var = self.var_calculator.calculate_portfolio_var(
                portfolio_weights, portfolio_returns_data, confidence_level
            )
            
            # Calculate portfolio beta
            portfolio_beta = self.beta_calculator.calculate_portfolio_beta(
                portfolio_weights, asset_betas
            )
            
            risk_metrics = {
                'portfolio_var': {
                    'var_95': portfolio_var.var_95,
                    'var_99': portfolio_var.var_99,
                    'expected_shortfall_95': portfolio_var.expected_shortfall_95,
                    'expected_shortfall_99': portfolio_var.expected_shortfall_99,
                    'method': portfolio_var.method
                },
                'portfolio_beta': portfolio_beta,
                'individual_assets': individual_metrics,
                'risk_attribution': self._calculate_risk_attribution(individual_metrics),
                'timestamp': datetime.now()
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive risk metrics: {e}")
            return {}
    
    def _calculate_risk_attribution(self, individual_metrics: Dict) -> Dict:
        """Calculate risk attribution by asset"""
        try:
            total_var_95 = sum(metrics['var_95'] * metrics['weight'] 
                             for metrics in individual_metrics.values())
            total_var_99 = sum(metrics['var_99'] * metrics['weight'] 
                             for metrics in individual_metrics.values())
            
            attribution = {}
            for symbol, metrics in individual_metrics.items():
                contribution_95 = (metrics['var_95'] * metrics['weight']) / total_var_95 if total_var_95 != 0 else 0
                contribution_99 = (metrics['var_99'] * metrics['weight']) / total_var_99 if total_var_99 != 0 else 0
                
                attribution[symbol] = {
                    'var_95_contribution': contribution_95,
                    'var_99_contribution': contribution_99,
                    'weight': metrics['weight'],
                    'beta': metrics['beta']
                }
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error calculating risk attribution: {e}")
            return {}
    
    def get_risk_dashboard_data(self) -> Dict:
        """Get data for risk dashboard"""
        try:
            return {
                'var_statistics': self.var_calculator.get_var_statistics(),
                'beta_statistics': self.beta_calculator.get_var_statistics(),
                'portfolio_symbols': list(self.portfolio_data.keys()),
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk dashboard data: {e}")
            return {}
