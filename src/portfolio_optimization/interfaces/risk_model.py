"""
Interface for risk models.

This interface ensures consistent risk calculations across all components
and enables easy swapping of risk models based on market conditions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


class IRiskModel(ABC):
    """
    Interface for portfolio risk models.
    
    All risk models must implement this interface to ensure consistent
    risk calculations and enable automatic model selection.
    """
    
    @abstractmethod
    def calculate_risk(
        self, 
        weights: np.ndarray, 
        returns: Optional[np.ndarray] = None,
        covariance_matrix: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate portfolio risk (volatility).
        
        Args:
            weights: Portfolio weights
            returns: Historical returns (optional)
            covariance_matrix: Asset covariance matrix (optional)
            
        Returns:
            Portfolio risk (annualized volatility)
        """
        pass
    
    @abstractmethod
    def get_risk_decomposition(
        self, 
        weights: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get detailed risk attribution for the portfolio.
        
        Args:
            weights: Portfolio weights
            asset_names: Names of assets (optional)
            
        Returns:
            Dictionary containing:
            - individual_risk: Risk contribution of each asset
            - marginal_risk: Marginal risk contribution
            - component_risk: Component risk contributions
            - diversification_ratio: Portfolio diversification ratio
        """
        pass
    
    @abstractmethod
    def calculate_var(
        self, 
        weights: np.ndarray, 
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            weights: Portfolio weights
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            time_horizon: Time horizon in days
            
        Returns:
            Value at Risk
        """
        pass
    
    @abstractmethod
    def calculate_expected_shortfall(
        self, 
        weights: np.ndarray, 
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            weights: Portfolio weights
            confidence_level: Confidence level (e.g., 0.95)
            time_horizon: Time horizon in days
            
        Returns:
            Expected Shortfall
        """
        pass
    
    @abstractmethod
    def calculate_beta(
        self, 
        weights: np.ndarray, 
        market_returns: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate portfolio beta relative to market.
        
        Args:
            weights: Portfolio weights
            market_returns: Market returns (optional)
            
        Returns:
            Portfolio beta
        """
        pass
    
    @abstractmethod
    def calculate_tracking_error(
        self, 
        weights: np.ndarray, 
        benchmark_weights: np.ndarray
    ) -> float:
        """
        Calculate tracking error relative to benchmark.
        
        Args:
            weights: Portfolio weights
            benchmark_weights: Benchmark weights
            
        Returns:
            Tracking error (annualized)
        """
        pass
    
    @abstractmethod
    def validate_risk_constraints(
        self, 
        weights: np.ndarray, 
        constraints: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate portfolio against risk constraints.
        
        Args:
            weights: Portfolio weights
            constraints: Dictionary of risk constraints
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        pass
    
    @abstractmethod
    def estimate_covariance_matrix(
        self, 
        returns: np.ndarray, 
        method: str = 'sample'
    ) -> np.ndarray:
        """
        Estimate covariance matrix from returns.
        
        Args:
            returns: Historical returns matrix
            method: Estimation method ('sample', 'shrinkage', 'factor')
            
        Returns:
            Estimated covariance matrix
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the risk model"""
        pass
    
    @property
    @abstractmethod
    def supported_metrics(self) -> List[str]:
        """Get list of supported risk metrics"""
        pass
    
    @property
    @abstractmethod
    def requires_historical_data(self) -> bool:
        """Whether this model requires historical return data"""
        pass
    
    def calculate_sharpe_ratio(
        self, 
        weights: np.ndarray, 
        expected_returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio for the portfolio.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns for each asset
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = self.calculate_risk(weights)
        
        if portfolio_risk == 0:
            return 0.0
        
        return (portfolio_return - risk_free_rate) / portfolio_risk
    
    def calculate_sortino_ratio(
        self, 
        weights: np.ndarray, 
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sortino ratio for the portfolio.
        
        Args:
            weights: Portfolio weights
            returns: Historical returns matrix
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sortino ratio
        """
        portfolio_returns = np.dot(returns, weights)
        excess_returns = portfolio_returns - risk_free_rate / 252  # Daily risk-free rate
        
        # Calculate downside deviation
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) == 0:
            downside_deviation = 0.0
        else:
            downside_deviation = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252)
        
        if downside_deviation == 0:
            return 0.0
        
        portfolio_return = np.mean(portfolio_returns) * 252  # Annualized
        return (portfolio_return - risk_free_rate) / downside_deviation
    
    def calculate_maximum_drawdown(
        self, 
        weights: np.ndarray, 
        returns: np.ndarray
    ) -> float:
        """
        Calculate maximum drawdown for the portfolio.
        
        Args:
            weights: Portfolio weights
            returns: Historical returns matrix
            
        Returns:
            Maximum drawdown
        """
        portfolio_returns = np.dot(returns, weights)
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)