"""
Mathematical utilities for portfolio optimization.

Provides common mathematical functions used across optimization algorithms
including matrix operations, statistical calculations, and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from scipy import linalg
from scipy.stats import norm

from .logger import get_logger


logger = get_logger('math_utils')


def ensure_positive_definite(
    matrix: np.ndarray, 
    min_eigenvalue: float = 1e-8
) -> np.ndarray:
    """
    Ensure a matrix is positive definite by adjusting eigenvalues.
    
    Args:
        matrix: Input matrix (should be symmetric)
        min_eigenvalue: Minimum eigenvalue to ensure
        
    Returns:
        Positive definite matrix
    """
    try:
        # Check if matrix is already positive definite
        eigenvalues, eigenvectors = linalg.eigh(matrix)
        
        if np.all(eigenvalues > min_eigenvalue):
            return matrix
        
        # Adjust eigenvalues to ensure positive definiteness
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        # Reconstruct matrix
        adjusted_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Ensure symmetry
        adjusted_matrix = (adjusted_matrix + adjusted_matrix.T) / 2
        
        logger.debug(f"Adjusted matrix to ensure positive definiteness")
        return adjusted_matrix
        
    except Exception as e:
        logger.error(f"Error ensuring positive definite matrix: {e}")
        # Fallback: add small value to diagonal
        return matrix + np.eye(matrix.shape[0]) * min_eigenvalue


def validate_correlation_matrix(correlation_matrix: np.ndarray) -> Tuple[bool, str]:
    """
    Validate a correlation matrix.
    
    Args:
        correlation_matrix: Matrix to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if matrix is square
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            return False, "Matrix is not square"
        
        # Check if diagonal elements are 1
        diagonal = np.diag(correlation_matrix)
        if not np.allclose(diagonal, 1.0, atol=1e-6):
            return False, "Diagonal elements are not 1"
        
        # Check if matrix is symmetric
        if not np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-6):
            return False, "Matrix is not symmetric"
        
        # Check if all elements are in [-1, 1]
        if np.any(correlation_matrix < -1.0) or np.any(correlation_matrix > 1.0):
            return False, "Elements outside [-1, 1] range"
        
        # Check if matrix is positive semi-definite
        eigenvalues = linalg.eigvals(correlation_matrix)
        if np.any(eigenvalues < -1e-8):
            return False, "Matrix is not positive semi-definite"
        
        return True, "Valid correlation matrix"
        
    except Exception as e:
        return False, f"Error validating matrix: {e}"


def normalize_weights(weights: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    """
    Normalize portfolio weights to sum to target value.
    
    Args:
        weights: Portfolio weights
        target_sum: Target sum for weights
        
    Returns:
        Normalized weights
    """
    try:
        # Ensure non-negative weights
        weights = np.maximum(weights, 0)
        
        # Check if all weights are zero
        if np.sum(weights) == 0:
            # Equal weights if all are zero
            return np.ones(len(weights)) * (target_sum / len(weights))
        
        # Normalize to target sum
        return weights * (target_sum / np.sum(weights))
        
    except Exception as e:
        logger.error(f"Error normalizing weights: {e}")
        # Fallback: equal weights
        return np.ones(len(weights)) * (target_sum / len(weights))


def calculate_portfolio_metrics(
    weights: np.ndarray,
    returns: np.ndarray,
    covariance_matrix: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics.
    
    Args:
        weights: Portfolio weights
        returns: Expected returns for each asset
        covariance_matrix: Asset covariance matrix (optional)
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Dictionary with portfolio metrics
    """
    try:
        metrics = {}
        
        # Portfolio return
        portfolio_return = np.dot(weights, returns)
        metrics['expected_return'] = portfolio_return
        
        # Portfolio risk (if covariance matrix provided)
        if covariance_matrix is not None:
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            metrics['volatility'] = portfolio_volatility
            
            # Sharpe ratio
            if portfolio_volatility > 0:
                metrics['sharpe_ratio'] = (portfolio_return - risk_free_rate) / portfolio_volatility
            else:
                metrics['sharpe_ratio'] = 0.0
        
        # Concentration metrics
        metrics['max_weight'] = np.max(weights)
        metrics['min_weight'] = np.min(weights)
        metrics['weight_concentration'] = np.sum(weights ** 2)  # Herfindahl index
        
        # Number of effective positions
        metrics['effective_positions'] = 1 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0
        
        # Diversification ratio (if covariance matrix provided)
        if covariance_matrix is not None:
            individual_volatilities = np.sqrt(np.diag(covariance_matrix))
            weighted_avg_volatility = np.dot(weights, individual_volatilities)
            if portfolio_volatility > 0:
                metrics['diversification_ratio'] = weighted_avg_volatility / portfolio_volatility
            else:
                metrics['diversification_ratio'] = 1.0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {e}")
        return {'error': str(e)}


def calculate_risk_contribution(
    weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate risk contribution of each asset to portfolio risk.
    
    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix
        
    Returns:
        Risk contribution for each asset
    """
    try:
        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        
        if portfolio_variance == 0:
            return np.zeros(len(weights))
        
        # Marginal risk contribution
        marginal_contrib = np.dot(covariance_matrix, weights)
        
        # Risk contribution
        risk_contrib = weights * marginal_contrib / portfolio_variance
        
        return risk_contrib
        
    except Exception as e:
        logger.error(f"Error calculating risk contribution: {e}")
        return np.zeros(len(weights))


def calculate_var(
    portfolio_returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        portfolio_returns: Historical portfolio returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        method: Calculation method ('historical' or 'parametric')
        
    Returns:
        Value at Risk
    """
    try:
        if method == 'historical':
            # Historical VaR
            return np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            z_score = norm.ppf(1 - confidence_level)
            return mean_return + z_score * std_return
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
            
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        return 0.0


def calculate_expected_shortfall(
    portfolio_returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    Args:
        portfolio_returns: Historical portfolio returns
        confidence_level: Confidence level (e.g., 0.95)
        
    Returns:
        Expected Shortfall
    """
    try:
        var = calculate_var(portfolio_returns, confidence_level, 'historical')
        tail_returns = portfolio_returns[portfolio_returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return np.mean(tail_returns)
        
    except Exception as e:
        logger.error(f"Error calculating Expected Shortfall: {e}")
        return 0.0


def calculate_maximum_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from return series.
    
    Args:
        returns: Return series
        
    Returns:
        Maximum drawdown
    """
    try:
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)
        
    except Exception as e:
        logger.error(f"Error calculating maximum drawdown: {e}")
        return 0.0


def shrink_covariance_matrix(
    sample_cov: np.ndarray,
    shrinkage_target: Optional[np.ndarray] = None,
    shrinkage_intensity: Optional[float] = None
) -> np.ndarray:
    """
    Apply shrinkage to covariance matrix estimation.
    
    Args:
        sample_cov: Sample covariance matrix
        shrinkage_target: Target matrix for shrinkage (default: identity)
        shrinkage_intensity: Shrinkage intensity (default: auto-calculated)
        
    Returns:
        Shrunk covariance matrix
    """
    try:
        n = sample_cov.shape[0]
        
        # Default shrinkage target (identity matrix scaled by average variance)
        if shrinkage_target is None:
            avg_variance = np.mean(np.diag(sample_cov))
            shrinkage_target = np.eye(n) * avg_variance
        
        # Auto-calculate shrinkage intensity if not provided
        if shrinkage_intensity is None:
            # Ledoit-Wolf shrinkage intensity estimation
            shrinkage_intensity = 0.1  # Simple default
        
        # Apply shrinkage
        shrunk_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * shrinkage_target
        
        return shrunk_cov
        
    except Exception as e:
        logger.error(f"Error applying covariance shrinkage: {e}")
        return sample_cov


def calculate_factor_loadings(
    returns: np.ndarray,
    factor_returns: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate factor loadings using linear regression.
    
    Args:
        returns: Asset returns matrix (assets x time)
        factor_returns: Factor returns matrix (factors x time)
        
    Returns:
        Tuple of (factor_loadings, residual_variance)
    """
    try:
        n_assets, n_periods = returns.shape
        n_factors = factor_returns.shape[0]
        
        factor_loadings = np.zeros((n_assets, n_factors))
        residual_variance = np.zeros(n_assets)
        
        for i in range(n_assets):
            # Linear regression: asset_return = alpha + beta * factor_returns + error
            X = factor_returns.T  # factors x time -> time x factors
            y = returns[i, :]     # time series for asset i
            
            # Add intercept term
            X_with_intercept = np.column_stack([np.ones(n_periods), X])
            
            # Solve using least squares
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            
            # Extract factor loadings (exclude intercept)
            factor_loadings[i, :] = coefficients[1:]
            
            # Calculate residual variance
            predicted = X_with_intercept @ coefficients
            residuals = y - predicted
            residual_variance[i] = np.var(residuals)
        
        return factor_loadings, residual_variance
        
    except Exception as e:
        logger.error(f"Error calculating factor loadings: {e}")
        return np.zeros((returns.shape[0], factor_returns.shape[0])), np.zeros(returns.shape[0])