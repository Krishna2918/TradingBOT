"""
Interface for portfolio optimization algorithms.

This interface ensures consistent behavior across all optimization methods
and enables easy swapping of algorithms based on market conditions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd


class IOptimizationAlgorithm(ABC):
    """
    Interface for portfolio optimization algorithms.
    
    All optimization algorithms must implement this interface to ensure
    consistent behavior and enable automatic algorithm selection.
    """
    
    @abstractmethod
    def optimize(
        self, 
        returns: np.ndarray, 
        constraints: Dict[str, Any],
        covariance_matrix: Optional[np.ndarray] = None,
        factor_loadings: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Optimize portfolio weights based on returns and constraints.
        
        Args:
            returns: Expected returns for each asset
            constraints: Dictionary of optimization constraints
            covariance_matrix: Asset covariance matrix (optional)
            factor_loadings: Factor loadings matrix (optional)
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Optimal portfolio weights as numpy array
            
        Raises:
            ConvergenceError: If optimization fails to converge
            ConstraintViolationError: If constraints are infeasible
        """
        pass
    
    @abstractmethod
    def get_optimization_info(self) -> Dict[str, Any]:
        """
        Get information about the last optimization run.
        
        Returns:
            Dictionary containing optimization metadata including:
            - algorithm_name: Name of the optimization algorithm
            - convergence_status: Whether optimization converged
            - iterations: Number of iterations required
            - objective_value: Final objective function value
            - computation_time: Time taken for optimization
            - constraints_satisfied: Whether all constraints were satisfied
        """
        pass
    
    @abstractmethod
    def validate_inputs(
        self, 
        returns: np.ndarray, 
        constraints: Dict[str, Any],
        covariance_matrix: Optional[np.ndarray] = None
    ) -> bool:
        """
        Validate input parameters before optimization.
        
        Args:
            returns: Expected returns for each asset
            constraints: Dictionary of optimization constraints
            covariance_matrix: Asset covariance matrix (optional)
            
        Returns:
            True if inputs are valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_constraints(self) -> List[str]:
        """
        Get list of constraints supported by this algorithm.
        
        Returns:
            List of supported constraint types
        """
        pass
    
    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Get the name of the optimization algorithm"""
        pass
    
    @property
    @abstractmethod
    def requires_covariance_matrix(self) -> bool:
        """Whether this algorithm requires a covariance matrix"""
        pass
    
    @property
    @abstractmethod
    def supports_factor_model(self) -> bool:
        """Whether this algorithm supports factor models"""
        pass
    
    def preprocess_data(
        self, 
        returns: np.ndarray, 
        covariance_matrix: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Preprocess data before optimization (optional override).
        
        Args:
            returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix (optional)
            
        Returns:
            Tuple of (processed_returns, processed_covariance_matrix)
        """
        return returns, covariance_matrix
    
    def postprocess_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Postprocess weights after optimization (optional override).
        
        Args:
            weights: Raw optimization weights
            
        Returns:
            Processed weights
        """
        # Ensure weights sum to 1 and are non-negative
        weights = np.maximum(weights, 0)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        return weights