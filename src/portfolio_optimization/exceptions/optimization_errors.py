"""
Custom exception classes for portfolio optimization errors.

These exceptions provide specific error types for different failure modes
in the optimization process, enabling better error handling and debugging.
"""

from typing import Optional, Dict, Any


class OptimizationError(Exception):
    """
    Base class for all portfolio optimization errors.
    
    This is the parent class for all optimization-related exceptions,
    providing common functionality for error reporting and debugging.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        error_str = self.message
        if self.error_code:
            error_str = f"[{self.error_code}] {error_str}"
        if self.context:
            error_str += f" Context: {self.context}"
        return error_str


class DataError(OptimizationError):
    """
    Exception raised for data-related errors.
    
    This includes missing data, stale data, data quality issues,
    and API connectivity problems.
    """
    
    def __init__(
        self, 
        message: str, 
        data_source: Optional[str] = None,
        symbols: Optional[list] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if data_source:
            context['data_source'] = data_source
        if symbols:
            context['symbols'] = symbols
        
        super().__init__(message, kwargs.get('error_code', 'DATA_ERROR'), context)
        self.data_source = data_source
        self.symbols = symbols


class ConstraintViolationError(OptimizationError):
    """
    Exception raised when portfolio constraints are violated.
    
    This includes position size limits, sector concentration limits,
    leverage constraints, and other risk management violations.
    """
    
    def __init__(
        self, 
        message: str, 
        constraint_type: Optional[str] = None,
        violated_constraints: Optional[list] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if constraint_type:
            context['constraint_type'] = constraint_type
        if violated_constraints:
            context['violated_constraints'] = violated_constraints
        
        super().__init__(message, kwargs.get('error_code', 'CONSTRAINT_VIOLATION'), context)
        self.constraint_type = constraint_type
        self.violated_constraints = violated_constraints or []


class ConvergenceError(OptimizationError):
    """
    Exception raised when optimization algorithms fail to converge.
    
    This includes numerical instability, infeasible problems,
    and optimization timeout errors.
    """
    
    def __init__(
        self, 
        message: str, 
        algorithm_name: Optional[str] = None,
        iterations: Optional[int] = None,
        objective_value: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if algorithm_name:
            context['algorithm_name'] = algorithm_name
        if iterations is not None:
            context['iterations'] = iterations
        if objective_value is not None:
            context['objective_value'] = objective_value
        
        super().__init__(message, kwargs.get('error_code', 'CONVERGENCE_ERROR'), context)
        self.algorithm_name = algorithm_name
        self.iterations = iterations
        self.objective_value = objective_value


class ResourceError(OptimizationError):
    """
    Exception raised for resource-related errors.
    
    This includes memory allocation failures, API rate limit exceeded,
    database connectivity issues, and system resource constraints.
    """
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if resource_type:
            context['resource_type'] = resource_type
        if current_usage is not None:
            context['current_usage'] = current_usage
        if limit is not None:
            context['limit'] = limit
        
        super().__init__(message, kwargs.get('error_code', 'RESOURCE_ERROR'), context)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class ValidationError(OptimizationError):
    """
    Exception raised for input validation errors.
    
    This includes invalid parameters, malformed data,
    and configuration validation failures.
    """
    
    def __init__(
        self, 
        message: str, 
        parameter_name: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if parameter_name:
            context['parameter_name'] = parameter_name
        if expected_type:
            context['expected_type'] = expected_type
        if actual_value is not None:
            context['actual_value'] = str(actual_value)
        
        super().__init__(message, kwargs.get('error_code', 'VALIDATION_ERROR'), context)
        self.parameter_name = parameter_name
        self.expected_type = expected_type
        self.actual_value = actual_value