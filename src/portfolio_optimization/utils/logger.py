"""
Logging utilities for the Portfolio Optimization Engine.

Provides centralized logging configuration to ensure consistent
logging across all components and prevent log conflicts.
"""

import logging
import os
import sys
from typing import Optional
from datetime import datetime
from pathlib import Path

from ..config.settings import get_config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Set up logging configuration for the optimization engine.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output logs to console
    """
    config = get_config()
    
    # Use config log level if not specified
    if log_level is None:
        log_level = config.performance.log_level
    
    # Create logs directory if it doesn't exist
    if log_file is None:
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / f'portfolio_optimization_{datetime.now().strftime("%Y%m%d")}.log'
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger('portfolio_optimization')
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(file_handler)
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component.
    
    Args:
        name: Name of the component/module
        
    Returns:
        Configured logger instance
    """
    # Ensure logging is set up
    if not logging.getLogger('portfolio_optimization').handlers:
        setup_logging()
    
    return logging.getLogger(f'portfolio_optimization.{name}')


class OptimizationLogger:
    """
    Specialized logger for optimization operations with performance tracking.
    """
    
    def __init__(self, component_name: str):
        self.logger = get_logger(component_name)
        self.component_name = component_name
        self._operation_start_times = {}
    
    def start_operation(self, operation_name: str, **context) -> None:
        """Start timing an operation"""
        self._operation_start_times[operation_name] = datetime.now()
        context_str = ', '.join(f'{k}={v}' for k, v in context.items())
        self.logger.info(f"Starting {operation_name} - {context_str}")
    
    def end_operation(self, operation_name: str, success: bool = True, **context) -> None:
        """End timing an operation"""
        if operation_name in self._operation_start_times:
            duration = datetime.now() - self._operation_start_times[operation_name]
            duration_ms = duration.total_seconds() * 1000
            
            status = "SUCCESS" if success else "FAILED"
            context_str = ', '.join(f'{k}={v}' for k, v in context.items())
            
            self.logger.info(
                f"Completed {operation_name} - {status} - "
                f"Duration: {duration_ms:.2f}ms - {context_str}"
            )
            
            del self._operation_start_times[operation_name]
        else:
            self.logger.warning(f"End operation called for {operation_name} without start")
    
    def log_performance_metrics(self, metrics: dict) -> None:
        """Log performance metrics"""
        metrics_str = ', '.join(f'{k}={v}' for k, v in metrics.items())
        self.logger.info(f"Performance metrics - {metrics_str}")
    
    def log_resource_usage(self, memory_mb: float, cpu_percent: float) -> None:
        """Log resource usage"""
        self.logger.debug(f"Resource usage - Memory: {memory_mb:.2f}MB, CPU: {cpu_percent:.1f}%")
    
    def log_api_call(self, provider: str, endpoint: str, symbols: list, success: bool) -> None:
        """Log API calls for rate limiting tracking"""
        status = "SUCCESS" if success else "FAILED"
        symbol_count = len(symbols) if symbols else 0
        self.logger.debug(
            f"API call - Provider: {provider}, Endpoint: {endpoint}, "
            f"Symbols: {symbol_count}, Status: {status}"
        )
    
    def log_optimization_result(
        self, 
        algorithm: str, 
        objective_value: float,
        iterations: int,
        convergence_status: str
    ) -> None:
        """Log optimization results"""
        self.logger.info(
            f"Optimization result - Algorithm: {algorithm}, "
            f"Objective: {objective_value:.6f}, Iterations: {iterations}, "
            f"Status: {convergence_status}"
        )
    
    def log_constraint_violation(self, constraint_type: str, details: str) -> None:
        """Log constraint violations"""
        self.logger.warning(f"Constraint violation - Type: {constraint_type}, Details: {details}")
    
    def log_fallback_activation(self, primary_method: str, fallback_method: str, reason: str) -> None:
        """Log fallback mechanism activation"""
        self.logger.warning(
            f"Fallback activated - Primary: {primary_method}, "
            f"Fallback: {fallback_method}, Reason: {reason}"
        )