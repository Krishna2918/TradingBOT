"""
Self-Learning Engine with Meta-Parameter Optimization

This module implements advanced self-learning capabilities including meta-parameter
optimization, confidence threshold tuning, and continuous improvement algorithms.
"""

import logging
import json
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import joblib

from src.config.mode_manager import get_current_mode, get_mode_config
from src.adaptive.configuration_manager import (
    get_adaptive_config_manager, PerformanceMetrics, ParameterConfig
)
from src.adaptive.performance_learning import (
    get_performance_learning_engine, TradeOutcome, IdentifiedPattern
)

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Optimization objectives."""
    MAXIMIZE_SHARPE_RATIO = "MAXIMIZE_SHARPE_RATIO"
    MAXIMIZE_PROFIT_FACTOR = "MAXIMIZE_PROFIT_FACTOR"
    MINIMIZE_DRAWDOWN = "MINIMIZE_DRAWDOWN"
    MAXIMIZE_WIN_RATE = "MAXIMIZE_WIN_RATE"
    MAXIMIZE_TOTAL_RETURN = "MAXIMIZE_TOTAL_RETURN"
    BALANCED_PERFORMANCE = "BALANCED_PERFORMANCE"

class MetaParameterType(Enum):
    """Types of meta-parameters."""
    LEARNING_RATE = "LEARNING_RATE"
    CONFIDENCE_THRESHOLD = "CONFIDENCE_THRESHOLD"
    RISK_TOLERANCE = "RISK_TOLERANCE"
    POSITION_SIZE_MULTIPLIER = "POSITION_SIZE_MULTIPLIER"
    STOP_LOSS_MULTIPLIER = "STOP_LOSS_MULTIPLIER"
    TAKE_PROFIT_MULTIPLIER = "TAKE_PROFIT_MULTIPLIER"
    ADAPTATION_SPEED = "ADAPTATION_SPEED"
    EXPLORATION_RATE = "EXPLORATION_RATE"

@dataclass
class MetaParameter:
    """A meta-parameter for optimization."""
    name: str
    parameter_type: MetaParameterType
    current_value: float
    min_value: float
    max_value: float
    default_value: float
    optimization_weight: float
    last_optimized: datetime
    optimization_history: List[Tuple[float, float]]  # (value, performance)
    is_active: bool

@dataclass
class OptimizationResult:
    """Result of a parameter optimization."""
    parameter_name: str
    old_value: float
    new_value: float
    performance_improvement: float
    optimization_method: str
    objective_function: OptimizationObjective
    iterations: int
    convergence: bool
    confidence: float
    timestamp: datetime
    validation_period: int  # days

@dataclass
class SelfLearningState:
    """State of the self-learning system."""
    total_optimizations: int
    successful_optimizations: int
    current_objective: OptimizationObjective
    learning_phase: str
    adaptation_speed: float
    exploration_rate: float
    last_optimization: datetime
    performance_trend: str
    confidence_level: float

class SelfLearningEngine:
    """Advanced self-learning engine with meta-parameter optimization."""
    
    def __init__(self, mode: str = None):
        self.mode = mode or get_current_mode()
        self.mode_config = get_mode_config()
        self.db_path = self._get_database_path()
        
        # Core components
        self.config_manager = get_adaptive_config_manager(self.mode)
        self.learning_engine = get_performance_learning_engine(self.mode)
        
        # Meta-parameters
        self.meta_parameters: Dict[str, MetaParameter] = {}
        self.optimization_results: List[OptimizationResult] = []
        self.learning_state = SelfLearningState(
            total_optimizations=0,
            successful_optimizations=0,
            current_objective=OptimizationObjective.BALANCED_PERFORMANCE,
            learning_phase="initialization",
            adaptation_speed=0.1,
            exploration_rate=0.2,
            last_optimization=None,
            performance_trend="unknown",
            confidence_level=0.5
        )
        
        # Optimization models
        self.gp_model = None
        self.performance_surrogate = None
        
        # Learning history
        self.parameter_performance_history: Dict[str, List[Tuple[float, float]]] = {}
        
        self._initialize_meta_parameters()
        self._initialize_database()
        self._load_learning_state()
        
        logger.info(f"Self-Learning Engine initialized for {self.mode} mode")
    
    def _get_database_path(self) -> str:
        """Get database path based on mode."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        return str(data_dir / f"self_learning_{self.mode.lower()}.db")
    
    def _initialize_meta_parameters(self):
        """Initialize meta-parameters for optimization."""
        base_config = getattr(self.mode_config, "meta_parameters", {})
        
        self.meta_parameters = {
            "learning_rate": MetaParameter(
                name="learning_rate",
                parameter_type=MetaParameterType.LEARNING_RATE,
                current_value=base_config.get("learning_rate", 0.1) if isinstance(base_config, dict) else 0.1,
                min_value=0.01,
                max_value=0.5,
                default_value=0.1,
                optimization_weight=0.3,
                last_optimized=datetime.now(),
                optimization_history=[],
                is_active=True
            ),
            "confidence_threshold": MetaParameter(
                name="confidence_threshold",
                parameter_type=MetaParameterType.CONFIDENCE_THRESHOLD,
                current_value=base_config.get("confidence_threshold", 0.7) if isinstance(base_config, dict) else 0.7,
                min_value=0.5,
                max_value=0.95,
                default_value=0.7,
                optimization_weight=0.4,
                last_optimized=datetime.now(),
                optimization_history=[],
                is_active=True
            ),
            "risk_tolerance": MetaParameter(
                name="risk_tolerance",
                parameter_type=MetaParameterType.RISK_TOLERANCE,
                current_value=base_config.get("risk_tolerance", 0.02) if isinstance(base_config, dict) else 0.02,
                min_value=0.005,
                max_value=0.05,
                default_value=0.02,
                optimization_weight=0.3,
                last_optimized=datetime.now(),
                optimization_history=[],
                is_active=True
            ),
            "position_size_multiplier": MetaParameter(
                name="position_size_multiplier",
                parameter_type=MetaParameterType.POSITION_SIZE_MULTIPLIER,
                current_value=base_config.get("position_size_multiplier", 1.0) if isinstance(base_config, dict) else 1.0,
                min_value=0.5,
                max_value=2.0,
                default_value=1.0,
                optimization_weight=0.2,
                last_optimized=datetime.now(),
                optimization_history=[],
                is_active=True
            ),
            "adaptation_speed": MetaParameter(
                name="adaptation_speed",
                parameter_type=MetaParameterType.ADAPTATION_SPEED,
                current_value=base_config.get("adaptation_speed", 0.1) if isinstance(base_config, dict) else 0.1,
                min_value=0.01,
                max_value=0.3,
                default_value=0.1,
                optimization_weight=0.1,
                last_optimized=datetime.now(),
                optimization_history=[],
                is_active=True
            ),
            "exploration_rate": MetaParameter(
                name="exploration_rate",
                parameter_type=MetaParameterType.EXPLORATION_RATE,
                current_value=base_config.get("exploration_rate", 0.2) if isinstance(base_config, dict) else 0.2,
                min_value=0.05,
                max_value=0.5,
                default_value=0.2,
                optimization_weight=0.1,
                last_optimized=datetime.now(),
                optimization_history=[],
                is_active=True
            )
        }
    
    def _initialize_database(self):
        """Initialize self-learning database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Meta-parameters table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS meta_parameters (
                        name TEXT PRIMARY KEY,
                        current_value REAL,
                        min_value REAL,
                        max_value REAL,
                        default_value REAL,
                        optimization_weight REAL,
                        last_optimized TEXT,
                        optimization_history TEXT,
                        is_active INTEGER
                    )
                """)
                
                # Optimization results table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        parameter_name TEXT,
                        old_value REAL,
                        new_value REAL,
                        performance_improvement REAL,
                        optimization_method TEXT,
                        objective_function TEXT,
                        iterations INTEGER,
                        convergence INTEGER,
                        confidence REAL,
                        timestamp TEXT,
                        validation_period INTEGER
                    )
                """)
                
                # Learning state table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_state (
                        id INTEGER PRIMARY KEY,
                        total_optimizations INTEGER,
                        successful_optimizations INTEGER,
                        current_objective TEXT,
                        learning_phase TEXT,
                        adaptation_speed REAL,
                        exploration_rate REAL,
                        last_optimization TEXT,
                        performance_trend TEXT,
                        confidence_level REAL,
                        updated_at TEXT
                    )
                """)
                
                conn.commit()
                logger.info("Self-learning database initialized")
                
        except Exception as e:
            logger.error(f"Error initializing self-learning database: {e}")
    
    def _load_learning_state(self):
        """Load learning state from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load meta-parameters
                cursor = conn.execute("SELECT * FROM meta_parameters")
                for row in cursor.fetchall():
                    name, current_value, min_value, max_value, default_value, weight, last_opt, history, is_active = row
                    if name in self.meta_parameters:
                        self.meta_parameters[name].current_value = current_value
                        self.meta_parameters[name].last_optimized = datetime.fromisoformat(last_opt) if last_opt else datetime.now()
                        self.meta_parameters[name].optimization_history = json.loads(history) if history else []
                        self.meta_parameters[name].is_active = bool(is_active)
                
                # Load optimization results
                cursor = conn.execute("SELECT * FROM optimization_results ORDER BY timestamp DESC LIMIT 100")
                for row in cursor.fetchall():
                    _, param_name, old_val, new_val, improvement, method, objective, iterations, convergence, confidence, timestamp, validation = row
                    result = OptimizationResult(
                        parameter_name=param_name,
                        old_value=old_val,
                        new_value=new_val,
                        performance_improvement=improvement,
                        optimization_method=method,
                        objective_function=OptimizationObjective(objective),
                        iterations=iterations,
                        convergence=bool(convergence),
                        confidence=confidence,
                        timestamp=datetime.fromisoformat(timestamp),
                        validation_period=validation
                    )
                    self.optimization_results.append(result)
                
                # Load learning state
                cursor = conn.execute("SELECT * FROM learning_state ORDER BY updated_at DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    _, total_opt, successful_opt, current_obj, phase, adapt_speed, explore_rate, last_opt, trend, confidence, updated_at = row
                    self.learning_state = SelfLearningState(
                        total_optimizations=total_opt,
                        successful_optimizations=successful_opt,
                        current_objective=OptimizationObjective(current_obj),
                        learning_phase=phase,
                        adaptation_speed=adapt_speed,
                        exploration_rate=explore_rate,
                        last_optimization=datetime.fromisoformat(last_opt) if last_opt else None,
                        performance_trend=trend,
                        confidence_level=confidence
                    )
                
                logger.info(f"Loaded learning state: {len(self.meta_parameters)} meta-parameters, "
                           f"{len(self.optimization_results)} optimization results")
                
        except Exception as e:
            logger.error(f"Error loading learning state: {e}")
    
    def _save_learning_state(self):
        """Save learning state to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Save meta-parameters
                for name, param in self.meta_parameters.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO meta_parameters 
                        (name, current_value, min_value, max_value, default_value, 
                         optimization_weight, last_optimized, optimization_history, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        name,
                        param.current_value,
                        param.min_value,
                        param.max_value,
                        param.default_value,
                        param.optimization_weight,
                        param.last_optimized.isoformat(),
                        json.dumps(param.optimization_history),
                        int(param.is_active)
                    ))
                
                # Save latest optimization result if any
                if self.optimization_results:
                    latest = self.optimization_results[-1]
                    conn.execute("""
                        INSERT INTO optimization_results 
                        (parameter_name, old_value, new_value, performance_improvement, 
                         optimization_method, objective_function, iterations, convergence, 
                         confidence, timestamp, validation_period)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        latest.parameter_name,
                        latest.old_value,
                        latest.new_value,
                        latest.performance_improvement,
                        latest.optimization_method,
                        latest.objective_function.value,
                        latest.iterations,
                        int(latest.convergence),
                        latest.confidence,
                        latest.timestamp.isoformat(),
                        latest.validation_period
                    ))
                
                # Save learning state
                conn.execute("""
                    INSERT OR REPLACE INTO learning_state 
                    (id, total_optimizations, successful_optimizations, current_objective, 
                     learning_phase, adaptation_speed, exploration_rate, last_optimization, 
                     performance_trend, confidence_level, updated_at)
                    VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.learning_state.total_optimizations,
                    self.learning_state.successful_optimizations,
                    self.learning_state.current_objective.value,
                    self.learning_state.learning_phase,
                    self.learning_state.adaptation_speed,
                    self.learning_state.exploration_rate,
                    self.learning_state.last_optimization.isoformat() if self.learning_state.last_optimization else None,
                    self.learning_state.performance_trend,
                    self.learning_state.confidence_level,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")
    
    async def optimize_parameters(self, objective: OptimizationObjective = None) -> Dict[str, Any]:
        """Optimize meta-parameters using advanced optimization techniques."""
        if objective:
            self.learning_state.current_objective = objective
        
        logger.info(f"Starting parameter optimization with objective: {self.learning_state.current_objective.value}")
        
        # Get current performance baseline
        current_performance = await self._get_current_performance()
        
        # Select parameters to optimize
        active_parameters = [name for name, param in self.meta_parameters.items() if param.is_active]
        
        if not active_parameters:
            logger.warning("No active parameters to optimize")
            return {"status": "no_active_parameters"}
        
        # Prepare optimization
        optimization_results = {}
        
        for param_name in active_parameters:
            try:
                result = await self._optimize_single_parameter(param_name, current_performance)
                optimization_results[param_name] = result
                
                if result["success"]:
                    self.learning_state.successful_optimizations += 1
                self.learning_state.total_optimizations += 1
                
            except Exception as e:
                logger.error(f"Error optimizing parameter {param_name}: {e}")
                optimization_results[param_name] = {"success": False, "error": str(e)}
        
        # Update learning state
        self.learning_state.last_optimization = datetime.now()
        self._update_learning_phase()
        self._save_learning_state()
        
        logger.info(f"Parameter optimization completed: {len([r for r in optimization_results.values() if r.get('success')])} successful")
        
        return {
            "status": "completed",
            "objective": self.learning_state.current_objective.value,
            "results": optimization_results,
            "learning_state": asdict(self.learning_state)
        }
    
    async def _get_current_performance(self) -> Dict[str, float]:
        """Get current performance metrics."""
        # Get recent performance from configuration manager
        recent_metrics = self.config_manager.performance_history[-5:] if self.config_manager.performance_history else []
        
        if not recent_metrics:
            return {
                "win_rate": 0.5,
                "profit_factor": 1.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0
            }
        
        # Calculate average metrics
        return {
            "win_rate": np.mean([m.win_rate for m in recent_metrics]),
            "profit_factor": np.mean([m.profit_factor for m in recent_metrics]),
            "sharpe_ratio": np.mean([m.sharpe_ratio for m in recent_metrics]),
            "max_drawdown": np.mean([m.max_drawdown for m in recent_metrics]),
            "total_return": np.mean([m.total_return for m in recent_metrics])
        }
    
    async def _optimize_single_parameter(self, param_name: str, baseline_performance: Dict[str, float]) -> Dict[str, Any]:
        """Optimize a single parameter using multiple methods."""
        param = self.meta_parameters[param_name]
        
        # Define objective function
        def objective_function(x):
            return self._evaluate_parameter_value(param_name, x[0], baseline_performance)
        
        # Try different optimization methods
        methods = [
            ("differential_evolution", self._optimize_with_differential_evolution),
            ("bayesian_optimization", self._optimize_with_bayesian),
            ("gradient_descent", self._optimize_with_gradient_descent)
        ]
        
        best_result = None
        best_performance = float('-inf')
        
        for method_name, method_func in methods:
            try:
                result = await method_func(param_name, objective_function, param.min_value, param.max_value)
                
                if result["success"] and result["performance"] > best_performance:
                    best_result = result
                    best_performance = result["performance"]
                    
            except Exception as e:
                logger.warning(f"Optimization method {method_name} failed for {param_name}: {e}")
        
        if best_result and best_result["success"]:
            # Apply the optimization
            old_value = param.current_value
            new_value = best_result["optimal_value"]
            
            param.current_value = new_value
            param.last_optimized = datetime.now()
            param.optimization_history.append((new_value, best_result["performance"]))
            
            # Record optimization result
            optimization_result = OptimizationResult(
                parameter_name=param_name,
                old_value=old_value,
                new_value=new_value,
                performance_improvement=best_result["performance"] - baseline_performance.get("sharpe_ratio", 0),
                optimization_method=best_result["method"],
                objective_function=self.learning_state.current_objective,
                iterations=best_result.get("iterations", 0),
                convergence=best_result.get("convergence", False),
                confidence=best_result.get("confidence", 0.5),
                timestamp=datetime.now(),
                validation_period=7
            )
            
            self.optimization_results.append(optimization_result)
            
            # Keep only last 100 results
            if len(self.optimization_results) > 100:
                self.optimization_results = self.optimization_results[-100:]
            
            logger.info(f"Optimized {param_name}: {old_value:.4f} → {new_value:.4f} "
                       f"(Performance: {best_result['performance']:.4f})")
            
            return {
                "success": True,
                "old_value": old_value,
                "new_value": new_value,
                "performance_improvement": best_result["performance"] - baseline_performance.get("sharpe_ratio", 0),
                "method": best_result["method"],
                "confidence": best_result.get("confidence", 0.5)
            }
        
        return {"success": False, "error": "All optimization methods failed"}
    
    def _evaluate_parameter_value(self, param_name: str, value: float, baseline_performance: Dict[str, float]) -> float:
        """Evaluate performance of a parameter value (simplified surrogate model)."""
        # This is a simplified evaluation - in practice, this would use historical data
        # or a trained surrogate model to predict performance
        
        param = self.meta_parameters[param_name]
        
        # Use optimization history if available
        if param.optimization_history:
            # Find closest historical value
            historical_values = [h[0] for h in param.optimization_history]
            historical_performances = [h[1] for h in param.optimization_history]
            
            if historical_values:
                # Simple interpolation
                distances = [abs(v - value) for v in historical_values]
                closest_idx = distances.index(min(distances))
                return historical_performances[closest_idx]
        
        # Default evaluation based on parameter type
        if param_name == "confidence_threshold":
            # Higher confidence threshold should improve win rate but reduce trade frequency
            return value * 0.8 + (1 - value) * 0.2
        elif param_name == "risk_tolerance":
            # Lower risk tolerance should reduce drawdown
            return (1 - value) * 0.6 + value * 0.4
        elif param_name == "learning_rate":
            # Moderate learning rate is usually best
            optimal = 0.1
            return 1.0 - abs(value - optimal) * 2
        else:
            # Default: prefer values closer to default
            return 1.0 - abs(value - param.default_value) / (param.max_value - param.min_value)
    
    async def _optimize_with_differential_evolution(self, param_name: str, objective_func: Callable, 
                                                   min_val: float, max_val: float) -> Dict[str, Any]:
        """Optimize using differential evolution."""
        try:
            result = differential_evolution(
                objective_func,
                bounds=[(min_val, max_val)],
                maxiter=50,
                popsize=15,
                seed=42
            )
            
            return {
                "success": result.success,
                "optimal_value": result.x[0],
                "performance": -result.fun,  # Minimize negative performance
                "method": "differential_evolution",
                "iterations": result.nit,
                "convergence": result.success,
                "confidence": 0.8 if result.success else 0.3
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _optimize_with_bayesian(self, param_name: str, objective_func: Callable, 
                                     min_val: float, max_val: float) -> Dict[str, Any]:
        """Optimize using Bayesian optimization with Gaussian Process."""
        try:
            # This is a simplified implementation
            # In practice, you'd use a proper Bayesian optimization library
            
            # Sample points for GP
            n_samples = 20
            x_samples = np.linspace(min_val, max_val, n_samples).reshape(-1, 1)
            y_samples = np.array([objective_func([x]) for x in x_samples.flatten()])
            
            # Fit GP
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
            gp.fit(x_samples, y_samples)
            
            # Find maximum
            x_test = np.linspace(min_val, max_val, 100).reshape(-1, 1)
            y_pred, y_std = gp.predict(x_test, return_std=True)
            
            # Use acquisition function (expected improvement)
            best_y = np.max(y_samples)
            improvement = y_pred - best_y
            acquisition = improvement / (y_std + 1e-9)
            
            optimal_idx = np.argmax(acquisition)
            optimal_value = x_test[optimal_idx][0]
            optimal_performance = y_pred[optimal_idx]
            
            return {
                "success": True,
                "optimal_value": optimal_value,
                "performance": optimal_performance,
                "method": "bayesian_optimization",
                "iterations": n_samples,
                "convergence": True,
                "confidence": 0.9
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _optimize_with_gradient_descent(self, param_name: str, objective_func: Callable, 
                                             min_val: float, max_val: float) -> Dict[str, Any]:
        """Optimize using gradient descent."""
        try:
            # Simple gradient descent implementation
            current_value = (min_val + max_val) / 2
            learning_rate = 0.01
            max_iterations = 100
            tolerance = 1e-6
            
            for iteration in range(max_iterations):
                # Numerical gradient
                eps = 1e-6
                grad = (objective_func([current_value + eps]) - objective_func([current_value - eps])) / (2 * eps)
                
                # Update
                new_value = current_value + learning_rate * grad
                
                # Clamp to bounds
                new_value = max(min_val, min(max_val, new_value))
                
                # Check convergence
                if abs(new_value - current_value) < tolerance:
                    break
                
                current_value = new_value
            
            final_performance = objective_func([current_value])
            
            return {
                "success": True,
                "optimal_value": current_value,
                "performance": final_performance,
                "method": "gradient_descent",
                "iterations": iteration + 1,
                "convergence": iteration < max_iterations - 1,
                "confidence": 0.7
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _update_learning_phase(self):
        """Update the learning phase based on performance and experience."""
        total_optimizations = self.learning_state.total_optimizations
        success_rate = (self.learning_state.successful_optimizations / total_optimizations 
                       if total_optimizations > 0 else 0)
        
        if total_optimizations < 10:
            self.learning_state.learning_phase = "exploration"
        elif total_optimizations < 50:
            self.learning_state.learning_phase = "learning"
        elif success_rate > 0.7:
            self.learning_state.learning_phase = "optimization"
        else:
            self.learning_state.learning_phase = "adaptation"
        
        # Update confidence level
        self.learning_state.confidence_level = min(0.95, 0.5 + success_rate * 0.3)
        
        # Update adaptation speed based on learning phase
        if self.learning_state.learning_phase == "exploration":
            self.learning_state.adaptation_speed = 0.2
        elif self.learning_state.learning_phase == "learning":
            self.learning_state.adaptation_speed = 0.15
        elif self.learning_state.learning_phase == "optimization":
            self.learning_state.adaptation_speed = 0.1
        else:  # adaptation
            self.learning_state.adaptation_speed = 0.05
    
    def get_meta_parameter(self, name: str) -> Optional[float]:
        """Get current value of a meta-parameter."""
        if name in self.meta_parameters:
            return self.meta_parameters[name].current_value
        return None
    
    def get_all_meta_parameters(self) -> Dict[str, float]:
        """Get all current meta-parameter values."""
        return {name: param.current_value for name, param in self.meta_parameters.items()}
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary."""
        return {
            "learning_state": asdict(self.learning_state),
            "meta_parameters": {name: {
                "current_value": param.current_value,
                "optimization_count": len(param.optimization_history),
                "last_optimized": param.last_optimized.isoformat(),
                "is_active": param.is_active
            } for name, param in self.meta_parameters.items()},
            "optimization_history": {
                "total_optimizations": len(self.optimization_results),
                "recent_optimizations": len([r for r in self.optimization_results 
                                           if r.timestamp > datetime.now() - timedelta(days=7)]),
                "success_rate": (len([r for r in self.optimization_results if r.performance_improvement > 0]) 
                               / len(self.optimization_results) if self.optimization_results else 0)
            }
        }
    
    def set_learning_objective(self, objective: OptimizationObjective):
        """Set the learning objective."""
        self.learning_state.current_objective = objective
        self._save_learning_state()
        logger.info(f"Learning objective set to: {objective.value}")
    
    def enable_parameter(self, param_name: str):
        """Enable a meta-parameter for optimization."""
        if param_name in self.meta_parameters:
            self.meta_parameters[param_name].is_active = True
            self._save_learning_state()
            logger.info(f"Enabled meta-parameter: {param_name}")
    
    def disable_parameter(self, param_name: str):
        """Disable a meta-parameter from optimization."""
        if param_name in self.meta_parameters:
            self.meta_parameters[param_name].is_active = False
            self._save_learning_state()
            logger.info(f"Disabled meta-parameter: {param_name}")
    
    def reset_learning_state(self):
        """Reset the learning state to initial values."""
        for param in self.meta_parameters.values():
            param.current_value = param.default_value
            param.optimization_history = []
            param.last_optimized = datetime.now()
        
        self.optimization_results = []
        self.learning_state = SelfLearningState(
            total_optimizations=0,
            successful_optimizations=0,
            current_objective=OptimizationObjective.BALANCED_PERFORMANCE,
            learning_phase="initialization",
            adaptation_speed=0.1,
            exploration_rate=0.2,
            last_optimization=None,
            performance_trend="unknown",
            confidence_level=0.5
        )
        
        self._save_learning_state()
        logger.info("Learning state reset to initial values")

# Global instance
_self_learning_engine = None

def get_self_learning_engine(mode: str = None) -> SelfLearningEngine:
    """Get the global self-learning engine instance."""
    global _self_learning_engine
    if _self_learning_engine is None or (mode and _self_learning_engine.mode != mode):
        _self_learning_engine = SelfLearningEngine(mode)
    return _self_learning_engine

def get_meta_parameter(name: str, mode: str = None) -> Optional[float]:
    """Get a meta-parameter value."""
    return get_self_learning_engine(mode).get_meta_parameter(name)

def get_all_meta_parameters(mode: str = None) -> Dict[str, float]:
    """Get all meta-parameter values."""
    return get_self_learning_engine(mode).get_all_meta_parameters()

async def optimize_parameters(objective: OptimizationObjective = None, mode: str = None) -> Dict[str, Any]:
    """Optimize parameters."""
    return await get_self_learning_engine(mode).optimize_parameters(objective)

def get_learning_summary(mode: str = None) -> Dict[str, Any]:
    """Get learning summary."""
    return get_self_learning_engine(mode).get_learning_summary()

def set_learning_objective(objective: OptimizationObjective, mode: str = None):
    """Set learning objective."""
    get_self_learning_engine(mode).set_learning_objective(objective)
