"""
Parallel Backtesting Engine
Implements high-performance parallel backtesting for multiple strategies and time periods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import asyncio
import time
import uuid
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class BacktestStatus(Enum):
    """Backtest status types"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BacktestType(Enum):
    """Backtest types"""
    STRATEGY_BACKTEST = "strategy_backtest"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"
    CROSS_VALIDATION = "cross_validation"

@dataclass
class BacktestConfig:
    """Backtest configuration"""
    backtest_id: str
    backtest_type: BacktestType
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    symbols: List[str]
    parameters: Dict[str, Any]
    data_source: str
    execution_config: Dict[str, Any]
    risk_config: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class BacktestResult:
    """Backtest result"""
    backtest_id: str
    status: BacktestStatus
    start_time: datetime
    end_time: datetime
    execution_time: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float
    var_99: float
    beta: float
    alpha: float
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    metrics: Dict[str, Any]
    error: Optional[str] = None

class ParallelBacktestingEngine:
    """High-performance parallel backtesting engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_workers = config.get('max_workers', mp.cpu_count())
        self.max_concurrent_backtests = config.get('max_concurrent_backtests', 4)
        self.backtest_queue = []
        self.running_backtests = {}
        self.completed_backtests = []
        self.failed_backtests = []
        self.backtest_results = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_backtests': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        
        # Data cache for performance
        self.data_cache = {}
        self.cache_size_limit = config.get('cache_size_limit', 1000)
        
        logger.info(f"Parallel Backtesting Engine initialized with {self.max_workers} workers")
    
    def submit_backtest(self, backtest_config: BacktestConfig) -> str:
        """Submit a backtest for execution"""
        try:
            # Generate backtest ID if not provided
            if not backtest_config.backtest_id:
                backtest_config.backtest_id = f"backtest_{uuid.uuid4().hex[:8]}"
            
            # Add to queue
            self.backtest_queue.append(backtest_config)
            
            logger.info(f"Backtest submitted: {backtest_config.backtest_id} ({backtest_config.backtest_type.value})")
            return backtest_config.backtest_id
            
        except Exception as e:
            logger.error(f"Error submitting backtest: {e}")
            return None
    
    def run_parallel_backtests(self, backtest_configs: List[BacktestConfig], 
                              max_workers: int = None) -> List[BacktestResult]:
        """Run multiple backtests in parallel"""
        try:
            if max_workers is None:
                max_workers = min(self.max_workers, len(backtest_configs))
            
            logger.info(f"Starting parallel backtesting with {max_workers} workers for {len(backtest_configs)} backtests")
            
            results = []
            
            # Use ProcessPoolExecutor for CPU-intensive backtesting
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all backtests
                future_to_config = {
                    executor.submit(self._run_single_backtest, config): config
                    for config in backtest_configs
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.status == BacktestStatus.COMPLETED:
                            logger.info(f"Backtest completed: {result.backtest_id}")
                        else:
                            logger.error(f"Backtest failed: {result.backtest_id} - {result.error}")
                            
                    except Exception as e:
                        logger.error(f"Backtest {config.backtest_id} failed with exception: {e}")
                        
                        # Create failed result
                        failed_result = BacktestResult(
                            backtest_id=config.backtest_id,
                            status=BacktestStatus.FAILED,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            execution_time=0.0,
                            total_return=0.0,
                            annualized_return=0.0,
                            sharpe_ratio=0.0,
                            max_drawdown=0.0,
                            win_rate=0.0,
                            total_trades=0,
                            profit_factor=0.0,
                            calmar_ratio=0.0,
                            sortino_ratio=0.0,
                            var_95=0.0,
                            var_99=0.0,
                            beta=0.0,
                            alpha=0.0,
                            equity_curve=pd.DataFrame(),
                            trade_log=pd.DataFrame(),
                            metrics={},
                            error=str(e)
                        )
                        results.append(failed_result)
            
            # Update performance metrics
            self._update_performance_metrics(results)
            
            logger.info(f"Parallel backtesting completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel backtesting: {e}")
            return []
    
    def run_parameter_optimization(self, base_config: BacktestConfig, 
                                 parameter_ranges: Dict[str, List[Any]],
                                 optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Run parameter optimization using parallel backtesting"""
        try:
            # Generate parameter combinations
            parameter_combinations = self._generate_parameter_combinations(parameter_ranges)
            
            logger.info(f"Starting parameter optimization with {len(parameter_combinations)} combinations")
            
            # Create backtest configs for each parameter combination
            backtest_configs = []
            for i, params in enumerate(parameter_combinations):
                config = BacktestConfig(
                    backtest_id=f"{base_config.backtest_id}_opt_{i}",
                    backtest_type=BacktestType.PARAMETER_OPTIMIZATION,
                    strategy_name=base_config.strategy_name,
                    start_date=base_config.start_date,
                    end_date=base_config.end_date,
                    initial_capital=base_config.initial_capital,
                    symbols=base_config.symbols,
                    parameters={**base_config.parameters, **params},
                    data_source=base_config.data_source,
                    execution_config=base_config.execution_config,
                    risk_config=base_config.risk_config,
                    metadata={**base_config.metadata, 'optimization_params': params}
                )
                backtest_configs.append(config)
            
            # Run parallel backtests
            results = self.run_parallel_backtests(backtest_configs)
            
            # Find best parameters
            best_result = self._find_best_parameters(results, optimization_metric)
            
            optimization_result = {
                'best_parameters': best_result.metadata.get('optimization_params', {}),
                'best_metric_value': getattr(best_result, optimization_metric, 0.0),
                'total_combinations': len(parameter_combinations),
                'successful_backtests': len([r for r in results if r.status == BacktestStatus.COMPLETED]),
                'optimization_metric': optimization_metric,
                'all_results': results
            }
            
            logger.info(f"Parameter optimization completed. Best {optimization_metric}: {optimization_result['best_metric_value']:.4f}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return {}
    
    def run_walk_forward_analysis(self, base_config: BacktestConfig,
                                training_period_months: int = 12,
                                testing_period_months: int = 3,
                                step_months: int = 1) -> List[BacktestResult]:
        """Run walk-forward analysis using parallel backtesting"""
        try:
            # Generate time periods for walk-forward analysis
            time_periods = self._generate_walk_forward_periods(
                base_config.start_date, base_config.end_date,
                training_period_months, testing_period_months, step_months
            )
            
            logger.info(f"Starting walk-forward analysis with {len(time_periods)} periods")
            
            # Create backtest configs for each period
            backtest_configs = []
            for i, (train_start, train_end, test_start, test_end) in enumerate(time_periods):
                config = BacktestConfig(
                    backtest_id=f"{base_config.backtest_id}_wf_{i}",
                    backtest_type=BacktestType.WALK_FORWARD,
                    strategy_name=base_config.strategy_name,
                    start_date=test_start,
                    end_date=test_end,
                    initial_capital=base_config.initial_capital,
                    symbols=base_config.symbols,
                    parameters=base_config.parameters,
                    data_source=base_config.data_source,
                    execution_config=base_config.execution_config,
                    risk_config=base_config.risk_config,
                    metadata={
                        **base_config.metadata,
                        'training_start': train_start,
                        'training_end': train_end,
                        'testing_start': test_start,
                        'testing_end': test_end,
                        'period_index': i
                    }
                )
                backtest_configs.append(config)
            
            # Run parallel backtests
            results = self.run_parallel_backtests(backtest_configs)
            
            logger.info(f"Walk-forward analysis completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {e}")
            return []
    
    def run_monte_carlo_simulation(self, base_config: BacktestConfig,
                                 num_simulations: int = 1000,
                                 confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Any]:
        """Run Monte Carlo simulation using parallel backtesting"""
        try:
            logger.info(f"Starting Monte Carlo simulation with {num_simulations} simulations")
            
            # Create backtest configs for each simulation
            backtest_configs = []
            for i in range(num_simulations):
                config = BacktestConfig(
                    backtest_id=f"{base_config.backtest_id}_mc_{i}",
                    backtest_type=BacktestType.MONTE_CARLO,
                    strategy_name=base_config.strategy_name,
                    start_date=base_config.start_date,
                    end_date=base_config.end_date,
                    initial_capital=base_config.initial_capital,
                    symbols=base_config.symbols,
                    parameters=base_config.parameters,
                    data_source=base_config.data_source,
                    execution_config=base_config.execution_config,
                    risk_config=base_config.risk_config,
                    metadata={
                        **base_config.metadata,
                        'simulation_index': i,
                        'random_seed': i
                    }
                )
                backtest_configs.append(config)
            
            # Run parallel backtests
            results = self.run_parallel_backtests(backtest_configs)
            
            # Analyze results
            monte_carlo_analysis = self._analyze_monte_carlo_results(results, confidence_levels)
            
            logger.info(f"Monte Carlo simulation completed: {len(results)} simulations")
            return monte_carlo_analysis
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {}
    
    def _run_single_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Run a single backtest (designed for multiprocessing)"""
        try:
            start_time = time.time()
            
            # Load data
            data = self._load_backtest_data(config)
            if data is None or data.empty:
                raise ValueError(f"No data available for backtest {config.backtest_id}")
            
            # Initialize backtest engine
            backtest_engine = self._create_backtest_engine(config)
            
            # Run backtest
            result = backtest_engine.run_backtest(data)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create result object
            backtest_result = BacktestResult(
                backtest_id=config.backtest_id,
                status=BacktestStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                execution_time=execution_time,
                total_return=result.get('total_return', 0.0),
                annualized_return=result.get('annualized_return', 0.0),
                sharpe_ratio=result.get('sharpe_ratio', 0.0),
                max_drawdown=result.get('max_drawdown', 0.0),
                win_rate=result.get('win_rate', 0.0),
                total_trades=result.get('total_trades', 0),
                profit_factor=result.get('profit_factor', 0.0),
                calmar_ratio=result.get('calmar_ratio', 0.0),
                sortino_ratio=result.get('sortino_ratio', 0.0),
                var_95=result.get('var_95', 0.0),
                var_99=result.get('var_99', 0.0),
                beta=result.get('beta', 0.0),
                alpha=result.get('alpha', 0.0),
                equity_curve=result.get('equity_curve', pd.DataFrame()),
                trade_log=result.get('trade_log', pd.DataFrame()),
                metrics=result.get('metrics', {})
            )
            
            return backtest_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create failed result
            failed_result = BacktestResult(
                backtest_id=config.backtest_id,
                status=BacktestStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                execution_time=execution_time,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                profit_factor=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                var_95=0.0,
                var_99=0.0,
                beta=0.0,
                alpha=0.0,
                equity_curve=pd.DataFrame(),
                trade_log=pd.DataFrame(),
                metrics={},
                error=str(e)
            )
            
            return failed_result
    
    def _load_backtest_data(self, config: BacktestConfig) -> pd.DataFrame:
        """Load data for backtesting"""
        try:
            # Check cache first
            cache_key = f"{config.symbols}_{config.start_date}_{config.end_date}_{config.data_source}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # Load data (this would interface with your data pipeline)
            # For now, generate sample data
            data = self._generate_sample_data(config)
            
            # Cache data
            if len(self.data_cache) < self.cache_size_limit:
                self.data_cache[cache_key] = data
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading backtest data: {e}")
            return pd.DataFrame()
    
    def _generate_sample_data(self, config: BacktestConfig) -> pd.DataFrame:
        """Generate sample data for backtesting"""
        try:
            # Generate date range
            date_range = pd.date_range(start=config.start_date, end=config.end_date, freq='D')
            
            # Generate sample price data
            data = []
            for symbol in config.symbols:
                # Generate random walk prices
                np.random.seed(hash(symbol) % 2**32)
                prices = 100 * np.cumprod(1 + np.random.normal(0.0001, 0.02, len(date_range)))
                
                for i, date in enumerate(date_range):
                    data.append({
                        'date': date,
                        'symbol': symbol,
                        'open': prices[i],
                        'high': prices[i] * (1 + abs(np.random.normal(0, 0.01))),
                        'low': prices[i] * (1 - abs(np.random.normal(0, 0.01))),
                        'close': prices[i],
                        'volume': np.random.randint(1000, 10000)
                    })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            return pd.DataFrame()
    
    def _create_backtest_engine(self, config: BacktestConfig):
        """Create backtest engine instance"""
        try:
            # This would create an actual backtest engine
            # For now, return a mock engine
            class MockBacktestEngine:
                def __init__(self, config):
                    self.config = config
                
                def run_backtest(self, data):
                    # Mock backtest results
                    return {
                        'total_return': np.random.normal(0.1, 0.2),
                        'annualized_return': np.random.normal(0.08, 0.15),
                        'sharpe_ratio': np.random.normal(1.2, 0.5),
                        'max_drawdown': abs(np.random.normal(0.1, 0.05)),
                        'win_rate': np.random.uniform(0.4, 0.7),
                        'total_trades': np.random.randint(50, 200),
                        'profit_factor': np.random.uniform(1.0, 2.0),
                        'calmar_ratio': np.random.normal(1.0, 0.3),
                        'sortino_ratio': np.random.normal(1.5, 0.4),
                        'var_95': abs(np.random.normal(0.03, 0.01)),
                        'var_99': abs(np.random.normal(0.05, 0.02)),
                        'beta': np.random.normal(1.0, 0.2),
                        'alpha': np.random.normal(0.02, 0.1),
                        'equity_curve': pd.DataFrame(),
                        'trade_log': pd.DataFrame(),
                        'metrics': {}
                    }
            
            return MockBacktestEngine(config)
            
        except Exception as e:
            logger.error(f"Error creating backtest engine: {e}")
            return None
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        try:
            import itertools
            
            # Get parameter names and values
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            # Generate all combinations
            combinations = list(itertools.product(*param_values))
            
            # Convert to list of dictionaries
            param_combinations = []
            for combo in combinations:
                param_dict = dict(zip(param_names, combo))
                param_combinations.append(param_dict)
            
            return param_combinations
            
        except Exception as e:
            logger.error(f"Error generating parameter combinations: {e}")
            return []
    
    def _generate_walk_forward_periods(self, start_date: datetime, end_date: datetime,
                                     training_months: int, testing_months: int, 
                                     step_months: int) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate walk-forward analysis periods"""
        try:
            periods = []
            current_date = start_date
            
            while current_date < end_date:
                # Training period
                train_start = current_date
                train_end = train_start + timedelta(days=training_months * 30)
                
                # Testing period
                test_start = train_end
                test_end = test_start + timedelta(days=testing_months * 30)
                
                # Check if testing period extends beyond end date
                if test_end > end_date:
                    break
                
                periods.append((train_start, train_end, test_start, test_end))
                
                # Move to next period
                current_date += timedelta(days=step_months * 30)
            
            return periods
            
        except Exception as e:
            logger.error(f"Error generating walk-forward periods: {e}")
            return []
    
    def _find_best_parameters(self, results: List[BacktestResult], metric: str) -> BacktestResult:
        """Find best parameters based on metric"""
        try:
            # Filter successful results
            successful_results = [r for r in results if r.status == BacktestStatus.COMPLETED]
            
            if not successful_results:
                return results[0] if results else None
            
            # Find result with best metric value
            best_result = max(successful_results, key=lambda r: getattr(r, metric, 0))
            
            return best_result
            
        except Exception as e:
            logger.error(f"Error finding best parameters: {e}")
            return results[0] if results else None
    
    def _analyze_monte_carlo_results(self, results: List[BacktestResult], 
                                   confidence_levels: List[float]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        try:
            # Filter successful results
            successful_results = [r for r in results if r.status == BacktestStatus.COMPLETED]
            
            if not successful_results:
                return {}
            
            # Extract metrics
            total_returns = [r.total_return for r in successful_results]
            sharpe_ratios = [r.sharpe_ratio for r in successful_results]
            max_drawdowns = [r.max_drawdown for r in successful_results]
            
            # Calculate statistics
            analysis = {
                'total_simulations': len(results),
                'successful_simulations': len(successful_results),
                'success_rate': len(successful_results) / len(results) if results else 0,
                'total_return': {
                    'mean': np.mean(total_returns),
                    'std': np.std(total_returns),
                    'min': np.min(total_returns),
                    'max': np.max(total_returns)
                },
                'sharpe_ratio': {
                    'mean': np.mean(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'min': np.min(sharpe_ratios),
                    'max': np.max(sharpe_ratios)
                },
                'max_drawdown': {
                    'mean': np.mean(max_drawdowns),
                    'std': np.std(max_drawdowns),
                    'min': np.min(max_drawdowns),
                    'max': np.max(max_drawdowns)
                }
            }
            
            # Calculate confidence intervals
            for confidence_level in confidence_levels:
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                analysis[f'confidence_interval_{int(confidence_level*100)}'] = {
                    'total_return': {
                        'lower': np.percentile(total_returns, lower_percentile),
                        'upper': np.percentile(total_returns, upper_percentile)
                    },
                    'sharpe_ratio': {
                        'lower': np.percentile(sharpe_ratios, lower_percentile),
                        'upper': np.percentile(sharpe_ratios, upper_percentile)
                    },
                    'max_drawdown': {
                        'lower': np.percentile(max_drawdowns, lower_percentile),
                        'upper': np.percentile(max_drawdowns, upper_percentile)
                    }
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Monte Carlo results: {e}")
            return {}
    
    def _update_performance_metrics(self, results: List[BacktestResult]):
        """Update performance metrics"""
        try:
            self.performance_metrics['total_backtests'] += len(results)
            self.performance_metrics['successful_backtests'] += len([r for r in results if r.status == BacktestStatus.COMPLETED])
            self.performance_metrics['failed_backtests'] += len([r for r in results if r.status == BacktestStatus.FAILED])
            
            # Update execution time metrics
            execution_times = [r.execution_time for r in results if r.execution_time > 0]
            if execution_times:
                self.performance_metrics['total_execution_time'] += sum(execution_times)
                self.performance_metrics['average_execution_time'] = (
                    self.performance_metrics['total_execution_time'] / 
                    self.performance_metrics['total_backtests']
                )
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_engine_statistics(self) -> Dict:
        """Get backtesting engine statistics"""
        try:
            return {
                'performance_metrics': self.performance_metrics,
                'queue_size': len(self.backtest_queue),
                'running_backtests': len(self.running_backtests),
                'completed_backtests': len(self.completed_backtests),
                'failed_backtests': len(self.failed_backtests),
                'max_workers': self.max_workers,
                'max_concurrent_backtests': self.max_concurrent_backtests,
                'cache_size': len(self.data_cache),
                'cache_size_limit': self.cache_size_limit
            }
            
        except Exception as e:
            logger.error(f"Error getting engine statistics: {e}")
            return {}
    
    def export_backtest_results(self, filepath: str, results: List[BacktestResult]):
        """Export backtest results to file"""
        try:
            export_data = {
                'export_info': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_results': len(results),
                    'successful_results': len([r for r in results if r.status == BacktestStatus.COMPLETED])
                },
                'results': [
                    {
                        'backtest_id': result.backtest_id,
                        'status': result.status.value,
                        'execution_time': result.execution_time,
                        'total_return': result.total_return,
                        'annualized_return': result.annualized_return,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'win_rate': result.win_rate,
                        'total_trades': result.total_trades,
                        'profit_factor': result.profit_factor,
                        'calmar_ratio': result.calmar_ratio,
                        'sortino_ratio': result.sortino_ratio,
                        'var_95': result.var_95,
                        'var_99': result.var_99,
                        'beta': result.beta,
                        'alpha': result.alpha,
                        'error': result.error
                    }
                    for result in results
                ],
                'engine_statistics': self.get_engine_statistics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported backtest results to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting backtest results: {e}")
    
    def clear_cache(self):
        """Clear data cache"""
        try:
            self.data_cache.clear()
            logger.info("Data cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
