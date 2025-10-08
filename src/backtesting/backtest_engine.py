"""
Backtesting Engine

Comprehensive backtesting framework with:
- Historical data validation
- Strategy performance testing
- Walk-forward optimization
- Monte Carlo simulation
- Stress testing
- Performance metrics (Sharpe, Sortino, Max DD)
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Backtest result data"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: float
    trades: List[Dict] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series())
    drawdown_curve: pd.Series = field(default_factory=lambda: pd.Series())

class BacktestEngine:
    """
    Backtesting Engine for Strategy Validation
    
    Features:
    - Historical data replay
    - Strategy execution simulation
    - Performance metrics calculation
    - Risk analysis
    - Walk-forward optimization
    - Monte Carlo simulation
    - Stress testing scenarios
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 0.05%
        risk_free_rate: float = 0.04  # 4% annual
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
        self.results: Dict[str, BacktestResult] = {}
        
        logger.info(" Backtest Engine initialized")
    
    def run_backtest(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        position_size: float = 0.1  # 10% per trade
    ) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            strategy_name: Name of strategy being tested
            data: Historical OHLCV data
            signals: Trading signals (1=buy, -1=sell, 0=hold)
            position_size: Position size as fraction of capital
        
        Returns:
            BacktestResult with performance metrics
        """
        
        logger.info(f" Running backtest for {strategy_name}")
        logger.info(f"   Period: {data.index[0]} to {data.index[-1]}")
        logger.info(f"   Data points: {len(data)}")
        
        # Initialize portfolio
        capital = self.initial_capital
        position = 0  # Current position (shares)
        trades = []
        equity_curve = []
        
        # Iterate through data
        for i in range(len(data)):
            date = data.index[i]
            price = data['close'].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Track equity
            equity = capital + (position * price)
            equity_curve.append({'date': date, 'equity': equity})
            
            # Execute signals
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size
                shares = (capital * position_size) / price
                cost = shares * price
                commission_cost = cost * self.commission
                slippage_cost = cost * self.slippage
                total_cost = cost + commission_cost + slippage_cost
                
                if total_cost <= capital:
                    position = shares
                    capital -= total_cost
                    
                    trades.append({
                        'type': 'buy',
                        'date': date,
                        'price': price,
                        'shares': shares,
                        'cost': total_cost,
                        'commission': commission_cost,
                        'slippage': slippage_cost
                    })
                    
                    logger.debug(f"   BUY: {shares:.2f} shares @ ${price:.2f}")
            
            elif signal == -1 and position > 0:  # Sell signal
                # Sell position
                proceeds = position * price
                commission_cost = proceeds * self.commission
                slippage_cost = proceeds * self.slippage
                net_proceeds = proceeds - commission_cost - slippage_cost
                
                capital += net_proceeds
                
                # Calculate P&L
                entry_trade = [t for t in trades if t['type'] == 'buy'][-1]
                pnl = net_proceeds - entry_trade['cost']
                pnl_pct = (pnl / entry_trade['cost']) * 100
                
                trades.append({
                    'type': 'sell',
                    'date': date,
                    'price': price,
                    'shares': position,
                    'proceeds': net_proceeds,
                    'commission': commission_cost,
                    'slippage': slippage_cost,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'entry_date': entry_trade['date'],
                    'entry_price': entry_trade['price'],
                    'duration': (date - entry_trade['date']).days
                })
                
                logger.debug(f"   SELL: {position:.2f} shares @ ${price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                
                position = 0
        
        # Close any open position at end
        if position > 0:
            price = data['close'].iloc[-1]
            proceeds = position * price
            commission_cost = proceeds * self.commission
            net_proceeds = proceeds - commission_cost
            capital += net_proceeds
            
            entry_trade = [t for t in trades if t['type'] == 'buy'][-1]
            pnl = net_proceeds - entry_trade['cost']
            
            trades.append({
                'type': 'sell',
                'date': data.index[-1],
                'price': price,
                'shares': position,
                'proceeds': net_proceeds,
                'commission': commission_cost,
                'pnl': pnl,
                'pnl_pct': (pnl / entry_trade['cost']) * 100,
                'entry_date': entry_trade['date'],
                'entry_price': entry_trade['price']
            })
        
        # Calculate metrics
        final_capital = capital + (position * data['close'].iloc[-1])
        result = self._calculate_metrics(
            strategy_name=strategy_name,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            trades=trades,
            equity_curve=equity_curve
        )
        
        # Store result
        self.results[strategy_name] = result
        
        logger.info(f" Backtest complete for {strategy_name}")
        logger.info(f"   Return: {result.total_return_pct:.2f}%")
        logger.info(f"   Sharpe: {result.sharpe_ratio:.2f}")
        logger.info(f"   Max DD: {result.max_drawdown_pct:.2f}%")
        logger.info(f"   Win Rate: {result.win_rate:.2f}%")
        
        return result
    
    def _calculate_metrics(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        final_capital: float,
        trades: List[Dict],
        equity_curve: List[Dict]
    ) -> BacktestResult:
        """Calculate performance metrics"""
        
        # Basic returns
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Trade statistics
        sell_trades = [t for t in trades if t['type'] == 'sell']
        total_trades = len(sell_trades)
        
        if total_trades == 0:
            # No completed trades
            return BacktestResult(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                total_return_pct=total_return_pct,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                avg_trade_duration=0.0,
                trades=trades
            )
        
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) <= 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum([t['pnl'] for t in winning_trades])
        total_losses = sum([abs(t['pnl']) for t in losing_trades])
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_duration = np.mean([t.get('duration', 0) for t in sell_trades]) if sell_trades else 0
        
        # Equity curve
        equity_df = pd.DataFrame(equity_curve)
        equity_series = pd.Series(equity_df['equity'].values, index=equity_df['date'])
        
        # Drawdown
        peak = equity_series.expanding().max()
        drawdown = equity_series - peak
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / initial_capital) * 100
        
        drawdown_pct = (drawdown / peak) * 100
        
        # Sharpe Ratio
        returns = equity_series.pct_change().dropna()
        if len(returns) > 1:
            excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio
        if len(returns) > 1:
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std()
            sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino_ratio = 0
        
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=avg_duration,
            trades=trades,
            equity_curve=equity_series,
            drawdown_curve=drawdown_pct
        )
    
    def monte_carlo_simulation(
        self,
        result: BacktestResult,
        num_simulations: int = 1000
    ) -> Dict:
        """
        Run Monte Carlo simulation on backtest results
        
        Args:
            result: BacktestResult to simulate
            num_simulations: Number of simulations to run
        
        Returns:
            Dictionary with simulation results
        """
        
        logger.info(f" Running Monte Carlo simulation ({num_simulations} runs)")
        
        if not result.trades:
            return {'error': 'No trades to simulate'}
        
        # Extract trade returns
        sell_trades = [t for t in result.trades if t['type'] == 'sell']
        trade_returns = [t.get('pnl_pct', 0) / 100 for t in sell_trades]
        
        if not trade_returns:
            return {'error': 'No completed trades'}
        
        # Run simulations
        final_returns = []
        max_drawdowns = []
        
        for _ in range(num_simulations):
            # Randomly sample trades with replacement
            simulated_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate cumulative return
            cumulative = result.initial_capital
            equity_curve = [cumulative]
            
            for ret in simulated_returns:
                cumulative *= (1 + ret)
                equity_curve.append(cumulative)
            
            final_returns.append((cumulative - result.initial_capital) / result.initial_capital * 100)
            
            # Calculate max drawdown
            equity_series = pd.Series(equity_curve)
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak * 100
            max_drawdowns.append(drawdown.min())
        
        # Calculate statistics
        return {
            'num_simulations': num_simulations,
            'mean_return': np.mean(final_returns),
            'median_return': np.median(final_returns),
            'std_return': np.std(final_returns),
            'percentile_5': np.percentile(final_returns, 5),
            'percentile_95': np.percentile(final_returns, 95),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),
            'probability_profit': (np.array(final_returns) > 0).mean() * 100
        }
    
    def stress_test(
        self,
        result: BacktestResult,
        scenarios: Dict[str, Dict] = None
    ) -> Dict:
        """
        Run stress tests on backtest results
        
        Args:
            result: BacktestResult to stress test
            scenarios: Custom stress test scenarios
        
        Returns:
            Dictionary with stress test results
        """
        
        logger.info(" Running stress tests")
        
        # Default scenarios
        if scenarios is None:
            scenarios = {
                'market_crash_20': {'shock': -0.20, 'duration': 5},
                'market_crash_30': {'shock': -0.30, 'duration': 10},
                'volatility_spike': {'vol_multiplier': 3.0},
                'commission_increase': {'commission_multiplier': 5.0},
                'slippage_increase': {'slippage_multiplier': 10.0}
            }
        
        stress_results = {}
        
        for scenario_name, params in scenarios.items():
            logger.info(f"   Testing: {scenario_name}")
            
            if 'shock' in params:
                # Market shock scenario
                shock_return = result.total_return * (1 + params['shock'])
                shock_capital = result.initial_capital + shock_return
                stress_results[scenario_name] = {
                    'final_capital': shock_capital,
                    'return_pct': (shock_return / result.initial_capital) * 100,
                    'impact': params['shock'] * 100
                }
            
            elif 'vol_multiplier' in params:
                # Increased volatility
                stress_results[scenario_name] = {
                    'sharpe_ratio': result.sharpe_ratio / params['vol_multiplier'],
                    'impact': f"{params['vol_multiplier']}x volatility"
                }
            
            elif 'commission_multiplier' in params:
                # Increased costs
                total_commission = sum([t.get('commission', 0) for t in result.trades])
                additional_cost = total_commission * (params['commission_multiplier'] - 1)
                adjusted_return = result.total_return - additional_cost
                stress_results[scenario_name] = {
                    'adjusted_return': adjusted_return,
                    'return_pct': (adjusted_return / result.initial_capital) * 100,
                    'impact': f"{params['commission_multiplier']}x commission"
                }
        
        return stress_results
    
    def walk_forward_analysis(
        self,
        data: pd.DataFrame,
        signals_func,
        train_window: int = 180,  # 6 months
        test_window: int = 30  # 1 month
    ) -> List[BacktestResult]:
        """
        Perform walk-forward optimization
        
        Args:
            data: Complete historical data
            signals_func: Function to generate signals
            train_window: Training period in days
            test_window: Testing period in days
        
        Returns:
            List of BacktestResult for each period
        """
        
        logger.info(" Running walk-forward analysis")
        
        results = []
        start_idx = 0
        
        while start_idx + train_window + test_window <= len(data):
            # Split data
            train_end = start_idx + train_window
            test_end = train_end + test_window
            
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]
            
            # Generate signals on train data (would optimize parameters here)
            # For now, just use test data
            test_signals = signals_func(test_data)
            
            # Run backtest on test period
            result = self.run_backtest(
                strategy_name=f"WF_{start_idx}_{test_end}",
                data=test_data,
                signals=test_signals
            )
            
            results.append(result)
            
            # Move window
            start_idx += test_window
        
        logger.info(f" Walk-forward analysis complete: {len(results)} periods")
        
        return results
    
    def compare_strategies(self, strategy_names: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple strategy results
        
        Args:
            strategy_names: List of strategy names to compare
        
        Returns:
            DataFrame with comparison metrics
        """
        
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        comparison = []
        
        for name in strategy_names:
            if name not in self.results:
                continue
            
            result = self.results[name]
            comparison.append({
                'Strategy': name,
                'Return %': result.total_return_pct,
                'Sharpe': result.sharpe_ratio,
                'Sortino': result.sortino_ratio,
                'Max DD %': result.max_drawdown_pct,
                'Win Rate %': result.win_rate,
                'Profit Factor': result.profit_factor,
                'Total Trades': result.total_trades
            })
        
        return pd.DataFrame(comparison)
    
    def save_results(self, filepath: str):
        """Save backtest results to file"""
        results_data = {}
        
        for name, result in self.results.items():
            results_data[name] = {
                'strategy_name': result.strategy_name,
                'start_date': result.start_date.isoformat(),
                'end_date': result.end_date.isoformat(),
                'initial_capital': result.initial_capital,
                'final_capital': result.final_capital,
                'total_return_pct': result.total_return_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown_pct': result.max_drawdown_pct,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f" Results saved to {filepath}")

# Global engine instance
_backtest_engine = None

def get_backtest_engine() -> BacktestEngine:
    """Get global backtest engine instance"""
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine()
    return _backtest_engine

