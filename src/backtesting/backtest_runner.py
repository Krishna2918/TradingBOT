"""
Production Backtest Engine

Comprehensive backtesting framework for evaluating trained models:
- Historical data replay
- P&L calculations with transaction costs
- Risk metrics (Sharpe, Sortino, Calmar, max drawdown)
- Trade journal and analytics
- Comparison with baselines (buy-and-hold, 60/40)

Author: Trading Bot Team
Date: October 28, 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class BacktestResult:
    """Results from a backtest run"""

    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int

    # Trade statistics
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: float

    # Portfolio metrics
    initial_capital: float
    final_capital: float
    max_capital: float
    min_capital: float

    # Time series
    portfolio_values: np.ndarray
    returns: np.ndarray
    drawdowns: np.ndarray
    positions: np.ndarray

    # Trade log
    trades: List[Dict]

    # Metadata
    start_date: str
    end_date: str
    duration_days: int
    strategy_name: str


class BacktestRunner:
    """
    Backtesting engine for model evaluation

    Features:
    - Realistic transaction costs and slippage
    - Position sizing constraints
    - Risk management rules
    - Detailed trade analytics
    - Baseline comparisons
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 0.05%
        max_position_size: float = 1.0,  # 100% of capital
        risk_free_rate: float = 0.02  # 2% annual
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate

    def run_backtest(
        self,
        df: pd.DataFrame,
        actions: np.ndarray,
        strategy_name: str = "Model"
    ) -> BacktestResult:
        """
        Run backtest with given actions

        Args:
            df: DataFrame with price data (must have 'close' column)
            actions: Array of actions for each timestep
                     0=Hold, 1-4=Buy(25%/50%/75%/100%), 5-8=Sell(25%/50%/75%/100%)
            strategy_name: Name of strategy for tracking

        Returns:
            BacktestResult with comprehensive metrics
        """

        # Initialize state
        cash = self.initial_capital
        shares = 0.0
        entry_price = 0.0

        portfolio_values = []
        returns_list = []
        positions_list = []
        trades = []

        # Get prices
        if 'close' in df.columns:
            prices = df['close'].values
        else:
            prices = df.iloc[:, 0].values  # Use first column

        # Simulate trading
        for t in range(len(actions)):
            action = actions[t]
            current_price = prices[t]

            # Apply slippage
            if action in [1, 2, 3, 4]:  # Buy
                execution_price = current_price * (1 + self.slippage)
            elif action in [5, 6, 7, 8]:  # Sell
                execution_price = current_price * (1 - self.slippage)
            else:
                execution_price = current_price

            # Execute action
            if action == 0:
                # Hold
                pass

            elif action in [1, 2, 3, 4]:
                # Buy
                allocation_pct = [0.25, 0.50, 0.75, 1.00][action - 1]
                max_buy_value = min(
                    cash * allocation_pct,
                    self.initial_capital * self.max_position_size - (shares * execution_price)
                )

                if max_buy_value > 0:
                    shares_to_buy = max_buy_value / execution_price
                    gross_cost = shares_to_buy * execution_price
                    transaction_fee = gross_cost * self.transaction_cost
                    total_cost = gross_cost + transaction_fee

                    if total_cost <= cash:
                        shares += shares_to_buy
                        cash -= total_cost

                        # Update entry price (weighted average)
                        if entry_price == 0:
                            entry_price = execution_price
                        else:
                            total_shares = shares
                            entry_price = (
                                (entry_price * (total_shares - shares_to_buy) +
                                 execution_price * shares_to_buy) / total_shares
                            )

                        trades.append({
                            'timestamp': t,
                            'date': df.index[t] if hasattr(df.index, 'date') else t,
                            'action': 'buy',
                            'shares': shares_to_buy,
                            'price': execution_price,
                            'cost': total_cost,
                            'portfolio_value': cash + shares * current_price
                        })

            elif action in [5, 6, 7, 8]:
                # Sell
                allocation_pct = [0.25, 0.50, 0.75, 1.00][action - 5]
                shares_to_sell = shares * allocation_pct

                if shares_to_sell > 0:
                    gross_proceeds = shares_to_sell * execution_price
                    transaction_fee = gross_proceeds * self.transaction_cost
                    net_proceeds = gross_proceeds - transaction_fee

                    shares -= shares_to_sell
                    cash += net_proceeds

                    # Reset entry price if fully sold
                    if shares < 1e-6:
                        entry_price = 0.0
                        shares = 0.0

                    trades.append({
                        'timestamp': t,
                        'date': df.index[t] if hasattr(df.index, 'date') else t,
                        'action': 'sell',
                        'shares': shares_to_sell,
                        'price': execution_price,
                        'proceeds': net_proceeds,
                        'portfolio_value': cash + shares * current_price
                    })

            # Update portfolio
            position_value = shares * current_price
            portfolio_value = cash + position_value

            portfolio_values.append(portfolio_value)
            positions_list.append(shares)

            # Calculate return
            if len(portfolio_values) > 1:
                ret = (portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
                returns_list.append(ret)

        # Convert to arrays
        portfolio_values = np.array(portfolio_values)
        returns = np.array(returns_list)
        positions = np.array(positions_list)

        # Calculate metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital

        # Annual return
        duration_days = len(portfolio_values)
        duration_years = duration_days / 252  # Trading days
        annual_return = (1 + total_return) ** (1 / duration_years) - 1 if duration_years > 0 else 0

        # Sharpe ratio
        if len(returns) > 0:
            daily_rf_rate = (1 + self.risk_free_rate) ** (1/252) - 1
            excess_returns = returns - daily_rf_rate
            sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio
        if len(returns) > 0:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                daily_rf_rate = (1 + self.risk_free_rate) ** (1/252) - 1
                excess_returns = returns - daily_rf_rate
                downside_std = np.std(downside_returns)
                sortino_ratio = np.mean(excess_returns) / (downside_std + 1e-10) * np.sqrt(252)
            else:
                sortino_ratio = sharpe_ratio
        else:
            sortino_ratio = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

        # Max drawdown duration
        is_drawdown = drawdowns < 0
        drawdown_durations = []
        current_duration = 0
        for dd in is_drawdown:
            if dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_durations.append(current_duration)
                current_duration = 0
        max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

        # Trade statistics
        num_trades = len(trades)

        # Calculate win rate and profit factor
        if num_trades >= 2:
            # Match buy-sell pairs
            wins = 0
            losses = 0
            total_profit = 0
            total_loss = 0
            trade_durations = []

            i = 0
            while i < len(trades) - 1:
                if trades[i]['action'] == 'buy':
                    # Find next sell
                    for j in range(i + 1, len(trades)):
                        if trades[j]['action'] == 'sell':
                            buy_price = trades[i]['price']
                            sell_price = trades[j]['price']
                            pnl = (sell_price - buy_price) / buy_price

                            if pnl > 0:
                                wins += 1
                                total_profit += pnl
                            else:
                                losses += 1
                                total_loss += abs(pnl)

                            # Trade duration
                            duration = trades[j]['timestamp'] - trades[i]['timestamp']
                            trade_durations.append(duration)

                            i = j
                            break
                    else:
                        break
                i += 1

            total_closed_trades = wins + losses
            win_rate = wins / total_closed_trades if total_closed_trades > 0 else 0
            avg_win = total_profit / wins if wins > 0 else 0
            avg_loss = total_loss / losses if losses > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            avg_trade_duration = 0.0

        # Create result
        result = BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=avg_trade_duration,
            initial_capital=self.initial_capital,
            final_capital=portfolio_values[-1],
            max_capital=np.max(portfolio_values),
            min_capital=np.min(portfolio_values),
            portfolio_values=portfolio_values,
            returns=returns,
            drawdowns=drawdowns,
            positions=positions,
            trades=trades,
            start_date=str(df.index[0]) if hasattr(df.index, 'date') else "0",
            end_date=str(df.index[-1]) if hasattr(df.index, 'date') else str(len(df)-1),
            duration_days=duration_days,
            strategy_name=strategy_name
        )

        return result

    def run_buy_and_hold_baseline(self, df: pd.DataFrame) -> BacktestResult:
        """Run buy-and-hold baseline strategy"""

        # Get prices
        if 'close' in df.columns:
            prices = df['close'].values
        else:
            prices = df.iloc[:, 0].values

        # Buy at first price, hold until end
        first_price = prices[0]
        shares = (self.initial_capital * (1 - self.transaction_cost)) / first_price

        portfolio_values = shares * prices
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Create dummy actions (all hold)
        actions = np.zeros(len(prices), dtype=int)
        actions[0] = 4  # Buy 100% at start

        result = self.run_backtest(df, actions, strategy_name="Buy-and-Hold")

        return result

    def compare_strategies(
        self,
        results: List[BacktestResult]
    ) -> pd.DataFrame:
        """
        Compare multiple backtest results

        Args:
            results: List of BacktestResult objects

        Returns:
            DataFrame with comparison metrics
        """

        comparison_data = []

        for result in results:
            comparison_data.append({
                'Strategy': result.strategy_name,
                'Total Return (%)': result.total_return * 100,
                'Annual Return (%)': result.annual_return * 100,
                'Sharpe Ratio': result.sharpe_ratio,
                'Sortino Ratio': result.sortino_ratio,
                'Calmar Ratio': result.calmar_ratio,
                'Max Drawdown (%)': result.max_drawdown * 100,
                'Win Rate (%)': result.win_rate * 100,
                'Profit Factor': result.profit_factor,
                'Num Trades': result.num_trades,
                'Avg Trade Duration': result.avg_trade_duration,
                'Final Capital': result.final_capital
            })

        df = pd.DataFrame(comparison_data)

        return df

    def save_results(
        self,
        result: BacktestResult,
        output_dir: Path = Path('results/backtests')
    ):
        """Save backtest results to disk"""

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary = {
            'strategy_name': result.strategy_name,
            'start_date': result.start_date,
            'end_date': result.end_date,
            'duration_days': result.duration_days,
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'calmar_ratio': result.calmar_ratio,
            'max_drawdown': result.max_drawdown,
            'max_drawdown_duration': result.max_drawdown_duration,
            'num_trades': result.num_trades,
            'win_rate': result.win_rate,
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'profit_factor': result.profit_factor,
            'avg_trade_duration': result.avg_trade_duration
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = output_dir / f'{result.strategy_name}_{timestamp}_summary.json'

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Backtest results saved to: {summary_path}")

        # Save trade log
        if result.trades:
            trades_df = pd.DataFrame(result.trades)
            trades_path = output_dir / f'{result.strategy_name}_{timestamp}_trades.csv'
            trades_df.to_csv(trades_path, index=False)
            print(f"Trade log saved to: {trades_path}")

        # Save time series
        timeseries_df = pd.DataFrame({
            'portfolio_value': result.portfolio_values,
            'drawdown': result.drawdowns,
            'position': result.positions
        })
        timeseries_path = output_dir / f'{result.strategy_name}_{timestamp}_timeseries.csv'
        timeseries_df.to_csv(timeseries_path, index=False)
        print(f"Time series saved to: {timeseries_path}")
