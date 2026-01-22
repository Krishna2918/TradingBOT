"""
Enhanced Trading Environment for Reinforcement Learning

Production-grade environment with:
- Complete observation generation (95 features from feature engineering)
- Advanced reward shaping (Sharpe-based, drawdown penalty, consistency bonus)
- Realistic transaction costs and slippage
- Multi-stock support
- Detailed performance tracking

Author: Trading Bot Team
Date: October 28, 2025
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


class EnhancedTradingEnvironment(gym.Env):
    """
    Enhanced Trading Environment for RL agents

    Features:
    - Full 95-feature observations (from feature engineering)
    - Sharpe-ratio-based reward shaping
    - Realistic slippage and transaction costs
    - Position tracking with portfolio constraints
    - Risk management penalties
    - Consistency rewards for stable returns

    Action Space (9 discrete actions):
    - 0: Hold
    - 1-4: Buy 25%, 50%, 75%, 100% of available cash
    - 5-8: Sell 25%, 50%, 75%, 100% of position

    Observation Space:
    - Market data: 95 technical features (OHLCV + indicators)
    - Portfolio state: 15 features (cash, positions, P&L, risk metrics)
    - Total: lookback_window × 110 features
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1% commission
        slippage: float = 0.0005,  # 0.05% slippage
        max_position_size: float = 1.0,  # 100% of capital (can be all-in)
        lookback_window: int = 60,  # 60 days for observation
        reward_scaling: float = 1.0,
        risk_free_rate: float = 0.02  # 2% annual risk-free rate
    ):
        super(EnhancedTradingEnvironment, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1

        # Environment state
        self.current_step = 0
        self.cash = initial_capital
        self.shares = 0.0
        self.entry_price = 0.0
        self.portfolio_value = initial_capital
        self.max_portfolio_value = initial_capital

        # Performance tracking
        self.trades = []
        self.portfolio_history = []
        self.returns_history = []
        self.action_history = []

        # Rolling metrics (for reward calculation)
        self.recent_returns = deque(maxlen=20)  # Last 20 days
        self.recent_rewards = deque(maxlen=20)

        # Define action space: 9 discrete actions
        self.action_space = spaces.Discrete(9)

        # Define observation space
        # Market features (assume 95 from feature engineering) + Portfolio features (15)
        market_features = df.shape[1] if len(df) > 0 else 95
        portfolio_features = 15
        total_features = market_features + portfolio_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, total_features),
            dtype=np.float32
        )

        logger.info(f"Enhanced Trading Environment initialized:")
        logger.info(f"  Timesteps: {len(df)}")
        logger.info(f"  Market features: {market_features}")
        logger.info(f"  Portfolio features: {portfolio_features}")
        logger.info(f"  Total features: {total_features}")
        logger.info(f"  Observation shape: ({self.lookback_window}, {total_features})")

    def _get_current_price(self, price_type: str = 'close') -> float:
        """Get current price with error handling"""
        try:
            if price_type in self.df.columns:
                return self.df[price_type].iloc[self.current_step]
            elif 'close' in self.df.columns:
                return self.df['close'].iloc[self.current_step]
            else:
                # Fallback: return first column value
                return self.df.iloc[self.current_step, 0]
        except:
            return 1.0  # Emergency fallback

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state)

        Returns:
            np.ndarray: [lookback_window, total_features]
                - Market data: All features from feature engineering (95)
                - Portfolio state: 15 features
        """

        # Ensure we have enough history
        start_idx = max(0, self.current_step - self.lookback_window + 1)
        end_idx = self.current_step + 1

        # Get market data (all columns from feature engineering)
        market_data = self.df.iloc[start_idx:end_idx].values  # [time_steps, market_features]

        # Pad if needed (for beginning of episode)
        if len(market_data) < self.lookback_window:
            padding_rows = self.lookback_window - len(market_data)
            padding = np.repeat(market_data[0:1], padding_rows, axis=0)
            market_data = np.vstack([padding, market_data])

        # Create portfolio features for each time step
        portfolio_features = []
        current_price = self._get_current_price()

        for i in range(self.lookback_window):
            # Use prices from each historical step
            # Calculate the index we need (going backwards from most recent)
            history_idx = -(self.lookback_window - i)

            # Check if we have enough history for this index
            if len(self.portfolio_history) >= (self.lookback_window - i):
                hist_portfolio_value = self.portfolio_history[history_idx]
            else:
                # Not enough history yet, use initial capital
                hist_portfolio_value = self.initial_capital

            position_value = self.shares * current_price
            total_value = self.cash + position_value

            # Calculate features
            portfolio_feat = [
                # 1. Cash ratio
                self.cash / self.initial_capital,

                # 2. Position ratio
                position_value / self.initial_capital if position_value > 0 else 0,

                # 3. Total value ratio
                total_value / self.initial_capital,

                # 4. Total return
                (total_value - self.initial_capital) / self.initial_capital,

                # 5. Drawdown
                (self.max_portfolio_value - total_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0,

                # 6. Number of shares normalized
                self.shares / (self.initial_capital / current_price) if current_price > 0 else 0,

                # 7. Entry price ratio
                self.entry_price / current_price if current_price > 0 and self.entry_price > 0 else 0,

                # 8. Unrealized P&L
                ((current_price - self.entry_price) * self.shares) / self.initial_capital if self.entry_price > 0 else 0,

                # 9. Number of trades normalized
                len(self.trades) / 100,

                # 10. Recent return (last 20 days avg)
                np.mean(list(self.recent_returns)) if len(self.recent_returns) > 0 else 0,

                # 11. Recent volatility
                np.std(list(self.recent_returns)) if len(self.recent_returns) > 1 else 0,

                # 12. Sharpe ratio (approximation)
                self._calculate_current_sharpe(),

                # 13. Win rate
                self._calculate_win_rate(),

                # 14. Average trade return
                self._calculate_avg_trade_return(),

                # 15. Position holding time
                (self.current_step - self.trades[-1]['step']) / 100 if self.trades and self.shares > 0 else 0
            ]

            portfolio_features.append(portfolio_feat)

        portfolio_features = np.array(portfolio_features, dtype=np.float32)

        # Combine market data and portfolio features
        observation = np.concatenate([market_data, portfolio_features], axis=1)

        # Ensure correct shape
        assert observation.shape == (self.lookback_window, market_data.shape[1] + 15), \
            f"Observation shape mismatch: {observation.shape}"

        return observation.astype(np.float32)

    def _calculate_current_sharpe(self) -> float:
        """Calculate current Sharpe ratio"""
        if len(self.recent_returns) < 2:
            return 0.0

        returns_array = np.array(list(self.recent_returns))
        excess_returns = returns_array - self.daily_risk_free_rate
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)

        if std_excess == 0:
            return 0.0

        sharpe = (mean_excess / std_excess) * np.sqrt(252)  # Annualized
        return float(np.clip(sharpe, -5, 5))  # Clip to reasonable range

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from completed trades"""
        if len(self.trades) < 2:
            return 0.5  # Neutral

        # Find matching buy-sell pairs
        wins = 0
        total_pairs = 0

        i = 0
        while i < len(self.trades) - 1:
            if self.trades[i]['action'] == 'buy' and i + 1 < len(self.trades):
                # Find next sell
                for j in range(i + 1, len(self.trades)):
                    if self.trades[j]['action'] == 'sell':
                        buy_price = self.trades[i]['price']
                        sell_price = self.trades[j]['price']
                        if sell_price > buy_price:
                            wins += 1
                        total_pairs += 1
                        i = j
                        break
                else:
                    break
            i += 1

        return wins / total_pairs if total_pairs > 0 else 0.5

    def _calculate_avg_trade_return(self) -> float:
        """Calculate average trade return"""
        if len(self.trades) < 2:
            return 0.0

        returns = []
        i = 0
        while i < len(self.trades) - 1:
            if self.trades[i]['action'] == 'buy' and i + 1 < len(self.trades):
                for j in range(i + 1, len(self.trades)):
                    if self.trades[j]['action'] == 'sell':
                        buy_price = self.trades[i]['price']
                        sell_price = self.trades[j]['price']
                        trade_return = (sell_price - buy_price) / buy_price
                        returns.append(trade_return)
                        i = j
                        break
                else:
                    break
            i += 1

        return np.mean(returns) if returns else 0.0

    def _calculate_reward(self, action: int, prev_portfolio_value: float) -> float:
        """
        Calculate reward using balanced, SB3-compatible approach.

        FIXED: All reward components are now properly scaled and clipped to [-1, +1]
        to work well with Stable-Baselines3's VecNormalize wrapper.

        Reward components:
        1. Return-based reward (main signal)
        2. Sharpe-based reward (risk-adjusted)
        3. Drawdown penalty (risk management)
        4. Transaction cost penalty
        """

        # 1. Calculate return
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.recent_returns.append(portfolio_return)

        # Return reward - scale to [-1, +1] range (10x = +-10% return maps to +-1)
        return_reward = np.clip(portfolio_return * 10, -1.0, 1.0)

        # 2. Sharpe-based reward (risk-adjusted bonus)
        sharpe_reward = 0.0
        if len(self.recent_returns) >= 5:
            sharpe = self._calculate_current_sharpe()
            # Clip Sharpe contribution to [-0.5, +0.5]
            sharpe_reward = np.clip(sharpe * 0.3, -0.5, 0.5)

        # 3. Drawdown penalty (only penalize significant drawdowns)
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = 0.0
        if drawdown > 0.05:  # Only penalize >5% drawdown
            # Scale penalty: 5% DD = -0.15, 20% DD = -0.6 (capped at -1.0)
            drawdown_penalty = -np.clip(drawdown * 3, 0.0, 1.0)

        # 4. Transaction cost penalty (small but non-zero)
        transaction_penalty = -0.05 if action != 0 else 0.0

        # Total reward (components sum to roughly [-2, +2] range)
        total_reward = (
            return_reward +
            sharpe_reward +
            drawdown_penalty +
            transaction_penalty
        )

        # Final clipping to ensure bounded rewards for SB3 stability
        total_reward = np.clip(total_reward * self.reward_scaling, -2.0, 2.0)

        # Store for metrics
        self.recent_rewards.append(total_reward)

        return float(total_reward)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment

        Returns:
            observation, reward, terminated, truncated, info
        """

        # Store previous portfolio value
        prev_portfolio_value = self.portfolio_value

        # Get current price
        current_price = self._get_current_price()

        # Apply slippage (price gets worse for us)
        if action in [1, 2, 3, 4]:  # Buy actions
            execution_price = current_price * (1 + self.slippage)
        elif action in [5, 6, 7, 8]:  # Sell actions
            execution_price = current_price * (1 - self.slippage)
        else:
            execution_price = current_price

        # Execute action
        if action == 0:
            # Hold - do nothing
            pass

        elif action in [1, 2, 3, 4]:
            # Buy actions
            allocation_pct = [0.25, 0.50, 0.75, 1.00][action - 1]

            # Calculate maximum buy
            max_buy_value = min(
                self.cash * allocation_pct,
                self.initial_capital * self.max_position_size - (self.shares * execution_price)
            )

            if max_buy_value > 0:
                # Calculate shares to buy
                shares_to_buy = max_buy_value / execution_price
                gross_cost = shares_to_buy * execution_price
                transaction_fee = gross_cost * self.transaction_cost
                total_cost = gross_cost + transaction_fee

                if total_cost <= self.cash:
                    self.shares += shares_to_buy
                    self.cash -= total_cost

                    # Update entry price (weighted average)
                    if self.entry_price == 0:
                        self.entry_price = execution_price
                    else:
                        total_shares = self.shares
                        self.entry_price = (
                            (self.entry_price * (total_shares - shares_to_buy) +
                             execution_price * shares_to_buy) / total_shares
                        )

                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'shares': shares_to_buy,
                        'price': execution_price,
                        'cost': total_cost,
                        'portfolio_value': self.portfolio_value
                    })

        elif action in [5, 6, 7, 8]:
            # Sell actions
            allocation_pct = [0.25, 0.50, 0.75, 1.00][action - 5]
            shares_to_sell = self.shares * allocation_pct

            if shares_to_sell > 0:
                gross_proceeds = shares_to_sell * execution_price
                transaction_fee = gross_proceeds * self.transaction_cost
                net_proceeds = gross_proceeds - transaction_fee

                self.shares -= shares_to_sell
                self.cash += net_proceeds

                # Reset entry price if fully sold
                if self.shares < 1e-6:
                    self.entry_price = 0.0
                    self.shares = 0.0

                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': execution_price,
                    'proceeds': net_proceeds,
                    'portfolio_value': self.portfolio_value
                })

        # Update portfolio value
        position_value = self.shares * current_price
        self.portfolio_value = self.cash + position_value
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)

        # Track performance
        self.portfolio_history.append(self.portfolio_value)
        self.action_history.append(action)

        # Calculate reward
        reward = self._calculate_reward(action, prev_portfolio_value)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = False
        truncated = False

        if self.current_step >= len(self.df) - 1:
            truncated = True

        # Bankruptcy check
        if self.portfolio_value < self.initial_capital * 0.1:  # Lost 90%
            terminated = True
            reward -= 10  # Large penalty for bankruptcy

        # Get next observation
        observation = self._get_observation()

        # Info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'shares': self.shares,
            'position_value': position_value,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'num_trades': len(self.trades),
            'sharpe_ratio': self._calculate_current_sharpe(),
            'win_rate': self._calculate_win_rate(),
            'max_drawdown': (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""

        super().reset(seed=seed)

        self.current_step = self.lookback_window
        self.cash = self.initial_capital
        self.shares = 0.0
        self.entry_price = 0.0
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital

        self.trades = []
        self.portfolio_history = [self.initial_capital]
        self.returns_history = []
        self.action_history = []
        self.recent_returns.clear()
        self.recent_rewards.clear()

        observation = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'shares': self.shares
        }

        return observation, info

    def render(self, mode='human'):
        """Render the environment"""

        if mode == 'human':
            current_price = self._get_current_price()
            position_value = self.shares * current_price

            print(f"Step: {self.current_step}/{len(self.df)}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Positions: {self.shares:.2f} shares (${position_value:,.2f})")
            print(f"Total Return: {(self.portfolio_value - self.initial_capital) / self.initial_capital * 100:.2f}%")
            print(f"Sharpe Ratio: {self._calculate_current_sharpe():.2f}")
            print(f"Trades: {len(self.trades)}")
            print("-" * 60)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""

        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital

        # Sharpe ratio
        sharpe_ratio = self._calculate_current_sharpe()

        # Sortino ratio (downside deviation)
        if len(self.recent_returns) > 0:
            returns_array = np.array(list(self.recent_returns))
            downside_returns = returns_array[returns_array < 0]

            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino_ratio = (np.mean(returns_array) - self.daily_risk_free_rate) / (downside_std + 1e-8) * np.sqrt(252)
            else:
                sortino_ratio = sharpe_ratio
        else:
            sortino_ratio = 0.0

        # Max drawdown
        if len(self.portfolio_history) > 0:
            portfolio_array = np.array(self.portfolio_history)
            running_max = np.maximum.accumulate(portfolio_array)
            drawdowns = (portfolio_array - running_max) / running_max
            max_drawdown = np.min(drawdowns)
        else:
            max_drawdown = 0.0

        # Win rate
        win_rate = self._calculate_win_rate()

        # Average trade return
        avg_trade_return = self._calculate_avg_trade_return()

        # Calmar ratio (return / max drawdown)
        calmar_ratio = total_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'num_trades': len(self.trades),
            'final_portfolio_value': self.portfolio_value,
            'avg_reward': np.mean(list(self.recent_rewards)) if len(self.recent_rewards) > 0 else 0.0
        }
