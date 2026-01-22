"""
Trading Environment for Reinforcement Learning

Custom Gym environment for training RL agents on trading
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Custom Trading Environment for RL agents
    
    State Space:
    - Market data (OHLCV, technical indicators)
    - Portfolio state (positions, cash, P&L)
    - Risk metrics (drawdown, Sharpe ratio)
    - Market regime indicators
    
    Action Space:
    - 0: Hold
    - 1: Buy (allocate % of capital)
    - 2: Sell (liquidate % of position)
    - 3: Close all positions
    
    Reward:
    - P&L + risk-adjusted metrics
    - Penalize large drawdowns
    - Reward consistent returns
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1%
        max_position_size: float = 0.2,  # 20% of capital
        lookback_window: int = 60
    ):
        super(TradingEnvironment, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        
        # Environment state
        self.current_step = 0
        self.cash = initial_capital
        self.positions = 0.0
        self.portfolio_value = initial_capital
        self.max_portfolio_value = initial_capital
        
        # Performance tracking
        self.trades = []
        self.portfolio_history = []
        self.returns_history = []
        
        # Define action space
        # 0: Hold, 1: Buy 25%, 2: Buy 50%, 3: Buy 75%, 4: Buy 100%
        # 5: Sell 25%, 6: Sell 50%, 7: Sell 75%, 8: Sell 100%
        self.action_space = spaces.Discrete(9)
        
        # Define observation space
        # Market features + portfolio features
        market_features = 50  # OHLCV + technical indicators
        portfolio_features = 10  # Cash, positions, P&L, etc.
        total_features = market_features + portfolio_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, total_features),
            dtype=np.float32
        )
        
        logger.info(f" Trading Environment initialized with {len(df)} timesteps")
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)"""
        
        # Ensure we have enough history
        start_idx = max(0, self.current_step - self.lookback_window + 1)
        end_idx = self.current_step + 1
        
        # Get market data
        market_data = self.df.iloc[start_idx:end_idx].copy()
        
        # Calculate features
        features = []
        
        for idx, row in market_data.iterrows():
            # Price features (normalized)
            price_features = [
                row['close'] / self.df['close'].mean() if 'close' in row else 0,
                row['open'] / self.df['open'].mean() if 'open' in row else 0,
                row['high'] / self.df['high'].mean() if 'high' in row else 0,
                row['low'] / self.df['low'].mean() if 'low' in row else 0,
                row['volume'] / self.df['volume'].mean() if 'volume' in row else 0,
            ]
            
            # Technical indicators (if available)
            technical_features = [
                row.get('rsi', 50) / 100,
                row.get('macd', 0) / 10,
                row.get('macd_signal', 0) / 10,
                row.get('bb_width', 0) / 10,
                row.get('atr', 0) / 10,
            ]
            
            # Placeholder for additional features (to reach 50 market features)
            padding = [0.0] * (40)
            
            # Portfolio features
            current_price = row['close'] if 'close' in row else self.df['close'].iloc[idx]
            position_value = self.positions * current_price if current_price > 0 else 0
            total_value = self.cash + position_value
            
            portfolio_features = [
                self.cash / self.initial_capital,
                self.positions / (self.initial_capital / current_price) if current_price > 0 else 0,
                position_value / self.initial_capital,
                total_value / self.initial_capital,
                (total_value - self.initial_capital) / self.initial_capital,  # Total return
                (self.max_portfolio_value - total_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0,  # Drawdown
                len(self.trades) / 100,  # Number of trades
                0.0,  # Placeholder: Sharpe ratio
                0.0,  # Placeholder: Win rate
                0.0,  # Placeholder: Risk-adjusted return
            ]
            
            # Combine all features
            all_features = price_features + technical_features + padding + portfolio_features
            features.append(all_features)
        
        # Pad if necessary
        while len(features) < self.lookback_window:
            features.insert(0, features[0] if features else [0.0] * (50 + 10))
        
        observation = np.array(features, dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, action: int, prev_portfolio_value: float) -> float:
        """
        Calculate reward for the agent
        
        Reward components:
        - P&L change
        - Risk-adjusted metrics
        - Drawdown penalty
        - Trade frequency penalty
        """
        
        # Calculate P&L change
        pnl_change = self.portfolio_value - prev_portfolio_value
        pnl_reward = pnl_change / self.initial_capital * 100  # Normalized P&L
        
        # Calculate drawdown penalty
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = -drawdown * 10 if drawdown > 0.05 else 0  # Penalize >5% drawdown
        
        # Trade frequency penalty (discourage overtrading)
        trade_penalty = -0.1 if action != 0 else 0
        
        # Calculate total reward
        total_reward = pnl_reward + drawdown_penalty + trade_penalty
        
        return total_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        
        # Store previous portfolio value
        prev_portfolio_value = self.portfolio_value
        
        # Get current price
        current_price = self.df['close'].iloc[self.current_step]
        
        # Execute action
        if action == 0:
            # Hold - do nothing
            pass
        
        elif action in [1, 2, 3, 4]:
            # Buy actions
            allocation_pct = [0.25, 0.50, 0.75, 1.00][action - 1]
            max_buy = min(
                self.cash * allocation_pct,
                self.initial_capital * self.max_position_size
            )
            
            if max_buy > 0:
                # Calculate shares to buy
                shares_to_buy = max_buy / current_price
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                
                if cost <= self.cash:
                    self.positions += shares_to_buy
                    self.cash -= cost
                    
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost
                    })
        
        elif action in [5, 6, 7, 8]:
            # Sell actions
            allocation_pct = [0.25, 0.50, 0.75, 1.00][action - 5]
            shares_to_sell = self.positions * allocation_pct
            
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.positions -= shares_to_sell
                self.cash += proceeds
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'proceeds': proceeds
                })
        
        # Update portfolio value
        position_value = self.positions * current_price
        self.portfolio_value = self.cash + position_value
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        
        # Track performance
        self.portfolio_history.append(self.portfolio_value)
        if len(self.portfolio_history) > 1:
            returns = (self.portfolio_value - self.portfolio_history[-2]) / self.portfolio_history[-2]
            self.returns_history.append(returns)
        
        # Calculate reward
        reward = self._calculate_reward(action, prev_portfolio_value)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        # Get observation
        observation = self._get_observation()
        
        # Info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'num_trades': len(self.trades)
        }
        
        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        
        self.current_step = self.lookback_window
        self.cash = self.initial_capital
        self.positions = 0.0
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        
        self.trades = []
        self.portfolio_history = [self.initial_capital]
        self.returns_history = []
        
        return self._get_observation()
    
    def render(self, mode='human'):
        """Render the environment"""
        
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Positions: {self.positions:.4f}")
            print(f"Total Return: {(self.portfolio_value - self.initial_capital) / self.initial_capital * 100:.2f}%")
            print(f"Trades: {len(self.trades)}")
            print("-" * 50)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe ratio
        if len(self.returns_history) > 0:
            returns_array = np.array(self.returns_history)
            sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-10) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        portfolio_array = np.array(self.portfolio_history)
        running_max = np.maximum.accumulate(portfolio_array)
        drawdowns = (portfolio_array - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Calculate win rate
        if len(self.trades) > 0:
            profitable_trades = sum(1 for trade in self.trades if trade.get('proceeds', 0) > trade.get('cost', 0))
            win_rate = profitable_trades / len(self.trades)
        else:
            win_rate = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'final_portfolio_value': self.portfolio_value
        }

