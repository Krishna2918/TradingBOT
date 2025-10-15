"""
Trading Environment for Reinforcement Learning

This module implements a trading environment that simulates market conditions
and provides the interface for reinforcement learning agents to interact with.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings

# Handle optional gymnasium import
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    gym = None
    spaces = None
    GYM_AVAILABLE = False
    warnings.warn("Gymnasium not installed. RL features will be limited.")

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env if GYM_AVAILABLE else object):
    """
    Trading environment for reinforcement learning.
    
    This environment simulates a trading scenario where an agent can
    buy, sell, or hold positions based on market data.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position_size: float = 1.0,
        lookback_window: int = 20,
        features: Optional[List[str]] = None
    ):
        """
        Initialize trading environment.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            initial_balance: Starting balance for trading
            transaction_cost: Cost per transaction (as fraction)
            max_position_size: Maximum position size (as fraction of balance)
            lookback_window: Number of historical observations to include in state
            features: List of feature columns to use
        """
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium required for RL. Install: pip install gymnasium")
        
        super(TradingEnvironment, self).__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        
        # Set up features
        if features is None:
            self.features = ['open', 'high', 'low', 'close', 'volume']
        else:
            self.features = features
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Prepare data
        self._prepare_data()
        
        # Set up action and observation spaces
        self._setup_spaces()
        
        # Initialize state
        self.reset()
        
        logger.info(f"Initialized Trading Environment with {len(self.data)} observations")
    
    def _prepare_data(self) -> None:
        """Prepare data for the environment."""
        # Normalize features
        for feature in self.features:
            if feature in self.data.columns:
                # Min-max normalization
                min_val = self.data[feature].min()
                max_val = self.data[feature].max()
                if max_val > min_val:
                    self.data[f'{feature}_norm'] = (self.data[feature] - min_val) / (max_val - min_val)
                else:
                    self.data[f'{feature}_norm'] = 0.0
        
        # Calculate returns
        self.data['returns'] = self.data['close'].pct_change()
        self.data['returns_norm'] = (self.data['returns'] - self.data['returns'].mean()) / self.data['returns'].std()
        
        # Fill NaN values
        self.data = self.data.fillna(method='ffill').fillna(0)
        
        logger.info("Data preparation completed")
    
    def _setup_spaces(self) -> None:
        """Set up action and observation spaces."""
        # Action space: [0: Hold, 1: Buy, 2: Sell]
        self.action_space = spaces.Discrete(3)
        
        # Observation space: normalized features + portfolio state
        n_features = len(self.features) + 2  # +2 for returns and portfolio state
        obs_dim = n_features * self.lookback_window + 3  # +3 for balance, position, unrealized_pnl
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Observation space: {self.observation_space}")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0  # Number of shares held
        self.position_value = 0.0  # Value of current position
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.total_transaction_costs = 0.0
        
        # Initialize portfolio history
        self.portfolio_history = []
        self.trade_history = []
        
        # Get initial observation
        observation = self._get_observation()
        
        logger.info("Environment reset")
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0: Hold, 1: Buy, 2: Sell)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Get current market data
        current_price = self.data.iloc[self.current_step]['close']
        current_volume = self.data.iloc[self.current_step]['volume']
        
        # Execute action
        reward = self._execute_action(action, current_price, current_volume)
        
        # Update portfolio
        self._update_portfolio(current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Create info dictionary
        info = {
            'balance': self.balance,
            'position': self.position,
            'position_value': self.position_value,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'total_transaction_costs': self.total_transaction_costs,
            'current_price': current_price,
            'portfolio_value': self.balance + self.position_value
        }
        
        return observation, reward, done, info
    
    def _execute_action(self, action: int, price: float, volume: float) -> float:
        """
        Execute the given action and return reward.
        
        Args:
            action: Action to execute
            price: Current market price
            volume: Current market volume
            
        Returns:
            Reward for the action
        """
        reward = 0.0
        
        if action == 1:  # Buy
            # Calculate maximum shares we can buy
            max_shares = (self.balance * self.max_position_size) / price
            shares_to_buy = min(max_shares, volume * 0.1)  # Limit to 10% of volume
            
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                transaction_cost = cost * self.transaction_cost
                total_cost = cost + transaction_cost
                
                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.position += shares_to_buy
                    self.total_trades += 1
                    self.total_transaction_costs += transaction_cost
                    
                    # Reward for taking action (encourage exploration)
                    reward += 0.01
        
        elif action == 2:  # Sell
            if self.position > 0:
                # Sell all position
                proceeds = self.position * price
                transaction_cost = proceeds * self.transaction_cost
                net_proceeds = proceeds - transaction_cost
                
                self.balance += net_proceeds
                self.position = 0.0
                self.total_trades += 1
                self.total_transaction_costs += transaction_cost
                
                # Reward for taking action (encourage exploration)
                reward += 0.01
        
        # Calculate portfolio return as reward
        portfolio_value = self.balance + self.position * price
        if hasattr(self, 'previous_portfolio_value'):
            portfolio_return = (portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
            reward += portfolio_return * 10  # Scale reward
        
        self.previous_portfolio_value = portfolio_value
        
        return reward
    
    def _update_portfolio(self, price: float) -> None:
        """Update portfolio values."""
        self.position_value = self.position * price
        self.unrealized_pnl = self.position_value - (self.position * self._get_average_buy_price())
        
        # Record portfolio state
        self.portfolio_history.append({
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'position_value': self.position_value,
            'unrealized_pnl': self.unrealized_pnl,
            'price': price,
            'portfolio_value': self.balance + self.position_value
        })
    
    def _get_average_buy_price(self) -> float:
        """Get average buy price for current position."""
        if self.position == 0:
            return 0.0
        
        # Simplified: use current price as proxy
        # In a more sophisticated implementation, you'd track actual buy prices
        return self.data.iloc[self.current_step]['close']
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Current observation vector
        """
        # Get historical features
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step + 1
        
        observation = []
        
        # Add normalized features
        for feature in self.features:
            if f'{feature}_norm' in self.data.columns:
                feature_data = self.data[f'{feature}_norm'].iloc[start_idx:end_idx].values
                # Pad with zeros if not enough history
                if len(feature_data) < self.lookback_window:
                    feature_data = np.pad(feature_data, (self.lookback_window - len(feature_data), 0), 'constant')
                observation.extend(feature_data)
        
        # Add returns
        returns_data = self.data['returns_norm'].iloc[start_idx:end_idx].values
        if len(returns_data) < self.lookback_window:
            returns_data = np.pad(returns_data, (self.lookback_window - len(returns_data), 0), 'constant')
        observation.extend(returns_data)
        
        # Add portfolio state
        observation.extend([
            self.balance / self.initial_balance,  # Normalized balance
            self.position / (self.initial_balance / self.data.iloc[self.current_step]['close']),  # Normalized position
            self.unrealized_pnl / self.initial_balance  # Normalized unrealized PnL
        ])
        
        return np.array(observation, dtype=np.float32)
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            portfolio_value = self.balance + self.position_value
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Position: {self.position:.2f} shares")
            print(f"Position Value: ${self.position_value:.2f}")
            print(f"Unrealized PnL: ${self.unrealized_pnl:.2f}")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Total Transaction Costs: ${self.total_transaction_costs:.2f}")
            print("-" * 50)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get portfolio history.
        
        Returns:
            DataFrame with portfolio history
        """
        return pd.DataFrame(self.portfolio_history)
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Returns:
            List of trade records
        """
        return self.trade_history.copy()
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.portfolio_history:
            return {}
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Calculate metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] - self.initial_balance) / self.initial_balance
        sharpe_ratio = portfolio_df['returns'].mean() / portfolio_df['returns'].std() * np.sqrt(252) if portfolio_df['returns'].std() > 0 else 0
        max_drawdown = (portfolio_df['portfolio_value'].cummax() - portfolio_df['portfolio_value']).max() / portfolio_df['portfolio_value'].cummax().max()
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'total_transaction_costs': self.total_transaction_costs,
            'final_portfolio_value': portfolio_df['portfolio_value'].iloc[-1]
        }
        
        return metrics


class MarketSimulator:
    """
    Market simulator for generating synthetic market data.
    
    This class can generate realistic market data for training
    reinforcement learning agents.
    """
    
    def __init__(
        self,
        n_periods: int = 1000,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0001,
        volume_base: int = 1000000
    ):
        """
        Initialize market simulator.
        
        Args:
            n_periods: Number of periods to simulate
            initial_price: Initial price
            volatility: Daily volatility
            drift: Daily drift
            volume_base: Base volume level
        """
        self.n_periods = n_periods
        self.initial_price = initial_price
        self.volatility = volatility
        self.drift = drift
        self.volume_base = volume_base
        
        logger.info(f"Initialized Market Simulator: {n_periods} periods")
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic market data.
        
        Returns:
            DataFrame with OHLCV data
        """
        # Generate price series using geometric Brownian motion
        dt = 1.0 / 252  # Daily time step
        returns = np.random.normal(self.drift * dt, self.volatility * np.sqrt(dt), self.n_periods)
        
        # Calculate prices
        prices = [self.initial_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[1:])  # Remove initial price
        
        # Generate OHLC data
        data = []
        for i, close in enumerate(prices):
            # Generate open (close to previous close)
            if i == 0:
                open_price = self.initial_price
            else:
                open_price = prices[i-1] * (1 + np.random.normal(0, 0.001))
            
            # Generate high and low
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.005)))
            
            # Generate volume
            volume = int(self.volume_base * (1 + np.random.normal(0, 0.3)))
            volume = max(volume, 1000)  # Minimum volume
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        
        logger.info(f"Generated {len(df)} periods of market data")
        
        return df
    
    def generate_trending_data(
        self,
        trend_direction: str = 'up',
        trend_strength: float = 0.001
    ) -> pd.DataFrame:
        """
        Generate trending market data.
        
        Args:
            trend_direction: 'up' or 'down'
            trend_strength: Strength of the trend
            
        Returns:
            DataFrame with trending OHLCV data
        """
        # Adjust drift based on trend
        if trend_direction == 'up':
            adjusted_drift = self.drift + trend_strength
        else:
            adjusted_drift = self.drift - trend_strength
        
        # Create temporary simulator with adjusted drift
        temp_simulator = MarketSimulator(
            n_periods=self.n_periods,
            initial_price=self.initial_price,
            volatility=self.volatility,
            drift=adjusted_drift,
            volume_base=self.volume_base
        )
        
        return temp_simulator.generate_data()
    
    def generate_volatile_data(
        self,
        volatility_multiplier: float = 2.0
    ) -> pd.DataFrame:
        """
        Generate high volatility market data.
        
        Args:
            volatility_multiplier: Multiplier for volatility
            
        Returns:
            DataFrame with volatile OHLCV data
        """
        # Create temporary simulator with increased volatility
        temp_simulator = MarketSimulator(
            n_periods=self.n_periods,
            initial_price=self.initial_price,
            volatility=self.volatility * volatility_multiplier,
            drift=self.drift,
            volume_base=self.volume_base
        )
        
        return temp_simulator.generate_data()

