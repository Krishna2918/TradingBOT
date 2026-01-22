"""
Reinforcement Learning Module

This module implements advanced reinforcement learning algorithms for
autonomous trading decision making, including Q-Learning, Deep Q-Networks (DQN),
and Policy Gradient methods.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import random
from collections import deque
import pickle
import os

from src.config.mode_manager import get_current_mode

logger = logging.getLogger(__name__)

@dataclass
class TradingState:
    """Represents the current trading state."""
    portfolio_value: float
    cash_balance: float
    position_count: int
    market_features: Dict[str, float]
    time_features: Dict[str, float]
    risk_metrics: Dict[str, float]

@dataclass
class TradingAction:
    """Represents a trading action."""
    action_type: str  # "BUY", "SELL", "HOLD"
    symbol: str
    quantity: int
    confidence: float
    reasoning: str

@dataclass
class TradingReward:
    """Represents a trading reward."""
    immediate_reward: float
    portfolio_change: float
    risk_penalty: float
    time_penalty: float
    total_reward: float

class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, state):
        """Forward pass through the network."""
        features = self.feature_extractor(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class TradingEnvironment:
    """Trading environment for reinforcement learning."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        self.current_step = 0
        self.max_steps = 1000
        self.initial_balance = 10000.0
        self.current_balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.positions = {}
        self.transaction_cost = 0.001  # 0.1% transaction cost
        
        # State and action spaces
        self.state_size = 50  # Market features + portfolio state
        self.action_size = 3  # BUY, SELL, HOLD
        
        # Available symbols
        self.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        logger.info(f"Trading environment initialized for {mode} mode")
    
    def reset(self) -> TradingState:
        """Reset the environment to initial state."""
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.positions = {}
        
        return self._get_state()
    
    def step(self, action: TradingAction) -> Tuple[TradingState, TradingReward, bool, Dict[str, Any]]:
        """Execute an action and return next state, reward, done, info."""
        self.current_step += 1
        
        # Execute action
        reward = self._execute_action(action)
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Get next state
        next_state = self._get_state()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or self.current_balance <= 0
        
        # Additional info
        info = {
            "portfolio_value": self.portfolio_value,
            "cash_balance": self.current_balance,
            "position_count": len(self.positions),
            "step": self.current_step
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action: TradingAction) -> TradingReward:
        """Execute a trading action and calculate reward."""
        immediate_reward = 0.0
        portfolio_change = 0.0
        risk_penalty = 0.0
        time_penalty = -0.01  # Small penalty for each step
        
        if action.action_type == "BUY":
            # Simulate buying
            price = self._get_current_price(action.symbol)
            cost = price * action.quantity
            transaction_cost = cost * self.transaction_cost
            
            if self.current_balance >= cost + transaction_cost:
                self.current_balance -= (cost + transaction_cost)
                
                if action.symbol in self.positions:
                    self.positions[action.symbol] += action.quantity
                else:
                    self.positions[action.symbol] = action.quantity
                
                immediate_reward = -transaction_cost  # Small penalty for transaction cost
            else:
                immediate_reward = -1.0  # Penalty for insufficient funds
        
        elif action.action_type == "SELL":
            # Simulate selling
            if action.symbol in self.positions and self.positions[action.symbol] >= action.quantity:
                price = self._get_current_price(action.symbol)
                proceeds = price * action.quantity
                transaction_cost = proceeds * self.transaction_cost
                
                self.current_balance += (proceeds - transaction_cost)
                self.positions[action.symbol] -= action.quantity
                
                if self.positions[action.symbol] == 0:
                    del self.positions[action.symbol]
                
                immediate_reward = -transaction_cost  # Small penalty for transaction cost
            else:
                immediate_reward = -1.0  # Penalty for invalid sell
        
        elif action.action_type == "HOLD":
            # No action, small reward for patience
            immediate_reward = 0.01
        
        # Calculate portfolio change
        old_portfolio_value = self.portfolio_value
        self._update_portfolio_value()
        portfolio_change = (self.portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Risk penalty for high volatility
        if len(self.positions) > 5:  # Too many positions
            risk_penalty = -0.1
        
        # Calculate total reward
        total_reward = immediate_reward + portfolio_change * 10 + risk_penalty + time_penalty
        
        return TradingReward(
            immediate_reward=immediate_reward,
            portfolio_change=portfolio_change,
            risk_penalty=risk_penalty,
            time_penalty=time_penalty,
            total_reward=total_reward
        )
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol (simulated)."""
        # In production, this would fetch real market data
        base_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0,
            "AMZN": 3000.0,
            "TSLA": 800.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        # Add some random variation
        variation = np.random.normal(0, 0.02)  # 2% standard deviation
        return base_price * (1 + variation)
    
    def _update_portfolio_value(self):
        """Update total portfolio value."""
        total_value = self.current_balance
        
        for symbol, quantity in self.positions.items():
            price = self._get_current_price(symbol)
            total_value += price * quantity
        
        self.portfolio_value = total_value
    
    def _get_state(self) -> TradingState:
        """Get current state representation."""
        # Market features (simulated)
        market_features = {
            "market_trend": np.random.uniform(-1, 1),
            "volatility": np.random.uniform(0, 0.1),
            "volume": np.random.uniform(0.5, 2.0),
            "rsi": np.random.uniform(0, 100),
            "macd": np.random.uniform(-2, 2),
            "sma_20": np.random.uniform(0.95, 1.05),
            "sma_50": np.random.uniform(0.95, 1.05),
            "bollinger_position": np.random.uniform(0, 1),
            "sentiment": np.random.uniform(-1, 1),
            "news_impact": np.random.uniform(-0.5, 0.5)
        }
        
        # Time features
        time_features = {
            "hour": self.current_step % 24 / 24.0,
            "day_of_week": (self.current_step // 24) % 7 / 7.0,
            "step_progress": self.current_step / self.max_steps
        }
        
        # Risk metrics
        risk_metrics = {
            "portfolio_concentration": len(self.positions) / 10.0,
            "cash_ratio": self.current_balance / self.portfolio_value,
            "leverage": 1.0 - (self.current_balance / self.portfolio_value),
            "drawdown": max(0, (self.initial_balance - self.portfolio_value) / self.initial_balance)
        }
        
        return TradingState(
            portfolio_value=self.portfolio_value,
            cash_balance=self.current_balance,
            position_count=len(self.positions),
            market_features=market_features,
            time_features=time_features,
            risk_metrics=risk_metrics
        )
    
    def state_to_vector(self, state: TradingState) -> np.ndarray:
        """Convert state to vector representation."""
        vector = []
        
        # Portfolio features
        vector.extend([
            state.portfolio_value / self.initial_balance,
            state.cash_balance / self.initial_balance,
            state.position_count / 10.0
        ])
        
        # Market features
        vector.extend(list(state.market_features.values()))
        
        # Time features
        vector.extend(list(state.time_features.values()))
        
        # Risk metrics
        vector.extend(list(state.risk_metrics.values()))
        
        # Pad or truncate to state_size
        while len(vector) < self.state_size:
            vector.append(0.0)
        
        return np.array(vector[:self.state_size], dtype=np.float32)

class DQNAgent:
    """Deep Q-Network agent for trading."""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32
        self.target_update = 10
        
        # Experience replay
        self.memory = ReplayBuffer()
        
        # Training tracking
        self.training_step = 0
        
        logger.info("DQN Agent initialized")
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose an action using epsilon-greedy policy."""
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
            logger.info(f"Model loaded from {filepath}")
        else:
            logger.warning(f"Model file {filepath} not found")

class PolicyGradientAgent:
    """Policy gradient agent for trading."""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Neural network
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Training tracking
        self.training_step = 0
        
        logger.info("Policy Gradient Agent initialized")
    
    def act(self, state: np.ndarray) -> int:
        """Choose an action using the policy network."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy_network(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            return action
    
    def train_step(self, states, actions, rewards):
        """Train the policy network."""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        
        # Get action probabilities
        action_probs = self.policy_network(states)
        
        # Calculate log probabilities
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        
        # Calculate policy loss (negative log likelihood weighted by rewards)
        policy_loss = -(log_probs * rewards).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        return policy_loss.item()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, filepath)
        logger.info(f"Policy model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_step = checkpoint['training_step']
            logger.info(f"Policy model loaded from {filepath}")
        else:
            logger.warning(f"Policy model file {filepath} not found")

class ReinforcementLearningTrainer:
    """Trainer for reinforcement learning agents."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        self.environment = TradingEnvironment(mode)
        self.dqn_agent = DQNAgent(self.environment.state_size, self.environment.action_size)
        self.policy_agent = PolicyGradientAgent(self.environment.state_size, self.environment.action_size)
        
        # Training configuration
        self.episodes = 1000
        self.max_steps_per_episode = 1000
        
        # Results tracking
        self.training_results = []
        
        logger.info(f"RL Trainer initialized for {mode} mode")
    
    def train_dqn(self, episodes: int = None) -> Dict[str, Any]:
        """Train the DQN agent."""
        if episodes is None:
            episodes = self.episodes
        
        logger.info(f"Starting DQN training for {episodes} episodes")
        
        episode_rewards = []
        episode_losses = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            state_vector = self.environment.state_to_vector(state)
            
            total_reward = 0
            total_loss = 0
            steps = 0
            
            for step in range(self.max_steps_per_episode):
                # Choose action
                action_idx = self.dqn_agent.act(state_vector, training=True)
                action = self._action_idx_to_action(action_idx)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                next_state_vector = self.environment.state_to_vector(next_state)
                
                # Store experience
                self.dqn_agent.remember(state_vector, action_idx, reward.total_reward, next_state_vector, done)
                
                # Train
                if len(self.dqn_agent.memory) > self.dqn_agent.batch_size:
                    loss = self.dqn_agent.replay()
                    total_loss += loss
                
                total_reward += reward.total_reward
                state_vector = next_state_vector
                steps += 1
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_losses.append(total_loss / max(steps, 1))
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_loss = np.mean(episode_losses[-100:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")
        
        # Save trained model
        model_path = f"models/dqn_model_{self.mode}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.dqn_agent.save_model(model_path)
        
        results = {
            "episodes": episodes,
            "final_avg_reward": np.mean(episode_rewards[-100:]),
            "final_avg_loss": np.mean(episode_losses[-100:]),
            "model_path": model_path
        }
        
        logger.info(f"DQN training completed. Final avg reward: {results['final_avg_reward']:.2f}")
        return results
    
    def train_policy_gradient(self, episodes: int = None) -> Dict[str, Any]:
        """Train the policy gradient agent."""
        if episodes is None:
            episodes = self.episodes
        
        logger.info(f"Starting Policy Gradient training for {episodes} episodes")
        
        episode_rewards = []
        episode_losses = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            state_vector = self.environment.state_to_vector(state)
            
            states = []
            actions = []
            rewards = []
            
            total_reward = 0
            
            for step in range(self.max_steps_per_episode):
                # Choose action
                action_idx = self.policy_agent.act(state_vector)
                action = self._action_idx_to_action(action_idx)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                next_state_vector = self.environment.state_to_vector(next_state)
                
                # Store experience
                states.append(state_vector)
                actions.append(action_idx)
                rewards.append(reward.total_reward)
                
                total_reward += reward.total_reward
                state_vector = next_state_vector
                
                if done:
                    break
            
            # Train on episode
            if states:
                # Calculate discounted rewards
                discounted_rewards = self._calculate_discounted_rewards(rewards)
                
                # Train
                loss = self.policy_agent.train_step(states, actions, discounted_rewards)
                episode_losses.append(loss)
            
            episode_rewards.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_loss = np.mean(episode_losses[-100:]) if episode_losses else 0
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")
        
        # Save trained model
        model_path = f"models/policy_model_{self.mode}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.policy_agent.save_model(model_path)
        
        results = {
            "episodes": episodes,
            "final_avg_reward": np.mean(episode_rewards[-100:]),
            "final_avg_loss": np.mean(episode_losses[-100:]) if episode_losses else 0,
            "model_path": model_path
        }
        
        logger.info(f"Policy Gradient training completed. Final avg reward: {results['final_avg_reward']:.2f}")
        return results
    
    def _action_idx_to_action(self, action_idx: int) -> TradingAction:
        """Convert action index to TradingAction."""
        action_types = ["HOLD", "BUY", "SELL"]
        action_type = action_types[action_idx]
        
        # For simplicity, use first symbol
        symbol = self.environment.symbols[0]
        quantity = 10  # Fixed quantity for now
        
        return TradingAction(
            action_type=action_type,
            symbol=symbol,
            quantity=quantity,
            confidence=0.8,
            reasoning=f"RL agent chose {action_type}"
        )
    
    def _calculate_discounted_rewards(self, rewards: List[float], gamma: float = 0.95) -> List[float]:
        """Calculate discounted rewards for policy gradient."""
        discounted = []
        running_total = 0
        
        for reward in reversed(rewards):
            running_total = reward + gamma * running_total
            discounted.insert(0, running_total)
        
        # Normalize rewards
        if discounted:
            mean_reward = np.mean(discounted)
            std_reward = np.std(discounted)
            if std_reward > 0:
                discounted = [(r - mean_reward) / std_reward for r in discounted]
        
        return discounted
    
    def evaluate_agent(self, agent_type: str = "dqn", episodes: int = 10) -> Dict[str, Any]:
        """Evaluate a trained agent."""
        logger.info(f"Evaluating {agent_type} agent for {episodes} episodes")
        
        if agent_type == "dqn":
            agent = self.dqn_agent
        elif agent_type == "policy":
            agent = self.policy_agent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        episode_rewards = []
        episode_returns = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            state_vector = self.environment.state_to_vector(state)
            
            total_reward = 0
            initial_value = self.environment.portfolio_value
            
            for step in range(self.max_steps_per_episode):
                # Choose action (no exploration)
                action_idx = agent.act(state_vector, training=False)
                action = self._action_idx_to_action(action_idx)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                next_state_vector = self.environment.state_to_vector(next_state)
                
                total_reward += reward.total_reward
                state_vector = next_state_vector
                
                if done:
                    break
            
            final_value = self.environment.portfolio_value
            episode_return = (final_value - initial_value) / initial_value
            
            episode_rewards.append(total_reward)
            episode_returns.append(episode_return)
        
        results = {
            "episodes": episodes,
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "avg_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "max_return": np.max(episode_returns),
            "min_return": np.min(episode_returns)
        }
        
        logger.info(f"Evaluation completed. Avg return: {results['avg_return']:.2%}")
        return results

# Global RL trainer instance
_rl_trainer: Optional[ReinforcementLearningTrainer] = None

def get_rl_trainer(mode: str = None) -> ReinforcementLearningTrainer:
    """Get the global RL trainer instance."""
    global _rl_trainer
    if _rl_trainer is None:
        if mode is None:
            mode = get_current_mode()
        _rl_trainer = ReinforcementLearningTrainer(mode)
    return _rl_trainer

def train_dqn_agent(episodes: int = 1000, mode: str = None) -> Dict[str, Any]:
    """Train DQN agent."""
    return get_rl_trainer(mode).train_dqn(episodes)

def train_policy_agent(episodes: int = 1000, mode: str = None) -> Dict[str, Any]:
    """Train policy gradient agent."""
    return get_rl_trainer(mode).train_policy_gradient(episodes)

def evaluate_rl_agent(agent_type: str = "dqn", episodes: int = 10, mode: str = None) -> Dict[str, Any]:
    """Evaluate RL agent."""
    return get_rl_trainer(mode).evaluate_agent(agent_type, episodes)
