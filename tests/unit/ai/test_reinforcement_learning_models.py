"""
Unit tests for reinforcement learning models.

This module contains comprehensive tests for all reinforcement learning models
including DQN, PPO, A3C, SAC, and the trading environment.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from ai.reinforcement_learning.trading_environment import TradingEnvironment, MarketSimulator
from ai.reinforcement_learning.dqn_agent import DQNAgent, DQNTrainer, DQNNetwork, ReplayBuffer
from ai.reinforcement_learning.ppo_agent import PPOAgent, PPOTrainer, PPONetwork, PPOBuffer
from ai.reinforcement_learning.a3c_agent import A3CAgent, A3CTrainer, A3CNetwork, A3CWorker
from ai.reinforcement_learning.sac_agent import SACAgent, SACTrainer, SACNetwork, Actor, Critic, SACBuffer
from ai.reinforcement_learning.reward_functions import (
    RewardFunction, SharpeReward, RiskAdjustedReward, 
    ProfitLossReward, TransactionCostReward, MomentumReward,
    CompositeReward, AdaptiveReward
)
from ai.reinforcement_learning.rl_manager import ReinforcementLearningManager

class TestTradingEnvironment:
    """Test Trading Environment."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(100).cumsum() * 0.1,
            'high': 101 + np.random.randn(100).cumsum() * 0.1,
            'low': 99 + np.random.randn(100).cumsum() * 0.1,
            'close': 100 + np.random.randn(100).cumsum() * 0.1,
            'volume': 1000000 + np.random.randint(-100000, 100000, 100)
        })
        return data
    
    @pytest.fixture
    def trading_env(self, sample_data):
        """Create trading environment instance."""
        return TradingEnvironment(
            data=sample_data,
            initial_balance=100000.0,
            transaction_cost=0.001,
            max_position_size=1.0,
            lookback_window=20
        )
    
    def test_trading_environment_initialization(self, trading_env):
        """Test trading environment initialization."""
        assert trading_env.initial_balance == 100000.0
        assert trading_env.transaction_cost == 0.001
        assert trading_env.max_position_size == 1.0
        assert trading_env.lookback_window == 20
        assert trading_env.action_space.n == 3  # Hold, Buy, Sell
        assert trading_env.observation_space.shape[0] > 0
    
    def test_reset_environment(self, trading_env):
        """Test environment reset."""
        observation = trading_env.reset()
        
        assert isinstance(observation, np.ndarray)
        assert len(observation) == trading_env.observation_space.shape[0]
        assert trading_env.balance == trading_env.initial_balance
        assert trading_env.position == 0.0
        assert trading_env.current_step == trading_env.lookback_window
    
    def test_step_environment(self, trading_env):
        """Test environment step."""
        state = trading_env.reset()
        
        # Test hold action
        next_state, reward, done, info = trading_env.step(0)  # Hold
        
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'balance' in info
        assert 'position' in info
        assert 'portfolio_value' in info
    
    def test_buy_action(self, trading_env):
        """Test buy action."""
        state = trading_env.reset()
        initial_balance = trading_env.balance
        
        # Buy action
        next_state, reward, done, info = trading_env.step(1)  # Buy
        
        assert trading_env.position > 0
        assert trading_env.balance < initial_balance
        assert info['position'] > 0
    
    def test_sell_action(self, trading_env):
        """Test sell action."""
        state = trading_env.reset()
        
        # First buy
        trading_env.step(1)  # Buy
        position_after_buy = trading_env.position
        
        # Then sell
        next_state, reward, done, info = trading_env.step(2)  # Sell
        
        assert trading_env.position == 0
        assert info['position'] == 0
    
    def test_calculate_metrics(self, trading_env):
        """Test metrics calculation."""
        state = trading_env.reset()
        
        # Run a few steps
        for _ in range(10):
            action = np.random.randint(0, 3)
            trading_env.step(action)
        
        metrics = trading_env.calculate_metrics()
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'total_trades' in metrics

class TestMarketSimulator:
    """Test Market Simulator."""
    
    @pytest.fixture
    def market_simulator(self):
        """Create market simulator instance."""
        return MarketSimulator(
            n_periods=100,
            initial_price=100.0,
            volatility=0.02,
            drift=0.0001
        )
    
    def test_market_simulator_initialization(self, market_simulator):
        """Test market simulator initialization."""
        assert market_simulator.n_periods == 100
        assert market_simulator.initial_price == 100.0
        assert market_simulator.volatility == 0.02
        assert market_simulator.drift == 0.0001
    
    def test_generate_data(self, market_simulator):
        """Test data generation."""
        data = market_simulator.generate_data()
        
        assert len(data) == 100
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
        assert 'date' in data.columns
        
        # Check data integrity
        assert (data['high'] >= data['low']).all()
        assert (data['high'] >= data['open']).all()
        assert (data['high'] >= data['close']).all()
        assert (data['low'] <= data['open']).all()
        assert (data['low'] <= data['close']).all()
    
    def test_generate_trending_data(self, market_simulator):
        """Test trending data generation."""
        up_data = market_simulator.generate_trending_data('up', 0.001)
        down_data = market_simulator.generate_trending_data('down', 0.001)
        
        assert len(up_data) == 100
        assert len(down_data) == 100
        
        # Check trend direction
        up_trend = up_data['close'].iloc[-1] - up_data['close'].iloc[0]
        down_trend = down_data['close'].iloc[-1] - down_data['close'].iloc[0]
        
        assert up_trend > down_trend
    
    def test_generate_volatile_data(self, market_simulator):
        """Test volatile data generation."""
        volatile_data = market_simulator.generate_volatile_data(2.0)
        
        assert len(volatile_data) == 100
        
        # Check increased volatility
        normal_data = market_simulator.generate_data()
        volatile_std = volatile_data['close'].pct_change().std()
        normal_std = normal_data['close'].pct_change().std()
        
        assert volatile_std > normal_std

class TestDQNNetwork:
    """Test DQN Network."""
    
    @pytest.fixture
    def dqn_network(self):
        """Create DQN network instance."""
        return DQNNetwork(
            input_size=100,
            hidden_sizes=[64, 32],
            output_size=3
        )
    
    def test_dqn_network_initialization(self, dqn_network):
        """Test DQN network initialization."""
        assert dqn_network.input_size == 100
        assert dqn_network.output_size == 3
        assert len(dqn_network.hidden_sizes) == 2
    
    def test_dqn_forward(self, dqn_network):
        """Test DQN forward pass."""
        input_tensor = torch.randn(1, 100)
        output = dqn_network(input_tensor)
        
        assert output.shape == (1, 3)
        assert isinstance(output, torch.Tensor)

class TestReplayBuffer:
    """Test Replay Buffer."""
    
    @pytest.fixture
    def replay_buffer(self):
        """Create replay buffer instance."""
        return ReplayBuffer(capacity=1000)
    
    def test_replay_buffer_initialization(self, replay_buffer):
        """Test replay buffer initialization."""
        assert replay_buffer.capacity == 1000
        assert len(replay_buffer) == 0
    
    def test_replay_buffer_push(self, replay_buffer):
        """Test adding experiences to buffer."""
        from ai.reinforcement_learning.dqn_agent import Experience
        
        experience = Experience(
            state=np.array([1, 2, 3]),
            action=1,
            reward=0.5,
            next_state=np.array([4, 5, 6]),
            done=False
        )
        
        replay_buffer.push(experience)
        
        assert len(replay_buffer) == 1
    
    def test_replay_buffer_sample(self, replay_buffer):
        """Test sampling from buffer."""
        from ai.reinforcement_learning.dqn_agent import Experience
        
        # Add multiple experiences
        for i in range(10):
            experience = Experience(
                state=np.array([i, i+1, i+2]),
                action=i % 3,
                reward=float(i),
                next_state=np.array([i+3, i+4, i+5]),
                done=i == 9
            )
            replay_buffer.push(experience)
        
        # Sample batch
        batch = replay_buffer.sample(5)
        
        assert len(batch) == 5
        assert all(isinstance(exp, Experience) for exp in batch)

class TestDQNAgent:
    """Test DQN Agent."""
    
    @pytest.fixture
    def dqn_agent(self):
        """Create DQN agent instance."""
        return DQNAgent(
            state_size=100,
            action_size=3,
            hidden_sizes=[64, 32],
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
    
    def test_dqn_agent_initialization(self, dqn_agent):
        """Test DQN agent initialization."""
        assert dqn_agent.state_size == 100
        assert dqn_agent.action_size == 3
        assert dqn_agent.epsilon == 1.0
        assert dqn_agent.epsilon_min == 0.01
        assert dqn_agent.gamma == 0.95
    
    def test_dqn_act(self, dqn_agent):
        """Test DQN action selection."""
        state = np.random.randn(100)
        
        # Test exploration (epsilon = 1.0)
        action = dqn_agent.act(state, training=True)
        
        assert isinstance(action, int)
        assert 0 <= action < 3
    
    def test_dqn_remember(self, dqn_agent):
        """Test DQN experience storage."""
        state = np.random.randn(100)
        action = 1
        reward = 0.5
        next_state = np.random.randn(100)
        done = False
        
        dqn_agent.remember(state, action, reward, next_state, done)
        
        assert len(dqn_agent.memory) == 1
    
    @patch('torch.optim.Adam')
    def test_dqn_replay(self, mock_optimizer, dqn_agent):
        """Test DQN replay training."""
        # Add experiences to memory
        for i in range(50):  # More than batch_size
            state = np.random.randn(100)
            action = i % 3
            reward = float(i)
            next_state = np.random.randn(100)
            done = i == 49
            
            dqn_agent.remember(state, action, reward, next_state, done)
        
        # Test replay
        loss = dqn_agent.replay()
        
        assert isinstance(loss, float)
        assert loss >= 0

class TestPPONetwork:
    """Test PPO Network."""
    
    @pytest.fixture
    def ppo_network(self):
        """Create PPO network instance."""
        return PPONetwork(
            state_size=100,
            action_size=3,
            hidden_sizes=[64, 32]
        )
    
    def test_ppo_network_initialization(self, ppo_network):
        """Test PPO network initialization."""
        assert ppo_network.state_size == 100
        assert ppo_network.action_size == 3
        assert len(ppo_network.hidden_sizes) == 2
    
    def test_ppo_forward(self, ppo_network):
        """Test PPO forward pass."""
        input_tensor = torch.randn(1, 100)
        action_logits, value = ppo_network(input_tensor)
        
        assert action_logits.shape == (1, 3)
        assert value.shape == (1, 1)
        assert isinstance(action_logits, torch.Tensor)
        assert isinstance(value, torch.Tensor)
    
    def test_ppo_get_action(self, ppo_network):
        """Test PPO action selection."""
        input_tensor = torch.randn(1, 100)
        action, log_prob, value = ppo_network.get_action(input_tensor)
        
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1, 1)
        assert 0 <= action.item() < 3

class TestPPOBuffer:
    """Test PPO Buffer."""
    
    @pytest.fixture
    def ppo_buffer(self):
        """Create PPO buffer instance."""
        return PPOBuffer(buffer_size=1000, state_size=100)
    
    def test_ppo_buffer_initialization(self, ppo_buffer):
        """Test PPO buffer initialization."""
        assert ppo_buffer.buffer_size == 1000
        assert ppo_buffer.state_size == 100
        assert ppo_buffer.size == 0
    
    def test_ppo_buffer_add(self, ppo_buffer):
        """Test adding experiences to PPO buffer."""
        state = np.random.randn(100)
        action = 1
        reward = 0.5
        value = 0.3
        log_prob = -0.7
        
        ppo_buffer.add(state, action, reward, value, log_prob)
        
        assert ppo_buffer.size == 1
    
    def test_ppo_buffer_compute_advantages(self, ppo_buffer):
        """Test advantage computation."""
        # Add some experiences
        for i in range(10):
            state = np.random.randn(100)
            action = i % 3
            reward = float(i)
            value = 0.5
            log_prob = -0.7
            
            ppo_buffer.add(state, action, reward, value, log_prob)
        
        # Compute advantages
        ppo_buffer.compute_advantages_and_returns()
        
        assert ppo_buffer.advantages.shape == (10,)
        assert ppo_buffer.returns.shape == (10,)

class TestPPOAgent:
    """Test PPO Agent."""
    
    @pytest.fixture
    def ppo_agent(self):
        """Create PPO agent instance."""
        return PPOAgent(
            state_size=100,
            action_size=3,
            hidden_sizes=[64, 32],
            learning_rate=3e-4,
            gamma=0.99
        )
    
    def test_ppo_agent_initialization(self, ppo_agent):
        """Test PPO agent initialization."""
        assert ppo_agent.state_size == 100
        assert ppo_agent.action_size == 3
        assert ppo_agent.gamma == 0.99
        assert ppo_agent.clip_ratio == 0.2
    
    def test_ppo_act(self, ppo_agent):
        """Test PPO action selection."""
        state = np.random.randn(100)
        action, log_prob, value = ppo_agent.act(state, training=True)
        
        assert isinstance(action, int)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert 0 <= action < 3
    
    def test_ppo_store_experience(self, ppo_agent):
        """Test PPO experience storage."""
        state = np.random.randn(100)
        action = 1
        reward = 0.5
        value = 0.3
        log_prob = -0.7
        
        ppo_agent.store_experience(state, action, reward, value, log_prob)
        
        assert ppo_agent.buffer.size == 1

class TestA3CNetwork:
    """Test A3C Network."""
    
    @pytest.fixture
    def a3c_network(self):
        """Create A3C network instance."""
        return A3CNetwork(
            state_size=100,
            action_size=3,
            hidden_sizes=[64, 32]
        )
    
    def test_a3c_network_initialization(self, a3c_network):
        """Test A3C network initialization."""
        assert a3c_network.state_size == 100
        assert a3c_network.action_size == 3
        assert len(a3c_network.hidden_sizes) == 2
    
    def test_a3c_forward(self, a3c_network):
        """Test A3C forward pass."""
        input_tensor = torch.randn(1, 100)
        action_logits, value = a3c_network(input_tensor)
        
        assert action_logits.shape == (1, 3)
        assert value.shape == (1, 1)
        assert isinstance(action_logits, torch.Tensor)
        assert isinstance(value, torch.Tensor)

class TestA3CAgent:
    """Test A3C Agent."""
    
    @pytest.fixture
    def a3c_agent(self):
        """Create A3C agent instance."""
        return A3CAgent(
            state_size=100,
            action_size=3,
            hidden_sizes=[64, 32],
            learning_rate=3e-4,
            gamma=0.99
        )
    
    def test_a3c_agent_initialization(self, a3c_agent):
        """Test A3C agent initialization."""
        assert a3c_agent.state_size == 100
        assert a3c_agent.action_size == 3
        assert a3c_agent.gamma == 0.99
        assert a3c_agent.beta == 0.01
    
    def test_a3c_act(self, a3c_agent):
        """Test A3C action selection."""
        state = np.random.randn(100)
        action, log_prob, value = a3c_agent.act(state, training=True)
        
        assert isinstance(action, int)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert 0 <= action < 3
    
    def test_a3c_compute_returns_and_advantages(self, a3c_agent):
        """Test A3C returns and advantages computation."""
        rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        dones = [False, False, False, False, True]
        
        returns, advantages = a3c_agent.compute_returns_and_advantages(rewards, values, dones)
        
        assert len(returns) == 5
        assert len(advantages) == 5
        assert all(isinstance(r, float) for r in returns)
        assert all(isinstance(a, float) for a in advantages)

class TestSACNetwork:
    """Test SAC Network."""
    
    @pytest.fixture
    def sac_network(self):
        """Create SAC network instance."""
        return SACNetwork(
            input_size=100,
            hidden_sizes=[64, 32],
            output_size=3
        )
    
    def test_sac_network_initialization(self, sac_network):
        """Test SAC network initialization."""
        assert sac_network.input_size == 100
        assert sac_network.output_size == 3
        assert len(sac_network.hidden_sizes) == 2
    
    def test_sac_forward(self, sac_network):
        """Test SAC forward pass."""
        input_tensor = torch.randn(1, 100)
        output = sac_network(input_tensor)
        
        assert output.shape == (1, 3)
        assert isinstance(output, torch.Tensor)

class TestActor:
    """Test Actor Network."""
    
    @pytest.fixture
    def actor(self):
        """Create Actor instance."""
        return Actor(
            state_size=100,
            action_size=3,
            hidden_sizes=[64, 32]
        )
    
    def test_actor_initialization(self, actor):
        """Test Actor initialization."""
        assert actor.state_size == 100
        assert actor.action_size == 3
        assert len(actor.hidden_sizes) == 2
    
    def test_actor_forward(self, actor):
        """Test Actor forward pass."""
        input_tensor = torch.randn(1, 100)
        output = actor(input_tensor)
        
        assert output.shape == (1, 3)
        assert isinstance(output, torch.Tensor)
    
    def test_actor_get_action(self, actor):
        """Test Actor action selection."""
        input_tensor = torch.randn(1, 100)
        action, log_prob = actor.get_action(input_tensor)
        
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert 0 <= action.item() < 3

class TestCritic:
    """Test Critic Network."""
    
    @pytest.fixture
    def critic(self):
        """Create Critic instance."""
        return Critic(
            state_size=100,
            action_size=3,
            hidden_sizes=[64, 32]
        )
    
    def test_critic_initialization(self, critic):
        """Test Critic initialization."""
        assert critic.state_size == 100
        assert critic.action_size == 3
        assert len(critic.hidden_sizes) == 2
    
    def test_critic_forward(self, critic):
        """Test Critic forward pass."""
        state_tensor = torch.randn(1, 100)
        action_tensor = torch.randn(1, 3)
        output = critic(state_tensor, action_tensor)
        
        assert output.shape == (1, 1)
        assert isinstance(output, torch.Tensor)

class TestSACBuffer:
    """Test SAC Buffer."""
    
    @pytest.fixture
    def sac_buffer(self):
        """Create SAC buffer instance."""
        return SACBuffer(capacity=1000)
    
    def test_sac_buffer_initialization(self, sac_buffer):
        """Test SAC buffer initialization."""
        assert sac_buffer.capacity == 1000
        assert len(sac_buffer) == 0
    
    def test_sac_buffer_push(self, sac_buffer):
        """Test adding experiences to SAC buffer."""
        state = np.random.randn(100)
        action = 1
        reward = 0.5
        next_state = np.random.randn(100)
        done = False
        
        sac_buffer.push(state, action, reward, next_state, done)
        
        assert len(sac_buffer) == 1
    
    def test_sac_buffer_sample(self, sac_buffer):
        """Test sampling from SAC buffer."""
        # Add multiple experiences
        for i in range(10):
            state = np.random.randn(100)
            action = i % 3
            reward = float(i)
            next_state = np.random.randn(100)
            done = i == 9
            
            sac_buffer.push(state, action, reward, next_state, done)
        
        # Sample batch
        states, actions, rewards, next_states, dones = sac_buffer.sample(5)
        
        assert states.shape == (5, 100)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert next_states.shape == (5, 100)
        assert dones.shape == (5,)

class TestSACAgent:
    """Test SAC Agent."""
    
    @pytest.fixture
    def sac_agent(self):
        """Create SAC agent instance."""
        return SACAgent(
            state_size=100,
            action_size=3,
            hidden_sizes=[64, 32],
            learning_rate=3e-4,
            gamma=0.99
        )
    
    def test_sac_agent_initialization(self, sac_agent):
        """Test SAC agent initialization."""
        assert sac_agent.state_size == 100
        assert sac_agent.action_size == 3
        assert sac_agent.gamma == 0.99
        assert sac_agent.tau == 0.005
        assert sac_agent.alpha == 0.2
    
    def test_sac_act(self, sac_agent):
        """Test SAC action selection."""
        state = np.random.randn(100)
        action = sac_agent.act(state, training=True)
        
        assert isinstance(action, int)
        assert 0 <= action < 3
    
    def test_sac_remember(self, sac_agent):
        """Test SAC experience storage."""
        state = np.random.randn(100)
        action = 1
        reward = 0.5
        next_state = np.random.randn(100)
        done = False
        
        sac_agent.remember(state, action, reward, next_state, done)
        
        assert len(sac_agent.memory) == 1

class TestRewardFunctions:
    """Test Reward Functions."""
    
    @pytest.fixture
    def sharpe_reward(self):
        """Create Sharpe reward function."""
        return SharpeReward(risk_free_rate=0.02, lookback_window=10)
    
    def test_sharpe_reward_initialization(self, sharpe_reward):
        """Test Sharpe reward initialization."""
        assert sharpe_reward.risk_free_rate == 0.02
        assert sharpe_reward.lookback_window == 10
        assert sharpe_reward.name == "sharpe_reward"
    
    def test_sharpe_reward_calculation(self, sharpe_reward):
        """Test Sharpe reward calculation."""
        portfolio_value = 100000
        previous_portfolio_value = 99000
        
        reward = sharpe_reward.calculate(
            portfolio_value=portfolio_value,
            previous_portfolio_value=previous_portfolio_value
        )
        
        assert isinstance(reward, float)
        assert len(sharpe_reward.returns_history) == 1
    
    @pytest.fixture
    def risk_adjusted_reward(self):
        """Create risk-adjusted reward function."""
        return RiskAdjustedReward(
            return_weight=1.0,
            volatility_weight=-0.5,
            drawdown_weight=-1.0
        )
    
    def test_risk_adjusted_reward_calculation(self, risk_adjusted_reward):
        """Test risk-adjusted reward calculation."""
        portfolio_value = 100000
        previous_portfolio_value = 99000
        
        reward = risk_adjusted_reward.calculate(
            portfolio_value=portfolio_value,
            previous_portfolio_value=previous_portfolio_value
        )
        
        assert isinstance(reward, float)
        assert len(risk_adjusted_reward.returns_history) == 1
    
    @pytest.fixture
    def profit_loss_reward(self):
        """Create profit/loss reward function."""
        return ProfitLossReward(scale_factor=1.0)
    
    def test_profit_loss_reward_calculation(self, profit_loss_reward):
        """Test profit/loss reward calculation."""
        portfolio_value = 100000
        previous_portfolio_value = 99000
        
        reward = profit_loss_reward.calculate(
            portfolio_value=portfolio_value,
            previous_portfolio_value=previous_portfolio_value
        )
        
        assert isinstance(reward, float)
        assert reward == 1000.0  # 100000 - 99000
    
    @pytest.fixture
    def composite_reward(self):
        """Create composite reward function."""
        sharpe = SharpeReward()
        profit_loss = ProfitLossReward()
        
        return CompositeReward([sharpe, profit_loss], [0.5, 0.5])
    
    def test_composite_reward_calculation(self, composite_reward):
        """Test composite reward calculation."""
        portfolio_value = 100000
        previous_portfolio_value = 99000
        
        reward = composite_reward.calculate(
            portfolio_value=portfolio_value,
            previous_portfolio_value=previous_portfolio_value
        )
        
        assert isinstance(reward, float)
        assert len(composite_reward.history) == 1

class TestReinforcementLearningManager:
    """Test Reinforcement Learning Manager."""
    
    @pytest.fixture
    def rl_manager(self):
        """Create RL manager instance."""
        return ReinforcementLearningManager()
    
    def test_rl_manager_initialization(self, rl_manager):
        """Test RL manager initialization."""
        assert rl_manager.model_dir is not None
        assert len(rl_manager.agent_configs) > 0
        assert len(rl_manager.agents) == 0
    
    def test_list_agents(self, rl_manager):
        """Test agent listing."""
        agents_info = rl_manager.list_agents()
        
        assert len(agents_info) > 0
        assert all('name' in info for info in agents_info)
        assert all('class' in info for info in agents_info)
        assert all('parameters' in info for info in agents_info)
    
    def test_get_agent_status(self, rl_manager):
        """Test agent status retrieval."""
        status = rl_manager.get_agent_status()
        
        assert 'total_agents' in status
        assert 'created_agents' in status
        assert 'trained_agents' in status
        assert 'agents' in status
    
    def test_generate_synthetic_data(self, rl_manager):
        """Test synthetic data generation."""
        data = rl_manager.generate_synthetic_data(n_periods=100)
        
        assert len(data) == 100
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
    
    def test_create_environment(self, rl_manager):
        """Test environment creation."""
        data = rl_manager.generate_synthetic_data(n_periods=100)
        environment = rl_manager.create_environment(data, "test_env")
        
        assert "test_env" in rl_manager.environments
        assert isinstance(environment, TradingEnvironment)
    
    def test_create_agent(self, rl_manager):
        """Test agent creation."""
        data = rl_manager.generate_synthetic_data(n_periods=100)
        environment = rl_manager.create_environment(data, "test_env")
        
        agent = rl_manager.create_agent("dqn", environment)
        
        assert "dqn" in rl_manager.agents
        assert isinstance(agent, DQNAgent)
    
    def test_get_reward_function(self, rl_manager):
        """Test reward function retrieval."""
        sharpe_reward = rl_manager.get_reward_function("sharpe")
        
        assert isinstance(sharpe_reward, SharpeReward)
        assert sharpe_reward.name == "sharpe_reward"
    
    def test_create_composite_reward(self, rl_manager):
        """Test composite reward creation."""
        composite_reward = rl_manager.create_composite_reward(
            ["sharpe", "profit_loss"],
            [0.5, 0.5],
            "test_composite"
        )
        
        assert isinstance(composite_reward, CompositeReward)
        assert "test_composite" in rl_manager.reward_functions

class TestIntegration:
    """Integration tests for reinforcement learning models."""
    
    @pytest.fixture
    def integration_data(self):
        """Create integration test data."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(200).cumsum() * 0.1,
            'high': 101 + np.random.randn(200).cumsum() * 0.1,
            'low': 99 + np.random.randn(200).cumsum() * 0.1,
            'close': 100 + np.random.randn(200).cumsum() * 0.1,
            'volume': 1000000 + np.random.randint(-100000, 100000, 200)
        })
        return data
    
    def test_end_to_end_dqn_pipeline(self, integration_data):
        """Test end-to-end DQN pipeline."""
        # Create environment
        environment = TradingEnvironment(integration_data)
        
        # Create agent
        agent = DQNAgent(
            state_size=environment.observation_space.shape[0],
            action_size=3,
            hidden_sizes=[64, 32]
        )
        
        # Test training loop
        state = environment.reset()
        
        for step in range(10):
            action = agent.act(state, training=True)
            next_state, reward, done, info = environment.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) >= agent.batch_size:
                agent.replay()
            
            state = next_state
            
            if done:
                break
        
        # Test evaluation
        state = environment.reset()
        action = agent.act(state, training=False)
        
        assert isinstance(action, int)
        assert 0 <= action < 3
    
    def test_end_to_end_ppo_pipeline(self, integration_data):
        """Test end-to-end PPO pipeline."""
        # Create environment
        environment = TradingEnvironment(integration_data)
        
        # Create agent
        agent = PPOAgent(
            state_size=environment.observation_space.shape[0],
            action_size=3,
            hidden_sizes=[64, 32]
        )
        
        # Test training loop
        state = environment.reset()
        
        for step in range(10):
            action, log_prob, value = agent.act(state, training=True)
            next_state, reward, done, info = environment.step(action)
            
            agent.store_experience(state, action, reward, value, log_prob)
            
            if agent.buffer.size >= agent.batch_size:
                agent.update()
            
            state = next_state
            
            if done:
                break
        
        # Test evaluation
        state = environment.reset()
        action, _, _ = agent.act(state, training=False)
        
        assert isinstance(action, int)
        assert 0 <= action < 3
    
    def test_rl_manager_integration(self, integration_data):
        """Test RL manager integration."""
        # Create manager
        manager = ReinforcementLearningManager()
        
        # Create environment
        environment = manager.create_environment(integration_data, "test_env")
        
        # Create agent
        agent = manager.create_agent("dqn", environment)
        
        # Test prediction
        state = environment.reset()
        action = agent.act(state, training=True)
        
        assert isinstance(action, int)
        assert 0 <= action < 3
        
        # Test ensemble prediction
        ensemble_result = manager.create_ensemble_prediction(state, ["dqn"])
        
        assert 'ensemble_action' in ensemble_result
        assert 'ensemble_confidence' in ensemble_result
        assert 'individual_predictions' in ensemble_result

