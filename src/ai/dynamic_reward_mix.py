"""
Dynamic Reward Mix
Allows real-time tuning between profit and risk objectives
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)

class RewardComponent(Enum):
    """Reward components"""
    RETURN = "return"
    DRAWDOWN = "drawdown"
    TURNOVER = "turnover"
    SHARPE = "sharpe"
    VOLATILITY = "volatility"
    CONSISTENCY = "consistency"

@dataclass
class RewardCoefficients:
    """Reward coefficient configuration"""
    return_coeff: float = 1.0
    drawdown_coeff: float = -0.5
    turnover_coeff: float = -0.1
    sharpe_coeff: float = 0.3
    volatility_coeff: float = -0.2
    consistency_coeff: float = 0.1
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PerformanceMetrics:
    """Performance metrics for reward calculation"""
    returns: List[float]
    drawdowns: List[float]
    turnover: float
    sharpe_ratio: float
    volatility: float
    consistency_score: float
    timestamp: datetime

class DynamicRewardMix:
    """Manages dynamic reward coefficients based on performance"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.coefficients = RewardCoefficients()
        self.performance_history = []
        self.reward_history = []
        
        # Performance thresholds
        self.sharpe_threshold = config.get('sharpe_threshold', 0.6)
        self.drawdown_threshold = config.get('drawdown_threshold', 0.1)
        self.volatility_threshold = config.get('volatility_threshold', 0.3)
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.decay_rate = config.get('decay_rate', 0.95)
        
        # Load saved coefficients
        self._load_coefficients()
        
        logger.info("Dynamic Reward Mix initialized")
    
    def calculate_reward(self, performance: PerformanceMetrics) -> float:
        """Calculate reward using current coefficients"""
        try:
            # Calculate component rewards
            return_reward = np.mean(performance.returns) * self.coefficients.return_coeff
            drawdown_reward = np.mean(performance.drawdowns) * self.coefficients.drawdown_coeff
            turnover_reward = performance.turnover * self.coefficients.turnover_coeff
            sharpe_reward = performance.sharpe_ratio * self.coefficients.sharpe_coeff
            volatility_reward = performance.volatility * self.coefficients.volatility_coeff
            consistency_reward = performance.consistency_score * self.coefficients.consistency_coeff
            
            # Total reward
            total_reward = (return_reward + drawdown_reward + turnover_reward + 
                          sharpe_reward + volatility_reward + consistency_reward)
            
            # Store reward history
            self.reward_history.append({
                'timestamp': performance.timestamp,
                'total_reward': total_reward,
                'components': {
                    'return': return_reward,
                    'drawdown': drawdown_reward,
                    'turnover': turnover_reward,
                    'sharpe': sharpe_reward,
                    'volatility': volatility_reward,
                    'consistency': consistency_reward
                },
                'coefficients': {
                    'return_coeff': self.coefficients.return_coeff,
                    'drawdown_coeff': self.coefficients.drawdown_coeff,
                    'turnover_coeff': self.coefficients.turnover_coeff,
                    'sharpe_coeff': self.coefficients.sharpe_coeff,
                    'volatility_coeff': self.coefficients.volatility_coeff,
                    'consistency_coeff': self.coefficients.consistency_coeff
                }
            })
            
            return total_reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def update_coefficients(self, performance: PerformanceMetrics):
        """Update reward coefficients based on performance"""
        try:
            # Store performance history
            self.performance_history.append(performance)
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            # Check if we need to adjust coefficients
            if self._should_adjust_coefficients(performance):
                self._adjust_coefficients(performance)
                self._save_coefficients()
                
                logger.info(f"Updated reward coefficients: {self.coefficients}")
            
        except Exception as e:
            logger.error(f"Error updating coefficients: {e}")
    
    def _should_adjust_coefficients(self, performance: PerformanceMetrics) -> bool:
        """Check if coefficients should be adjusted"""
        try:
            # Adjust if Sharpe ratio is below threshold
            if performance.sharpe_ratio < self.sharpe_threshold:
                return True
            
            # Adjust if drawdown is too high
            if np.mean(performance.drawdowns) > self.drawdown_threshold:
                return True
            
            # Adjust if volatility is too high
            if performance.volatility > self.volatility_threshold:
                return True
            
            # Adjust if performance is consistently poor
            if len(self.performance_history) >= 10:
                recent_sharpe = np.mean([p.sharpe_ratio for p in self.performance_history[-10:]])
                if recent_sharpe < self.sharpe_threshold * 0.8:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking adjustment criteria: {e}")
            return False
    
    def _adjust_coefficients(self, performance: PerformanceMetrics):
        """Adjust reward coefficients based on performance"""
        try:
            # Calculate gradients for each coefficient
            gradients = self._calculate_gradients(performance)
            
            # Update coefficients using gradient descent with momentum
            self.coefficients.return_coeff += self.learning_rate * gradients['return']
            self.coefficients.drawdown_coeff += self.learning_rate * gradients['drawdown']
            self.coefficients.turnover_coeff += self.learning_rate * gradients['turnover']
            self.coefficients.sharpe_coeff += self.learning_rate * gradients['sharpe']
            self.coefficients.volatility_coeff += self.learning_rate * gradients['volatility']
            self.coefficients.consistency_coeff += self.learning_rate * gradients['consistency']
            
            # Apply bounds to prevent extreme values
            self._apply_bounds()
            
            # Update timestamp
            self.coefficients.timestamp = datetime.now()
            
        except Exception as e:
            logger.error(f"Error adjusting coefficients: {e}")
    
    def _calculate_gradients(self, performance: PerformanceMetrics) -> Dict[str, float]:
        """Calculate gradients for coefficient adjustment"""
        try:
            gradients = {}
            
            # Return coefficient gradient
            if performance.sharpe_ratio < self.sharpe_threshold:
                # Increase return weight if Sharpe is low
                gradients['return'] = 0.1
            else:
                gradients['return'] = -0.05
            
            # Drawdown coefficient gradient
            if np.mean(performance.drawdowns) > self.drawdown_threshold:
                # Increase drawdown penalty if drawdown is high
                gradients['drawdown'] = -0.2
            else:
                gradients['drawdown'] = 0.05
            
            # Turnover coefficient gradient
            if performance.turnover > 0.5:  # High turnover
                # Increase turnover penalty
                gradients['turnover'] = -0.1
            else:
                gradients['turnover'] = 0.02
            
            # Sharpe coefficient gradient
            if performance.sharpe_ratio < self.sharpe_threshold:
                # Increase Sharpe weight if Sharpe is low
                gradients['sharpe'] = 0.2
            else:
                gradients['sharpe'] = -0.05
            
            # Volatility coefficient gradient
            if performance.volatility > self.volatility_threshold:
                # Increase volatility penalty if volatility is high
                gradients['volatility'] = -0.15
            else:
                gradients['volatility'] = 0.05
            
            # Consistency coefficient gradient
            if performance.consistency_score < 0.5:
                # Increase consistency weight if consistency is low
                gradients['consistency'] = 0.1
            else:
                gradients['consistency'] = -0.02
            
            return gradients
            
        except Exception as e:
            logger.error(f"Error calculating gradients: {e}")
            return {key: 0.0 for key in ['return', 'drawdown', 'turnover', 'sharpe', 'volatility', 'consistency']}
    
    def _apply_bounds(self):
        """Apply bounds to coefficients to prevent extreme values"""
        try:
            # Return coefficient bounds
            self.coefficients.return_coeff = np.clip(self.coefficients.return_coeff, 0.1, 2.0)
            
            # Drawdown coefficient bounds
            self.coefficients.drawdown_coeff = np.clip(self.coefficients.drawdown_coeff, -2.0, -0.1)
            
            # Turnover coefficient bounds
            self.coefficients.turnover_coeff = np.clip(self.coefficients.turnover_coeff, -0.5, 0.0)
            
            # Sharpe coefficient bounds
            self.coefficients.sharpe_coeff = np.clip(self.coefficients.sharpe_coeff, 0.0, 1.0)
            
            # Volatility coefficient bounds
            self.coefficients.volatility_coeff = np.clip(self.coefficients.volatility_coeff, -1.0, 0.0)
            
            # Consistency coefficient bounds
            self.coefficients.consistency_coeff = np.clip(self.coefficients.consistency_coeff, 0.0, 0.5)
            
        except Exception as e:
            logger.error(f"Error applying bounds: {e}")
    
    def _save_coefficients(self):
        """Save coefficients to file"""
        try:
            config_dir = self.config.get('config_dir', 'config')
            os.makedirs(config_dir, exist_ok=True)
            
            filepath = os.path.join(config_dir, 'reward_coefficients.json')
            
            coefficients_data = {
                'return_coeff': self.coefficients.return_coeff,
                'drawdown_coeff': self.coefficients.drawdown_coeff,
                'turnover_coeff': self.coefficients.turnover_coeff,
                'sharpe_coeff': self.coefficients.sharpe_coeff,
                'volatility_coeff': self.coefficients.volatility_coeff,
                'consistency_coeff': self.coefficients.consistency_coeff,
                'timestamp': self.coefficients.timestamp.isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(coefficients_data, f, indent=2)
            
            logger.info(f"Saved reward coefficients to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving coefficients: {e}")
    
    def _load_coefficients(self):
        """Load coefficients from file"""
        try:
            config_dir = self.config.get('config_dir', 'config')
            filepath = os.path.join(config_dir, 'reward_coefficients.json')
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    coefficients_data = json.load(f)
                
                self.coefficients.return_coeff = coefficients_data.get('return_coeff', 1.0)
                self.coefficients.drawdown_coeff = coefficients_data.get('drawdown_coeff', -0.5)
                self.coefficients.turnover_coeff = coefficients_data.get('turnover_coeff', -0.1)
                self.coefficients.sharpe_coeff = coefficients_data.get('sharpe_coeff', 0.3)
                self.coefficients.volatility_coeff = coefficients_data.get('volatility_coeff', -0.2)
                self.coefficients.consistency_coeff = coefficients_data.get('consistency_coeff', 0.1)
                
                timestamp_str = coefficients_data.get('timestamp')
                if timestamp_str:
                    self.coefficients.timestamp = datetime.fromisoformat(timestamp_str)
                
                logger.info(f"Loaded reward coefficients from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading coefficients: {e}")
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get current reward coefficients"""
        return {
            'return_coeff': self.coefficients.return_coeff,
            'drawdown_coeff': self.coefficients.drawdown_coeff,
            'turnover_coeff': self.coefficients.turnover_coeff,
            'sharpe_coeff': self.coefficients.sharpe_coeff,
            'volatility_coeff': self.coefficients.volatility_coeff,
            'consistency_coeff': self.coefficients.consistency_coeff
        }
    
    def set_coefficients(self, coefficients: Dict[str, float]):
        """Set reward coefficients (for GPT-5 overrides)"""
        try:
            if 'return_coeff' in coefficients:
                self.coefficients.return_coeff = coefficients['return_coeff']
            if 'drawdown_coeff' in coefficients:
                self.coefficients.drawdown_coeff = coefficients['drawdown_coeff']
            if 'turnover_coeff' in coefficients:
                self.coefficients.turnover_coeff = coefficients['turnover_coeff']
            if 'sharpe_coeff' in coefficients:
                self.coefficients.sharpe_coeff = coefficients['sharpe_coeff']
            if 'volatility_coeff' in coefficients:
                self.coefficients.volatility_coeff = coefficients['volatility_coeff']
            if 'consistency_coeff' in coefficients:
                self.coefficients.consistency_coeff = coefficients['consistency_coeff']
            
            self.coefficients.timestamp = datetime.now()
            self._apply_bounds()
            self._save_coefficients()
            
            logger.info(f"Updated reward coefficients: {self.get_coefficients()}")
            
        except Exception as e:
            logger.error(f"Error setting coefficients: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for analysis"""
        try:
            if not self.performance_history:
                return {}
            
            recent_performance = self.performance_history[-10:]  # Last 10 periods
            
            return {
                'avg_sharpe': np.mean([p.sharpe_ratio for p in recent_performance]),
                'avg_drawdown': np.mean([np.mean(p.drawdowns) for p in recent_performance]),
                'avg_volatility': np.mean([p.volatility for p in recent_performance]),
                'avg_turnover': np.mean([p.turnover for p in recent_performance]),
                'avg_consistency': np.mean([p.consistency_score for p in recent_performance]),
                'coefficients': self.get_coefficients(),
                'last_update': self.coefficients.timestamp
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def reset_coefficients(self):
        """Reset coefficients to default values"""
        try:
            self.coefficients = RewardCoefficients()
            self._save_coefficients()
            logger.info("Reset reward coefficients to default values")
            
        except Exception as e:
            logger.error(f"Error resetting coefficients: {e}")

class RewardMixManager:
    """Manages dynamic reward mixing for RL agents"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.reward_mix = DynamicRewardMix(config)
        self.performance_tracker = {}
        
        logger.info("Reward Mix Manager initialized")
    
    def calculate_agent_reward(self, agent_id: str, performance: PerformanceMetrics) -> float:
        """Calculate reward for a specific RL agent"""
        try:
            # Calculate base reward
            reward = self.reward_mix.calculate_reward(performance)
            
            # Store performance for this agent
            if agent_id not in self.performance_tracker:
                self.performance_tracker[agent_id] = []
            
            self.performance_tracker[agent_id].append(performance)
            
            # Keep only recent history
            if len(self.performance_tracker[agent_id]) > 50:
                self.performance_tracker[agent_id] = self.performance_tracker[agent_id][-50:]
            
            # Update coefficients based on overall performance
            self.reward_mix.update_coefficients(performance)
            
            return reward
            
        except Exception as e:
            logger.error(f"Error calculating agent reward: {e}")
            return 0.0
    
    def get_agent_performance(self, agent_id: str) -> Dict:
        """Get performance summary for a specific agent"""
        try:
            if agent_id not in self.performance_tracker:
                return {}
            
            performances = self.performance_tracker[agent_id]
            if not performances:
                return {}
            
            return {
                'agent_id': agent_id,
                'avg_sharpe': np.mean([p.sharpe_ratio for p in performances]),
                'avg_drawdown': np.mean([np.mean(p.drawdowns) for p in performances]),
                'avg_volatility': np.mean([p.volatility for p in performances]),
                'avg_turnover': np.mean([p.turnover for p in performances]),
                'avg_consistency': np.mean([p.consistency_score for p in performances]),
                'total_periods': len(performances)
            }
            
        except Exception as e:
            logger.error(f"Error getting agent performance: {e}")
            return {}
    
    def get_all_agents_performance(self) -> Dict[str, Dict]:
        """Get performance summary for all agents"""
        try:
            all_performance = {}
            for agent_id in self.performance_tracker.keys():
                all_performance[agent_id] = self.get_agent_performance(agent_id)
            
            return all_performance
            
        except Exception as e:
            logger.error(f"Error getting all agents performance: {e}")
            return {}
    
    def update_coefficients_from_gpt5(self, coefficients: Dict[str, float]):
        """Update coefficients from GPT-5 recommendations"""
        try:
            self.reward_mix.set_coefficients(coefficients)
            logger.info(f"Updated reward coefficients from GPT-5: {coefficients}")
            
        except Exception as e:
            logger.error(f"Error updating coefficients from GPT-5: {e}")
    
    def get_reward_analysis(self) -> Dict:
        """Get comprehensive reward analysis"""
        try:
            return {
                'current_coefficients': self.reward_mix.get_coefficients(),
                'performance_summary': self.reward_mix.get_performance_summary(),
                'agents_performance': self.get_all_agents_performance(),
                'reward_history': self.reward_mix.reward_history[-20:]  # Last 20 rewards
            }
            
        except Exception as e:
            logger.error(f"Error getting reward analysis: {e}")
            return {}
