"""
Regime Policy Manager - Configuration Management

This module manages regime-specific trading policies loaded from YAML configuration.
"""

import logging
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass

from src.ai.regime_detection import MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class RegimePolicy:
    """Represents regime-specific trading policy."""
    regime: MarketRegime
    ensemble_weights: Dict[str, float]
    kelly_adjustments: Dict[str, float]
    atr_brackets: Dict[str, float]
    lookback_windows: Dict[str, int]
    position_management: Dict[str, Any]
    risk_management: Dict[str, float]

class RegimePolicyManager:
    """Manages regime-specific trading policies."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Regime Policy Manager."""
        if config_path is None:
            config_path = Path(__file__).parent / "regime_policies.yaml"
        
        self.config_path = Path(config_path)
        self.policies: Dict[MarketRegime, RegimePolicy] = {}
        self.feature_flags: Dict[str, bool] = {}
        self.transition_rules: Dict[str, Any] = {}
        
        self._load_policies()
        logger.info("Regime Policy Manager initialized")
    
    def _load_policies(self) -> None:
        """Load regime policies from YAML configuration."""
        try:
            if not self.config_path.exists():
                logger.error(f"Regime policies file not found: {self.config_path}")
                self._create_default_policies()
                return
            
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Load feature flags
            self.feature_flags = config.get('feature_flags', {})
            
            # Load transition rules
            self.transition_rules = config.get('regime_transitions', {})
            
            # Load regime-specific policies
            ensemble_weights = config.get('ensemble_weights', {})
            kelly_adjustments = config.get('kelly_adjustments', {})
            atr_brackets = config.get('atr_brackets', {})
            lookback_windows = config.get('lookback_windows', {})
            position_management = config.get('position_management', {})
            risk_management = config.get('risk_management', {})
            
            # Create policy objects for each regime
            for regime_name in MarketRegime:
                regime_key = regime_name.value
                
                policy = RegimePolicy(
                    regime=regime_name,
                    ensemble_weights=ensemble_weights.get(regime_key, {}),
                    kelly_adjustments=kelly_adjustments.get(regime_key, {}),
                    atr_brackets=atr_brackets.get(regime_key, {}),
                    lookback_windows=lookback_windows.get(regime_key, {}),
                    position_management=position_management.get(regime_key, {}),
                    risk_management=risk_management.get(regime_key, {})
                )
                
                self.policies[regime_name] = policy
            
            logger.info(f"Loaded regime policies for {len(self.policies)} regimes")
            
        except Exception as e:
            logger.error(f"Error loading regime policies: {e}")
            self._create_default_policies()
    
    def _create_default_policies(self) -> None:
        """Create default policies when configuration loading fails."""
        logger.warning("Creating default regime policies")
        
        # Default feature flags
        self.feature_flags = {
            'enable_regime_weights': True,
            'enable_regime_kelly': True,
            'enable_regime_brackets': True,
            'enable_regime_lookbacks': True,
            'enable_regime_risk': True,
            'enable_regime_positions': True,
            'enable_regime_transitions': True
        }
        
        # Default transition rules
        self.transition_rules = {
            'min_regime_duration': 3,
            'regime_change_threshold': 0.7,
            'transition_smoothing': 0.3,
            'regime_persistence': 5
        }
        
        # Default policies for each regime
        default_ensemble_weights = {
            'technical_analyst': 0.3,
            'sentiment_analyst': 0.25,
            'fundamental_analyst': 0.2,
            'risk_analyst': 0.15,
            'market_regime_analyst': 0.1
        }
        
        default_kelly_adjustments = {
            'kelly_multiplier': 1.0,
            'max_position_risk': 0.02,
            'confidence_threshold': 0.7
        }
        
        default_atr_brackets = {
            'atr_multiplier': 2.0,
            'r_multiple': 1.5,
            'min_stop_loss_percent': 0.02,
            'max_stop_loss_percent': 0.10
        }
        
        default_lookback_windows = {
            'price_lookback': 20,
            'volume_lookback': 15,
            'volatility_lookback': 14,
            'sentiment_lookback': 7
        }
        
        default_position_management = {
            'max_positions': 10,
            'max_holding_days': 25,
            'rebalance_frequency': 4,
            'stop_loss_trailing': False
        }
        
        default_risk_management = {
            'max_portfolio_risk': 0.18,
            'daily_loss_limit': 0.025,
            'drawdown_threshold': 0.045,
            'correlation_limit': 0.65
        }
        
        # Create default policies for all regimes
        for regime in MarketRegime:
            policy = RegimePolicy(
                regime=regime,
                ensemble_weights=default_ensemble_weights.copy(),
                kelly_adjustments=default_kelly_adjustments.copy(),
                atr_brackets=default_atr_brackets.copy(),
                lookback_windows=default_lookback_windows.copy(),
                position_management=default_position_management.copy(),
                risk_management=default_risk_management.copy()
            )
            self.policies[regime] = policy
    
    def get_ensemble_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get ensemble weights for a specific regime."""
        if not self.feature_flags.get('enable_regime_weights', True):
            return self._get_default_ensemble_weights()
        
        policy = self.policies.get(regime)
        if policy and policy.ensemble_weights:
            return policy.ensemble_weights
        
        return self._get_default_ensemble_weights()
    
    def get_kelly_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """Get Kelly sizing adjustments for a specific regime."""
        if not self.feature_flags.get('enable_regime_kelly', True):
            return self._get_default_kelly_adjustments()
        
        policy = self.policies.get(regime)
        if policy and policy.kelly_adjustments:
            return policy.kelly_adjustments
        
        return self._get_default_kelly_adjustments()
    
    def get_atr_brackets(self, regime: MarketRegime) -> Dict[str, float]:
        """Get ATR bracket parameters for a specific regime."""
        if not self.feature_flags.get('enable_regime_brackets', True):
            return self._get_default_atr_brackets()
        
        policy = self.policies.get(regime)
        if policy and policy.atr_brackets:
            return policy.atr_brackets
        
        return self._get_default_atr_brackets()
    
    def get_lookback_windows(self, regime: MarketRegime) -> Dict[str, int]:
        """Get lookback windows for a specific regime."""
        if not self.feature_flags.get('enable_regime_lookbacks', True):
            return self._get_default_lookback_windows()
        
        policy = self.policies.get(regime)
        if policy and policy.lookback_windows:
            return policy.lookback_windows
        
        return self._get_default_lookback_windows()
    
    def get_position_management(self, regime: MarketRegime) -> Dict[str, Any]:
        """Get position management parameters for a specific regime."""
        if not self.feature_flags.get('enable_regime_positions', True):
            return self._get_default_position_management()
        
        policy = self.policies.get(regime)
        if policy and policy.position_management:
            return policy.position_management
        
        return self._get_default_position_management()
    
    def get_risk_management(self, regime: MarketRegime) -> Dict[str, float]:
        """Get risk management parameters for a specific regime."""
        if not self.feature_flags.get('enable_regime_risk', True):
            return self._get_default_risk_management()
        
        policy = self.policies.get(regime)
        if policy and policy.risk_management:
            return policy.risk_management
        
        return self._get_default_risk_management()
    
    def get_transition_rules(self) -> Dict[str, Any]:
        """Get regime transition rules."""
        return self.transition_rules.copy()
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature flag is enabled."""
        return self.feature_flags.get(feature, False)
    
    def _get_default_ensemble_weights(self) -> Dict[str, float]:
        """Get default ensemble weights."""
        return {
            'technical_analyst': 0.3,
            'sentiment_analyst': 0.25,
            'fundamental_analyst': 0.2,
            'risk_analyst': 0.15,
            'market_regime_analyst': 0.1
        }
    
    def _get_default_kelly_adjustments(self) -> Dict[str, float]:
        """Get default Kelly adjustments."""
        return {
            'kelly_multiplier': 1.0,
            'max_position_risk': 0.02,
            'confidence_threshold': 0.7
        }
    
    def _get_default_atr_brackets(self) -> Dict[str, float]:
        """Get default ATR brackets."""
        return {
            'atr_multiplier': 2.0,
            'r_multiple': 1.5,
            'min_stop_loss_percent': 0.02,
            'max_stop_loss_percent': 0.10
        }
    
    def _get_default_lookback_windows(self) -> Dict[str, int]:
        """Get default lookback windows."""
        return {
            'price_lookback': 20,
            'volume_lookback': 15,
            'volatility_lookback': 14,
            'sentiment_lookback': 7
        }
    
    def _get_default_position_management(self) -> Dict[str, Any]:
        """Get default position management."""
        return {
            'max_positions': 10,
            'max_holding_days': 25,
            'rebalance_frequency': 4,
            'stop_loss_trailing': False
        }
    
    def _get_default_risk_management(self) -> Dict[str, float]:
        """Get default risk management."""
        return {
            'max_portfolio_risk': 0.18,
            'daily_loss_limit': 0.025,
            'drawdown_threshold': 0.045,
            'correlation_limit': 0.65
        }
    
    def reload_policies(self) -> None:
        """Reload policies from configuration file."""
        logger.info("Reloading regime policies")
        self._load_policies()
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of all policies."""
        summary = {
            'feature_flags': self.feature_flags,
            'transition_rules': self.transition_rules,
            'regimes': {}
        }
        
        for regime, policy in self.policies.items():
            summary['regimes'][regime.value] = {
                'ensemble_weights': policy.ensemble_weights,
                'kelly_adjustments': policy.kelly_adjustments,
                'atr_brackets': policy.atr_brackets,
                'lookback_windows': policy.lookback_windows,
                'position_management': policy.position_management,
                'risk_management': policy.risk_management
            }
        
        return summary

# Global regime policy manager instance
_regime_policy_manager: Optional[RegimePolicyManager] = None

def get_regime_policy_manager() -> RegimePolicyManager:
    """Get the global regime policy manager instance."""
    global _regime_policy_manager
    if _regime_policy_manager is None:
        _regime_policy_manager = RegimePolicyManager()
    return _regime_policy_manager

def get_ensemble_weights(regime: MarketRegime) -> Dict[str, float]:
    """Get ensemble weights for a specific regime."""
    return get_regime_policy_manager().get_ensemble_weights(regime)

def get_kelly_adjustments(regime: MarketRegime) -> Dict[str, float]:
    """Get Kelly sizing adjustments for a specific regime."""
    return get_regime_policy_manager().get_kelly_adjustments(regime)

def get_atr_brackets(regime: MarketRegime) -> Dict[str, float]:
    """Get ATR bracket parameters for a specific regime."""
    return get_regime_policy_manager().get_atr_brackets(regime)

def get_lookback_windows(regime: MarketRegime) -> Dict[str, int]:
    """Get lookback windows for a specific regime."""
    return get_regime_policy_manager().get_lookback_windows(regime)

def get_position_management(regime: MarketRegime) -> Dict[str, Any]:
    """Get position management parameters for a specific regime."""
    return get_regime_policy_manager().get_position_management(regime)

def get_risk_management(regime: MarketRegime) -> Dict[str, float]:
    """Get risk management parameters for a specific regime."""
    return get_regime_policy_manager().get_risk_management(regime)

def is_feature_enabled(feature: str) -> bool:
    """Check if a feature flag is enabled."""
    return get_regime_policy_manager().is_feature_enabled(feature)
