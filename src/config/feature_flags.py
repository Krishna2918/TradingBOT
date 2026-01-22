"""
Feature Flags System - Runtime Feature Management

This module provides runtime feature flag management for safe rollout
and gradual enablement of new features in production.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class FeatureStatus(Enum):
    """Feature flag status enumeration."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    ROLLING_OUT = "rolling_out"
    ROLLBACK = "rollback"

@dataclass
class FeatureFlag:
    """Feature flag configuration."""
    name: str
    status: FeatureStatus
    description: str
    rollout_percentage: float = 0.0
    enabled_for_users: List[str] = None
    disabled_for_users: List[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    dependencies: List[str] = None
    metrics_threshold: Optional[Dict[str, float]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.enabled_for_users is None:
            self.enabled_for_users = []
        if self.disabled_for_users is None:
            self.disabled_for_users = []
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

class FeatureFlagManager:
    """Manages feature flags and their runtime configuration."""
    
    def __init__(self, config_file: str = "data/feature_flags.json"):
        """
        Initialize the feature flag manager.
        
        Args:
            config_file: Path to the feature flags configuration file
        """
        self.config_file = config_file
        self.flags: Dict[str, FeatureFlag] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        
        # Load existing configuration
        self._load_config()
        
        # Initialize default feature flags
        self._initialize_default_flags()
        
        logger.info(f"FeatureFlagManager initialized with {len(self.flags)} flags")
    
    def _load_config(self) -> None:
        """Load feature flags configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    
                for flag_name, flag_data in data.get('flags', {}).items():
                    # Convert string status to enum
                    if 'status' in flag_data:
                        flag_data['status'] = FeatureStatus(flag_data['status'])
                    
                    # Convert datetime strings back to datetime objects
                    for date_field in ['start_time', 'end_time', 'created_at', 'updated_at']:
                        if date_field in flag_data and flag_data[date_field]:
                            flag_data[date_field] = datetime.fromisoformat(flag_data[date_field])
                    
                    self.flags[flag_name] = FeatureFlag(**flag_data)
                
                self.metrics = data.get('metrics', {})
                logger.info(f"Loaded {len(self.flags)} feature flags from {self.config_file}")
            else:
                logger.info(f"Feature flags config file {self.config_file} not found, using defaults")
                
        except Exception as e:
            logger.error(f"Failed to load feature flags config: {e}")
    
    def _save_config(self) -> None:
        """Save feature flags configuration to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Convert to serializable format
            data = {
                'flags': {},
                'metrics': self.metrics,
                'last_updated': datetime.now().isoformat()
            }
            
            for flag_name, flag in self.flags.items():
                flag_dict = asdict(flag)
                # Convert enum to string
                flag_dict['status'] = flag.status.value
                # Convert datetime objects to ISO strings
                for date_field in ['start_time', 'end_time', 'created_at', 'updated_at']:
                    if flag_dict[date_field]:
                        flag_dict[date_field] = flag_dict[date_field].isoformat()
                
                data['flags'][flag_name] = flag_dict
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved feature flags config to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save feature flags config: {e}")
    
    def _initialize_default_flags(self) -> None:
        """Initialize default feature flags for the trading system."""
        default_flags = [
            FeatureFlag(
                name="adaptive_weights",
                status=FeatureStatus.DISABLED,
                description="Enable adaptive ensemble weights based on model performance",
                rollout_percentage=0.0,
                dependencies=[],
                metrics_threshold={
                    "accuracy_threshold": 0.6,
                    "brier_threshold": 0.3
                }
            ),
            FeatureFlag(
                name="confidence_calibration",
                status=FeatureStatus.DISABLED,
                description="Enable Bayesian confidence calibration for risk management",
                rollout_percentage=0.0,
                dependencies=[],
                metrics_threshold={
                    "calibration_error_threshold": 0.1,
                    "min_trades_threshold": 50
                }
            ),
            FeatureFlag(
                name="drawdown_aware_kelly",
                status=FeatureStatus.DISABLED,
                description="Enable drawdown-aware Kelly position sizing",
                rollout_percentage=0.0,
                dependencies=[],
                metrics_threshold={
                    "max_drawdown_threshold": 0.05,
                    "kelly_scale_threshold": 0.3
                }
            ),
            FeatureFlag(
                name="atr_brackets",
                status=FeatureStatus.DISABLED,
                description="Enable ATR-based stop loss and take profit brackets",
                rollout_percentage=0.0,
                dependencies=[],
                metrics_threshold={
                    "atr_accuracy_threshold": 0.7,
                    "bracket_success_rate": 0.6
                }
            ),
            FeatureFlag(
                name="regime_awareness",
                status=FeatureStatus.DISABLED,
                description="Enable market regime detection and adaptation",
                rollout_percentage=0.0,
                dependencies=[],
                metrics_threshold={
                    "regime_accuracy_threshold": 0.8,
                    "transition_detection_delay": 5.0
                }
            ),
            FeatureFlag(
                name="data_quality_gates",
                status=FeatureStatus.ENABLED,
                description="Enable data quality validation before trading decisions",
                rollout_percentage=100.0,
                dependencies=[],
                metrics_threshold={
                    "quality_threshold": 0.8,
                    "max_missing_data": 0.1
                }
            ),
            FeatureFlag(
                name="api_budget_management",
                status=FeatureStatus.ENABLED,
                description="Enable API budget management and rate limiting",
                rollout_percentage=100.0,
                dependencies=[],
                metrics_threshold={
                    "rate_limit_hit_threshold": 0.05,
                    "response_time_threshold": 5.0
                }
            ),
            FeatureFlag(
                name="enhanced_monitoring",
                status=FeatureStatus.ENABLED,
                description="Enable enhanced system monitoring and metrics",
                rollout_percentage=100.0,
                dependencies=[],
                metrics_threshold={
                    "uptime_threshold": 0.99,
                    "latency_threshold": 2.0
                }
            )
        ]
        
        # Add default flags if they don't exist
        for flag in default_flags:
            if flag.name not in self.flags:
                self.flags[flag.name] = flag
                logger.info(f"Added default feature flag: {flag.name}")
        
        # Save configuration
        self._save_config()
    
    def is_enabled(self, flag_name: str, user_id: str = "default") -> bool:
        """
        Check if a feature flag is enabled for a specific user.
        
        Args:
            flag_name: Name of the feature flag
            user_id: User identifier for user-specific flags
            
        Returns:
            True if the feature is enabled, False otherwise
        """
        if flag_name not in self.flags:
            logger.warning(f"Feature flag '{flag_name}' not found")
            return False
        
        flag = self.flags[flag_name]
        
        # Check if explicitly disabled for this user
        if user_id in flag.disabled_for_users:
            return False
        
        # Check if explicitly enabled for this user
        if user_id in flag.enabled_for_users:
            return True
        
        # Check status
        if flag.status == FeatureStatus.DISABLED:
            return False
        elif flag.status == FeatureStatus.ENABLED:
            return True
        elif flag.status == FeatureStatus.ROLLING_OUT:
            # Check rollout percentage
            return self._should_rollout_for_user(user_id, flag.rollout_percentage)
        elif flag.status == FeatureStatus.ROLLBACK:
            return False
        
        return False
    
    def _should_rollout_for_user(self, user_id: str, rollout_percentage: float) -> bool:
        """
        Determine if a user should receive a feature based on rollout percentage.
        
        Args:
            user_id: User identifier
            rollout_percentage: Percentage of users to enable the feature for
            
        Returns:
            True if the user should receive the feature
        """
        # Simple hash-based rollout (deterministic)
        import hashlib
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return (user_hash % 100) < rollout_percentage
    
    def enable_feature(self, flag_name: str, rollout_percentage: float = 100.0) -> bool:
        """
        Enable a feature flag.
        
        Args:
            flag_name: Name of the feature flag
            rollout_percentage: Percentage of users to enable for (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        if flag_name not in self.flags:
            logger.error(f"Feature flag '{flag_name}' not found")
            return False
        
        flag = self.flags[flag_name]
        flag.status = FeatureStatus.ROLLING_OUT if rollout_percentage < 100.0 else FeatureStatus.ENABLED
        flag.rollout_percentage = rollout_percentage
        flag.updated_at = datetime.now()
        
        self._save_config()
        logger.info(f"Enabled feature flag '{flag_name}' with {rollout_percentage}% rollout")
        return True
    
    def disable_feature(self, flag_name: str) -> bool:
        """
        Disable a feature flag.
        
        Args:
            flag_name: Name of the feature flag
            
        Returns:
            True if successful, False otherwise
        """
        if flag_name not in self.flags:
            logger.error(f"Feature flag '{flag_name}' not found")
            return False
        
        flag = self.flags[flag_name]
        flag.status = FeatureStatus.DISABLED
        flag.rollout_percentage = 0.0
        flag.updated_at = datetime.now()
        
        self._save_config()
        logger.info(f"Disabled feature flag '{flag_name}'")
        return True
    
    def rollback_feature(self, flag_name: str) -> bool:
        """
        Rollback a feature flag (emergency disable).
        
        Args:
            flag_name: Name of the feature flag
            
        Returns:
            True if successful, False otherwise
        """
        if flag_name not in self.flags:
            logger.error(f"Feature flag '{flag_name}' not found")
            return False
        
        flag = self.flags[flag_name]
        flag.status = FeatureStatus.ROLLBACK
        flag.rollout_percentage = 0.0
        flag.updated_at = datetime.now()
        
        self._save_config()
        logger.warning(f"Rolled back feature flag '{flag_name}'")
        return True
    
    def get_feature_status(self, flag_name: str) -> Optional[FeatureFlag]:
        """
        Get the status of a feature flag.
        
        Args:
            flag_name: Name of the feature flag
            
        Returns:
            FeatureFlag object or None if not found
        """
        return self.flags.get(flag_name)
    
    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """
        Get all feature flags.
        
        Returns:
            Dictionary of all feature flags
        """
        return self.flags.copy()
    
    def update_metrics(self, flag_name: str, metrics: Dict[str, Any]) -> None:
        """
        Update metrics for a feature flag.
        
        Args:
            flag_name: Name of the feature flag
            metrics: Metrics data to update
        """
        if flag_name not in self.metrics:
            self.metrics[flag_name] = {}
        
        self.metrics[flag_name].update(metrics)
        self.metrics[flag_name]['last_updated'] = datetime.now().isoformat()
        
        # Check if metrics exceed thresholds and trigger rollback
        self._check_metrics_thresholds(flag_name)
        
        self._save_config()
    
    def _check_metrics_thresholds(self, flag_name: str) -> None:
        """
        Check if metrics exceed thresholds and trigger rollback if needed.
        
        Args:
            flag_name: Name of the feature flag
        """
        if flag_name not in self.flags or flag_name not in self.metrics:
            return
        
        flag = self.flags[flag_name]
        metrics = self.metrics[flag_name]
        
        if not flag.metrics_threshold:
            return
        
        # Check each threshold
        for threshold_name, threshold_value in flag.metrics_threshold.items():
            if threshold_name in metrics:
                metric_value = metrics[threshold_name]
                
                # Check if metric exceeds threshold (assuming higher is worse)
                if isinstance(metric_value, (int, float)) and isinstance(threshold_value, (int, float)):
                    if metric_value > threshold_value:
                        logger.warning(f"Feature flag '{flag_name}' metric '{threshold_name}' "
                                     f"({metric_value}) exceeds threshold ({threshold_value})")
                        
                        # Auto-rollback if enabled
                        if os.environ.get('AUTO_ROLLBACK_ON_THRESHOLD', 'false').lower() == 'true':
                            self.rollback_feature(flag_name)
                            logger.error(f"Auto-rollback triggered for feature flag '{flag_name}'")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all feature flag metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        summary = {
            'total_flags': len(self.flags),
            'enabled_flags': len([f for f in self.flags.values() if f.status == FeatureStatus.ENABLED]),
            'rolling_out_flags': len([f for f in self.flags.values() if f.status == FeatureStatus.ROLLING_OUT]),
            'disabled_flags': len([f for f in self.flags.values() if f.status == FeatureStatus.DISABLED]),
            'rollback_flags': len([f for f in self.flags.values() if f.status == FeatureStatus.ROLLBACK]),
            'flags_with_metrics': len(self.metrics),
            'last_updated': datetime.now().isoformat()
        }
        
        return summary

# Global instance
_feature_flag_manager: Optional[FeatureFlagManager] = None

def get_feature_flag_manager() -> FeatureFlagManager:
    """Get the global feature flag manager instance."""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager

def is_feature_enabled(flag_name: str, user_id: str = "default") -> bool:
    """Check if a feature is enabled."""
    manager = get_feature_flag_manager()
    return manager.is_enabled(flag_name, user_id)

def enable_feature(flag_name: str, rollout_percentage: float = 100.0) -> bool:
    """Enable a feature flag."""
    manager = get_feature_flag_manager()
    return manager.enable_feature(flag_name, rollout_percentage)

def disable_feature(flag_name: str) -> bool:
    """Disable a feature flag."""
    manager = get_feature_flag_manager()
    return manager.disable_feature(flag_name)

def rollback_feature(flag_name: str) -> bool:
    """Rollback a feature flag."""
    manager = get_feature_flag_manager()
    return manager.rollback_feature(flag_name)

def get_feature_status(flag_name: str) -> Optional[FeatureFlag]:
    """Get feature flag status."""
    manager = get_feature_flag_manager()
    return manager.get_feature_status(flag_name)

def update_feature_metrics(flag_name: str, metrics: Dict[str, Any]) -> None:
    """Update feature flag metrics."""
    manager = get_feature_flag_manager()
    manager.update_metrics(flag_name, metrics)
