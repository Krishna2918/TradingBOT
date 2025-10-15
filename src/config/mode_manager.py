"""
Mode Manager - Dual Mode Architecture (LIVE/DEMO)

This module manages the dual mode operation of the AI trading system,
ensuring identical functionality in both LIVE and DEMO modes with
separate data isolation.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class ModeConfig:
    """Configuration for a specific mode."""
    mode: str
    database_path: str
    order_execution: str  # 'REAL' or 'PAPER'
    balance_tracking: bool
    risk_limits: Dict[str, float]
    description: str
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for checking if attribute exists."""
        return hasattr(self, key) or key in self.risk_limits
    
    def __getitem__(self, key: str):
        """Support dictionary-style access."""
        if hasattr(self, key):
            return getattr(self, key)
        elif key in self.risk_limits:
            return self.risk_limits[key]
        raise KeyError(f"Key '{key}' not found in ModeConfig")

class ModeManager:
    """Manages dual mode operation (LIVE/DEMO) for the trading system."""
    
    def __init__(self, config_path: str = "config/mode_config.json"):
        """Initialize Mode Manager with configuration."""
        self.config_path = config_path
        self.current_mode = "DEMO"  # Default to DEMO for safety
        self.modes = self._load_mode_configs()
        self._validate_modes()
        
        logger.info(f"Mode Manager initialized. Current mode: {self.current_mode}")
    
    def _load_mode_configs(self) -> Dict[str, ModeConfig]:
        """Load mode configurations from file."""
        default_configs = {
            "LIVE": ModeConfig(
                mode="LIVE",
                database_path="data/trading_live.db",
                order_execution="REAL",
                balance_tracking=True,
                risk_limits={
                    "max_position_size": 0.10,  # 10% max per position
                    "max_daily_drawdown": 0.05,  # 5% max daily loss
                    "max_portfolio_risk": 0.20,  # 20% max total risk
                    "min_confidence": 0.75,  # 75% min confidence for trades
                    "max_positions": 10
                },
                description="Live trading with real money and real orders"
            ),
            "DEMO": ModeConfig(
                mode="DEMO",
                database_path="data/trading_demo.db",
                order_execution="PAPER",
                balance_tracking=True,
                risk_limits={
                    "max_position_size": 0.10,  # Same limits as LIVE
                    "max_daily_drawdown": 0.05,
                    "max_portfolio_risk": 0.20,
                    "min_confidence": 0.70,  # Slightly lower for testing
                    "max_positions": 10
                },
                description="Demo trading with paper money and simulated orders"
            )
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Load from file if exists, otherwise create with defaults
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    return {
                        mode: ModeConfig(**config) 
                        for mode, config in config_data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load mode config: {e}. Using defaults.")
        
        # Save default configs
        self._save_mode_configs(default_configs)
        return default_configs
    
    def _save_mode_configs(self, configs: Dict[str, ModeConfig]) -> None:
        """Save mode configurations to file."""
        try:
            config_data = {
                mode: {
                    "mode": config.mode,
                    "database_path": config.database_path,
                    "order_execution": config.order_execution,
                    "balance_tracking": config.balance_tracking,
                    "risk_limits": config.risk_limits,
                    "description": config.description
                }
                for mode, config in configs.items()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info("Mode configurations saved successfully")
        except Exception as e:
            logger.error(f"Failed to save mode configs: {e}")
    
    def _validate_modes(self) -> None:
        """Validate that all required modes are configured."""
        required_modes = ["LIVE", "DEMO"]
        for mode in required_modes:
            if mode not in self.modes:
                raise ValueError(f"Required mode '{mode}' not configured")
        
        # Validate database paths
        for mode, config in self.modes.items():
            db_dir = os.path.dirname(config.database_path)
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Database directory ensured for {mode}: {db_dir}")
    
    def get_current_mode(self) -> str:
        """Get the current active mode."""
        return self.current_mode
    
    def set_mode(self, mode: str) -> bool:
        """Set the current mode with validation."""
        if mode not in self.modes:
            logger.error(f"Invalid mode: {mode}. Available modes: {list(self.modes.keys())}")
            return False
        
        old_mode = self.current_mode
        self.current_mode = mode
        
        # Log mode change
        logger.info(f"Mode changed from {old_mode} to {mode}")
        
        # Validate mode switch
        if not self.validate_mode():
            logger.error(f"Mode validation failed for {mode}. Reverting to {old_mode}")
            self.current_mode = old_mode
            return False
        
        return True
    
    def validate_mode(self) -> bool:
        """Validate the current mode configuration."""
        try:
            config = self.modes[self.current_mode]
            
            # Check database path
            if not os.path.exists(os.path.dirname(config.database_path)):
                logger.error(f"Database directory does not exist: {config.database_path}")
                return False
            
            # Check risk limits
            required_limits = ["max_position_size", "max_daily_drawdown", "max_portfolio_risk", "min_confidence", "max_positions"]
            for limit in required_limits:
                if limit not in config.risk_limits:
                    logger.error(f"Missing risk limit: {limit}")
                    return False
                
                if not isinstance(config.risk_limits[limit], (int, float)):
                    logger.error(f"Invalid risk limit type for {limit}")
                    return False
            
            # Validate risk limit ranges
            if not (0 < config.risk_limits["max_position_size"] <= 1):
                logger.error("max_position_size must be between 0 and 1")
                return False
            
            if not (0 < config.risk_limits["max_daily_drawdown"] <= 1):
                logger.error("max_daily_drawdown must be between 0 and 1")
                return False
            
            if not (0 < config.risk_limits["min_confidence"] <= 1):
                logger.error("min_confidence must be between 0 and 1")
                return False
            
            logger.info(f"Mode validation successful for {self.current_mode}")
            return True
            
        except Exception as e:
            logger.error(f"Mode validation error: {e}")
            return False
    
    def get_mode_config(self, mode: Optional[str] = None) -> ModeConfig:
        """Get configuration for the current mode."""
        return self.modes[self.current_mode]
    
    def get_database_path(self) -> str:
        """Get database path for the current mode."""
        return self.modes[self.current_mode].database_path
    
    def is_live_mode(self) -> bool:
        """Check if currently in LIVE mode."""
        return self.current_mode == "LIVE"
    
    def is_demo_mode(self) -> bool:
        """Check if currently in DEMO mode."""
        return self.current_mode == "DEMO"
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get risk limits for the current mode."""
        return self.modes[self.current_mode].risk_limits.copy()
    
    def get_order_execution_type(self) -> str:
        """Get order execution type for the current mode."""
        return self.modes[self.current_mode].order_execution
    
    def is_real_trading(self) -> bool:
        """Check if real trading is enabled (LIVE mode)."""
        return self.is_live_mode() and self.modes[self.current_mode].order_execution == "REAL"
    
    def is_paper_trading(self) -> bool:
        """Check if paper trading is enabled (DEMO mode)."""
        return self.is_demo_mode() or self.modes[self.current_mode].order_execution == "PAPER"
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Get comprehensive mode information."""
        config = self.modes[self.current_mode]
        return {
            "current_mode": self.current_mode,
            "database_path": config.database_path,
            "order_execution": config.order_execution,
            "balance_tracking": config.balance_tracking,
            "risk_limits": config.risk_limits,
            "description": config.description,
            "is_live": self.is_live_mode(),
            "is_demo": self.is_demo_mode(),
            "is_real_trading": self.is_real_trading(),
            "is_paper_trading": self.is_paper_trading()
        }
    
    def toggle_mode(self) -> bool:
        """Toggle between LIVE and DEMO modes."""
        new_mode = "DEMO" if self.current_mode == "LIVE" else "LIVE"
        return self.set_mode(new_mode)
    
    def validate_mode_switch(self, from_mode: str, to_mode: str) -> bool:
        """Validate a mode switch operation."""
        if from_mode not in self.modes or to_mode not in self.modes:
            logger.error(f"Invalid mode switch: {from_mode} -> {to_mode}")
            return False
        
        # Additional validation logic can be added here
        # For example, checking if there are open positions that need to be handled
        
        logger.info(f"Mode switch validation passed: {from_mode} -> {to_mode}")
        return True

# Global mode manager instance
_mode_manager: Optional[ModeManager] = None

def get_mode_manager() -> ModeManager:
    """Get the global mode manager instance."""
    global _mode_manager
    if _mode_manager is None:
        _mode_manager = ModeManager()
    return _mode_manager

def get_current_mode() -> str:
    """Get the current mode."""
    return get_mode_manager().get_current_mode()

def set_mode(mode: str) -> bool:
    """Set the current mode."""
    return get_mode_manager().set_mode(mode)

def is_live_mode() -> bool:
    """Check if in LIVE mode."""
    return get_mode_manager().is_live_mode()

def is_demo_mode() -> bool:
    """Check if in DEMO mode."""
    return get_mode_manager().is_demo_mode()

def get_database_path() -> str:
    """Get database path for current mode."""
    return get_mode_manager().get_database_path()

def get_risk_limits() -> Dict[str, float]:
    """Get risk limits for current mode."""
    return get_mode_manager().get_risk_limits()

def is_real_trading() -> bool:
    """Check if real trading is enabled."""
    return get_mode_manager().is_real_trading()

def is_paper_trading() -> bool:
    """Check if paper trading is enabled."""
    return get_mode_manager().is_paper_trading()

def get_mode_config() -> ModeConfig:
    """Get configuration for current mode."""
    return get_mode_manager().get_mode_config()
