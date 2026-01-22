"""
Kill Switches Module
Implements emergency stop mechanisms for various risk scenarios
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import yaml
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class KillSwitchState:
    """Current kill switch state"""
    active: bool
    reason: str
    activated_at: Optional[datetime]
    deactivated_at: Optional[datetime]
    manual_override: bool

class KillSwitchManager:
    """Manages kill switches and emergency stops"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['risk']
        
        self.kill_switches = self.config['kill_switches']
        self.kill_switch_state = KillSwitchState(
            active=False,
            reason="",
            activated_at=None,
            deactivated_at=None,
            manual_override=False
        )
        
        self.emergency_actions = []
        self.recovery_actions = []
    
    def activate_kill_switch(self, reason: str, source: str = "system"):
        """Activate kill switch for specified reason"""
        if self.kill_switch_state.active:
            logger.warning(f"Kill switch already active: {self.kill_switch_state.reason}")
            return
        
        self.kill_switch_state.active = True
        self.kill_switch_state.reason = reason
        self.kill_switch_state.activated_at = datetime.now()
        self.kill_switch_state.manual_override = False
        
        logger.critical(f"KILL SWITCH ACTIVATED: {reason} (Source: {source})")
        
        # Execute emergency actions
        self.execute_emergency_actions(reason)
    
    def deactivate_kill_switch(self, reason: str = "manual_deactivation"):
        """Deactivate kill switch"""
        if not self.kill_switch_state.active:
            logger.warning("Kill switch not active")
            return
        
        self.kill_switch_state.active = False
        self.kill_switch_state.deactivated_at = datetime.now()
        
        logger.info(f"Kill switch deactivated: {reason}")
        
        # Execute recovery actions
        self.execute_recovery_actions()
    
    def execute_emergency_actions(self, reason: str):
        """Execute emergency actions when kill switch is activated"""
        actions = {
            "daily_loss_limit": [
                "close_all_positions",
                "disable_new_orders",
                "notify_risk_team"
            ],
            "max_drawdown": [
                "close_risky_positions",
                "reduce_leverage",
                "activate_hedging"
            ],
            "margin_utilization": [
                "force_position_reduction",
                "disable_leverage",
                "emergency_capital_injection"
            ],
            "broker_failure": [
                "switch_to_backup_broker",
                "pause_trading",
                "verify_connectivity"
            ],
            "data_feed_stall": [
                "restart_data_pipeline",
                "use_backup_feeds",
                "pause_trading"
            ]
        }
        
        if reason in actions:
            for action in actions[reason]:
                logger.info(f"Executing emergency action: {action}")
                self.emergency_actions.append({
                    "action": action,
                    "executed_at": datetime.now(),
                    "status": "executed"
                })
    
    def execute_recovery_actions(self):
        """Execute recovery actions when kill switch is deactivated"""
        recovery_actions = [
            "verify_system_health",
            "check_risk_metrics",
            "resume_trading_gradually",
            "monitor_performance"
        ]
        
        for action in recovery_actions:
            logger.info(f"Executing recovery action: {action}")
            self.recovery_actions.append({
                "action": action,
                "executed_at": datetime.now(),
                "status": "executed"
            })
    
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is currently active"""
        return self.kill_switch_state.active
    
    def get_kill_switch_state(self) -> KillSwitchState:
        """Get current kill switch state"""
        return self.kill_switch_state
    
    def manual_override(self, reason: str):
        """Manual override of kill switch (admin only)"""
        if self.kill_switch_state.active:
            self.kill_switch_state.manual_override = True
            logger.warning(f"Manual override activated: {reason}")
            self.deactivate_kill_switch("manual_override")
    
    def get_emergency_actions(self) -> List[Dict]:
        """Get list of executed emergency actions"""
        return self.emergency_actions
    
    def get_recovery_actions(self) -> List[Dict]:
        """Get list of executed recovery actions"""
        return self.recovery_actions

# Test the kill switch manager
if __name__ == "__main__":
    manager = KillSwitchManager("config/risk_config.yaml")
    
    # Test kill switch activation
    manager.activate_kill_switch("daily_loss_limit", "risk_monitor")
    print(f"Kill switch active: {manager.is_kill_switch_active()}")
    
    # Test deactivation
    manager.deactivate_kill_switch("manual_review")
    print(f"Kill switch active: {manager.is_kill_switch_active()}")
    
    # Test emergency actions
    print(f"Emergency actions: {len(manager.get_emergency_actions())}")
    print(f"Recovery actions: {len(manager.get_recovery_actions())}")

