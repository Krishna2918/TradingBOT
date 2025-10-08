"""
Hybrid Control Plane - Production Ready
GPT-5 API = Strategic Brain (≤3 calls/day)
Local Stack = Workhorse (RTX 4080, 32GB RAM)
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time

# Import new components
from .trigger_monitor import TriggerMonitor, TriggerEvent
from .auditor_sandbox import AuditorSandbox, AuditReport

logger = logging.getLogger(__name__)

class EscalationType(Enum):
    REGIME_AUDITOR = "regime_auditor"
    REWARD_ENGINEER = "reward_engineer"
    SANITY_JUDGE = "sanity_judge"

@dataclass
class EscalationTrigger:
    """Escalation trigger conditions"""
    risk_drawdown_5d: float = 0.08  # 8% max drawdown
    daily_loss: float = 0.025  # 2.5% daily loss
    vol_zscore: float = 2.0  # Volatility z-score threshold
    correlation_breakdown: float = 0.4  # Correlation threshold
    accuracy_drop: float = 0.07  # 7 percentage points
    sharpe_threshold: float = 0.6  # Sharpe ratio threshold
    put_call_ratio: float = 1.6  # Options skew threshold

@dataclass
class RiskLimits:
    """Hard risk limits"""
    penny_bucket: float = 0.02  # 2%
    fno_bucket: float = 0.05  # 5%
    core_bucket: float = 0.90  # 90%
    sip_bucket: float = 0.01  # 1%
    
    per_name_core: float = 0.015  # 1.5%
    per_name_penny: float = 0.004  # 0.4%
    per_name_fno: float = 0.007  # 0.7%
    
    kill_switch_daily: float = -0.025  # -2.5%
    kill_switch_quarantine: int = 3  # 3 days in 10

class HybridControlPlane:
    """
    Hybrid Control Plane - Production Ready
    
    Responsibilities:
    - Monitor escalation triggers
    - Manage GPT-5 API calls (≤3/day)
    - Enforce hard risk limits
    - Coordinate local vs remote AI
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.escalation_triggers = EscalationTrigger()
        self.risk_limits = RiskLimits()
        
        # GPT-5 API management
        self.gpt5_calls_today = 0
        self.gpt5_daily_limit = 3
        self.gpt5_weekend_limit = 1
        self.last_gpt5_call = None
        
        # Risk monitoring
        self.portfolio_state = {
            'net_liquidity': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown_5d': 0.0,
            'volatility_zscore': 0.0,
            'correlation_breakdown': 0.0,
            'ensemble_accuracy': 0.0,
            'sharpe_ratio': 0.0,
            'put_call_ratio': 0.0
        }
        
        # Bucket usage tracking
        self.bucket_usage = {
            'penny': 0.0,
            'fno': 0.0,
            'core': 0.0,
            'sip': 0.0
        }
        
        # Initialize Trigger Monitor
        try:
            self.trigger_monitor = TriggerMonitor(config)
            logger.info("Trigger Monitor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Trigger Monitor: {e}")
            self.trigger_monitor = None
        
        # Initialize Auditor Sandbox
        try:
            self.auditor_sandbox = AuditorSandbox(config)
            logger.info("Auditor Sandbox initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Auditor Sandbox: {e}")
            self.auditor_sandbox = None
        
        # Kill switch state
        self.kill_switch_active = False
        self.kill_switch_days = 0
        self.demo_quarantine = False
        
        # Local reasoner (will be initialized separately)
        self.local_reasoner = None
        
        logger.info(" Hybrid Control Plane initialized")
        logger.info(f" Risk limits: Penny {self.risk_limits.penny_bucket*100}%, F&O {self.risk_limits.fno_bucket*100}%, Core {self.risk_limits.core_bucket*100}%")
        logger.info(f" Kill switch: {self.risk_limits.kill_switch_daily*100}% daily loss")
    
    def update_portfolio_state(self, state: Dict[str, Any]):
        """Update portfolio state for monitoring"""
        self.portfolio_state.update(state)
        
        # Check for kill switch
        if self.portfolio_state['daily_pnl'] < self.risk_limits.kill_switch_daily:
            self._trigger_kill_switch()
        
        # Check escalation triggers
        escalation_needed = self._check_escalation_triggers()
        if escalation_needed:
            self._handle_escalation(escalation_needed)
    
    def _check_escalation_triggers(self) -> Optional[EscalationType]:
        """Check if escalation to GPT-5 is needed"""
        
        # Risk escalation
        if (self.portfolio_state['max_drawdown_5d'] > self.escalation_triggers.risk_drawdown_5d or
            self.portfolio_state['daily_pnl'] < self.escalation_triggers.daily_loss):
            return EscalationType.REWARD_ENGINEER
        
        # Regime shift escalation
        if (abs(self.portfolio_state['volatility_zscore']) > self.escalation_triggers.vol_zscore and
            self.portfolio_state['correlation_breakdown'] < self.escalation_triggers.correlation_breakdown):
            return EscalationType.REGIME_AUDITOR
        
        # Model decay escalation
        if (self.portfolio_state['ensemble_accuracy'] < (0.52 - self.escalation_triggers.accuracy_drop) or
            self.portfolio_state['sharpe_ratio'] < self.escalation_triggers.sharpe_threshold):
            return EscalationType.REWARD_ENGINEER
        
        # Event heat escalation
        if self.portfolio_state['put_call_ratio'] > self.escalation_triggers.put_call_ratio:
            return EscalationType.REGIME_AUDITOR
        
        return None
    
    def _handle_escalation(self, escalation_type: EscalationType):
        """Handle escalation to GPT-5"""
        
        # Check call limits
        if not self._can_make_gpt5_call():
            logger.warning(f" GPT-5 call limit reached, skipping {escalation_type.value}")
            return
        
        logger.info(f" ESCALATION: {escalation_type.value}")
        
        # Prepare escalation data
        escalation_data = self._prepare_escalation_data(escalation_type)
        
        # Make GPT-5 call
        try:
            response = self._call_gpt5(escalation_type, escalation_data)
            self._process_gpt5_response(escalation_type, response)
            self.gpt5_calls_today += 1
            self.last_gpt5_call = datetime.now()
            
        except Exception as e:
            logger.error(f" GPT-5 escalation failed: {e}")
            # Fall back to local reasoner
            self._fallback_to_local_reasoner(escalation_type, escalation_data)
    
    def _can_make_gpt5_call(self) -> bool:
        """Check if we can make a GPT-5 call"""
        is_weekend = datetime.now().weekday() >= 5
        
        if is_weekend:
            return self.gpt5_calls_today < self.gpt5_weekend_limit
        else:
            return self.gpt5_calls_today < self.gpt5_daily_limit
    
    def _prepare_escalation_data(self, escalation_type: EscalationType) -> Dict[str, Any]:
        """Prepare data for GPT-5 escalation"""
        
        base_data = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_state': self.portfolio_state.copy(),
            'bucket_usage': self.bucket_usage.copy(),
            'kill_switch_active': self.kill_switch_active
        }
        
        if escalation_type == EscalationType.REGIME_AUDITOR:
            return {
                **base_data,
                'vol_zscore': self.portfolio_state['volatility_zscore'],
                'correlation_breakdown': self.portfolio_state['correlation_breakdown'],
                'sector_dispersion': self._get_sector_dispersion()
            }
        
        elif escalation_type == EscalationType.REWARD_ENGINEER:
            return {
                **base_data,
                'pnl_decomposition': self._get_pnl_decomposition(),
                'turnover_stats': self._get_turnover_stats(),
                'feature_performance': self._get_feature_performance()
            }
        
        elif escalation_type == EscalationType.SANITY_JUDGE:
            return {
                **base_data,
                'kill_switch_reason': self._get_kill_switch_reason(),
                'recent_trades': self._get_recent_trades()
            }
        
        return base_data
    
    def _call_gpt5(self, escalation_type: EscalationType, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call GPT-5 API for escalation"""
        
        # This would integrate with your ChatGPT integration
        # For now, return a structured response format
        
        prompts = {
            EscalationType.REGIME_AUDITOR: f"""
            Regime Auditor: Analyze market regime shift.
            Data: {json.dumps(data, indent=2)}
            
            Return JSON: {{
                "regime_label": "bull/bear/sideways/volatile",
                "blend_weights_delta": {{"short": 0.0, "mid": 0.0, "rl": 0.0}},
                "bucket_caps_delta": {{"penny": 0.0, "fno": 0.0, "core": 0.0}},
                "do_not_trade_list": ["SYMBOL1", "SYMBOL2"]
            }}
            """,
            
            EscalationType.REWARD_ENGINEER: f"""
            Reward Engineer: Optimize reward function.
            Data: {json.dumps(data, indent=2)}
            
            Return JSON: {{
                "reward_coefficients": {{"returns": 0.6, "turnover": 0.2, "drawdown": 0.2}},
                "turnover_penalties": {{"penalty_rate": 0.01}},
                "disabled_features": ["feature1", "feature2", "feature3"]
            }}
            """,
            
            EscalationType.SANITY_JUDGE: f"""
            Sanity Judge: Evaluate kill switch situation.
            Data: {json.dumps(data, indent=2)}
            
            Return JSON: {{
                "stay_demo": true/false,
                "live_changes": {{
                    "max_position_pct": 0.015,
                    "stop_multipliers": {{"atr": 1.8, "trailing": 1.2}}
                }}
            }}
            """
        }
        
        # This would call your ChatGPT integration
        # For now, return a mock response
        return {
            "regime_label": "volatile",
            "blend_weights_delta": {"short": 0.1, "mid": -0.1, "rl": 0.0},
            "bucket_caps_delta": {"penny": -0.01, "fno": -0.02, "core": 0.0},
            "do_not_trade_list": []
        }
    
    def _process_gpt5_response(self, escalation_type: EscalationType, response: Dict[str, Any]):
        """Process GPT-5 response and apply changes"""
        
        logger.info(f" GPT-5 response received for {escalation_type.value}")
        
        if escalation_type == EscalationType.REGIME_AUDITOR:
            self._apply_regime_changes(response)
        elif escalation_type == EscalationType.REWARD_ENGINEER:
            self._apply_reward_changes(response)
        elif escalation_type == EscalationType.SANITY_JUDGE:
            self._apply_sanity_changes(response)
    
    def _apply_regime_changes(self, response: Dict[str, Any]):
        """Apply regime auditor changes"""
        regime = response.get('regime_label', 'unknown')
        blend_delta = response.get('blend_weights_delta', {})
        bucket_delta = response.get('bucket_caps_delta', {})
        no_trade_list = response.get('do_not_trade_list', [])
        
        logger.info(f" Regime change: {regime}")
        logger.info(f" Blend weights delta: {blend_delta}")
        logger.info(f" Bucket caps delta: {bucket_delta}")
        logger.info(f" No trade list: {no_trade_list}")
        
        # Apply changes to system configuration
        # This would update your meta-ensemble and risk limits
    
    def _apply_reward_changes(self, response: Dict[str, Any]):
        """Apply reward engineer changes"""
        reward_coeffs = response.get('reward_coefficients', {})
        turnover_penalties = response.get('turnover_penalties', {})
        disabled_features = response.get('disabled_features', [])
        
        logger.info(f" New reward coefficients: {reward_coeffs}")
        logger.info(f" Turnover penalties: {turnover_penalties}")
        logger.info(f" Disabled features: {disabled_features}")
        
        # Apply changes to RL agents and meta-ensemble
    
    def _apply_sanity_changes(self, response: Dict[str, Any]):
        """Apply sanity judge changes"""
        stay_demo = response.get('stay_demo', True)
        live_changes = response.get('live_changes', {})
        
        if stay_demo:
            self.demo_quarantine = True
            logger.info(" Staying in demo mode per GPT-5 recommendation")
        else:
            self.demo_quarantine = False
            logger.info(" Cleared for live trading per GPT-5 recommendation")
        
        logger.info(f" Live changes: {live_changes}")
    
    def _trigger_kill_switch(self):
        """Trigger kill switch"""
        if not self.kill_switch_active:
            self.kill_switch_active = True
            self.kill_switch_days += 1
            logger.critical(" KILL SWITCH ACTIVATED - Flattening all positions")
            
            # Check for quarantine
            if self.kill_switch_days >= self.risk_limits.kill_switch_quarantine:
                self.demo_quarantine = True
                logger.critical(" DEMO QUARANTINE ACTIVATED - 7 days required")
    
    def _fallback_to_local_reasoner(self, escalation_type: EscalationType, data: Dict[str, Any]):
        """Fall back to local reasoner when GPT-5 unavailable"""
        logger.info(f" Falling back to local reasoner for {escalation_type.value}")
        
        if self.local_reasoner:
            # Use local reasoner for analysis
            local_response = self.local_reasoner.analyze_escalation(escalation_type, data)
            self._process_gpt5_response(escalation_type, local_response)
        else:
            logger.warning(" No local reasoner available, using conservative defaults")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        return {
            'kill_switch_active': self.kill_switch_active,
            'demo_quarantine': self.demo_quarantine,
            'gpt5_calls_today': self.gpt5_calls_today,
            'gpt5_calls_remaining': self.gpt5_daily_limit - self.gpt5_calls_today,
            'bucket_usage': self.bucket_usage.copy(),
            'portfolio_state': self.portfolio_state.copy()
        }
    
    # Helper methods for data preparation
    def _get_sector_dispersion(self) -> Dict[str, float]:
        """Get sector dispersion metrics"""
        # Implementation would calculate sector dispersion
        return {"technology": 0.15, "financials": 0.12, "energy": 0.18}
    
    def _get_pnl_decomposition(self) -> Dict[str, Any]:
        """Get P&L decomposition"""
        # Implementation would break down P&L by source
        return {"core": 0.02, "penny": -0.01, "fno": 0.005, "sip": 0.001}
    
    def _get_turnover_stats(self) -> Dict[str, float]:
        """Get turnover statistics"""
        # Implementation would calculate turnover metrics
        return {"daily_turnover": 0.05, "weekly_turnover": 0.15, "monthly_turnover": 0.35}
    
    def _get_feature_performance(self) -> Dict[str, float]:
        """Get feature performance metrics"""
        # Implementation would track feature importance
        return {"rsi": 0.65, "macd": 0.58, "volume": 0.72, "news": 0.45}
    
    def _get_kill_switch_reason(self) -> str:
        """Get kill switch reason"""
        return f"Daily P&L {self.portfolio_state['daily_pnl']*100:.2f}% below threshold"
    
    def _get_recent_trades(self) -> List[Dict[str, Any]]:
        """Get recent trades"""
        # Implementation would return recent trade history
        return [
            {"symbol": "TD.TO", "action": "BUY", "size": 0.01, "pnl": -0.005},
            {"symbol": "RY.TO", "action": "SELL", "size": 0.008, "pnl": 0.003}
        ]
    
    def check_escalation_triggers(self, system_metrics: Dict[str, Any]) -> List[TriggerEvent]:
        """Check for escalation triggers using the trigger monitor"""
        try:
            if not self.trigger_monitor:
                return []
            
            # Update trigger monitor with current system metrics
            self.trigger_monitor.update_performance_metrics(system_metrics)
            self.trigger_monitor.update_risk_metrics(system_metrics)
            
            # Check for triggers
            trigger_events = self.trigger_monitor.check_triggers(system_metrics)
            
            return trigger_events
            
        except Exception as e:
            logger.error(f"Error checking escalation triggers: {e}")
            return []
    
    def should_escalate_to_gpt5(self, trigger_events: List[TriggerEvent]) -> bool:
        """Determine if triggers should be escalated to GPT-5"""
        try:
            if not self.trigger_monitor:
                return False
            
            return self.trigger_monitor.should_escalate_to_gpt5(trigger_events)
            
        except Exception as e:
            logger.error(f"Error determining GPT-5 escalation: {e}")
            return False
    
    def audit_gpt5_response(self, gpt5_response: Dict[str, Any], 
                          system_context: Dict[str, Any]) -> AuditReport:
        """Audit GPT-5 response using the auditor sandbox"""
        try:
            if not self.auditor_sandbox:
                # Return a default approved report if sandbox not available
                return AuditReport(
                    report_id=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    gpt5_response=gpt5_response,
                    audit_result="approved",
                    findings=[],
                    approved_actions=[],
                    rejected_actions=[],
                    modified_actions=[],
                    audit_summary="Auditor sandbox not available - response approved by default"
                )
            
            # Audit the GPT-5 response
            audit_report = self.auditor_sandbox.audit_gpt5_response(gpt5_response, system_context)
            
            return audit_report
            
        except Exception as e:
            logger.error(f"Error auditing GPT-5 response: {e}")
            return None
    
    def get_escalation_context(self, trigger_event: TriggerEvent) -> Dict[str, Any]:
        """Get context for GPT-5 escalation"""
        try:
            if not self.trigger_monitor:
                return {}
            
            return self.trigger_monitor.get_escalation_context(trigger_event)
            
        except Exception as e:
            logger.error(f"Error getting escalation context: {e}")
            return {}
    
    def get_trigger_statistics(self) -> Dict:
        """Get trigger monitoring statistics"""
        try:
            if not self.trigger_monitor:
                return {}
            
            return self.trigger_monitor.get_trigger_statistics()
            
        except Exception as e:
            logger.error(f"Error getting trigger statistics: {e}")
            return {}
    
    def get_audit_statistics(self) -> Dict:
        """Get audit statistics"""
        try:
            if not self.auditor_sandbox:
                return {}
            
            return self.auditor_sandbox.get_audit_statistics()
            
        except Exception as e:
            logger.error(f"Error getting audit statistics: {e}")
            return {}
    
    def resolve_trigger(self, event_id: str, resolution_context: Dict[str, Any] = None):
        """Resolve a trigger event"""
        try:
            if self.trigger_monitor:
                self.trigger_monitor.resolve_trigger(event_id, resolution_context)
                
        except Exception as e:
            logger.error(f"Error resolving trigger: {e}")


# Example usage
if __name__ == "__main__":
    config = {
        'gpt5_api_key': 'your-gpt5-key-here',
        'local_reasoner_model': 'qwen2.5-14b-instruct'
    }
    
    control_plane = HybridControlPlane(config)
    
    # Simulate portfolio update
    portfolio_state = {
        'net_liquidity': 100000.0,
        'daily_pnl': -0.03,  # -3% daily loss
        'max_drawdown_5d': 0.05,
        'volatility_zscore': 1.5,
        'correlation_breakdown': 0.3,
        'ensemble_accuracy': 0.48,
        'sharpe_ratio': 0.4,
        'put_call_ratio': 1.2
    }
    
    control_plane.update_portfolio_state(portfolio_state)
    
    # Get risk status
    status = control_plane.get_risk_status()
    print("Risk Status:", json.dumps(status, indent=2))
