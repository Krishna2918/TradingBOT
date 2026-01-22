"""
Leverage Governance Module
Manages dynamic leverage adjustments based on market conditions and risk metrics
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional
import yaml
import requests

logger = logging.getLogger(__name__)

@dataclass
class LeverageState:
    """Current leverage allocation state"""
    current_leverage: float
    max_leverage: float
    vix_level: float
    margin_utilization: float
    instrument_leverages: Dict[str, float]

class LeverageGovernor:
    """Manages dynamic leverage adjustments"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['risk']
        
        self.leverage_state = LeverageState(
            current_leverage=self.config['leverage']['base_leverage'],
            max_leverage=self.config['leverage']['max_leverage'],
            vix_level=15.0,  # Default VIX level
            margin_utilization=0.0,
            instrument_leverages={}
        )
        
        self.vix_adjustments = self.config['leverage']['vix_adjustments']
        self.margin_buffer = 0.85  # 85% margin utilization limit
    
    def update_vix_level(self, vix: float):
        """Update VIX level and adjust leverage accordingly"""
        self.leverage_state.vix_level = vix
        self.adjust_leverage_for_vix()
    
    def adjust_leverage_for_vix(self):
        """Adjust leverage based on VIX level"""
        base_leverage = self.config['leverage']['base_leverage']
        
        if self.leverage_state.vix_level > 25:
            adjustment = self.vix_adjustments.get('25+', 0.5)
        elif self.leverage_state.vix_level > 20:
            adjustment = self.vix_adjustments.get('20-25', 0.8)
        elif self.leverage_state.vix_level > 15:
            adjustment = self.vix_adjustments.get('15-20', 0.9)
        else:
            adjustment = 1.0
        
        new_leverage = base_leverage * adjustment
        new_leverage = min(new_leverage, self.leverage_state.max_leverage)
        
        if new_leverage != self.leverage_state.current_leverage:
            logger.info(f"Leverage adjusted from {self.leverage_state.current_leverage} to {new_leverage} (VIX: {self.leverage_state.vix_level})")
            self.leverage_state.current_leverage = new_leverage
    
    def update_margin_utilization(self, utilization: float):
        """Update margin utilization and check limits"""
        self.leverage_state.margin_utilization = utilization
        
        if utilization > self.margin_buffer:
            logger.warning(f"Margin utilization high: {utilization:.2%}")
            self.force_leverage_reduction()
    
    def force_leverage_reduction(self):
        """Force leverage reduction when margin utilization is high"""
        current = self.leverage_state.current_leverage
        reduced = current * 0.7  # Reduce by 30%
        
        logger.info(f"Force leverage reduction: {current} -> {reduced}")
        self.leverage_state.current_leverage = reduced
    
    def get_strategy_leverage(self, strategy: str) -> float:
        """Get leverage for specific strategy"""
        strategy_config = self.config.get('strategies', {}).get(strategy, {})
        base_leverage = strategy_config.get('max_leverage', self.leverage_state.current_leverage)
        
        # Apply VIX adjustment
        if self.leverage_state.vix_level > 25:
            return base_leverage * 0.5
        elif self.leverage_state.vix_level > 20:
            return base_leverage * 0.8
        elif self.leverage_state.vix_level > 15:
            return base_leverage * 0.9
        else:
            return base_leverage
    
    def get_leverage_state(self) -> LeverageState:
        """Get current leverage state"""
        return self.leverage_state
    
    def is_leverage_safe(self) -> bool:
        """Check if current leverage is within safe limits"""
        return (self.leverage_state.current_leverage <= self.leverage_state.max_leverage and
                self.leverage_state.margin_utilization <= self.margin_buffer)

# Test the leverage governor
if __name__ == "__main__":
    governor = LeverageGovernor("config/risk_config.yaml")
    
    # Test VIX adjustments
    test_vix_levels = [12, 18, 22, 28]
    for vix in test_vix_levels:
        governor.update_vix_level(vix)
        print(f"VIX {vix}: Leverage {governor.leverage_state.current_leverage}")
    
    # Test margin utilization
    governor.update_margin_utilization(0.87)
    print(f"High margin: Leverage {governor.leverage_state.current_leverage}")

