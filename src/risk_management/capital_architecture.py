"""
Capital Architecture Manager - 4-Bucket System
Implements the research plan's capital allocation framework
"""

import logging
import yaml
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

# Import new risk management components
from .var_beta_tracker import VaRBetaTracker
from .dynamic_bucket_scaling import DynamicBucketScaler

logger = logging.getLogger(__name__)


@dataclass
class BucketAllocation:
    """Represents a capital allocation bucket"""
    name: str
    percentage: float
    max_capital: float
    current_capital: float
    instruments: List[str]
    risk_rules: Dict
    positions: Dict[str, float]  # symbol -> position_value


@dataclass
class SIPAllocation:
    """Represents SIP (Systematic Investment Plan) allocation"""
    target_etf: str
    profit_percentage: float
    min_profit_threshold: float
    max_daily_allocation: float
    total_allocated: float
    daily_allocations: List[Dict]


class CapitalArchitectureManager:
    """
    Manages the 4-bucket capital allocation system
    """
    
    def __init__(self, config_path: str = "config/capital_architecture.yaml"):
        self.config = self._load_config(config_path)
        self.total_capital = self.config['capital_architecture']['total_capital']
        
        # Initialize buckets
        self.buckets = self._initialize_buckets()
        
        # Initialize VaR/Beta tracker
        try:
            self.var_beta_tracker = VaRBetaTracker(self.config)
            logger.info("VaR/Beta Tracker initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize VaR/Beta Tracker: {e}")
            self.var_beta_tracker = None
        
        # Initialize Dynamic Bucket Scaler
        try:
            self.dynamic_scaler = DynamicBucketScaler(self.config)
            logger.info("Dynamic Bucket Scaler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Dynamic Bucket Scaler: {e}")
            self.dynamic_scaler = None
        
        # Initialize SIP
        self.sip = self._initialize_sip()
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.consecutive_losses = 0
        self.last_loss_time = None
        
        logger.info(" Capital Architecture Manager initialized")
        logger.info(f" Total Capital: ${self.total_capital:,.2f} CAD")
        logger.info(f" Buckets: {len(self.buckets)}")
        logger.info(f" SIP Target: {self.sip.target_etf}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load capital architecture configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f" Failed to load capital config: {e}")
            return {}
    
    def _initialize_buckets(self) -> Dict[str, BucketAllocation]:
        """Initialize all capital allocation buckets"""
        buckets = {}
        bucket_configs = self.config['capital_architecture']['buckets']
        
        for bucket_name, config in bucket_configs.items():
            if bucket_name == 'sip_buffer':
                continue  # SIP is handled separately
            
            max_capital = self.total_capital * config['percentage']
            
            buckets[bucket_name] = BucketAllocation(
                name=bucket_name,
                percentage=config['percentage'],
                max_capital=max_capital,
                current_capital=max_capital,  # Start with full allocation
                instruments=config['instruments'],
                risk_rules=config['risk_rules'],
                positions={}
            )
            
            logger.info(f" {bucket_name}: ${max_capital:,.2f} ({config['percentage']*100:.1f}%)")
        
        return buckets
    
    def _initialize_sip(self) -> SIPAllocation:
        """Initialize SIP allocation"""
        sip_config = self.config['capital_architecture']['sip_config']
        
        return SIPAllocation(
            target_etf=sip_config['target_etf'],
            profit_percentage=sip_config['profit_percentage'],
            min_profit_threshold=sip_config['min_profit_threshold'],
            max_daily_allocation=sip_config['max_daily_allocation'],
            total_allocated=0.0,
            daily_allocations=[]
        )
    
    def get_bucket_for_symbol(self, symbol: str) -> Optional[str]:
        """Determine which bucket a symbol belongs to"""
        for bucket_name, bucket in self.buckets.items():
            if symbol in bucket.instruments:
                return bucket_name
        return None
    
    def can_allocate_position(self, symbol: str, position_size: float) -> Tuple[bool, str]:
        """Check if a position can be allocated to the appropriate bucket"""
        bucket_name = self.get_bucket_for_symbol(symbol)
        
        if not bucket_name:
            return False, f"Symbol {symbol} not found in any bucket"
        
        bucket = self.buckets[bucket_name]
        risk_rules = bucket.risk_rules
        
        # Check position size limits
        max_position_size = self.total_capital * risk_rules.get('max_position_size', 0.1)
        if position_size > max_position_size:
            return False, f"Position size ${position_size:,.2f} exceeds limit ${max_position_size:,.2f}"
        
        # Check available capital
        if position_size > bucket.current_capital:
            return False, f"Insufficient capital in {bucket_name}: ${bucket.current_capital:,.2f}"
        
        # Check maximum positions
        max_positions = risk_rules.get('max_positions', 10)
        if len(bucket.positions) >= max_positions:
            return False, f"Maximum positions reached in {bucket_name}: {max_positions}"
        
        return True, "Allocation approved"
    
    def allocate_position(self, symbol: str, position_size: float) -> bool:
        """Allocate a position to the appropriate bucket"""
        can_allocate, reason = self.can_allocate_position(symbol, position_size)
        
        if not can_allocate:
            logger.warning(f" Cannot allocate {symbol}: {reason}")
            return False
        
        bucket_name = self.get_bucket_for_symbol(symbol)
        bucket = self.buckets[bucket_name]
        
        # Allocate position
        bucket.positions[symbol] = position_size
        bucket.current_capital -= position_size
        
        logger.info(f" Allocated {symbol}: ${position_size:,.2f} to {bucket_name}")
        logger.info(f" {bucket_name} remaining capital: ${bucket.current_capital:,.2f}")
        
        return True
    
    def deallocate_position(self, symbol: str, position_value: float) -> bool:
        """Deallocate a position and return capital to bucket"""
        bucket_name = self.get_bucket_for_symbol(symbol)
        
        if not bucket_name:
            logger.warning(f" Symbol {symbol} not found in any bucket")
            return False
        
        bucket = self.buckets[bucket_name]
        
        if symbol not in bucket.positions:
            logger.warning(f" Position {symbol} not found in {bucket_name}")
            return False
        
        # Deallocate position
        original_size = bucket.positions[symbol]
        bucket.current_capital += position_value
        del bucket.positions[symbol]
        
        pnl = position_value - original_size
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        logger.info(f" Deallocated {symbol}: ${position_value:,.2f} from {bucket_name}")
        logger.info(f" P&L: ${pnl:+,.2f}")
        logger.info(f" {bucket_name} available capital: ${bucket.current_capital:,.2f}")
        
        return True
    
    def process_sip_allocation(self, daily_profit: float) -> float:
        """Process SIP allocation based on daily profit"""
        if daily_profit <= 0:
            return 0.0
        
        if daily_profit < self.sip.min_profit_threshold:
            return 0.0
        
        # Calculate SIP allocation (1% of profit)
        sip_amount = daily_profit * self.sip.profit_percentage
        
        # Apply daily limit
        sip_amount = min(sip_amount, self.sip.max_daily_allocation)
        
        if sip_amount > 0:
            # Record SIP allocation
            allocation = {
                'date': datetime.now().isoformat(),
                'profit': daily_profit,
                'sip_amount': sip_amount,
                'target_etf': self.sip.target_etf
            }
            
            self.sip.daily_allocations.append(allocation)
            self.sip.total_allocated += sip_amount
            
            logger.info(f" SIP Allocation: ${sip_amount:,.2f} to {self.sip.target_etf}")
            logger.info(f" Total SIP Allocated: ${self.sip.total_allocated:,.2f}")
        
        return sip_amount
    
    def check_risk_limits(self) -> Dict[str, bool]:
        """Check all risk management limits"""
        risk_config = self.config['capital_architecture']['risk_management']
        risk_status = {}
        
        # Daily loss limit
        daily_loss_limit = risk_config['daily_loss_limit']
        if self.daily_pnl < -self.total_capital * daily_loss_limit:
            risk_status['daily_loss_exceeded'] = True
            logger.warning(f" Daily loss limit exceeded: ${self.daily_pnl:,.2f}")
        else:
            risk_status['daily_loss_exceeded'] = False
        
        # Maximum drawdown
        max_drawdown = risk_config['max_drawdown']
        if self.total_pnl < -self.total_capital * max_drawdown:
            risk_status['max_drawdown_exceeded'] = True
            logger.warning(f" Maximum drawdown exceeded: ${self.total_pnl:,.2f}")
        else:
            risk_status['max_drawdown_exceeded'] = False
        
        # Consecutive losses
        consecutive_limit = risk_config['consecutive_loss_limit']
        if self.consecutive_losses >= consecutive_limit:
            risk_status['consecutive_losses_exceeded'] = True
            logger.warning(f" Consecutive losses exceeded: {self.consecutive_losses}")
        else:
            risk_status['consecutive_losses_exceeded'] = False
        
        return risk_status
    
    def should_pause_trading(self) -> Tuple[bool, str]:
        """Determine if trading should be paused based on risk limits"""
        risk_status = self.check_risk_limits()
        
        if risk_status['daily_loss_exceeded']:
            return True, "Daily loss limit exceeded"
        
        if risk_status['max_drawdown_exceeded']:
            return True, "Maximum drawdown exceeded"
        
        if risk_status['consecutive_losses_exceeded']:
            return True, "Consecutive losses exceeded"
        
        return False, "Trading allowed"
    
    def get_bucket_summary(self) -> Dict:
        """Get summary of all buckets"""
        summary = {
            'total_capital': self.total_capital,
            'total_allocated': sum(bucket.max_capital - bucket.current_capital for bucket in self.buckets.values()),
            'total_available': sum(bucket.current_capital for bucket in self.buckets.values()),
            'buckets': {}
        }
        
        for bucket_name, bucket in self.buckets.items():
            summary['buckets'][bucket_name] = {
                'max_capital': bucket.max_capital,
                'current_capital': bucket.current_capital,
                'allocated': bucket.max_capital - bucket.current_capital,
                'positions': len(bucket.positions),
                'position_symbols': list(bucket.positions.keys())
            }
        
        return summary
    
    def get_sip_summary(self) -> Dict:
        """Get SIP allocation summary"""
        return {
            'target_etf': self.sip.target_etf,
            'total_allocated': self.sip.total_allocated,
            'daily_allocations_count': len(self.sip.daily_allocations),
            'last_allocation': self.sip.daily_allocations[-1] if self.sip.daily_allocations else None
        }
    
    def reset_daily_pnl(self):
        """Reset daily P&L (called at start of each trading day)"""
        self.daily_pnl = 0.0
        logger.info(" Daily P&L reset")
    
    def record_loss(self):
        """Record a loss and update consecutive loss counter"""
        self.consecutive_losses += 1
        self.last_loss_time = datetime.now()
        logger.info(f" Loss recorded. Consecutive losses: {self.consecutive_losses}")
    
    def record_win(self):
        """Record a win and reset consecutive loss counter"""
        self.consecutive_losses = 0
        logger.info(" Win recorded. Consecutive losses reset to 0")
    
    def update_risk_metrics(self, symbol: str, returns: pd.Series, 
                          benchmark_returns: pd.Series = None):
        """Update risk metrics for VaR/Beta tracking"""
        try:
            if self.var_beta_tracker:
                self.var_beta_tracker.update_portfolio_data(symbol, returns, benchmark_returns)
                logger.debug(f"Updated risk metrics for {symbol}")
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def calculate_portfolio_risk_metrics(self) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            if not self.var_beta_tracker:
                return {}
            
            # Get current portfolio weights
            portfolio_weights = {}
            for bucket_name, bucket in self.buckets.items():
                if bucket.current_capital > 0:
                    portfolio_weights[bucket_name] = bucket.current_capital / self.total_capital
            
            # Calculate risk metrics
            risk_metrics = self.var_beta_tracker.calculate_comprehensive_risk_metrics(
                portfolio_weights
            )
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {}
    
    def execute_dynamic_scaling(self) -> Dict:
        """Execute dynamic bucket scaling based on performance and risk"""
        try:
            if not self.dynamic_scaler:
                return {'scaled': False, 'reason': 'Dynamic scaler not available'}
            
            # Get current allocation
            current_allocation = {}
            for bucket_name, bucket in self.buckets.items():
                current_allocation[bucket_name] = bucket.max_capital / self.total_capital
            
            # Execute scaling
            scaling_result = self.dynamic_scaler.execute_scaling(current_allocation)
            
            if scaling_result['scaled']:
                # Apply new allocation
                new_allocation = scaling_result['new_allocation']
                self._apply_new_allocation(new_allocation)
                
                logger.info(f"Dynamic scaling applied: {scaling_result['reason']}")
            
            return scaling_result
            
        except Exception as e:
            logger.error(f"Error executing dynamic scaling: {e}")
            return {'scaled': False, 'reason': f'Error: {str(e)}'}
    
    def _apply_new_allocation(self, new_allocation: Dict[str, float]):
        """Apply new allocation to buckets"""
        try:
            for bucket_name, percentage in new_allocation.items():
                if bucket_name in self.buckets:
                    new_max_capital = self.total_capital * percentage
                    self.buckets[bucket_name].max_capital = new_max_capital
                    
                    # Adjust current capital if it exceeds new max
                    if self.buckets[bucket_name].current_capital > new_max_capital:
                        excess = self.buckets[bucket_name].current_capital - new_max_capital
                        self.buckets[bucket_name].current_capital = new_max_capital
                        logger.warning(f"Reduced {bucket_name} capital by {excess:.2f} due to scaling")
            
            logger.info("Applied new allocation to buckets")
            
        except Exception as e:
            logger.error(f"Error applying new allocation: {e}")
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics for dynamic scaling"""
        try:
            if self.dynamic_scaler:
                self.dynamic_scaler.update_performance_metrics(metrics)
                logger.debug("Updated performance metrics for dynamic scaling")
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_risk_dashboard_data(self) -> Dict:
        """Get comprehensive risk dashboard data"""
        try:
            dashboard_data = {
                'capital_allocation': self.get_allocation_summary(),
                'sip_summary': self.get_sip_summary(),
                'performance_metrics': {
                    'daily_pnl': self.daily_pnl,
                    'consecutive_losses': self.consecutive_losses,
                    'last_loss_time': self.last_loss_time.isoformat() if self.last_loss_time else None
                }
            }
            
            # Add VaR/Beta data if available
            if self.var_beta_tracker:
                dashboard_data['risk_metrics'] = self.var_beta_tracker.get_risk_dashboard_data()
                dashboard_data['portfolio_risk'] = self.calculate_portfolio_risk_metrics()
            
            # Add dynamic scaling data if available
            if self.dynamic_scaler:
                dashboard_data['scaling_statistics'] = self.dynamic_scaler.get_scaling_statistics()
                dashboard_data['scaling_recommendations'] = self.dynamic_scaler.get_scaling_recommendations(
                    {name: bucket.max_capital / self.total_capital 
                     for name, bucket in self.buckets.items()}
                )
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting risk dashboard data: {e}")
            return {}


if __name__ == "__main__":
    # Test the capital architecture manager
    logging.basicConfig(level=logging.INFO)
    
    manager = CapitalArchitectureManager()
    
    # Test allocation
    print("\n" + "="*50)
    print("TESTING CAPITAL ALLOCATION")
    print("="*50)
    
    # Test penny stock allocation
    success = manager.allocate_position("AI.TO", 500)
    print(f"AI.TO allocation: {'' if success else ''}")
    
    # Test core stock allocation
    success = manager.allocate_position("RY.TO", 5000)
    print(f"RY.TO allocation: {'' if success else ''}")
    
    # Test SIP allocation
    sip_amount = manager.process_sip_allocation(1000)
    print(f"SIP allocation: ${sip_amount:,.2f}")
    
    # Get summaries
    bucket_summary = manager.get_bucket_summary()
    sip_summary = manager.get_sip_summary()
    
    print(f"\n Bucket Summary:")
    for bucket_name, bucket_info in bucket_summary['buckets'].items():
        print(f"  {bucket_name}: ${bucket_info['allocated']:,.2f} allocated, {bucket_info['positions']} positions")
    
    print(f"\n SIP Summary:")
    print(f"  Total Allocated: ${sip_summary['total_allocated']:,.2f}")
    print(f"  Target ETF: {sip_summary['target_etf']}")
