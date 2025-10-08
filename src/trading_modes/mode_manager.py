"""
Trading Mode Manager
Handles switching between Live and Demo modes with shared AI learning
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading mode enumeration"""
    DEMO = "demo"
    LIVE = "live"

class ModeManager:
    """
    Manages trading mode switching between Live and Demo
    
    Features:
    - Seamless mode switching
    - Separate capital tracking for each mode
    - Shared AI learning from both modes
    - Trade information sharing for continuous improvement
    - Risk controls for live mode
    - Full simulation capabilities for demo mode
    """
    
    def __init__(self, config_path: str = "config/mode_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        self.current_mode = TradingMode.DEMO  # Default to demo mode
        self.mode_history = []
        
        # Separate accounts for each mode
        self.demo_account = {
            'capital': self.config['demo']['starting_capital'],
            'positions': {},
            'trades': [],
            'performance': {}
        }
        
        self.live_account = {
            'capital': self.config['live']['starting_capital'],
            'positions': {},
            'trades': [],
            'performance': {}
        }
        
        # Shared AI learning database
        self.shared_learning = {
            'trades': [],  # All trades from both modes
            'patterns': [],  # Discovered patterns
            'strategies': {},  # Strategy performance
            'ai_insights': []  # AI learning insights
        }
        
        logger.info(f" Mode Manager initialized in {self.current_mode.value.upper()} mode")
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'demo': {
                    'starting_capital': 100000.0,
                    'allow_partial_shares': True,
                    'allow_options': True,
                    'allow_margin': False,
                    'max_position_size': 0.20,
                    'daily_loss_limit': 0.05
                },
                'live': {
                    'starting_capital': 0.0,  # Must be set by user
                    'allow_partial_shares': True,
                    'allow_options': True,
                    'allow_margin': False,
                    'max_position_size': 0.10,
                    'daily_loss_limit': 0.03,
                    'require_confirmation': True
                },
                'ai_learning': {
                    'share_trades': True,
                    'learn_from_both_modes': True,
                    'focus_on_profitability': True,
                    'minimize_losses': True
                }
            }
    
    def switch_mode(self, new_mode: TradingMode) -> Dict[str, any]:
        """
        Switch between Live and Demo modes
        
        Returns status and mode information
        """
        
        if new_mode == self.current_mode:
            logger.info(f" Already in {new_mode.value.upper()} mode")
            return {
                'success': True,
                'message': f"Already in {new_mode.value.upper()} mode",
                'mode': new_mode.value
            }
        
        # Validate switch to live mode
        if new_mode == TradingMode.LIVE:
            if self.live_account['capital'] <= 0:
                logger.error(" Cannot switch to LIVE mode: No capital allocated")
                return {
                    'success': False,
                    'message': "Cannot switch to LIVE mode: No capital allocated. Please set live capital first.",
                    'mode': self.current_mode.value
                }
            
            # Additional safety check
            logger.warning(" Switching to LIVE mode - real money will be used!")
        
        # Record mode change
        self.mode_history.append({
            'timestamp': datetime.now(),
            'from_mode': self.current_mode.value,
            'to_mode': new_mode.value
        })
        
        # Switch mode
        old_mode = self.current_mode
        self.current_mode = new_mode
        
        logger.info(f" Switched from {old_mode.value.upper()} to {new_mode.value.upper()} mode")
        
        return {
            'success': True,
            'message': f"Successfully switched to {new_mode.value.upper()} mode",
            'mode': new_mode.value,
            'account_info': self.get_current_account_info()
        }
    
    def get_current_mode(self) -> TradingMode:
        """Get current trading mode"""
        return self.current_mode
    
    def is_live_mode(self) -> bool:
        """Check if currently in live mode"""
        return self.current_mode == TradingMode.LIVE
    
    def is_demo_mode(self) -> bool:
        """Check if currently in demo mode"""
        return self.current_mode == TradingMode.DEMO
    
    def get_current_account_info(self) -> Dict[str, any]:
        """Get current account information"""
        
        if self.current_mode == TradingMode.DEMO:
            account = self.demo_account
        else:
            account = self.live_account
        
        return {
            'mode': self.current_mode.value,
            'capital': account['capital'],
            'positions': account['positions'],
            'num_trades': len(account['trades']),
            'performance': account['performance']
        }
    
    def record_trade(
        self,
        trade_data: Dict[str, any],
        mode: Optional[TradingMode] = None
    ):
        """
        Record a trade in the appropriate account and shared learning
        
        Args:
            trade_data: Trade information
            mode: Trading mode (default: current mode)
        """
        
        mode = mode or self.current_mode
        
        # Add metadata
        trade_data['mode'] = mode.value
        trade_data['timestamp'] = trade_data.get('timestamp', datetime.now())
        
        # Record in mode-specific account
        if mode == TradingMode.DEMO:
            self.demo_account['trades'].append(trade_data)
        else:
            self.live_account['trades'].append(trade_data)
        
        # Record in shared learning database
        if self.config.get('ai_learning', {}).get('share_trades', True):
            self.shared_learning['trades'].append(trade_data)
            logger.info(f" Trade recorded and shared for AI learning: {trade_data.get('symbol')} {trade_data.get('action')}")
    
    def get_shared_learning_data(self) -> Dict[str, any]:
        """
        Get shared learning data from both modes
        
        Returns comprehensive learning insights for AI improvement
        """
        
        # Analyze trades from both modes
        all_trades = self.shared_learning['trades']
        
        if not all_trades:
            return {
                'total_trades': 0,
                'insights': []
            }
        
        # Separate by mode
        demo_trades = [t for t in all_trades if t.get('mode') == 'demo']
        live_trades = [t for t in all_trades if t.get('mode') == 'live']
        
        # Calculate performance metrics
        demo_performance = self._calculate_performance(demo_trades)
        live_performance = self._calculate_performance(live_trades)
        
        # Identify successful patterns
        successful_patterns = self._identify_successful_patterns(all_trades)
        
        # Identify areas for improvement
        improvement_areas = self._identify_improvement_areas(all_trades)
        
        # Calculate learning metrics
        learning_insights = {
            'total_trades': len(all_trades),
            'demo_trades': len(demo_trades),
            'live_trades': len(live_trades),
            'demo_performance': demo_performance,
            'live_performance': live_performance,
            'successful_patterns': successful_patterns,
            'improvement_areas': improvement_areas,
            'profitability_trend': self._calculate_profitability_trend(all_trades),
            'loss_reduction_progress': self._calculate_loss_reduction(all_trades),
            'cross_mode_learning': self._analyze_cross_mode_learning(demo_trades, live_trades)
        }
        
        return learning_insights
    
    def _calculate_performance(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics for trades"""
        
        if not trades:
            return {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        avg_profit = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades)
        }
    
    def _identify_successful_patterns(self, trades: List[Dict]) -> List[Dict]:
        """Identify successful trading patterns"""
        
        patterns = []
        
        # Group by strategy
        strategy_performance = {}
        for trade in trades:
            strategy = trade.get('strategy', 'unknown')
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(trade)
        
        # Analyze each strategy
        for strategy, strategy_trades in strategy_performance.items():
            performance = self._calculate_performance(strategy_trades)
            
            if performance['win_rate'] > 0.6 and performance['profit_factor'] > 1.5:
                patterns.append({
                    'type': 'strategy',
                    'name': strategy,
                    'win_rate': performance['win_rate'],
                    'profit_factor': performance['profit_factor'],
                    'num_trades': len(strategy_trades),
                    'avg_profit': performance['avg_profit']
                })
        
        # Group by time of day
        time_performance = {}
        for trade in trades:
            hour = trade.get('timestamp', datetime.now()).hour
            time_bucket = f"{hour:02d}:00"
            if time_bucket not in time_performance:
                time_performance[time_bucket] = []
            time_performance[time_bucket].append(trade)
        
        # Analyze time-based patterns
        for time_bucket, time_trades in time_performance.items():
            performance = self._calculate_performance(time_trades)
            
            if performance['win_rate'] > 0.65:
                patterns.append({
                    'type': 'time_of_day',
                    'time': time_bucket,
                    'win_rate': performance['win_rate'],
                    'num_trades': len(time_trades)
                })
        
        return patterns
    
    def _identify_improvement_areas(self, trades: List[Dict]) -> List[Dict]:
        """Identify areas for improvement"""
        
        improvement_areas = []
        
        # Analyze losing trades
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        if losing_trades:
            # Group by strategy
            strategy_losses = {}
            for trade in losing_trades:
                strategy = trade.get('strategy', 'unknown')
                if strategy not in strategy_losses:
                    strategy_losses[strategy] = []
                strategy_losses[strategy].append(trade)
            
            # Identify problematic strategies
            for strategy, strategy_trades in strategy_losses.items():
                avg_loss = sum(t.get('pnl', 0) for t in strategy_trades) / len(strategy_trades)
                
                if len(strategy_trades) > 3 and avg_loss < -100:
                    improvement_areas.append({
                        'area': 'strategy',
                        'name': strategy,
                        'issue': 'High loss rate',
                        'num_losing_trades': len(strategy_trades),
                        'avg_loss': avg_loss,
                        'recommendation': f"Review {strategy} strategy parameters or reduce position sizing"
                    })
        
        # Analyze holding periods
        trades_with_duration = [t for t in trades if 'duration' in t]
        if trades_with_duration:
            long_duration_losses = [t for t in trades_with_duration if t.get('pnl', 0) < 0 and t.get('duration', 0) > 3600]
            
            if len(long_duration_losses) > 5:
                improvement_areas.append({
                    'area': 'risk_management',
                    'issue': 'Holding losing positions too long',
                    'num_trades': len(long_duration_losses),
                    'recommendation': 'Implement tighter stop losses for positions held > 1 hour'
                })
        
        return improvement_areas
    
    def _calculate_profitability_trend(self, trades: List[Dict]) -> Dict[str, any]:
        """Calculate profitability trend over time"""
        
        if not trades:
            return {'trend': 'neutral', 'daily_improvement': 0.0}
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.get('timestamp', datetime.now()))
        
        # Calculate daily P&L
        daily_pnl = {}
        for trade in sorted_trades:
            date = trade.get('timestamp', datetime.now()).date()
            if date not in daily_pnl:
                daily_pnl[date] = 0.0
            daily_pnl[date] += trade.get('pnl', 0)
        
        # Calculate trend
        if len(daily_pnl) > 1:
            pnl_values = list(daily_pnl.values())
            first_half = sum(pnl_values[:len(pnl_values)//2]) / (len(pnl_values)//2)
            second_half = sum(pnl_values[len(pnl_values)//2:]) / (len(pnl_values) - len(pnl_values)//2)
            
            improvement = ((second_half - first_half) / abs(first_half)) * 100 if first_half != 0 else 0
            
            if improvement > 10:
                trend = 'improving'
            elif improvement < -10:
                trend = 'declining'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'daily_improvement': improvement,
                'first_half_avg': first_half,
                'second_half_avg': second_half
            }
        
        return {'trend': 'insufficient_data', 'daily_improvement': 0.0}
    
    def _calculate_loss_reduction(self, trades: List[Dict]) -> Dict[str, any]:
        """Calculate loss reduction progress over time"""
        
        if not trades:
            return {'progress': 'no_data', 'reduction_percentage': 0.0}
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.get('timestamp', datetime.now()))
        
        # Split into two halves
        mid_point = len(sorted_trades) // 2
        first_half = sorted_trades[:mid_point]
        second_half = sorted_trades[mid_point:]
        
        # Calculate average loss per losing trade
        first_half_losses = [t.get('pnl', 0) for t in first_half if t.get('pnl', 0) < 0]
        second_half_losses = [t.get('pnl', 0) for t in second_half if t.get('pnl', 0) < 0]
        
        if first_half_losses and second_half_losses:
            avg_first_loss = sum(first_half_losses) / len(first_half_losses)
            avg_second_loss = sum(second_half_losses) / len(second_half_losses)
            
            reduction = ((avg_first_loss - avg_second_loss) / abs(avg_first_loss)) * 100
            
            return {
                'progress': 'improving' if reduction > 0 else 'needs_attention',
                'reduction_percentage': reduction,
                'first_half_avg_loss': avg_first_loss,
                'second_half_avg_loss': avg_second_loss
            }
        
        return {'progress': 'insufficient_data', 'reduction_percentage': 0.0}
    
    def _analyze_cross_mode_learning(self, demo_trades: List[Dict], live_trades: List[Dict]) -> Dict[str, any]:
        """Analyze learning transfer between demo and live modes"""
        
        if not demo_trades or not live_trades:
            return {
                'learning_transfer': 'insufficient_data',
                'demo_to_live_improvement': 0.0
            }
        
        # Calculate performance in each mode
        demo_perf = self._calculate_performance(demo_trades)
        live_perf = self._calculate_performance(live_trades)
        
        # Compare key metrics
        win_rate_diff = live_perf['win_rate'] - demo_perf['win_rate']
        profit_factor_diff = live_perf['profit_factor'] - demo_perf['profit_factor']
        
        # Determine learning effectiveness
        if win_rate_diff > 0.05 and profit_factor_diff > 0.2:
            learning_transfer = 'excellent'
            message = "Demo practice is translating well to live trading"
        elif win_rate_diff > 0:
            learning_transfer = 'good'
            message = "Positive learning transfer from demo to live"
        elif win_rate_diff > -0.05:
            learning_transfer = 'moderate'
            message = "Some learning transfer, but room for improvement"
        else:
            learning_transfer = 'needs_improvement'
            message = "Live performance lagging demo - review risk management"
        
        return {
            'learning_transfer': learning_transfer,
            'win_rate_difference': win_rate_diff,
            'profit_factor_difference': profit_factor_diff,
            'message': message,
            'demo_performance': demo_perf,
            'live_performance': live_perf
        }
    
    def get_mode_comparison(self) -> Dict[str, any]:
        """Get side-by-side comparison of demo and live modes"""
        
        demo_perf = self._calculate_performance(self.demo_account['trades'])
        live_perf = self._calculate_performance(self.live_account['trades'])
        
        return {
            'demo': {
                'capital': self.demo_account['capital'],
                'num_trades': len(self.demo_account['trades']),
                'performance': demo_perf
            },
            'live': {
                'capital': self.live_account['capital'],
                'num_trades': len(self.live_account['trades']),
                'performance': live_perf
            },
            'comparison': {
                'win_rate_diff': live_perf['win_rate'] - demo_perf['win_rate'],
                'pnl_diff': live_perf['total_pnl'] - demo_perf['total_pnl'],
                'profit_factor_diff': live_perf['profit_factor'] - demo_perf['profit_factor']
            }
        }
    
    def export_learning_data(self, file_path: str):
        """Export shared learning data for analysis"""
        
        learning_data = self.get_shared_learning_data()
        
        with open(file_path, 'w') as f:
            json.dump(learning_data, f, indent=2, default=str)
        
        logger.info(f" Learning data exported to {file_path}")
    
    def set_live_capital(self, capital: float) -> bool:
        """Set live trading capital"""
        
        if capital <= 0:
            logger.error(" Live capital must be positive")
            return False
        
        self.live_account['capital'] = capital
        logger.info(f" Live capital set to ${capital:,.2f}")
        return True

# Global mode manager instance
_mode_manager_instance = None

def get_mode_manager() -> ModeManager:
    """Get global mode manager instance"""
    global _mode_manager_instance
    if _mode_manager_instance is None:
        _mode_manager_instance = ModeManager()
    return _mode_manager_instance

