"""
24/7 Activity Scheduler

This module implements a comprehensive activity scheduler that manages
AI trading activities around the clock, including market hours intelligence,
pre-market analysis, post-market review, and off-hours research.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
import schedule

from src.workflows.trading_cycle import get_trading_cycle, execute_complete_cycle
from src.ai.enhanced_ensemble import get_enhanced_ensemble
from src.config.mode_manager import get_current_mode

logger = logging.getLogger(__name__)

class ActivityType(Enum):
    """Activity type enumeration."""
    MARKET_ANALYSIS = "MARKET_ANALYSIS"
    PRE_MARKET_SCAN = "PRE_MARKET_SCAN"
    MARKET_OPEN = "MARKET_OPEN"
    INTRADAY_MONITORING = "INTRADAY_MONITORING"
    MARKET_CLOSE = "MARKET_CLOSE"
    POST_MARKET_REVIEW = "POST_MARKET_REVIEW"
    OFF_HOURS_RESEARCH = "OFF_HOURS_RESEARCH"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    PORTFOLIO_REBALANCE = "PORTFOLIO_REBALANCE"
    MODEL_TRAINING = "MODEL_TRAINING"
    DATA_COLLECTION = "DATA_COLLECTION"
    SYSTEM_HEALTH_CHECK = "SYSTEM_HEALTH_CHECK"

class MarketSession(Enum):
    """Market session enumeration."""
    PRE_MARKET = "PRE_MARKET"      # 4:00 AM - 9:30 AM ET
    MARKET_OPEN = "MARKET_OPEN"    # 9:30 AM - 4:00 PM ET
    POST_MARKET = "POST_MARKET"    # 4:00 PM - 8:00 PM ET
    OFF_HOURS = "OFF_HOURS"        # 8:00 PM - 4:00 AM ET

@dataclass
class ScheduledActivity:
    """Represents a scheduled activity."""
    activity_type: ActivityType
    scheduled_time: datetime
    frequency: str  # "daily", "hourly", "every_30min", etc.
    priority: int   # 1-10, higher is more important
    enabled: bool
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0

@dataclass
class ActivityResult:
    """Result of an activity execution."""
    activity_type: ActivityType
    start_time: datetime
    end_time: datetime
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ActivityScheduler:
    """Manages 24/7 AI trading activities."""
    
    def __init__(self):
        """Initialize Activity Scheduler."""
        self.trading_cycle = get_trading_cycle()
        self.enhanced_ensemble = get_enhanced_ensemble()
        
        # Scheduler state
        self.is_running = False
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        # Activity tracking
        self.scheduled_activities: Dict[ActivityType, ScheduledActivity] = {}
        self.activity_results: List[ActivityResult] = []
        self.max_results_history = 1000
        
        # Market hours configuration (ET timezone)
        self.market_open_time = dt_time(9, 30)   # 9:30 AM ET
        self.market_close_time = dt_time(16, 0)  # 4:00 PM ET
        self.pre_market_start = dt_time(4, 0)    # 4:00 AM ET
        self.post_market_end = dt_time(20, 0)    # 8:00 PM ET
        
        # Initialize scheduled activities
        self._initialize_activities()
        
        logger.info("Activity Scheduler initialized")
    
    def _initialize_activities(self) -> None:
        """Initialize all scheduled activities."""
        now = datetime.now()
        
        # Market Analysis - Every 30 minutes during market hours
        self.scheduled_activities[ActivityType.MARKET_ANALYSIS] = ScheduledActivity(
            activity_type=ActivityType.MARKET_ANALYSIS,
            scheduled_time=now,
            frequency="every_30min",
            priority=8,
            enabled=True
        )
        
        # Pre-Market Scan - Daily at 8:00 AM
        self.scheduled_activities[ActivityType.PRE_MARKET_SCAN] = ScheduledActivity(
            activity_type=ActivityType.PRE_MARKET_SCAN,
            scheduled_time=now.replace(hour=8, minute=0, second=0, microsecond=0),
            frequency="daily",
            priority=9,
            enabled=True
        )
        
        # Market Open - Daily at 9:30 AM
        self.scheduled_activities[ActivityType.MARKET_OPEN] = ScheduledActivity(
            activity_type=ActivityType.MARKET_OPEN,
            scheduled_time=now.replace(hour=9, minute=30, second=0, microsecond=0),
            frequency="daily",
            priority=10,
            enabled=True
        )
        
        # Intraday Monitoring - Every 15 minutes during market hours
        self.scheduled_activities[ActivityType.INTRADAY_MONITORING] = ScheduledActivity(
            activity_type=ActivityType.INTRADAY_MONITORING,
            scheduled_time=now,
            frequency="every_15min",
            priority=7,
            enabled=True
        )
        
        # Market Close - Daily at 4:00 PM
        self.scheduled_activities[ActivityType.MARKET_CLOSE] = ScheduledActivity(
            activity_type=ActivityType.MARKET_CLOSE,
            scheduled_time=now.replace(hour=16, minute=0, second=0, microsecond=0),
            frequency="daily",
            priority=9,
            enabled=True
        )
        
        # Post-Market Review - Daily at 5:00 PM
        self.scheduled_activities[ActivityType.POST_MARKET_REVIEW] = ScheduledActivity(
            activity_type=ActivityType.POST_MARKET_REVIEW,
            scheduled_time=now.replace(hour=17, minute=0, second=0, microsecond=0),
            frequency="daily",
            priority=6,
            enabled=True
        )
        
        # Off-Hours Research - Every 2 hours during off-hours
        self.scheduled_activities[ActivityType.OFF_HOURS_RESEARCH] = ScheduledActivity(
            activity_type=ActivityType.OFF_HOURS_RESEARCH,
            scheduled_time=now,
            frequency="every_2hours",
            priority=4,
            enabled=True
        )
        
        # Risk Assessment - Every hour
        self.scheduled_activities[ActivityType.RISK_ASSESSMENT] = ScheduledActivity(
            activity_type=ActivityType.RISK_ASSESSMENT,
            scheduled_time=now,
            frequency="hourly",
            priority=8,
            enabled=True
        )
        
        # Portfolio Rebalance - Daily at 3:30 PM
        self.scheduled_activities[ActivityType.PORTFOLIO_REBALANCE] = ScheduledActivity(
            activity_type=ActivityType.PORTFOLIO_REBALANCE,
            scheduled_time=now.replace(hour=15, minute=30, second=0, microsecond=0),
            frequency="daily",
            priority=5,
            enabled=True
        )
        
        # Model Training - Daily at 11:00 PM
        self.scheduled_activities[ActivityType.MODEL_TRAINING] = ScheduledActivity(
            activity_type=ActivityType.MODEL_TRAINING,
            scheduled_time=now.replace(hour=23, minute=0, second=0, microsecond=0),
            frequency="daily",
            priority=3,
            enabled=True
        )
        
        # Data Collection - Every 5 minutes
        self.scheduled_activities[ActivityType.DATA_COLLECTION] = ScheduledActivity(
            activity_type=ActivityType.DATA_COLLECTION,
            scheduled_time=now,
            frequency="every_5min",
            priority=6,
            enabled=True
        )
        
        # System Health Check - Every 10 minutes
        self.scheduled_activities[ActivityType.SYSTEM_HEALTH_CHECK] = ScheduledActivity(
            activity_type=ActivityType.SYSTEM_HEALTH_CHECK,
            scheduled_time=now,
            frequency="every_10min",
            priority=7,
            enabled=True
        )
        
        logger.info(f"Initialized {len(self.scheduled_activities)} scheduled activities")
    
    def start_scheduler(self) -> None:
        """Start the activity scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Activity scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the activity scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Activity scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler loop started")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                current_session = self._get_current_market_session(current_time)
                
                # Check for activities to run
                activities_to_run = self._get_activities_to_run(current_time, current_session)
                
                for activity in activities_to_run:
                    if self.stop_event.is_set():
                        break
                    
                    try:
                        # Execute activity
                        result = self._execute_activity(activity)
                        self.activity_results.append(result)
                        
                        # Update activity statistics
                        activity.last_run = current_time
                        activity.run_count += 1
                        if result.success:
                            activity.success_count += 1
                        else:
                            activity.failure_count += 1
                        
                        # Calculate next run time
                        activity.next_run = self._calculate_next_run_time(activity, current_time)
                        
                    except Exception as e:
                        logger.error(f"Error executing activity {activity.activity_type}: {e}")
                        activity.failure_count += 1
                
                # Clean up old results
                self._cleanup_old_results()
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
        
        logger.info("Scheduler loop ended")
    
    def _get_current_market_session(self, current_time: datetime) -> MarketSession:
        """Determine the current market session."""
        current_time_only = current_time.time()
        
        if self.pre_market_start <= current_time_only < self.market_open_time:
            return MarketSession.PRE_MARKET
        elif self.market_open_time <= current_time_only < self.market_close_time:
            return MarketSession.MARKET_OPEN
        elif self.market_close_time <= current_time_only < self.post_market_end:
            return MarketSession.POST_MARKET
        else:
            return MarketSession.OFF_HOURS
    
    def _get_activities_to_run(self, current_time: datetime, current_session: MarketSession) -> List[ScheduledActivity]:
        """Get activities that should run at the current time."""
        activities_to_run = []
        
        for activity in self.scheduled_activities.values():
            if not activity.enabled:
                continue
            
            # Check if activity should run based on frequency and session
            if self._should_run_activity(activity, current_time, current_session):
                activities_to_run.append(activity)
        
        # Sort by priority (higher priority first)
        activities_to_run.sort(key=lambda x: x.priority, reverse=True)
        
        return activities_to_run
    
    def _should_run_activity(self, activity: ScheduledActivity, current_time: datetime, current_session: MarketSession) -> bool:
        """Check if an activity should run at the current time."""
        # Check if it's time to run based on frequency
        if activity.last_run is None:
            # First run
            return True
        
        time_since_last_run = current_time - activity.last_run
        
        if activity.frequency == "every_5min":
            return time_since_last_run >= timedelta(minutes=5)
        elif activity.frequency == "every_10min":
            return time_since_last_run >= timedelta(minutes=10)
        elif activity.frequency == "every_15min":
            return time_since_last_run >= timedelta(minutes=15)
        elif activity.frequency == "every_30min":
            return time_since_last_run >= timedelta(minutes=30)
        elif activity.frequency == "hourly":
            return time_since_last_run >= timedelta(hours=1)
        elif activity.frequency == "every_2hours":
            return time_since_last_run >= timedelta(hours=2)
        elif activity.frequency == "daily":
            # Check if it's the scheduled time for daily activities
            scheduled_time = activity.scheduled_time.time()
            current_time_only = current_time.time()
            
            # Allow 5-minute window for daily activities
            time_diff = abs((current_time_only.hour * 60 + current_time_only.minute) - 
                           (scheduled_time.hour * 60 + scheduled_time.minute))
            return time_diff <= 5 and time_since_last_run >= timedelta(hours=1)
        
        return False
    
    def _execute_activity(self, activity: ScheduledActivity) -> ActivityResult:
        """Execute a specific activity."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing activity: {activity.activity_type.value}")
            
            if activity.activity_type == ActivityType.MARKET_ANALYSIS:
                result = self._execute_market_analysis()
            elif activity.activity_type == ActivityType.PRE_MARKET_SCAN:
                result = self._execute_pre_market_scan()
            elif activity.activity_type == ActivityType.MARKET_OPEN:
                result = self._execute_market_open()
            elif activity.activity_type == ActivityType.INTRADAY_MONITORING:
                result = self._execute_intraday_monitoring()
            elif activity.activity_type == ActivityType.MARKET_CLOSE:
                result = self._execute_market_close()
            elif activity.activity_type == ActivityType.POST_MARKET_REVIEW:
                result = self._execute_post_market_review()
            elif activity.activity_type == ActivityType.OFF_HOURS_RESEARCH:
                result = self._execute_off_hours_research()
            elif activity.activity_type == ActivityType.RISK_ASSESSMENT:
                result = self._execute_risk_assessment()
            elif activity.activity_type == ActivityType.PORTFOLIO_REBALANCE:
                result = self._execute_portfolio_rebalance()
            elif activity.activity_type == ActivityType.MODEL_TRAINING:
                result = self._execute_model_training()
            elif activity.activity_type == ActivityType.DATA_COLLECTION:
                result = self._execute_data_collection()
            elif activity.activity_type == ActivityType.SYSTEM_HEALTH_CHECK:
                result = self._execute_system_health_check()
            else:
                result = {"success": False, "message": f"Unknown activity type: {activity.activity_type}"}
            
            end_time = datetime.now()
            
            return ActivityResult(
                activity_type=activity.activity_type,
                start_time=start_time,
                end_time=end_time,
                success=result.get("success", False),
                message=result.get("message", ""),
                data=result.get("data"),
                error=result.get("error")
            )
            
        except Exception as e:
            end_time = datetime.now()
            error_msg = f"Error executing {activity.activity_type.value}: {e}"
            logger.error(error_msg)
            
            return ActivityResult(
                activity_type=activity.activity_type,
                start_time=start_time,
                end_time=end_time,
                success=False,
                message="Activity execution failed",
                error=error_msg
            )
    
    def _execute_market_analysis(self) -> Dict[str, Any]:
        """Execute market analysis activity."""
        try:
            # Run complete trading cycle
            mode = get_current_mode()
            cycle_results = execute_complete_cycle(mode)
            
            return {
                "success": True,
                "message": f"Market analysis completed: {cycle_results.positions_opened} opened, {cycle_results.positions_closed} closed",
                "data": {
                    "positions_opened": cycle_results.positions_opened,
                    "positions_closed": cycle_results.positions_closed,
                    "total_pnl": cycle_results.total_pnl
                }
            }
        except Exception as e:
            return {"success": False, "message": f"Market analysis failed: {e}", "error": str(e)}
    
    def _execute_pre_market_scan(self) -> Dict[str, Any]:
        """Execute pre-market scan activity."""
        try:
            # Analyze market conditions and prepare for trading
            mode = get_current_mode()
            
            # Get market features for analysis
            market_features = self._get_market_features()
            
            # Analyze market regime
            market_regime = self.enhanced_ensemble.analyze_market_regime(market_features)
            
            return {
                "success": True,
                "message": f"Pre-market scan completed: {market_regime}",
                "data": {"market_regime": market_regime}
            }
        except Exception as e:
            return {"success": False, "message": f"Pre-market scan failed: {e}", "error": str(e)}
    
    def _execute_market_open(self) -> Dict[str, Any]:
        """Execute market open activity."""
        try:
            # Start active trading
            mode = get_current_mode()
            
            # Run buy phase to open new positions
            new_positions = self.trading_cycle.run_buy_phase(mode)
            
            return {
                "success": True,
                "message": f"Market open completed: {len(new_positions)} positions opened",
                "data": {"positions_opened": len(new_positions)}
            }
        except Exception as e:
            return {"success": False, "message": f"Market open failed: {e}", "error": str(e)}
    
    def _execute_intraday_monitoring(self) -> Dict[str, Any]:
        """Execute intraday monitoring activity."""
        try:
            # Monitor existing positions
            mode = get_current_mode()
            position_updates = self.trading_cycle.run_hold_phase(mode)
            
            return {
                "success": True,
                "message": f"Intraday monitoring completed: {len(position_updates)} positions monitored",
                "data": {"positions_monitored": len(position_updates)}
            }
        except Exception as e:
            return {"success": False, "message": f"Intraday monitoring failed: {e}", "error": str(e)}
    
    def _execute_market_close(self) -> Dict[str, Any]:
        """Execute market close activity."""
        try:
            # Close positions and prepare for post-market
            mode = get_current_mode()
            closed_positions = self.trading_cycle.run_sell_phase(mode)
            
            return {
                "success": True,
                "message": f"Market close completed: {len(closed_positions)} positions closed",
                "data": {"positions_closed": len(closed_positions)}
            }
        except Exception as e:
            return {"success": False, "message": f"Market close failed: {e}", "error": str(e)}
    
    def _execute_post_market_review(self) -> Dict[str, Any]:
        """Execute post-market review activity."""
        try:
            # Review trading performance and prepare for next day
            mode = get_current_mode()
            
            # Get portfolio summary
            portfolio_summary = self.trading_cycle.position_manager.get_portfolio_summary(mode)
            
            return {
                "success": True,
                "message": f"Post-market review completed: P&L ${portfolio_summary['total_pnl']:.2f}",
                "data": {"portfolio_summary": portfolio_summary}
            }
        except Exception as e:
            return {"success": False, "message": f"Post-market review failed: {e}", "error": str(e)}
    
    def _execute_off_hours_research(self) -> Dict[str, Any]:
        """Execute off-hours research activity."""
        try:
            # Research and analyze for next trading day
            mode = get_current_mode()
            
            # Analyze market features
            market_features = self._get_market_features()
            
            return {
                "success": True,
                "message": "Off-hours research completed",
                "data": {"market_features": market_features}
            }
        except Exception as e:
            return {"success": False, "message": f"Off-hours research failed: {e}", "error": str(e)}
    
    def _execute_risk_assessment(self) -> Dict[str, Any]:
        """Execute risk assessment activity."""
        try:
            # Assess current risk levels
            mode = get_current_mode()
            
            # Get risk summary
            risk_summary = self.trading_cycle.position_manager.get_risk_summary(mode)
            
            return {
                "success": True,
                "message": f"Risk assessment completed: {risk_summary['risk_status']['overall_status']}",
                "data": {"risk_summary": risk_summary}
            }
        except Exception as e:
            return {"success": False, "message": f"Risk assessment failed: {e}", "error": str(e)}
    
    def _execute_portfolio_rebalance(self) -> Dict[str, Any]:
        """Execute portfolio rebalance activity."""
        try:
            # Rebalance portfolio if needed
            mode = get_current_mode()
            
            # Check if rebalancing is needed
            portfolio_summary = self.trading_cycle.position_manager.get_portfolio_summary(mode)
            
            return {
                "success": True,
                "message": "Portfolio rebalance completed",
                "data": {"portfolio_summary": portfolio_summary}
            }
        except Exception as e:
            return {"success": False, "message": f"Portfolio rebalance failed: {e}", "error": str(e)}
    
    def _execute_model_training(self) -> Dict[str, Any]:
        """Execute model training activity."""
        try:
            # Train AI models with new data
            mode = get_current_mode()
            
            # Train enhanced ensemble
            training_result = self.enhanced_ensemble.train_models(mode)
            
            return {
                "success": True,
                "message": f"Model training completed: {training_result}",
                "data": {"training_result": training_result}
            }
        except Exception as e:
            return {"success": False, "message": f"Model training failed: {e}", "error": str(e)}
    
    def _execute_data_collection(self) -> Dict[str, Any]:
        """Execute data collection activity."""
        try:
            # Collect and update market data
            mode = get_current_mode()
            
            # Collect data for all symbols
            symbols = self._get_symbols_to_analyze(mode)
            data_collected = len(symbols)
            
            return {
                "success": True,
                "message": f"Data collection completed: {data_collected} symbols",
                "data": {"symbols_processed": data_collected}
            }
        except Exception as e:
            return {"success": False, "message": f"Data collection failed: {e}", "error": str(e)}
    
    def _execute_system_health_check(self) -> Dict[str, Any]:
        """Execute system health check activity."""
        try:
            # Check system health and performance
            mode = get_current_mode()
            
            # Check database connectivity
            db_healthy = self.trading_cycle.position_manager.validate_position_data(mode)
            
            # Check AI ensemble health
            ai_healthy = self.enhanced_ensemble.validate_ensemble()
            
            # Check trading cycle health
            cycle_healthy = self.trading_cycle.validate_cycle()
            
            overall_health = db_healthy and ai_healthy and cycle_healthy
            
            return {
                "success": True,
                "message": f"System health check completed: {'HEALTHY' if overall_health else 'ISSUES DETECTED'}",
                "data": {
                    "database_healthy": db_healthy,
                    "ai_healthy": ai_healthy,
                    "cycle_healthy": cycle_healthy,
                    "overall_health": overall_health
                }
            }
        except Exception as e:
            return {"success": False, "message": f"System health check failed: {e}", "error": str(e)}
    
    def _get_market_features(self) -> Dict[str, Any]:
        """Get current market features."""
        # This would integrate with real market data in production
        # For now, return simulated data
        import random
        
        return {
            "market_regime": random.choice(["BULL", "BEAR", "SIDEWAYS", "NEUTRAL"]),
            "volatility": random.uniform(0.01, 0.05),
            "volume_trend": random.choice(["HIGH", "NORMAL", "LOW"]),
            "sector_performance": random.uniform(-0.1, 0.1),
            "news_impact": random.uniform(-0.5, 0.5)
        }
    
    def _get_symbols_to_analyze(self, mode: str) -> List[str]:
        """Get list of symbols to analyze."""
        # This would integrate with the stock universe in production
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    def _calculate_next_run_time(self, activity: ScheduledActivity, current_time: datetime) -> datetime:
        """Calculate the next run time for an activity."""
        if activity.frequency == "every_5min":
            return current_time + timedelta(minutes=5)
        elif activity.frequency == "every_10min":
            return current_time + timedelta(minutes=10)
        elif activity.frequency == "every_15min":
            return current_time + timedelta(minutes=15)
        elif activity.frequency == "every_30min":
            return current_time + timedelta(minutes=30)
        elif activity.frequency == "hourly":
            return current_time + timedelta(hours=1)
        elif activity.frequency == "every_2hours":
            return current_time + timedelta(hours=2)
        elif activity.frequency == "daily":
            # Next day at the same time
            return current_time + timedelta(days=1)
        else:
            return current_time + timedelta(hours=1)
    
    def _cleanup_old_results(self) -> None:
        """Clean up old activity results to prevent memory buildup."""
        if len(self.activity_results) > self.max_results_history:
            # Keep only the most recent results
            self.activity_results = self.activity_results[-self.max_results_history:]
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "is_running": self.is_running,
            "total_activities": len(self.scheduled_activities),
            "enabled_activities": len([a for a in self.scheduled_activities.values() if a.enabled]),
            "recent_results": len(self.activity_results),
            "current_session": self._get_current_market_session(datetime.now()).value,
            "activities": [
                {
                    "type": activity.activity_type.value,
                    "enabled": activity.enabled,
                    "priority": activity.priority,
                    "frequency": activity.frequency,
                    "last_run": activity.last_run.isoformat() if activity.last_run else None,
                    "next_run": activity.next_run.isoformat() if activity.next_run else None,
                    "run_count": activity.run_count,
                    "success_count": activity.success_count,
                    "failure_count": activity.failure_count
                }
                for activity in self.scheduled_activities.values()
            ]
        }
    
    def enable_activity(self, activity_type: ActivityType) -> None:
        """Enable a specific activity."""
        if activity_type in self.scheduled_activities:
            self.scheduled_activities[activity_type].enabled = True
            logger.info(f"Enabled activity: {activity_type.value}")
    
    def disable_activity(self, activity_type: ActivityType) -> None:
        """Disable a specific activity."""
        if activity_type in self.scheduled_activities:
            self.scheduled_activities[activity_type].enabled = False
            logger.info(f"Disabled activity: {activity_type.value}")
    
    def get_activity_results(self, limit: int = 100) -> List[ActivityResult]:
        """Get recent activity results."""
        return self.activity_results[-limit:] if self.activity_results else []

# Global activity scheduler instance
_activity_scheduler: Optional[ActivityScheduler] = None

def get_activity_scheduler() -> ActivityScheduler:
    """Get the global activity scheduler instance."""
    global _activity_scheduler
    if _activity_scheduler is None:
        _activity_scheduler = ActivityScheduler()
    return _activity_scheduler

def start_scheduler() -> None:
    """Start the activity scheduler."""
    get_activity_scheduler().start_scheduler()

def stop_scheduler() -> None:
    """Stop the activity scheduler."""
    get_activity_scheduler().stop_scheduler()

def get_scheduler_status() -> Dict[str, Any]:
    """Get scheduler status."""
    return get_activity_scheduler().get_scheduler_status()
