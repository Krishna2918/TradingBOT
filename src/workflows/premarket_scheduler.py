"""
Pre-Market Data Collection Scheduler

Ensures all data is collected and analyzed BEFORE market open at 9:30 AM ET.

Schedule:
- 6:00 AM ET: Start data collection
- 6:30 AM: Fetch US futures, overnight moves
- 7:00 AM: Currency rates, commodity prices
- 7:30 AM: News sentiment, social media scan
- 8:00 AM: Insider trades, institutional flows
- 8:30 AM: Run AI analysis, generate signals
- 9:00 AM: Final pre-market summary ready
- 9:30 AM: Market opens - execute strategies

For Canadian TSX market hours: 9:30 AM - 4:00 PM ET
"""

import logging
import asyncio
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import schedule
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ScheduledTask:
    """A scheduled pre-market task"""
    name: str
    scheduled_time: dt_time  # Time in ET
    task_func: Callable
    priority: int = 5  # 1 = highest priority
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300  # 5 minute default timeout
    retry_count: int = 2
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = None
    started_at: datetime = None
    completed_at: datetime = None


class PreMarketScheduler:
    """
    Orchestrates pre-market data collection and analysis

    Ensures Canadian market trading is ready before 9:30 AM ET
    """

    # Canadian market hours (Eastern Time)
    MARKET_OPEN = dt_time(9, 30)   # 9:30 AM ET
    MARKET_CLOSE = dt_time(16, 0)  # 4:00 PM ET

    # Pre-market schedule
    PREMARKET_START = dt_time(6, 0)  # 6:00 AM ET

    def __init__(
        self,
        output_dir: str = "data/premarket",
        demo_mode: bool = True
    ):
        """
        Initialize pre-market scheduler

        Args:
            output_dir: Directory to store pre-market analysis
            demo_mode: If True, use simulated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.demo_mode = demo_mode

        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_results: Dict[str, Any] = {}
        self.is_running = False
        self._lock = threading.Lock()

        # Initialize data collectors
        self._init_collectors()

        # Register default tasks
        self._register_default_tasks()

        logger.info(f"PreMarket Scheduler initialized (demo_mode={demo_mode})")
        logger.info(f"Output directory: {self.output_dir}")

    def _init_collectors(self):
        """Initialize data collection modules"""
        try:
            from src.data_services.currency_tracker import (
                CurrencyTracker, USPreMarketTracker
            )
            self.currency_tracker = CurrencyTracker(demo_mode=self.demo_mode)
            self.premarket_tracker = USPreMarketTracker(demo_mode=self.demo_mode)
        except ImportError as e:
            logger.warning(f"Could not import currency_tracker: {e}")
            self.currency_tracker = None
            self.premarket_tracker = None

        try:
            from src.data_services.social_sentiment import SocialSentimentTracker
            self.social_tracker = SocialSentimentTracker(demo_mode=self.demo_mode)
        except ImportError:
            self.social_tracker = None

        try:
            from src.data_services.insider_trades import InsiderTradesTracker
            self.insider_tracker = InsiderTradesTracker(demo_mode=self.demo_mode)
        except ImportError:
            self.insider_tracker = None

        try:
            from src.data_services.whale_tracker import WhaleTracker
            self.whale_tracker = WhaleTracker(demo_mode=self.demo_mode)
        except ImportError:
            self.whale_tracker = None

        try:
            from src.data_services.weather_commodities import WeatherCommodityTracker
            self.weather_tracker = WeatherCommodityTracker(demo_mode=self.demo_mode)
        except ImportError:
            self.weather_tracker = None

        try:
            from src.event_awareness.event_calendar import EventCalendar
            self.event_calendar = EventCalendar() if hasattr(self, 'EventCalendar') else None
        except ImportError:
            self.event_calendar = None

    def _register_default_tasks(self):
        """Register the default pre-market task schedule"""

        # 6:00 AM - Start overnight data collection
        self.register_task(ScheduledTask(
            name="overnight_futures",
            scheduled_time=dt_time(6, 0),
            task_func=self._collect_overnight_futures,
            priority=1,
            timeout_seconds=180
        ))

        # 6:30 AM - US Pre-market data
        self.register_task(ScheduledTask(
            name="us_premarket",
            scheduled_time=dt_time(6, 30),
            task_func=self._collect_us_premarket,
            priority=1,
            dependencies=["overnight_futures"],
            timeout_seconds=180
        ))

        # 7:00 AM - Currency and commodity prices
        self.register_task(ScheduledTask(
            name="currency_commodities",
            scheduled_time=dt_time(7, 0),
            task_func=self._collect_currency_commodities,
            priority=2,
            timeout_seconds=180
        ))

        # 7:30 AM - News and social sentiment
        self.register_task(ScheduledTask(
            name="sentiment_analysis",
            scheduled_time=dt_time(7, 30),
            task_func=self._collect_sentiment,
            priority=2,
            timeout_seconds=300
        ))

        # 8:00 AM - Insider trades and institutional flows
        self.register_task(ScheduledTask(
            name="institutional_flows",
            scheduled_time=dt_time(8, 0),
            task_func=self._collect_institutional_data,
            priority=3,
            timeout_seconds=180
        ))

        # 8:00 AM - Events calendar check
        self.register_task(ScheduledTask(
            name="events_calendar",
            scheduled_time=dt_time(8, 0),
            task_func=self._check_events_calendar,
            priority=3,
            timeout_seconds=120
        ))

        # 8:30 AM - Run AI analysis
        self.register_task(ScheduledTask(
            name="ai_analysis",
            scheduled_time=dt_time(8, 30),
            task_func=self._run_ai_analysis,
            priority=1,
            dependencies=["us_premarket", "currency_commodities", "sentiment_analysis"],
            timeout_seconds=600  # AI analysis may take longer
        ))

        # 9:00 AM - Generate final pre-market summary
        self.register_task(ScheduledTask(
            name="premarket_summary",
            scheduled_time=dt_time(9, 0),
            task_func=self._generate_summary,
            priority=1,
            dependencies=["ai_analysis", "institutional_flows", "events_calendar"],
            timeout_seconds=120
        ))

    def register_task(self, task: ScheduledTask):
        """Register a task with the scheduler"""
        self.tasks[task.name] = task
        logger.debug(f"Registered task: {task.name} at {task.scheduled_time}")

    def run_now(self) -> Dict[str, Any]:
        """
        Run all pre-market tasks immediately (for testing or late start)

        Returns:
            Dict with all task results
        """
        logger.info("Running all pre-market tasks NOW...")

        with self._lock:
            self.is_running = True

        # Sort tasks by priority and dependencies
        ordered_tasks = self._get_execution_order()

        for task_name in ordered_tasks:
            task = self.tasks[task_name]
            self._execute_task(task)

        with self._lock:
            self.is_running = False

        # Generate and return summary
        return self._compile_results()

    def _get_execution_order(self) -> List[str]:
        """Get tasks in correct execution order based on dependencies"""
        ordered = []
        remaining = set(self.tasks.keys())

        while remaining:
            # Find tasks with all dependencies satisfied
            ready = []
            for name in remaining:
                task = self.tasks[name]
                deps_satisfied = all(d in ordered for d in task.dependencies)
                if deps_satisfied:
                    ready.append((task.priority, task.scheduled_time, name))

            if not ready:
                # Circular dependency or missing dependency
                logger.error(f"Cannot resolve task order. Remaining: {remaining}")
                break

            # Sort by priority, then scheduled time
            ready.sort()

            for _, _, name in ready:
                ordered.append(name)
                remaining.remove(name)

        return ordered

    def _execute_task(self, task: ScheduledTask):
        """Execute a single task"""
        logger.info(f"Executing task: {task.name}")
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        for attempt in range(task.retry_count + 1):
            try:
                result = task.task_func()
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                self.task_results[task.name] = result
                logger.info(f"Task {task.name} completed successfully")
                return

            except Exception as e:
                logger.error(f"Task {task.name} failed (attempt {attempt + 1}): {e}")
                task.error = str(e)

                if attempt < task.retry_count:
                    logger.info(f"Retrying task {task.name}...")
                    continue

        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        logger.error(f"Task {task.name} failed after {task.retry_count + 1} attempts")

    def _compile_results(self) -> Dict[str, Any]:
        """Compile all task results into final summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'status': 'complete',
            'tasks': {},
            'data': self.task_results,
            'warnings': [],
            'errors': []
        }

        for name, task in self.tasks.items():
            summary['tasks'][name] = {
                'status': task.status.value,
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'error': task.error
            }

            if task.status == TaskStatus.FAILED:
                summary['errors'].append(f"{name}: {task.error}")
                summary['status'] = 'partial'

        # Save to file
        output_file = self.output_dir / f"premarket_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Pre-market summary saved to {output_file}")
        return summary

    # ========== Task Implementation Functions ==========

    def _collect_overnight_futures(self) -> Dict[str, Any]:
        """Collect overnight futures data"""
        if self.premarket_tracker:
            return self.premarket_tracker.get_premarket_data()
        return {'status': 'skipped', 'reason': 'tracker not available'}

    def _collect_us_premarket(self) -> Dict[str, Any]:
        """Collect US pre-market data and gap prediction"""
        if self.premarket_tracker:
            return {
                'futures': self.premarket_tracker.get_premarket_data(),
                'tsx_gap_prediction': self.premarket_tracker.predict_tsx_gap(),
                'sector_outlook': self.premarket_tracker.get_sector_outlook()
            }
        return {'status': 'skipped', 'reason': 'tracker not available'}

    def _collect_currency_commodities(self) -> Dict[str, Any]:
        """Collect currency and commodity data"""
        result = {}

        if self.currency_tracker:
            result['currencies'] = self.currency_tracker.get_current_rates()
            result['usdcad'] = self.currency_tracker.get_usdcad()

        if self.weather_tracker:
            # Get weather/commodity impacts for major Canadian stocks
            symbols = ['CNQ.TO', 'SU.TO', 'ENB.TO', 'ABX.TO', 'RY.TO']
            result['commodity_impacts'] = self.weather_tracker.get_impact_score(symbols)

        return result if result else {'status': 'skipped'}

    def _collect_sentiment(self) -> Dict[str, Any]:
        """Collect news and social sentiment"""
        result = {}

        # Canadian stock universe for sentiment
        symbols = [
            'RY.TO', 'TD.TO', 'BNS.TO', 'CNQ.TO', 'SU.TO',
            'ENB.TO', 'ABX.TO', 'SHOP.TO', 'BCE.TO', 'CNR.TO'
        ]

        if self.social_tracker:
            result['social_sentiment'] = self.social_tracker.get_social_sentiment(symbols)

        # Add news sentiment if available
        try:
            from src.data_services.free_apis_integration import FreeAPIsIntegration
            config = {'api_keys': {}, 'rate_limits': {}}
            news_api = FreeAPIsIntegration(config)
            result['news_sentiment'] = news_api.get_news_sentiment(symbols[:5])
        except Exception as e:
            logger.warning(f"Could not fetch news sentiment: {e}")

        return result if result else {'status': 'skipped'}

    def _collect_institutional_data(self) -> Dict[str, Any]:
        """Collect insider trades and whale activity"""
        result = {}

        symbols = ['RY.TO', 'TD.TO', 'CNQ.TO', 'SU.TO', 'ENB.TO', 'ABX.TO']

        if self.insider_tracker:
            result['insider_trades'] = self.insider_tracker.get_insider_sentiment(symbols)

        if self.whale_tracker:
            result['whale_activity'] = self.whale_tracker.get_whale_activity(symbols)

        return result if result else {'status': 'skipped'}

    def _check_events_calendar(self) -> Dict[str, Any]:
        """Check for important events today"""
        today = datetime.now().date()

        # Key events to watch
        events = {
            'earnings_today': [],
            'economic_releases': [],
            'central_bank': [],
            'dividends': [],
            'warnings': []
        }

        # Check for Bank of Canada meetings (typically 8 per year)
        # This is a simplified check - real implementation would use API
        if self.event_calendar:
            try:
                upcoming = self.event_calendar.get_upcoming_events(hours_ahead=24)
                for event in upcoming:
                    if event.event_type == 'central_bank':
                        events['central_bank'].append(event.to_dict())
                    elif event.event_type == 'earnings':
                        events['earnings_today'].append(event.to_dict())
                    elif event.event_type == 'economic':
                        events['economic_releases'].append(event.to_dict())
            except Exception as e:
                logger.warning(f"Error checking event calendar: {e}")

        # Add warning if high-impact events
        if events['central_bank']:
            events['warnings'].append("Bank of Canada announcement today - expect volatility")
        if len(events['earnings_today']) > 5:
            events['warnings'].append(f"Heavy earnings day: {len(events['earnings_today'])} reports")

        return events

    def _run_ai_analysis(self) -> Dict[str, Any]:
        """Run AI analysis on collected data"""
        logger.info("Running AI analysis on pre-market data...")

        analysis = {
            'market_regime': 'normal',
            'risk_level': 'medium',
            'recommended_exposure': 0.8,
            'sector_recommendations': {},
            'top_picks': [],
            'avoid_list': [],
            'key_insights': []
        }

        # Analyze TSX gap prediction
        if 'us_premarket' in self.task_results:
            premarket = self.task_results['us_premarket']
            gap = premarket.get('tsx_gap_prediction', {})

            if gap.get('direction') == 'down' and gap.get('predicted_gap_pct', 0) < -1:
                analysis['market_regime'] = 'risk_off'
                analysis['risk_level'] = 'high'
                analysis['recommended_exposure'] = 0.5
                analysis['key_insights'].append(
                    f"Expecting {gap['predicted_gap_pct']:.1f}% gap down - reduce exposure"
                )
            elif gap.get('direction') == 'up' and gap.get('predicted_gap_pct', 0) > 1:
                analysis['market_regime'] = 'risk_on'
                analysis['key_insights'].append(
                    f"Expecting {gap['predicted_gap_pct']:.1f}% gap up - favorable conditions"
                )

            # Sector outlook
            sectors = premarket.get('sector_outlook', {})
            for sector, outlook in sectors.items():
                if outlook.get('outlook') == 'bullish':
                    analysis['sector_recommendations'][sector] = 'overweight'
                elif outlook.get('outlook') == 'bearish':
                    analysis['sector_recommendations'][sector] = 'underweight'
                else:
                    analysis['sector_recommendations'][sector] = 'neutral'

        # Check currency impact
        if 'currency_commodities' in self.task_results:
            fx_data = self.task_results['currency_commodities']
            usdcad = fx_data.get('usdcad', {})

            if usdcad.get('trend') == 'strengthening':
                analysis['key_insights'].append(
                    "CAD weakening - favor exporters (energy, mining)"
                )
            elif usdcad.get('trend') == 'weakening':
                analysis['key_insights'].append(
                    "CAD strengthening - favor importers and banks"
                )

        # Check sentiment
        if 'sentiment_analysis' in self.task_results:
            sentiment = self.task_results['sentiment_analysis']
            social = sentiment.get('social_sentiment', {})

            # Find stocks with strong positive/negative sentiment
            for symbol, data in social.items():
                score = data.get('score', 0)
                if score > 0.5 and data.get('trending', False):
                    analysis['top_picks'].append({
                        'symbol': symbol,
                        'reason': 'Strong positive social sentiment + trending'
                    })
                elif score < -0.3:
                    analysis['avoid_list'].append({
                        'symbol': symbol,
                        'reason': 'Negative social sentiment'
                    })

        # Check for event risks
        if 'events_calendar' in self.task_results:
            events = self.task_results['events_calendar']
            warnings = events.get('warnings', [])
            for warning in warnings:
                analysis['key_insights'].append(f"⚠️ {warning}")

            if events.get('central_bank'):
                analysis['risk_level'] = 'high'
                analysis['recommended_exposure'] = min(analysis['recommended_exposure'], 0.6)

        return analysis

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate final pre-market summary"""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'market_open': '9:30 AM ET',
            'ready_for_trading': True,
            'data_quality': 'good',
            'ai_analysis': self.task_results.get('ai_analysis', {}),
            'quick_view': {}
        }

        # Create quick view
        ai = self.task_results.get('ai_analysis', {})
        summary['quick_view'] = {
            'market_regime': ai.get('market_regime', 'unknown'),
            'risk_level': ai.get('risk_level', 'unknown'),
            'recommended_exposure': f"{ai.get('recommended_exposure', 0.8) * 100:.0f}%",
            'top_sectors': [
                s for s, r in ai.get('sector_recommendations', {}).items()
                if r == 'overweight'
            ],
            'avoid_sectors': [
                s for s, r in ai.get('sector_recommendations', {}).items()
                if r == 'underweight'
            ],
            'key_insights': ai.get('key_insights', [])[:5]
        }

        # Check data quality
        failed_tasks = [
            name for name, task in self.tasks.items()
            if task.status == TaskStatus.FAILED
        ]
        if failed_tasks:
            summary['data_quality'] = 'partial'
            summary['failed_tasks'] = failed_tasks

        if len(failed_tasks) > 3:
            summary['ready_for_trading'] = False
            summary['data_quality'] = 'poor'

        return summary

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            'is_running': self.is_running,
            'tasks': {
                name: {
                    'status': task.status.value,
                    'scheduled': task.scheduled_time.isoformat()
                }
                for name, task in self.tasks.items()
            },
            'next_market_open': self.MARKET_OPEN.isoformat(),
            'premarket_start': self.PREMARKET_START.isoformat()
        }


# Convenience function
def run_premarket_collection(demo_mode: bool = True) -> Dict[str, Any]:
    """
    Run complete pre-market data collection

    Call this before 9:30 AM ET to prepare for Canadian market trading

    Args:
        demo_mode: Use simulated data if True

    Returns:
        Complete pre-market analysis summary
    """
    scheduler = PreMarketScheduler(demo_mode=demo_mode)
    return scheduler.run_now()


if __name__ == "__main__":
    # Run pre-market collection
    logging.basicConfig(level=logging.INFO)
    result = run_premarket_collection(demo_mode=True)

    print("\n" + "="*60)
    print("PRE-MARKET SUMMARY")
    print("="*60)

    quick = result.get('data', {}).get('premarket_summary', {}).get('quick_view', {})
    print(f"\nMarket Regime: {quick.get('market_regime', 'N/A')}")
    print(f"Risk Level: {quick.get('risk_level', 'N/A')}")
    print(f"Recommended Exposure: {quick.get('recommended_exposure', 'N/A')}")

    print(f"\nTop Sectors: {', '.join(quick.get('top_sectors', ['N/A']))}")
    print(f"Avoid Sectors: {', '.join(quick.get('avoid_sectors', ['N/A']))}")

    print("\nKey Insights:")
    for insight in quick.get('key_insights', ['None']):
        print(f"  • {insight}")

    print("\n" + "="*60)
