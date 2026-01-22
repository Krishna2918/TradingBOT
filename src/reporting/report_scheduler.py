"""
Automated Report Scheduler
Schedules and generates reports automatically based on configured intervals
"""

import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Optional
import threading

from .report_generator import get_report_generator

logger = logging.getLogger(__name__)

class ReportScheduler:
    """
    Automated report scheduler
    
    Schedules:
    - Daily reports: Every day at 6:00 PM (after market close)
    - Weekly reports: Every Friday at 7:00 PM
    - Biweekly reports: Every other Friday at 7:30 PM
    - Monthly reports: 1st day of month at 8:00 PM
    - Quarterly reports: Last day of quarter at 8:00 PM
    - Yearly reports: December 31st at 11:00 PM
    """
    
    def __init__(self):
        self.report_generator = get_report_generator()
        self.is_running = False
        self.scheduler_thread = None
        
        logger.info("⏰ Report Scheduler initialized")
    
    def setup_schedules(self):
        """Setup all report schedules"""
        
        # Daily report - 6:00 PM EST (after market close)
        schedule.every().day.at("18:00").do(self._run_daily_report)
        logger.info(" Daily report scheduled for 6:00 PM EST")
        
        # Weekly report - Every Friday at 7:00 PM
        schedule.every().friday.at("19:00").do(self._run_weekly_report)
        logger.info(" Weekly report scheduled for Fridays at 7:00 PM EST")
        
        # Biweekly report - Every other Friday at 7:30 PM
        # (Will check if it's the right week)
        schedule.every().friday.at("19:30").do(self._run_biweekly_report)
        logger.info(" Biweekly report scheduled for alternating Fridays at 7:30 PM EST")
        
        # Monthly report - 1st of month at 8:00 PM
        schedule.every().day.at("20:00").do(self._run_monthly_report_if_first_day)
        logger.info(" Monthly report scheduled for 1st of month at 8:00 PM EST")
        
        # Quarterly report - Last day of quarter at 8:00 PM
        schedule.every().day.at("20:00").do(self._run_quarterly_report_if_quarter_end)
        logger.info(" Quarterly report scheduled for end of quarter at 8:00 PM EST")
        
        # Yearly report - December 31st at 11:00 PM
        schedule.every().day.at("23:00").do(self._run_yearly_report_if_year_end)
        logger.info(" Yearly report scheduled for Dec 31st at 11:00 PM EST")
    
    def _run_daily_report(self):
        """Run daily report"""
        try:
            logger.info(" Running scheduled daily report...")
            report = self.report_generator.generate_daily_report()
            logger.info(" Daily report completed successfully")
            return report
        except Exception as e:
            logger.error(f" Daily report failed: {e}")
    
    def _run_weekly_report(self):
        """Run weekly report"""
        try:
            logger.info(" Running scheduled weekly report...")
            report = self.report_generator.generate_weekly_report()
            logger.info(" Weekly report completed successfully")
            return report
        except Exception as e:
            logger.error(f" Weekly report failed: {e}")
    
    def _run_biweekly_report(self):
        """Run biweekly report (every other week)"""
        try:
            # Check if it's the right week (even week numbers)
            week_number = datetime.now().isocalendar()[1]
            if week_number % 2 == 0:
                logger.info(" Running scheduled biweekly report...")
                # Generate using same method as weekly but with 14-day period
                report = self.report_generator.generate_weekly_report()  # Adapt for biweekly
                logger.info(" Biweekly report completed successfully")
                return report
            else:
                logger.debug("⏭ Skipping biweekly report (not the right week)")
        except Exception as e:
            logger.error(f" Biweekly report failed: {e}")
    
    def _run_monthly_report_if_first_day(self):
        """Run monthly report if it's the first day of the month"""
        try:
            if datetime.now().day == 1:
                logger.info(" Running scheduled monthly report...")
                report = self.report_generator.generate_monthly_report()
                logger.info(" Monthly report completed successfully")
                return report
        except Exception as e:
            logger.error(f" Monthly report failed: {e}")
    
    def _run_quarterly_report_if_quarter_end(self):
        """Run quarterly report if it's the last day of the quarter"""
        try:
            today = datetime.now()
            month = today.month
            
            # Check if it's the last day of a quarter month (March, June, September, December)
            if month in [3, 6, 9, 12]:
                # Check if it's the last day of the month
                next_day = today + timedelta(days=1)
                if next_day.month != month:  # If next day is in a different month
                    logger.info(" Running scheduled quarterly report...")
                    report = self.report_generator.generate_quarterly_report()
                    logger.info(" Quarterly report completed successfully")
                    return report
        except Exception as e:
            logger.error(f" Quarterly report failed: {e}")
    
    def _run_yearly_report_if_year_end(self):
        """Run yearly report if it's December 31st"""
        try:
            today = datetime.now()
            if today.month == 12 and today.day == 31:
                logger.info(" Running scheduled yearly report...")
                report = self.report_generator.generate_yearly_report()
                logger.info(" Yearly report completed successfully")
                return report
        except Exception as e:
            logger.error(f" Yearly report failed: {e}")
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning(" Scheduler is already running")
            return
        
        self.setup_schedules()
        self.is_running = True
        
        # Run scheduler in a separate thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(" Report scheduler started")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        logger.info(" Scheduler loop started")
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info(" Report scheduler stopped")
    
    def generate_on_demand(self, report_type: str, **kwargs):
        """Generate a report on demand"""
        
        logger.info(f" Generating on-demand {report_type} report...")
        
        try:
            if report_type == 'daily':
                return self.report_generator.generate_daily_report(**kwargs)
            elif report_type == 'weekly':
                return self.report_generator.generate_weekly_report(**kwargs)
            elif report_type == 'monthly':
                return self.report_generator.generate_monthly_report(**kwargs)
            elif report_type == 'quarterly':
                return self.report_generator.generate_quarterly_report(**kwargs)
            elif report_type == 'yearly':
                return self.report_generator.generate_yearly_report(**kwargs)
            else:
                logger.error(f" Unknown report type: {report_type}")
                return None
        except Exception as e:
            logger.error(f" On-demand report generation failed: {e}")
            return None

# Global scheduler instance
_scheduler_instance = None

def get_report_scheduler() -> ReportScheduler:
    """Get global report scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = ReportScheduler()
    return _scheduler_instance

