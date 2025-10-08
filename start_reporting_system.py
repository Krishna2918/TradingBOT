"""
Automated Reporting System Startup Script

Starts the automated report scheduler and generates initial reports
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Start the automated reporting system"""
    
    print("=" * 70)
    print("🚀 Automated Reporting System Starting...")
    print("=" * 70)
    print("📊 Report Types:")
    print("   • Daily: Every day at 6:00 PM EST")
    print("   • Weekly: Every Friday at 7:00 PM EST")
    print("   • Biweekly: Every other Friday at 7:30 PM EST")
    print("   • Monthly: 1st of month at 8:00 PM EST")
    print("   • Quarterly: End of quarter at 8:00 PM EST")
    print("   • Yearly: December 31st at 11:00 PM EST")
    print("=" * 70)
    print("🧠 AI Learning Features:")
    print("   • Learns from every report")
    print("   • Extracts insights automatically")
    print("   • Updates parameters daily")
    print("   • Implements corrections immediately")
    print("   • Tracks improvements over time")
    print("=" * 70)
    print("📁 Report Locations:")
    print("   • Daily: reports/daily/")
    print("   • Weekly: reports/weekly/")
    print("   • Biweekly: reports/biweekly/")
    print("   • Monthly: reports/monthly/")
    print("   • Quarterly: reports/quarterly/")
    print("   • Yearly: reports/yearly/")
    print("=" * 70)
    print()
    
    try:
        # Import reporting modules
        from src.reporting import get_report_generator, get_report_scheduler
        
        # Initialize report generator
        logger.info("📊 Initializing report generator...")
        generator = get_report_generator()
        logger.info("✅ Report generator initialized")
        
        # Generate initial daily report
        logger.info("📊 Generating initial daily report...")
        daily_report = generator.generate_daily_report()
        logger.info("✅ Initial daily report generated")
        
        # Initialize and start scheduler
        logger.info("⏰ Initializing report scheduler...")
        scheduler = get_report_scheduler()
        scheduler.start()
        logger.info("✅ Report scheduler started")
        
        print("=" * 70)
        print("✅ Automated Reporting System is now running!")
        print("=" * 70)
        print("📊 Current Status:")
        print(f"   • System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}")
        print(f"   • Reports Directory: reports/")
        print(f"   • AI Learning Database: data/ai_learning_database.json")
        print("=" * 70)
        print("📈 What Happens Next:")
        print("   1. Reports are generated automatically on schedule")
        print("   2. AI analyzes each report for insights")
        print("   3. Learning database is updated with findings")
        print("   4. Parameters are adjusted for improvement")
        print("   5. Changes are applied to trading strategies")
        print("=" * 70)
        print("🔍 Monitor Reports:")
        print("   • View reports in: reports/ directory")
        print("   • Check AI learning: data/ai_learning_database.json")
        print("   • Review dashboard: Trading Bot Dashboard")
        print("=" * 70)
        print("⌨️  Press Ctrl+C to stop the reporting system")
        print("=" * 70)
        print()
        
        # Keep running
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n")
            print("=" * 70)
            print("🛑 Stopping Automated Reporting System...")
            print("=" * 70)
            scheduler.stop()
            print("✅ Reporting system stopped gracefully")
            print("=" * 70)
    
    except Exception as e:
        logger.error(f"❌ Failed to start reporting system: {e}")
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("  1. All dependencies are installed (pip install -r requirements.txt)")
        print("  2. src/reporting/ directory exists")
        print("  3. Configuration files are present")
        sys.exit(1)

if __name__ == '__main__':
    main()

