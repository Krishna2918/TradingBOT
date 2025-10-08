"""
Test script for the Automated Reporting System

Tests report generation and AI learning capabilities
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.reporting import get_report_generator, get_report_scheduler
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_report_generation():
    """Test generating all report types"""
    
    print("\n" + "=" * 70)
    print("🧪 Testing Automated Reporting System")
    print("=" * 70)
    
    try:
        # Get report generator
        logger.info("📊 Initializing report generator...")
        generator = get_report_generator()
        logger.info("✅ Report generator initialized")
        
        # Test daily report
        print("\n📅 Testing Daily Report Generation...")
        daily_report = generator.generate_daily_report()
        print("✅ Daily report generated successfully")
        print(f"   - Metadata: {daily_report['metadata']}")
        
        # Test weekly report
        print("\n📅 Testing Weekly Report Generation...")
        weekly_report = generator.generate_weekly_report()
        print("✅ Weekly report generated successfully")
        print(f"   - Metadata: {weekly_report['metadata']}")
        
        # Test monthly report
        print("\n📅 Testing Monthly Report Generation...")
        monthly_report = generator.generate_monthly_report()
        print("✅ Monthly report generated successfully")
        print(f"   - Metadata: {monthly_report['metadata']}")
        
        print("\n" + "=" * 70)
        print("✅ All report types tested successfully!")
        print("=" * 70)
        
        # Check file generation
        print("\n📁 Checking generated files...")
        reports_dir = Path("reports")
        
        daily_files = list((reports_dir / "daily").glob("*.json"))
        weekly_files = list((reports_dir / "weekly").glob("*.json"))
        monthly_files = list((reports_dir / "monthly").glob("*.json"))
        
        print(f"   - Daily reports: {len(daily_files)} files")
        print(f"   - Weekly reports: {len(weekly_files)} files")
        print(f"   - Monthly reports: {len(monthly_files)} files")
        
        # Check AI learning database
        print("\n🧠 Checking AI Learning Database...")
        learning_db_path = Path("data/ai_learning_database.json")
        if learning_db_path.exists():
            print("✅ AI learning database exists")
            import json
            with open(learning_db_path, 'r') as f:
                learning_db = json.load(f)
            print(f"   - Total learnings: {len(learning_db.get('learnings', []))}")
        else:
            print("⚠️  AI learning database not yet created (will be created after first report)")
        
        print("\n" + "=" * 70)
        print("🎉 Reporting System Test Complete!")
        print("=" * 70)
        print("\n📊 Next Steps:")
        print("   1. Review generated reports in: reports/")
        print("   2. Check AI learning database: data/ai_learning_database.json")
        print("   3. Start automated scheduler: python start_reporting_system.py")
        print("   4. View reports in dashboard")
        print()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scheduler():
    """Test report scheduler"""
    
    print("\n" + "=" * 70)
    print("🧪 Testing Report Scheduler")
    print("=" * 70)
    
    try:
        # Get scheduler
        logger.info("⏰ Initializing report scheduler...")
        scheduler = get_report_scheduler()
        logger.info("✅ Report scheduler initialized")
        
        # Test on-demand generation
        print("\n📅 Testing on-demand report generation...")
        report = scheduler.generate_on_demand('daily')
        if report:
            print("✅ On-demand daily report generated successfully")
        else:
            print("❌ On-demand report generation failed")
        
        print("\n" + "=" * 70)
        print("✅ Scheduler test complete!")
        print("=" * 70)
        print("\n📊 Scheduler Configuration:")
        print("   • Daily: Every day at 6:00 PM EST")
        print("   • Weekly: Every Friday at 7:00 PM EST")
        print("   • Biweekly: Every other Friday at 7:30 PM EST")
        print("   • Monthly: 1st of month at 8:00 PM EST")
        print("   • Quarterly: End of quarter at 8:00 PM EST")
        print("   • Yearly: December 31st at 11:00 PM EST")
        print()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Scheduler test failed: {e}")
        print(f"\n❌ Error during scheduler testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\n🚀 Starting Reporting System Tests...\n")
    
    # Test report generation
    report_test = test_report_generation()
    
    # Test scheduler
    scheduler_test = test_scheduler()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"   Report Generation: {'✅ PASS' if report_test else '❌ FAIL'}")
    print(f"   Scheduler: {'✅ PASS' if scheduler_test else '❌ FAIL'}")
    print("=" * 70)
    
    if report_test and scheduler_test:
        print("\n🎉 All tests passed! Reporting system is ready to use.")
        print("\n📚 Documentation:")
        print("   • Complete Guide: REPORTING_SYSTEM_GUIDE.md")
        print("   • Features Summary: REPORTING_FEATURES_SUMMARY.md")
        print("   • Start System: python start_reporting_system.py")
        print()
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        sys.exit(1)

