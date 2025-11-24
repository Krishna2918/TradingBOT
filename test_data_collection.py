#!/usr/bin/env python3
"""
Test Data Collection System

Quick test to verify the data collection system is working properly.
Tests symbol verification, data fetching, storage, and progress tracking.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collection import (
    SymbolManager, 
    ProgressTracker, 
    StorageManager, 
    HistoricalAppender,
    DataValidator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_symbol_manager():
    """Test symbol manager functionality"""
    print("\n" + "="*60)
    print("🧪 TESTING SYMBOL MANAGER")
    print("="*60)
    
    manager = SymbolManager()
    
    # Test symbol loading
    all_symbols = manager.get_all_symbols()
    print(f"📊 Total symbols loaded: {len(all_symbols)}")
    
    # Show breakdown by priority
    priority_symbols = manager.get_symbols_by_priority()
    for priority, symbols in priority_symbols.items():
        print(f"   {priority}: {len(symbols)} symbols")
        print(f"   Examples: {', '.join(symbols[:3])}")
    
    # Test symbol verification (just a few)
    test_symbols = ["RY.TO", "TD.TO", "SHOP.TO"]
    print(f"\n🔍 Testing symbol verification with: {test_symbols}")
    
    verification_results = {}
    for symbol in test_symbols:
        result = manager.verify_symbol_availability(symbol)
        verification_results[symbol] = result
        status = "✅ Available" if result else "❌ Not available"
        print(f"   {symbol}: {status}")
    
    return verification_results

def test_progress_tracker():
    """Test progress tracking functionality"""
    print("\n" + "="*60)
    print("🧪 TESTING PROGRESS TRACKER")
    print("="*60)
    
    tracker = ProgressTracker()
    
    # Test logging progress
    print("📝 Testing progress logging...")
    tracker.log_progress("RY.TO", "1d", 5000, 2.5, 0, 0.95)
    tracker.log_progress("TD.TO", "1d", 4800, 2.3, 1, 0.90)
    tracker.log_progress("SHOP.TO", "5m", 1200, 0.8, 0, 0.85)
    
    # Get progress summary
    summary = tracker.get_progress_summary()
    print(f"📊 Progress Summary:")
    print(f"   Overall completion: {summary['overall_progress']['completion_percentage']:.1f}%")
    print(f"   Total data: {summary['overall_progress']['total_data_gb']:.2f} GB")
    print(f"   Future steps: {len(summary['future_steps'])}")
    
    return summary

def test_storage_manager():
    """Test storage manager functionality"""
    print("\n" + "="*60)
    print("🧪 TESTING STORAGE MANAGER")
    print("="*60)
    
    import pandas as pd
    import numpy as np
    
    storage = StorageManager()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1D')
    sample_data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(95, 115, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Test save and load
    print("💾 Testing save/load operations...")
    
    save_success = storage.save_to_parquet(sample_data, "TEST.TO", "1d")
    print(f"   Save test: {'✅ Success' if save_success else '❌ Failed'}")
    
    loaded_data = storage.load_from_parquet("TEST.TO", "1d")
    load_success = not loaded_data.empty  # Just check if data was loaded
    print(f"   Load test: {'✅ Success' if load_success else '❌ Failed'}")
    
    # Test quality validation
    quality_metrics = storage.validate_data_quality("TEST.TO", "1d")
    print(f"   Quality score: {quality_metrics['quality_score']:.3f}")
    
    # Test storage summary
    summary = storage.get_storage_summary()
    print(f"📊 Storage Summary:")
    print(f"   Total files: {summary['total_files']}")
    print(f"   Total size: {summary['total_size_mb']:.2f} MB")
    print(f"   Symbols: {len(summary['symbols'])}")
    
    return save_success and load_success

def test_data_validator():
    """Test data validator functionality"""
    print("\n" + "="*60)
    print("🧪 TESTING DATA VALIDATOR")
    print("="*60)
    
    import pandas as pd
    import numpy as np
    
    validator = DataValidator()
    
    # Create sample data with some issues
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1D')
    sample_data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(95, 115, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Introduce some data quality issues for testing
    sample_data.loc[sample_data.index[2], 'High'] = sample_data.loc[sample_data.index[2], 'Low'] - 1  # High < Low
    sample_data.loc[sample_data.index[4], 'Volume'] = -1000  # Negative volume
    
    # Test validation
    result = validator.validate_dataframe(sample_data, "TEST.TO", "1d")
    
    print(f"📊 Validation Results:")
    print(f"   Valid: {'✅ Yes' if result.is_valid else '❌ No'}")
    print(f"   Quality Score: {result.quality_score:.3f}")
    print(f"   Issues: {len(result.issues)}")
    print(f"   Warnings: {len(result.warnings)}")
    
    if result.issues:
        print("   Issues found:")
        for issue in result.issues:
            print(f"     - {issue}")
    
    if result.warnings:
        print("   Warnings:")
        for warning in result.warnings:
            print(f"     - {warning}")
    
    return result

def test_historical_appender():
    """Test historical data collection (limited test)"""
    print("\n" + "="*60)
    print("🧪 TESTING HISTORICAL APPENDER")
    print("="*60)
    
    appender = HistoricalAppender(max_workers=2)
    
    # Test with just one symbol to avoid rate limiting
    test_symbol = "RY.TO"
    print(f"📥 Testing data collection for {test_symbol}")
    
    try:
        # Test daily data collection
        result = appender.fetch_daily_data(test_symbol)
        print(f"   Daily data: {'✅ Success' if result else '❌ Failed'}")
        
        if result:
            # Test aggregation
            agg_result = appender.aggregate_higher_timeframes(test_symbol)
            print(f"   Aggregation: {'✅ Success' if agg_result else '❌ Failed'}")
        
        # Get collection summary
        summary = appender.get_collection_summary()
        print(f"📊 Collection Summary:")
        if 'storage' in summary:
            print(f"   Files created: {summary['storage']['total_files']}")
            print(f"   Data size: {summary['storage']['total_size_mb']:.2f} MB")
        
        return result
        
    except Exception as e:
        print(f"❌ Collection test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test of all components"""
    print("🚀 STARTING COMPREHENSIVE DATA COLLECTION SYSTEM TEST")
    print("="*80)
    
    results = {}
    
    try:
        # Test each component
        results['symbol_manager'] = test_symbol_manager()
        results['progress_tracker'] = test_progress_tracker()
        results['storage_manager'] = test_storage_manager()
        results['data_validator'] = test_data_validator()
        results['historical_appender'] = test_historical_appender()
        
        # Overall results
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        success_count = 0
        total_tests = 0
        
        for component, result in results.items():
            if component == 'symbol_manager':
                # Check if any symbols were verified
                success = any(result.values()) if isinstance(result, dict) else bool(result)
            elif component == 'progress_tracker':
                # Check if progress was logged
                success = isinstance(result, dict) and 'overall_progress' in result
            elif component == 'data_validator':
                # Check if validation ran
                success = hasattr(result, 'quality_score')
            else:
                success = bool(result)
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {component.replace('_', ' ').title()}: {status}")
            
            if success:
                success_count += 1
            total_tests += 1
        
        # Final verdict
        print(f"\n🎉 FINAL RESULT: {success_count}/{total_tests} components working")
        
        if success_count == total_tests:
            print("✅ ALL SYSTEMS OPERATIONAL - Ready for full data collection!")
        elif success_count >= total_tests * 0.8:
            print("⚠️ MOSTLY OPERATIONAL - Minor issues detected")
        else:
            print("❌ SYSTEM ISSUES - Review failed components")
        
        return success_count / total_tests
        
    except Exception as e:
        print(f"❌ COMPREHENSIVE TEST FAILED: {e}")
        return 0.0

if __name__ == "__main__":
    # Run the comprehensive test
    success_rate = run_comprehensive_test()
    
    # Exit with appropriate code
    if success_rate >= 0.8:
        print(f"\n🎯 System ready for production use (success rate: {success_rate:.1%})")
        sys.exit(0)
    else:
        print(f"\n⚠️ System needs attention (success rate: {success_rate:.1%})")
        sys.exit(1)