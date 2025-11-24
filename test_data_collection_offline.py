#!/usr/bin/env python3
"""
Offline Data Collection System Test

Tests the data collection infrastructure without requiring network access.
Uses mock data to verify all components work correctly.
"""

import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collection import (
    SymbolManager, 
    ProgressTracker, 
    StorageManager, 
    DataValidator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_mock_financial_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Create realistic mock financial data"""
    
    # Generate dates (business days only)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days*1.5)  # Extra days to account for weekends
    dates = pd.bdate_range(start=start_date, end=end_date)[:days]
    
    # Generate realistic price data with random walk
    np.random.seed(42)  # For reproducible results
    
    base_price = 100.0
    volatility = 0.02  # 2% daily volatility
    
    # Generate returns
    returns = np.random.normal(0.0005, volatility, len(dates))  # Slight upward drift
    
    # Calculate prices
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC data
    ohlc_data = []
    
    for i, price in enumerate(prices):
        # Generate realistic OHLC
        daily_range = abs(np.random.normal(0, volatility * 0.5))  # Intraday range
        
        high = price * (1 + daily_range)
        low = price * (1 - daily_range)
        
        # Ensure Close is within High/Low
        close = np.random.uniform(low, high)
        
        # Open is usually close to previous close
        if i > 0:
            prev_close = ohlc_data[i-1]['Close']
            open_price = prev_close * (1 + np.random.normal(0, volatility * 0.3))
            open_price = np.clip(open_price, low, high)  # Ensure within range
        else:
            open_price = price
        
        ohlc_data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': int(np.random.uniform(1000000, 5000000))
        })
    
    # Create DataFrame
    data = pd.DataFrame(ohlc_data, index=dates)
    
    return data

def test_symbol_manager_offline():
    """Test symbol manager without network calls"""
    print("\n" + "="*60)
    print("🧪 TESTING SYMBOL MANAGER (OFFLINE)")
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
    
    # Test symbol list structure
    expected_count = 100  # We expect around 100 symbols
    success = 90 <= len(all_symbols) <= 110  # Allow some variance
    
    print(f"\n✅ Symbol list structure: {'PASS' if success else 'FAIL'}")
    return success

def test_progress_tracker_offline():
    """Test progress tracking functionality"""
    print("\n" + "="*60)
    print("🧪 TESTING PROGRESS TRACKER (OFFLINE)")
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
    
    # Test database functionality
    incomplete = tracker.get_incomplete_symbols(95.0)
    print(f"   Incomplete symbols: {len(incomplete)}")
    
    success = (
        'overall_progress' in summary and
        summary['overall_progress']['completion_percentage'] > 0 and
        len(summary['future_steps']) > 0
    )
    
    print(f"\n✅ Progress tracking: {'PASS' if success else 'FAIL'}")
    return success

def test_storage_manager_offline():
    """Test storage manager with mock data"""
    print("\n" + "="*60)
    print("🧪 TESTING STORAGE MANAGER (OFFLINE)")
    print("="*60)
    
    storage = StorageManager()
    
    # Create mock data
    mock_data = create_mock_financial_data("TEST.TO", days=50)
    print(f"📊 Created mock data: {len(mock_data)} rows")
    
    # Test save and load
    print("💾 Testing save/load operations...")
    
    save_success = storage.save_to_parquet(mock_data, "TEST.TO", "1d")
    print(f"   Save test: {'✅ Success' if save_success else '❌ Failed'}")
    
    loaded_data = storage.load_from_parquet("TEST.TO", "1d")
    load_success = not loaded_data.empty
    print(f"   Load test: {'✅ Success' if load_success else '❌ Failed'}")
    
    if load_success:
        print(f"   Loaded {len(loaded_data)} rows")
    
    # Test append functionality
    new_mock_data = create_mock_financial_data("TEST.TO", days=10)
    append_success = storage.append_to_parquet(new_mock_data, "TEST.TO", "1d")
    print(f"   Append test: {'✅ Success' if append_success else '❌ Failed'}")
    
    # Test quality validation
    quality_metrics = storage.validate_data_quality("TEST.TO", "1d")
    print(f"   Quality score: {quality_metrics['quality_score']:.3f}")
    
    # Test storage summary
    summary = storage.get_storage_summary()
    print(f"📊 Storage Summary:")
    print(f"   Total files: {summary['total_files']}")
    print(f"   Total size: {summary['total_size_mb']:.2f} MB")
    print(f"   Symbols: {len(summary['symbols'])}")
    
    success = save_success and load_success and append_success
    print(f"\n✅ Storage operations: {'PASS' if success else 'FAIL'}")
    return success

def test_data_validator_offline():
    """Test data validator with mock data"""
    print("\n" + "="*60)
    print("🧪 TESTING DATA VALIDATOR (OFFLINE)")
    print("="*60)
    
    validator = DataValidator()
    
    # Create good mock data
    good_data = create_mock_financial_data("GOOD.TO", days=30)
    
    # Create problematic mock data
    bad_data = create_mock_financial_data("BAD.TO", days=30)
    
    # Introduce issues
    bad_data.loc[bad_data.index[5], 'High'] = bad_data.loc[bad_data.index[5], 'Low'] - 1  # High < Low
    bad_data.loc[bad_data.index[10], 'Volume'] = -1000  # Negative volume
    bad_data.loc[bad_data.index[15], 'Close'] = 0  # Zero price
    
    # Test validation on good data
    good_result = validator.validate_dataframe(good_data, "GOOD.TO", "1d")
    print(f"📊 Good Data Validation:")
    print(f"   Valid: {'✅ Yes' if good_result.is_valid else '❌ No'}")
    print(f"   Quality Score: {good_result.quality_score:.3f}")
    print(f"   Issues: {len(good_result.issues)}")
    
    # Test validation on bad data
    bad_result = validator.validate_dataframe(bad_data, "BAD.TO", "1d")
    print(f"\n📊 Bad Data Validation:")
    print(f"   Valid: {'✅ Yes' if bad_result.is_valid else '❌ No'}")
    print(f"   Quality Score: {bad_result.quality_score:.3f}")
    print(f"   Issues: {len(bad_result.issues)}")
    
    if bad_result.issues:
        print("   Issues detected:")
        for issue in bad_result.issues[:3]:  # Show first 3 issues
            print(f"     - {issue}")
    
    # Success if good data passes and bad data fails appropriately
    success = (
        good_result.is_valid and 
        good_result.quality_score > 0.8 and
        not bad_result.is_valid and
        len(bad_result.issues) > 0
    )
    
    print(f"\n✅ Data validation: {'PASS' if success else 'FAIL'}")
    return success

def test_integration_offline():
    """Test integration between components"""
    print("\n" + "="*60)
    print("🧪 TESTING COMPONENT INTEGRATION (OFFLINE)")
    print("="*60)
    
    # Initialize all components
    symbol_manager = SymbolManager()
    progress_tracker = ProgressTracker()
    storage_manager = StorageManager()
    validator = DataValidator()
    
    # Get a few symbols
    symbols = symbol_manager.get_all_symbols()[:3]
    print(f"📊 Testing integration with symbols: {symbols}")
    
    success_count = 0
    
    for symbol in symbols:
        try:
            # Create mock data
            mock_data = create_mock_financial_data(symbol, days=20)
            
            # Store data
            storage_success = storage_manager.save_to_parquet(mock_data, symbol, "1d")
            
            if storage_success:
                # Validate data
                validation_result = validator.validate_dataframe(mock_data, symbol, "1d")
                
                # Log progress
                file_path = storage_manager.get_file_path(symbol, "1d")
                data_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0.0
                
                progress_tracker.log_progress(
                    symbol, "1d", len(mock_data), data_size_mb, 
                    0 if validation_result.is_valid else 1, 
                    validation_result.quality_score
                )
                
                success_count += 1
                print(f"   ✅ {symbol}: Stored, validated, progress logged")
            else:
                print(f"   ❌ {symbol}: Storage failed")
                
        except Exception as e:
            print(f"   ❌ {symbol}: Integration error - {e}")
    
    # Get final summary
    summary = progress_tracker.get_progress_summary()
    storage_summary = storage_manager.get_storage_summary()
    
    print(f"\n📊 Integration Results:")
    print(f"   Successful integrations: {success_count}/{len(symbols)}")
    print(f"   Total files created: {storage_summary['total_files']}")
    print(f"   Overall completion: {summary['overall_progress']['completion_percentage']:.1f}%")
    
    success = success_count >= len(symbols) * 0.8  # 80% success rate
    print(f"\n✅ Component integration: {'PASS' if success else 'FAIL'}")
    return success

def run_offline_comprehensive_test():
    """Run comprehensive offline test of all components"""
    print("🚀 STARTING OFFLINE DATA COLLECTION SYSTEM TEST")
    print("="*80)
    print("Note: This test uses mock data to verify system functionality")
    print("="*80)
    
    results = {}
    
    try:
        # Test each component
        results['symbol_manager'] = test_symbol_manager_offline()
        results['progress_tracker'] = test_progress_tracker_offline()
        results['storage_manager'] = test_storage_manager_offline()
        results['data_validator'] = test_data_validator_offline()
        results['integration'] = test_integration_offline()
        
        # Overall results
        print("\n" + "="*80)
        print("🎯 OFFLINE TEST RESULTS")
        print("="*80)
        
        success_count = 0
        total_tests = len(results)
        
        for component, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {component.replace('_', ' ').title()}: {status}")
            
            if success:
                success_count += 1
        
        # Final verdict
        success_rate = success_count / total_tests
        print(f"\n🎉 FINAL RESULT: {success_count}/{total_tests} components working ({success_rate:.1%})")
        
        if success_rate == 1.0:
            print("✅ ALL SYSTEMS OPERATIONAL - Infrastructure ready for data collection!")
            print("\n📋 Next Steps:")
            print("   1. Verify internet connectivity for yfinance")
            print("   2. Test with real market data")
            print("   3. Start historical data collection")
        elif success_rate >= 0.8:
            print("⚠️ MOSTLY OPERATIONAL - Minor issues detected")
            print("\n📋 Recommendations:")
            print("   1. Review failed components")
            print("   2. Fix any infrastructure issues")
            print("   3. Proceed with caution")
        else:
            print("❌ SYSTEM ISSUES - Review failed components")
            print("\n📋 Required Actions:")
            print("   1. Fix critical infrastructure issues")
            print("   2. Re-run tests until all pass")
            print("   3. Do not proceed to data collection")
        
        return success_rate
        
    except Exception as e:
        print(f"❌ COMPREHENSIVE TEST FAILED: {e}")
        return 0.0

if __name__ == "__main__":
    # Run the offline comprehensive test
    success_rate = run_offline_comprehensive_test()
    
    # Exit with appropriate code
    if success_rate >= 0.8:
        print(f"\n🎯 Infrastructure ready (success rate: {success_rate:.1%})")
        sys.exit(0)
    else:
        print(f"\n⚠️ Infrastructure needs fixes (success rate: {success_rate:.1%})")
        sys.exit(1)