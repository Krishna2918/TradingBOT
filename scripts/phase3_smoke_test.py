"""
Phase 3 Smoke Test
==================

Quick smoke test to verify data quality validation, provenance tracking, and quality gates.
"""

import logging
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def test_data_quality_validator():
    """Test Data Quality Validator functionality."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from validation.data_quality import get_data_quality_validator
        
        validator = get_data_quality_validator()
        
        # Test initialization
        assert len(validator.validation_rules) > 0, "Should have validation rules"
        assert "open" in validator.validation_rules, "Should have rules for open"
        assert "rsi" in validator.validation_rules, "Should have rules for RSI"
        
        # Test clean data validation
        clean_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'rsi': [45.0, 50.0, 55.0, 60.0, 65.0],
            'adx': [25.0, 30.0, 35.0, 40.0, 45.0],
            'atr': [1.5, 1.6, 1.7, 1.8, 1.9]
        })
        
        report = validator.validate_dataframe(clean_data, "TEST")
        assert report.overall_score >= 0.8, f"Clean data should have high quality, got {report.overall_score}"
        assert len(report.violations) == 0, f"Clean data should have no violations, got {len(report.violations)}"
        
        logger.info("Data Quality Validator test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Data Quality Validator test: FAIL - {e}")
        return False


def test_data_quality_validation_dirty_data():
    """Test data quality validation with dirty data."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from validation.data_quality import get_data_quality_validator
        
        validator = get_data_quality_validator()
        
        # Create dirty test data
        dirty_data = pd.DataFrame({
            'open': [100.0, -50.0, 102.0, np.nan, 104.0],  # Negative value and NaN
            'high': [101.0, 102.0, 50.0, 104.0, 105.0],    # High < open
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000000, -500000, 1200000, 1300000, 1400000],  # Negative volume
            'rsi': [150.0, 50.0, 55.0, 60.0, 65.0],        # RSI > 100
            'adx': [25.0, 30.0, 35.0, 40.0, 45.0],
            'atr': [1.5, 1.6, 1.7, 1.8, 1.9]
        })
        
        report = validator.validate_dataframe(dirty_data, "TEST")
        assert report.overall_score < 0.7, f"Dirty data should have low quality, got {report.overall_score}"
        assert len(report.violations) > 0, f"Dirty data should have violations, got {len(report.violations)}"
        
        # Check for specific violations
        violation_types = [v.violation_type for v in report.violations]
        assert "NEGATIVE_VALUES" in violation_types, "Should detect negative values"
        assert "OUT_OF_RANGE" in violation_types, "Should detect out of range values"
        
        logger.info("Data quality validation dirty data test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Data quality validation dirty data test: FAIL - {e}")
        return False


def test_should_skip_sizing_logic():
    """Test should_skip_sizing logic."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from validation.data_quality import get_data_quality_validator, QualityReport, QualityLevel
        
        validator = get_data_quality_validator()
        
        # Test with high quality data
        high_quality_report = QualityReport(
            symbol="TEST",
            timestamp=datetime.now(),
            overall_score=0.95,
            quality_level=QualityLevel.EXCELLENT,
            violations=[],
            column_scores={},
            missing_data_summary={},
            statistical_summary={},
            recommendations=[]
        )
        
        should_skip_high = validator.should_skip_sizing(high_quality_report, threshold=0.7)
        assert not should_skip_high, "Should not skip sizing for high quality data"
        
        # Test with low quality data
        low_quality_report = QualityReport(
            symbol="TEST",
            timestamp=datetime.now(),
            overall_score=0.5,
            quality_level=QualityLevel.CRITICAL,
            violations=[],
            column_scores={},
            missing_data_summary={},
            statistical_summary={},
            recommendations=[]
        )
        
        should_skip_low = validator.should_skip_sizing(low_quality_report, threshold=0.7)
        assert should_skip_low, "Should skip sizing for low quality data"
        
        logger.info("Should skip sizing logic test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Should skip sizing logic test: FAIL - {e}")
        return False


def test_database_provenance_tracking():
    """Test database provenance tracking functionality."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from config.database import get_database_manager
        
        db_manager = get_database_manager()
        
        # Test logging data provenance
        provenance_id = db_manager.log_data_provenance(
            symbol="TEST",
            data_type="price_data",
            source="alpha_vantage",
            source_metadata={"test": "data", "rows": 100},
            quality_score=0.95,
            quality_level="excellent"
        )
        
        assert provenance_id > 0, "Should return valid provenance ID"
        
        # Test retrieving provenance history
        history = db_manager.get_data_provenance_history(symbol="TEST", limit=10)
        assert len(history) > 0, "Should retrieve provenance history"
        
        logger.info("Database provenance tracking test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Database provenance tracking test: FAIL - {e}")
        return False


def test_database_quality_violations_tracking():
    """Test database quality violations tracking functionality."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from config.database import get_database_manager
        
        db_manager = get_database_manager()
        
        # Test logging quality violation
        violation_id = db_manager.log_data_quality_violation(
            symbol="TEST",
            violation_type="NEGATIVE_VALUES",
            severity="HIGH",
            description="Found negative values in price data",
            column_name="open",
            violation_value=5,
            expected_range=(0.0, 10000.0)
        )
        
        assert violation_id > 0, "Should return valid violation ID"
        
        # Test retrieving quality violations
        violations = db_manager.get_data_quality_violations(symbol="TEST", limit=10)
        assert len(violations) > 0, "Should retrieve quality violations"
        
        logger.info("Database quality violations tracking test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Database quality violations tracking test: FAIL - {e}")
        return False


def test_quality_gate_integration():
    """Test quality gate integration with system monitor."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from validation.data_quality import get_data_quality_validator, should_skip_sizing
        from monitoring.system_monitor import get_system_monitor
        
        validator = get_data_quality_validator()
        system_monitor = get_system_monitor()
        
        # Create test data with quality issues
        test_data = pd.DataFrame({
            'open': [100.0, -50.0, 102.0, np.nan, 104.0],
            'high': [101.0, 102.0, 50.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000000, -500000, 1200000, 1300000, 1400000],
            'rsi': [150.0, 50.0, 55.0, 60.0, 65.0],
            'adx': [25.0, 30.0, 35.0, 40.0, 45.0],
            'atr': [1.5, 1.6, 1.7, 1.8, 1.9]
        })
        
        # Validate data quality
        quality_report = validator.validate_dataframe(test_data, "TEST")
        
        # Test quality gate logic
        should_skip = should_skip_sizing(quality_report, threshold=0.7)
        assert should_skip, "Should skip sizing for poor quality data"
        
        # Test system monitor integration
        system_monitor.log_phase_event("buy_phase", "quality_gate_failed", "DEMO", {
            "symbol": "TEST",
            "quality_score": quality_report.overall_score,
            "quality_level": quality_report.quality_level.value,
            "violations": len(quality_report.violations)
        })
        
        logger.info("Quality gate integration test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Quality gate integration test: FAIL - {e}")
        return False


def test_bollinger_bands_validation():
    """Test Bollinger Bands relationship validation."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from validation.data_quality import get_data_quality_validator
        
        validator = get_data_quality_validator()
        
        # Create data with Bollinger Band violations
        bb_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'bb_upper': [102.0, 103.0, 104.0, 105.0, 106.0],
            'bb_middle': [100.0, 101.0, 102.0, 103.0, 104.0],
            'bb_lower': [98.0, 99.0, 100.0, 101.0, 102.0],  # This is actually correct: lower < middle
            'rsi': [45.0, 50.0, 55.0, 60.0, 65.0],
            'adx': [25.0, 30.0, 35.0, 40.0, 45.0],
            'atr': [1.5, 1.6, 1.7, 1.8, 1.9]
        })
        
        # Create actual violation: make bb_lower > bb_middle
        bb_data.loc[2, 'bb_lower'] = 103.0  # This should be < bb_middle (102.0) but we set it to 103.0
        
        # Validate Bollinger Bands data
        report = validator.validate_dataframe(bb_data, "TEST")
        
        # Should detect Bollinger Band violations
        bb_violations = [v for v in report.violations if v.violation_type == "BOLLINGER_VIOLATION"]
        assert len(bb_violations) > 0, "Should detect Bollinger Band violations"
        
        logger.info("Bollinger Bands validation test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Bollinger Bands validation test: FAIL - {e}")
        return False


def main():
    """Run all Phase 3 smoke tests."""
    results = {}
    
    logger.info("Starting Phase 3 Smoke Test...")
    start_time = time.time()
    
    # Run all tests
    logger.info("Testing Data Quality Validator...")
    results["Data Quality Validator"] = test_data_quality_validator()
    
    logger.info("Testing Data Quality Validation Dirty Data...")
    results["Data Quality Validation Dirty Data"] = test_data_quality_validation_dirty_data()
    
    logger.info("Testing Should Skip Sizing Logic...")
    results["Should Skip Sizing Logic"] = test_should_skip_sizing_logic()
    
    logger.info("Testing Database Provenance Tracking...")
    results["Database Provenance Tracking"] = test_database_provenance_tracking()
    
    logger.info("Testing Database Quality Violations Tracking...")
    results["Database Quality Violations Tracking"] = test_database_quality_violations_tracking()
    
    logger.info("Testing Quality Gate Integration...")
    results["Quality Gate Integration"] = test_quality_gate_integration()
    
    logger.info("Testing Bollinger Bands Validation...")
    results["Bollinger Bands Validation"] = test_bollinger_bands_validation()
    
    # Calculate results
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info("==================================================")
    logger.info("PHASE 3 SMOKE TEST SUMMARY")
    logger.info("==================================================")
    logger.info(f"Duration: {time.time() - start_time:.1f}s")
    logger.info(f"Tests: {passed_tests}/{total_tests} passed")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    for test_name, status in results.items():
        logger.info(f"{test_name}: {'PASS' if status else 'FAIL'}")
    
    # Save results
    import json
    results_file = ROOT / "logs" / "phase3_smoke_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    if success_rate == 100:
        logger.info("PHASE 3 SMOKE TEST PASSED")
        return 0
    else:
        logger.error("PHASE 3 SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
