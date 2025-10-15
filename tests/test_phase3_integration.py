"""
Phase 3 Integration Tests
=========================

Integration tests for Phase 3: Data Contracts & Quality Gates
Tests data quality validation, provenance tracking, and quality gate enforcement.
"""

import logging
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


class TestPhase3Integration:
    """Test Phase 3 data quality validation and quality gates integration."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Add src to path
        import sys
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
    
    def test_data_quality_validator_initialization(self):
        """Test Data Quality Validator initialization and configuration."""
        try:
            from src.validation.data_quality import get_data_quality_validator
            
            validator = get_data_quality_validator()
            
            # Test that validation rules are initialized
            assert len(validator.validation_rules) > 0, "Should have validation rules"
            
            # Test specific validation rules
            expected_columns = ["open", "high", "low", "close", "volume", "rsi", "adx", "atr"]
            for column in expected_columns:
                assert column in validator.validation_rules, f"Should have rules for {column}"
            
            # Test quality thresholds
            assert validator.quality_thresholds["excellent"] == 0.90
            assert validator.quality_thresholds["good"] == 0.80
            assert validator.quality_thresholds["fair"] == 0.70
            assert validator.quality_thresholds["poor"] == 0.60
            
            # Test missing data thresholds
            assert "price_data" in validator.missing_thresholds
            assert "volume_data" in validator.missing_thresholds
            assert "technical_indicators" in validator.missing_thresholds
            
            logger.info("Data Quality Validator initialization test passed")
            
        except Exception as e:
            logger.error(f"Data Quality Validator initialization test failed: {e}")
            raise
    
    def test_data_quality_validation_clean_data(self):
        """Test data quality validation with clean data."""
        try:
            from src.validation.data_quality import get_data_quality_validator
            
            validator = get_data_quality_validator()
            
            # Create clean test data
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
            
            # Validate clean data
            report = validator.validate_dataframe(clean_data, "TEST")
            
            # Should have high quality score
            assert report.overall_score >= 0.8, f"Clean data should have high quality score, got {report.overall_score}"
            assert report.quality_level.value in ["excellent", "good"], f"Clean data should be excellent or good, got {report.quality_level.value}"
            assert len(report.violations) == 0, f"Clean data should have no violations, got {len(report.violations)}"
            
            logger.info("Data quality validation clean data test passed")
            
        except Exception as e:
            logger.error(f"Data quality validation clean data test failed: {e}")
            raise
    
    def test_data_quality_validation_dirty_data(self):
        """Test data quality validation with dirty data."""
        try:
            from src.validation.data_quality import get_data_quality_validator
            
            validator = get_data_quality_validator()
            
            # Create dirty test data with various issues
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
            
            # Validate dirty data
            report = validator.validate_dataframe(dirty_data, "TEST")
            
            # Should have low quality score
            assert report.overall_score < 0.7, f"Dirty data should have low quality score, got {report.overall_score}"
            assert report.quality_level.value in ["poor", "critical"], f"Dirty data should be poor or critical, got {report.quality_level.value}"
            assert len(report.violations) > 0, f"Dirty data should have violations, got {len(report.violations)}"
            
            # Check for specific violations
            violation_types = [v.violation_type for v in report.violations]
            assert "NEGATIVE_VALUES" in violation_types, "Should detect negative values"
            assert "OUT_OF_RANGE" in violation_types, "Should detect out of range values"
            assert "OHLC_VIOLATION" in violation_types, "Should detect OHLC violations"
            
            logger.info("Data quality validation dirty data test passed")
            
        except Exception as e:
            logger.error(f"Data quality validation dirty data test failed: {e}")
            raise
    
    def test_data_quality_validation_missing_data(self):
        """Test data quality validation with missing data."""
        try:
            from src.validation.data_quality import get_data_quality_validator
            
            validator = get_data_quality_validator()
            
            # Create data with excessive missing values
            missing_data = pd.DataFrame({
                'open': [100.0, np.nan, np.nan, np.nan, np.nan],  # 80% missing
                'high': [101.0, 102.0, 103.0, 104.0, 105.0],
                'low': [99.0, 100.0, 101.0, 102.0, 103.0],
                'close': [100.5, 101.5, 102.5, 103.5, 104.5],
                'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
                'rsi': [45.0, 50.0, 55.0, 60.0, 65.0],
                'adx': [25.0, 30.0, 35.0, 40.0, 45.0],
                'atr': [1.5, 1.6, 1.7, 1.8, 1.9]
            })
            
            # Validate missing data
            report = validator.validate_dataframe(missing_data, "TEST")
            
            # Should detect excessive missing data
            missing_violations = [v for v in report.violations if v.violation_type == "EXCESSIVE_MISSING_DATA"]
            assert len(missing_violations) > 0, "Should detect excessive missing data"
            
            # Check missing data summary
            assert "open" in report.missing_data_summary, "Should track missing data by column"
            assert report.missing_data_summary["open"] == 4, "Should count 4 missing values for open"
            
            logger.info("Data quality validation missing data test passed")
            
        except Exception as e:
            logger.error(f"Data quality validation missing data test failed: {e}")
            raise
    
    def test_bollinger_bands_validation(self):
        """Test Bollinger Bands relationship validation."""
        try:
            from src.validation.data_quality import get_data_quality_validator
            
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
                'bb_lower': [98.0, 99.0, 100.0, 101.0, 102.0],  # Violation: lower > middle
                'rsi': [45.0, 50.0, 55.0, 60.0, 65.0],
                'adx': [25.0, 30.0, 35.0, 40.0, 45.0],
                'atr': [1.5, 1.6, 1.7, 1.8, 1.9]
            })
            
            # Validate Bollinger Bands data
            report = validator.validate_dataframe(bb_data, "TEST")
            
            # Should detect Bollinger Band violations
            bb_violations = [v for v in report.violations if v.violation_type == "BOLLINGER_VIOLATION"]
            assert len(bb_violations) > 0, "Should detect Bollinger Band violations"
            
            logger.info("Bollinger Bands validation test passed")
            
        except Exception as e:
            logger.error(f"Bollinger Bands validation test failed: {e}")
            raise
    
    def test_should_skip_sizing_logic(self):
        """Test should_skip_sizing logic."""
        try:
            from src.validation.data_quality import get_data_quality_validator, QualityReport, QualityLevel
            
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
            
            logger.info("Should skip sizing logic test passed")
            
        except Exception as e:
            logger.error(f"Should skip sizing logic test failed: {e}")
            raise
    
    def test_database_provenance_tracking(self):
        """Test database provenance tracking functionality."""
        try:
            from src.config.database import get_database_manager
            
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
            
            # Check that our logged entry is in the history
            found_entry = False
            for entry in history:
                if entry["symbol"] == "TEST" and entry["data_type"] == "price_data":
                    found_entry = True
                    assert entry["source"] == "alpha_vantage"
                    assert entry["quality_score"] == 0.95
                    assert entry["quality_level"] == "excellent"
                    break
            
            assert found_entry, "Should find logged provenance entry in history"
            
            logger.info("Database provenance tracking test passed")
            
        except Exception as e:
            logger.error(f"Database provenance tracking test failed: {e}")
            raise
    
    def test_database_quality_violations_tracking(self):
        """Test database quality violations tracking functionality."""
        try:
            from src.config.database import get_database_manager
            
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
            
            # Check that our logged violation is in the results
            found_violation = False
            for violation in violations:
                if violation["symbol"] == "TEST" and violation["violation_type"] == "NEGATIVE_VALUES":
                    found_violation = True
                    assert violation["severity"] == "HIGH"
                    assert violation["column_name"] == "open"
                    assert violation["violation_value"] == "5"
                    break
            
            assert found_violation, "Should find logged violation in results"
            
            logger.info("Database quality violations tracking test passed")
            
        except Exception as e:
            logger.error(f"Database quality violations tracking test failed: {e}")
            raise
    
    def test_quality_gate_integration(self):
        """Test quality gate integration with trading cycle."""
        try:
            from src.validation.data_quality import get_data_quality_validator, should_skip_sizing
            from src.monitoring.system_monitor import get_system_monitor
            
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
            
            logger.info("Quality gate integration test passed")
            
        except Exception as e:
            logger.error(f"Quality gate integration test failed: {e}")
            raise
    
    def test_property_tests_on_real_data(self):
        """Test property tests on real data constraints."""
        try:
            from src.validation.data_quality import get_data_quality_validator
            
            validator = get_data_quality_validator()
            
            # Create realistic market data
            real_data = pd.DataFrame({
                'open': [100.0, 101.0, 102.0, 103.0, 104.0],
                'high': [101.0, 102.0, 103.0, 104.0, 105.0],
                'low': [99.0, 100.0, 101.0, 102.0, 103.0],
                'close': [100.5, 101.5, 102.5, 103.5, 104.5],
                'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
                'rsi': [45.0, 50.0, 55.0, 60.0, 65.0],
                'adx': [25.0, 30.0, 35.0, 40.0, 45.0],
                'atr': [1.5, 1.6, 1.7, 1.8, 1.9],
                'bb_upper': [102.0, 103.0, 104.0, 105.0, 106.0],
                'bb_middle': [100.0, 101.0, 102.0, 103.0, 104.0],
                'bb_lower': [98.0, 99.0, 100.0, 101.0, 102.0]
            })
            
            # Validate real data
            report = validator.validate_dataframe(real_data, "AAPL")
            
            # Property tests - these constraints should always hold
            assert report.overall_score >= 0.0, "Quality score should be non-negative"
            assert report.overall_score <= 1.0, "Quality score should be <= 1.0"
            assert report.quality_level.value in ["excellent", "good", "fair", "poor", "critical"], "Quality level should be valid"
            assert len(report.violations) >= 0, "Violations count should be non-negative"
            assert len(report.column_scores) >= 0, "Column scores should be non-negative"
            assert len(report.missing_data_summary) >= 0, "Missing data summary should be non-negative"
            
            # Test that all column scores are in valid range
            for column, score in report.column_scores.items():
                assert 0.0 <= score <= 1.0, f"Column score for {column} should be in [0,1], got {score}"
            
            # Test that missing data counts are non-negative
            for column, count in report.missing_data_summary.items():
                assert count >= 0, f"Missing data count for {column} should be non-negative, got {count}"
            
            logger.info("Property tests on real data test passed")
            
        except Exception as e:
            logger.error(f"Property tests on real data test failed: {e}")
            raise


def test_phase3_smoke_test():
    """Run Phase 3 smoke test to verify all components work together."""
    try:
        logger.info("Starting Phase 3 smoke test...")
        
        # Test all integration points
        test_suite = TestPhase3Integration()
        test_suite.setup_method()
        
        # Run all tests
        test_suite.test_data_quality_validator_initialization()
        test_suite.test_data_quality_validation_clean_data()
        test_suite.test_data_quality_validation_dirty_data()
        test_suite.test_data_quality_validation_missing_data()
        test_suite.test_bollinger_bands_validation()
        test_suite.test_should_skip_sizing_logic()
        test_suite.test_database_provenance_tracking()
        test_suite.test_database_quality_violations_tracking()
        test_suite.test_quality_gate_integration()
        test_suite.test_property_tests_on_real_data()
        
        logger.info("Phase 3 smoke test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Phase 3 smoke test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_phase3_smoke_test()
    if success:
        print("✅ Phase 3 smoke test passed")
        exit(0)
    else:
        print("❌ Phase 3 smoke test failed")
        exit(1)
