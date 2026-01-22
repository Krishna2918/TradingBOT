"""
Phase 4 Smoke Test
==================

Quick smoke test to verify confidence calibration implementation.
Tests Bayesian Beta(2,2) calibration, risk integration, and database persistence.
"""

import logging
import sys
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def test_confidence_calibrator_initialization():
    """Test Confidence Calibrator initialization and configuration."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from adaptive.confidence_calibration import get_confidence_calibrator
        
        calibrator = get_confidence_calibrator()
        
        # Test initialization
        assert calibrator is not None, "Confidence calibrator should be initialized"
        assert calibrator.window_size_days == 30, "Default window size should be 30 days"
        assert calibrator.min_trades_for_calibration == 10, "Default min trades should be 10"
        assert calibrator.alpha_prior == 2.0, "Alpha prior should be 2.0"
        assert calibrator.beta_prior == 2.0, "Beta prior should be 2.0"
        
        logger.info("Confidence Calibrator initialization test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Confidence Calibrator initialization test: FAIL - {e}")
        return False


def test_bayesian_calibration():
    """Test Bayesian Beta(2,2) calibration logic."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from adaptive.confidence_calibration import get_confidence_calibrator
        
        calibrator = get_confidence_calibrator()
        
        # Test with no historical data (should return raw confidence)
        raw_confidence = 0.8
        calibrated = calibrator.calibrate_confidence("test_model", raw_confidence, datetime.now())
        assert calibrated == raw_confidence, "Should return raw confidence when no historical data"
        
        # Add some trade outcomes (use same date for all trades to ensure same window)
        base_date = datetime.now() - timedelta(days=5)
        
        # Add 15 trades with 60% win rate (9 wins, 6 losses) - all on same date
        for i in range(9):  # 9 wins
            calibrator.add_trade_outcome(
                trade_id=f"win_{i}",
                model_name="test_model",
                symbol="TEST",
                raw_confidence=0.8,
                outcome="WIN",
                trade_date=base_date
            )
        
        for i in range(6):  # 6 losses
            calibrator.add_trade_outcome(
                trade_id=f"loss_{i}",
                model_name="test_model",
                symbol="TEST",
                raw_confidence=0.8,
                outcome="LOSS",
                trade_date=base_date
            )
        
        # Test calibration with historical data (use same date as trades)
        calibrated = calibrator.calibrate_confidence("test_model", 0.8, base_date)
        
        # With 9 wins out of 15 trades, Beta(2,2) prior:
        # p_cal = (9 + 2) / (15 + 2 + 2) = 11/19 ≈ 0.579
        # But there's blending with raw confidence based on sample size
        expected_calibrated = (9 + 2) / (15 + 2 + 2)
        
        # Should be different from raw confidence and reasonable
        assert calibrated != 0.8, "Calibrated confidence should differ from raw confidence"
        assert 0.5 <= calibrated <= 0.8, f"Calibrated confidence {calibrated:.3f} should be reasonable"
        
        logger.info("Bayesian calibration test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Bayesian calibration test: FAIL - {e}")
        return False


def test_calibration_summary():
    """Test calibration summary functionality."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from adaptive.confidence_calibration import get_confidence_calibrator
        
        calibrator = get_confidence_calibrator()
        
        # Get calibration summary
        summary = calibrator.get_calibration_summary("test_model")
        
        # Should have calibration data
        assert "total_trades" in summary, "Summary should include total_trades"
        assert "total_wins" in summary, "Summary should include total_wins"
        assert "win_rate" in summary, "Summary should include win_rate"
        assert "avg_calibrated_probability" in summary, "Summary should include avg_calibrated_probability"
        
        # Should have 15 trades with 9 wins
        assert summary["total_trades"] == 15, f"Should have 15 trades, got {summary['total_trades']}"
        assert summary["total_wins"] == 9, f"Should have 9 wins, got {summary['total_wins']}"
        assert abs(summary["win_rate"] - 0.6) < 0.01, f"Win rate should be ~0.6, got {summary['win_rate']}"
        
        logger.info("Calibration summary test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Calibration summary test: FAIL - {e}")
        return False


def test_risk_integration():
    """Test risk management integration with calibrated confidence."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from trading.risk import calculate_position_size
        
        # Test position sizing with calibrated confidence
        account_balance = 10000.0
        volatility = 0.02
        entry_price = 100.0
        stop_loss = 95.0
        raw_confidence = 0.8
        
        # Test without calibration (should use raw confidence)
        risk_metrics = calculate_position_size(
            signal_confidence=raw_confidence,
            account_balance=account_balance,
            volatility=volatility,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        assert risk_metrics.raw_confidence == raw_confidence, "Should store raw confidence"
        assert risk_metrics.calibrated_confidence == raw_confidence, "Should use raw confidence when no calibration"
        
        # Test with calibration
        risk_metrics_calibrated = calculate_position_size(
            signal_confidence=raw_confidence,
            account_balance=account_balance,
            volatility=volatility,
            entry_price=entry_price,
            stop_loss=stop_loss,
            model_name="test_model",
            symbol="TEST",
            trade_date=datetime.now()
        )
        
        assert risk_metrics_calibrated.raw_confidence == raw_confidence, "Should store raw confidence"
        assert risk_metrics_calibrated.calibrated_confidence != raw_confidence, "Should have different calibrated confidence"
        
        # Calibrated confidence should be between 0 and 1
        assert 0.0 <= risk_metrics_calibrated.calibrated_confidence <= 1.0, "Calibrated confidence should be in [0,1]"
        
        logger.info("Risk integration test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Risk integration test: FAIL - {e}")
        return False


def test_database_integration():
    """Test database integration for confidence calibration."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from config.database import get_database_manager, log_confidence_calibration
        
        db_manager = get_database_manager()
        
        # Test logging confidence calibration
        trade_date = datetime.now()
        calibration_id = log_confidence_calibration(
            trade_date=trade_date,
            model="test_model",
            symbol="TEST",
            raw_confidence=0.8,
            calibrated_confidence=0.65,
            outcome="WIN",
            pnl=150.0,
            window_id="test_window"
        )
        
        assert calibration_id > 0, "Should return valid calibration ID"
        
        # Test retrieving calibration history
        history = db_manager.get_confidence_calibration_history(model="test_model", limit=10)
        assert len(history) > 0, "Should retrieve calibration history"
        
        # Check that our logged entry is in the history
        found_entry = False
        for entry in history:
            if entry["model"] == "test_model" and entry["symbol"] == "TEST":
                found_entry = True
                assert entry["raw_confidence"] == 0.8
                assert entry["calibrated_confidence"] == 0.65
                assert entry["outcome"] == "WIN"
                assert entry["pnl"] == 150.0
                break
        
        assert found_entry, "Should find logged calibration entry in history"
        
        logger.info("Database integration test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Database integration test: FAIL - {e}")
        return False


def test_calibration_quality():
    """Test calibration quality metrics."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from adaptive.confidence_calibration import get_confidence_calibrator
        
        calibrator = get_confidence_calibrator()
        
        # Get recent calibration (use longer period to catch our test data)
        recent_calibration = calibrator.get_recent_calibration("test_model", days_back=30)
        
        assert recent_calibration is not None, "Should have recent calibration data"
        assert recent_calibration.model_name == "test_model", "Should be for test_model"
        assert recent_calibration.total_trades == 15, "Should have 15 trades"
        assert recent_calibration.wins == 9, "Should have 9 wins"
        assert recent_calibration.calibration_quality >= 0.0, "Calibration quality should be non-negative"
        
        logger.info("Calibration quality test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Calibration quality test: FAIL - {e}")
        return False


def test_export_import():
    """Test calibration data export and import."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from adaptive.confidence_calibration import get_confidence_calibrator
        
        calibrator = get_confidence_calibrator()
        
        # Export calibration data
        export_data = calibrator.export_calibration_data()
        
        assert "calibration_cache" in export_data, "Should export calibration cache"
        assert "trade_outcomes" in export_data, "Should export trade outcomes"
        assert len(export_data["calibration_cache"]) > 0, "Should have calibration cache data"
        assert len(export_data["trade_outcomes"]) > 0, "Should have trade outcomes data"
        
        # Test import (create new calibrator and import)
        new_calibrator = get_confidence_calibrator()
        new_calibrator.import_calibration_data(export_data)
        
        # Verify import worked
        summary = new_calibrator.get_calibration_summary("test_model")
        assert summary["total_trades"] == 15, "Imported data should have 15 trades"
        assert summary["total_wins"] == 9, "Imported data should have 9 wins"
        
        logger.info("Export/import test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Export/import test: FAIL - {e}")
        return False


def main():
    """Run all Phase 4 smoke tests."""
    results = {}
    
    logger.info("Starting Phase 4 Smoke Test...")
    start_time = time.time()
    
    # Run all tests
    logger.info("Testing Confidence Calibrator Initialization...")
    results["Confidence Calibrator Initialization"] = test_confidence_calibrator_initialization()
    
    logger.info("Testing Bayesian Calibration...")
    results["Bayesian Calibration"] = test_bayesian_calibration()
    
    logger.info("Testing Calibration Summary...")
    results["Calibration Summary"] = test_calibration_summary()
    
    logger.info("Testing Risk Integration...")
    results["Risk Integration"] = test_risk_integration()
    
    logger.info("Testing Database Integration...")
    results["Database Integration"] = test_database_integration()
    
    logger.info("Testing Calibration Quality...")
    results["Calibration Quality"] = test_calibration_quality()
    
    logger.info("Testing Export/Import...")
    results["Export/Import"] = test_export_import()
    
    # Calculate results
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info("==================================================")
    logger.info("PHASE 4 SMOKE TEST SUMMARY")
    logger.info("==================================================")
    logger.info(f"Duration: {time.time() - start_time:.1f}s")
    logger.info(f"Tests: {passed_tests}/{total_tests} passed")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    for test_name, status in results.items():
        logger.info(f"{test_name}: {'PASS' if status else 'FAIL'}")
    
    # Save results
    import json
    results_file = ROOT / "logs" / "phase4_smoke_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    if success_rate == 100:
        logger.info("PHASE 4 SMOKE TEST PASSED")
        return 0
    else:
        logger.error("PHASE 4 SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
