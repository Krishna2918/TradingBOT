#!/usr/bin/env python3
"""
Phase 4 Smoke Test - Fixed Version
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def populate_calibrator_cache():
    """Populate calibrator cache with test data."""
    try:
        from adaptive.confidence_calibration import get_confidence_calibrator
        from config.database import execute_query
        
        calibrator = get_confidence_calibrator()
        
        # Get calibration data from database
        result = execute_query(
            "SELECT * FROM confidence_calibration WHERE model = 'test_model'", 
            (), "DEMO"
        )
        
        # Group by window_id
        window_data = {}
        for record in result:
            window_id = record['window_id']
            if window_id not in window_data:
                window_data[window_id] = []
            window_data[window_id].append(record)
        
        # Create calibration data for each window
        for window_id, records in window_data.items():
            if len(records) >= 3:  # Need at least 3 trades for calibration
                wins = sum(1 for r in records if r['outcome'] == 'WIN')
                losses = sum(1 for r in records if r['outcome'] == 'LOSS')
                total_trades = len(records)
                raw_confidence_sum = sum(r['raw_confidence'] for r in records)
                
                # Calculate calibrated probability using Bayesian approach
                alpha_prior = 2.0
                beta_prior = 2.0
                calibrated_prob = (wins + alpha_prior) / (total_trades + alpha_prior + beta_prior)
                
                # Create calibration data
                from adaptive.confidence_calibration import CalibrationData
                calibration_data = CalibrationData(
                    model_name="test_model",
                    window_id=window_id,
                    total_trades=total_trades,
                    wins=wins,
                    losses=losses,
                    raw_confidence_sum=raw_confidence_sum,
                    calibrated_probability=calibrated_prob,
                    calibration_quality=0.8,  # Default quality
                    last_updated=datetime.now(),
                    window_start=datetime.now() - timedelta(days=30),
                    window_end=datetime.now()
                )
                
                # Add to cache
                calibration_key = f"test_model_{window_id}"
                calibrator.calibration_cache[calibration_key] = calibration_data
        
        logger.info("Calibrator cache populated with test data")
        return True
        
    except Exception as e:
        logger.error(f"Failed to populate calibrator cache: {e}")
        return False

def test_risk_integration():
    """Test risk management integration with calibrated confidence."""
    try:
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

def main():
    """Run Phase 4 smoke test with fixes."""
    start_time = time.time()
    
    logger.info("Starting Phase 4 Smoke Test (Fixed Version)...")
    
    # Populate calibrator cache first
    logger.info("Populating calibrator cache...")
    if not populate_calibrator_cache():
        logger.error("Failed to populate calibrator cache")
        return False
    
    # Test risk integration
    logger.info("Testing Risk Integration...")
    result = test_risk_integration()
    
    duration = time.time() - start_time
    
    if result:
        logger.info("=" * 50)
        logger.info("PHASE 4 SMOKE TEST (FIXED) SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Duration: {duration:.1f}s")
        logger.info("Risk Integration: PASS")
        logger.info("PHASE 4 SMOKE TEST PASSED")
        return True
    else:
        logger.error("PHASE 4 SMOKE TEST FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

