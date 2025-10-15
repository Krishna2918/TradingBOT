#!/usr/bin/env python3
"""
Debug Risk Integration
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def debug_risk_integration():
    """Debug risk integration."""
    try:
        from trading.risk import calculate_position_size
        from adaptive.confidence_calibration import get_confidence_calibrator
        
        calibrator = get_confidence_calibrator()
        
        print(f"Calibrator cache keys: {list(calibrator.calibration_cache.keys())}")
        print(f"Min trades for calibration: {calibrator.min_trades_for_calibration}")
        
        for key, cal_data in calibrator.calibration_cache.items():
            print(f"  {key}: {cal_data.total_trades} trades, calibrated_prob={cal_data.calibrated_probability:.3f}")
        
        # Test calibration directly
        test_date = datetime.now()
        raw_confidence = 0.8
        calibrated = calibrator.calibrate_confidence("test_model", raw_confidence, test_date)
        
        print(f"\nDirect calibration test:")
        print(f"  Raw confidence: {raw_confidence}")
        print(f"  Calibrated confidence: {calibrated}")
        print(f"  Different: {raw_confidence != calibrated}")
        
        # Test through risk manager
        print(f"\nRisk manager test:")
        risk_metrics = calculate_position_size(
            signal_confidence=raw_confidence,
            account_balance=10000.0,
            volatility=0.02,
            entry_price=100.0,
            stop_loss=95.0,
            model_name="test_model",
            symbol="TEST",
            trade_date=test_date
        )
        
        print(f"  Raw confidence: {risk_metrics.raw_confidence}")
        print(f"  Calibrated confidence: {risk_metrics.calibrated_confidence}")
        print(f"  Different: {risk_metrics.raw_confidence != risk_metrics.calibrated_confidence}")
        
        return risk_metrics.raw_confidence != risk_metrics.calibrated_confidence
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = debug_risk_integration()
    print(f"\nTest result: {result}")
