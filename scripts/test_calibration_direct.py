#!/usr/bin/env python3
"""
Test Calibration Directly
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_calibration_direct():
    """Test calibration directly."""
    try:
        from adaptive.confidence_calibration import get_confidence_calibrator
        
        calibrator = get_confidence_calibrator()
        
        # Check cache
        print(f"Cache keys: {list(calibrator.calibration_cache.keys())}")
        
        # Test calibration
        raw_confidence = 0.8
        calibrated = calibrator.calibrate_confidence("test_model", raw_confidence, datetime.now())
        
        print(f"Raw confidence: {raw_confidence}")
        print(f"Calibrated confidence: {calibrated}")
        print(f"Different: {raw_confidence != calibrated}")
        
        return raw_confidence != calibrated
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    # First populate the cache
    import subprocess
    subprocess.run([sys.executable, "scripts/fix_calibrator_cache.py"])
    
    # Then test
    result = test_calibration_direct()
    print(f"Test result: {result}")
