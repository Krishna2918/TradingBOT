#!/usr/bin/env python3
"""
Debug Calibration Data
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def debug_calibration_data():
    """Debug calibration data in database."""
    try:
        from config.database import execute_query
        
        # Check what's in the confidence_calibration table
        result = execute_query("SELECT * FROM confidence_calibration", (), "DEMO")
        
        print(f"Found {len(result)} calibration records:")
        for record in result:
            print(f"  - Model: {record['model']}, Window: {record['window_id']}, Raw: {record['raw_confidence']}")
        
        # Check what window the calibrator is looking for
        now = datetime.now()
        window_start = now - timedelta(days=30)
        expected_window = f"{window_start.strftime('%Y%m%d')}_{now.strftime('%Y%m%d')}"
        print(f"\nExpected window ID: {expected_window}")
        
        # Check if there's data for test_model
        test_model_data = execute_query(
            "SELECT * FROM confidence_calibration WHERE model = 'test_model'", 
            (), "DEMO"
        )
        print(f"\nFound {len(test_model_data)} records for test_model")
        
        return True
        
    except Exception as e:
        print(f"Debug failed: {e}")
        return False

if __name__ == "__main__":
    debug_calibration_data()
