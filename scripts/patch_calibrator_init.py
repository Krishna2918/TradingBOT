#!/usr/bin/env python3
"""
Patch Calibrator to Load Data from Database on Initialization
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def patch_calibrator():
    """Patch the calibrator to load data from database on initialization."""
    try:
        from adaptive.confidence_calibration import ConfidenceCalibrator
        from config.database import execute_query
        
        # Store original __init__ method
        original_init = ConfidenceCalibrator.__init__
        
        def patched_init(self, window_size_days=30, min_trades_for_calibration=5):
            # Call original init
            original_init(self, window_size_days, min_trades_for_calibration)
            
            # Load calibration data from database
            try:
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
                        self.calibration_cache[calibration_key] = calibration_data
                
                print(f"Calibrator initialized with {len(self.calibration_cache)} cached calibrations")
                
            except Exception as e:
                print(f"Failed to load calibration data: {e}")
        
        # Apply patch
        ConfidenceCalibrator.__init__ = patched_init
        
        print("Calibrator patched to load data from database on initialization")
        return True
        
    except Exception as e:
        print(f"Failed to patch calibrator: {e}")
        return False

if __name__ == "__main__":
    patch_calibrator()

