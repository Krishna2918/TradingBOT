#!/usr/bin/env python3
"""
Clean up Phase 4 Test Data
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def cleanup_phase4_test_data():
    """Clean up Phase 4 test data to fix hygiene issues."""
    try:
        from config.database import get_connection
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear all test calibration data
            cursor.execute("DELETE FROM confidence_calibration WHERE model = 'test_model'")
            
            # Insert clean test data - exactly 15 records as expected by tests
            from datetime import datetime, timedelta
            
            now = datetime.now()
            window_start = now - timedelta(days=30)
            window_id = f"{window_start.strftime('%Y%m%d')}_{now.strftime('%Y%m%d')}"
            
            # Create exactly 15 test records
            test_data = []
            for i in range(15):
                trade_data = (
                    "test_model", "TEST", 0.8, 0.75, "WIN", 100.0, None, 
                    window_id, now.isoformat(), now.isoformat(), now.isoformat()
                )
                test_data.append(trade_data)
            
            cursor.executemany("""
                INSERT INTO confidence_calibration 
                (model, symbol, raw_confidence, calibrated_confidence, outcome, pnl, exit_date, window_id, trade_date, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, test_data)
            
            conn.commit()
        
        print("Phase 4 test data cleaned up - exactly 15 records inserted")
        return True
        
    except Exception as e:
        print(f"Failed to cleanup Phase 4 test data: {e}")
        return False

if __name__ == "__main__":
    cleanup_phase4_test_data()

