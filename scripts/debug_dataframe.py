#!/usr/bin/env python3
"""
Debug DataFrame Creation from Market Data
"""

import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def debug_dataframe():
    """Debug DataFrame creation from market data query."""
    try:
        from config.database import execute_query
        
        # Query for recent market data (same as regime detection)
        query = """
            SELECT date, open, high, low, close, volume, atr
            FROM market_data 
            WHERE symbol = ? AND mode = ?
            ORDER BY date DESC
            LIMIT ?
        """
        
        result = execute_query(query, ("SPY", "DEMO", 10), "DEMO")
        
        print(f"Query result type: {type(result)}")
        print(f"Query result length: {len(result) if result else 0}")
        
        if result:
            print(f"First row type: {type(result[0])}")
            print(f"First row: {result[0]}")
            print(f"First row keys: {list(result[0].keys()) if hasattr(result[0], 'keys') else 'No keys'}")
            
            # Convert to DataFrame
            df = pd.DataFrame(result)
            print(f"\nDataFrame shape: {df.shape}")
            print(f"DataFrame columns: {list(df.columns)}")
            print(f"DataFrame dtypes:\n{df.dtypes}")
            print(f"\nDataFrame head:\n{df.head()}")
            
            # Check if date column exists
            if 'date' in df.columns:
                print(f"\nDate column found! Sample values: {df['date'].head().tolist()}")
            else:
                print(f"\nDate column NOT found! Available columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"Failed to debug DataFrame: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_dataframe()

