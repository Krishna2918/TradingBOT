#!/usr/bin/env python3
"""
Debug Market Data Table
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def debug_market_data():
    """Debug the market_data table structure and content."""
    try:
        from config.database import get_connection
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Check table structure
            cursor.execute("PRAGMA table_info(market_data)")
            columns = cursor.fetchall()
            print("Market Data Table Structure:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Check sample data
            cursor.execute("SELECT * FROM market_data LIMIT 3")
            rows = cursor.fetchall()
            print(f"\nSample Data ({len(rows)} rows):")
            for row in rows:
                print(f"  {row}")
            
            # Check column names specifically
            cursor.execute("SELECT * FROM market_data LIMIT 1")
            row = cursor.fetchone()
            if row:
                cursor.execute("PRAGMA table_info(market_data)")
                column_info = cursor.fetchall()
                column_names = [col[1] for col in column_info]
                print(f"\nColumn names: {column_names}")
                print(f"Row data: {row}")
                
                # Check if date column exists and its type
                date_col_index = None
                for i, col in enumerate(column_info):
                    if col[1] == 'date':
                        date_col_index = i
                        print(f"Date column found at index {i}, type: {col[2]}")
                        break
                
                if date_col_index is not None:
                    print(f"Date value: {row[date_col_index]} (type: {type(row[date_col_index])})")
        
        return True
        
    except Exception as e:
        print(f"Failed to debug market data: {e}")
        return False

if __name__ == "__main__":
    debug_market_data()

