#!/usr/bin/env python3
"""
Fix Market Data Schema - Date Column Issue
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def fix_market_data_schema():
    """Fix market_data table schema and data format."""
    try:
        from config.database import get_connection
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Drop and recreate market_data table with correct schema
            cursor.execute("DROP TABLE IF EXISTS market_data")
            
            cursor.execute("""
                CREATE TABLE market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    atr REAL,
                    mode TEXT DEFAULT 'DEMO',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert properly formatted sample data
            base_date = datetime.now() - timedelta(days=10)
            sample_data = []
            
            for i in range(10):
                trade_date = base_date + timedelta(days=i)
                date_str = trade_date.strftime('%Y-%m-%d')
                
                # Generate realistic OHLC data
                base_price = 100.0 + i * 0.5
                open_price = base_price
                high_price = base_price + 1.0
                low_price = base_price - 0.5
                close_price = base_price + 0.2
                volume = 1000000 + i * 100000
                atr = 1.5 + i * 0.1
                
                sample_data.append((
                    "SPY", date_str, open_price, high_price, low_price, 
                    close_price, volume, atr, "DEMO"
                ))
            
            cursor.executemany("""
                INSERT INTO market_data (symbol, date, open, high, low, close, volume, atr, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, sample_data)
            
            conn.commit()
        
        print("Market data table schema fixed with proper date format")
        return True
        
    except Exception as e:
        print(f"Failed to fix market data schema: {e}")
        return False

if __name__ == "__main__":
    fix_market_data_schema()

