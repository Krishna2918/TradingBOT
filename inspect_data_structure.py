"""
Inspect Data Structure

Quick script to check the actual column names and structure of our collected data.
"""

import pandas as pd
from pathlib import Path

def inspect_data():
    """Inspect the structure of collected data files"""
    print("Data Structure Inspection")
    print("=" * 50)
    
    data_dir = Path("TrainingData")
    
    # Check daily data
    daily_files = list(data_dir.glob("daily/*.parquet"))
    if daily_files:
        print(f"\nDaily Data Sample ({daily_files[0].name}):")
        df = pd.read_parquet(daily_files[0])
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Index: {df.index}")
        print(f"First few rows:")
        print(df.head())
    
    # Check intraday data
    intraday_files = list(data_dir.glob("intraday/*.parquet"))
    if intraday_files:
        print(f"\nIntraday Data Sample ({intraday_files[0].name}):")
        df = pd.read_parquet(intraday_files[0])
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Index: {df.index}")
        print(f"First few rows:")
        print(df.head())
    
    # Check indicators data
    indicator_files = list(data_dir.glob("indicators/*.parquet"))
    if indicator_files:
        print(f"\nIndicator Data Sample ({indicator_files[0].name}):")
        df = pd.read_parquet(indicator_files[0])
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Index: {df.index}")
        print(f"First few rows:")
        print(df.head())

if __name__ == "__main__":
    inspect_data()