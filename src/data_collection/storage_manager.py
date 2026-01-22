"""
Storage Manager - Efficient Parquet-based Data Storage

Manages organized storage of historical and intraday data in Parquet format.
Handles incremental updates, deduplication, and data quality validation.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
import os

logger = logging.getLogger(__name__)

class StorageManager:
    """Efficient Parquet-based storage for financial data"""
    
    def __init__(self, base_path: str = "PastData"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create timeframe directories
        self.timeframe_dirs = {
            "1d": self.base_path / "daily",
            "1wk": self.base_path / "weekly", 
            "1mo": self.base_path / "monthly",
            "3mo": self.base_path / "quarterly",
            "1y": self.base_path / "yearly",
            "1min": self.base_path / "intraday",
            "5min": self.base_path / "intraday",
            "15min": self.base_path / "intraday", 
            "30min": self.base_path / "intraday"
        }
        
        for dir_path in self.timeframe_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_file_path(self, symbol: str, timeframe: str) -> Path:
        """Get the file path for a symbol/timeframe combination"""
        if timeframe in ["1min", "5min", "15min", "30min"]:
            # Intraday data: organized by symbol subdirectories
            symbol_dir = self.base_path / "intraday" / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            return symbol_dir / f"{timeframe}.parquet"
        else:
            # Higher timeframes: direct files in timeframe directories
            return self.timeframe_dirs[timeframe] / f"{symbol}.parquet"
    
    def save_to_parquet(self, data: pd.DataFrame, symbol: str, timeframe: str, 
                       validate: bool = True) -> bool:
        """Save DataFrame to Parquet file"""
        try:
            if data.empty:
                logger.warning(f"Empty DataFrame for {symbol} {timeframe}")
                return False
            
            # Validate data if requested
            if validate:
                data = self._validate_and_clean_data(data, symbol, timeframe)
                if data.empty:
                    logger.error(f"Data validation failed for {symbol} {timeframe}")
                    return False
            
            # Get file path
            file_path = self.get_file_path(symbol, timeframe)
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'Date' in data.columns:
                    data.set_index('Date', inplace=True)
                elif 'Datetime' in data.columns:
                    data.set_index('Datetime', inplace=True)
            
            # Sort by index (datetime)
            data = data.sort_index()
            
            # Save to Parquet with compression
            data.to_parquet(file_path, compression='snappy', index=True)
            
            # Calculate file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"💾 Saved {symbol} {timeframe}: {len(data)} rows, {file_size_mb:.2f} MB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save {symbol} {timeframe}: {e}")
            return False
    
    def append_to_parquet(self, new_data: pd.DataFrame, symbol: str, timeframe: str,
                         validate: bool = True) -> bool:
        """Append new data to existing Parquet file without duplication"""
        try:
            if new_data.empty:
                logger.warning(f"Empty new data for {symbol} {timeframe}")
                return False
            
            file_path = self.get_file_path(symbol, timeframe)
            
            # Load existing data if file exists
            if file_path.exists():
                existing_data = pd.read_parquet(file_path)
                
                # Ensure datetime index for both datasets
                if not isinstance(existing_data.index, pd.DatetimeIndex):
                    if 'Date' in existing_data.columns:
                        existing_data.set_index('Date', inplace=True)
                    elif 'Datetime' in existing_data.columns:
                        existing_data.set_index('Datetime', inplace=True)
                
                if not isinstance(new_data.index, pd.DatetimeIndex):
                    if 'Date' in new_data.columns:
                        new_data.set_index('Date', inplace=True)
                    elif 'Datetime' in new_data.columns:
                        new_data.set_index('Datetime', inplace=True)
                
                # Remove duplicates by index (datetime)
                combined_data = pd.concat([existing_data, new_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data = combined_data.sort_index()
                
                logger.info(f"📈 Appending to {symbol} {timeframe}: {len(existing_data)} + {len(new_data)} = {len(combined_data)} rows")
            else:
                combined_data = new_data.copy()
                logger.info(f"🆕 Creating new file for {symbol} {timeframe}: {len(combined_data)} rows")
            
            # Validate combined data
            if validate:
                combined_data = self._validate_and_clean_data(combined_data, symbol, timeframe)
                if combined_data.empty:
                    logger.error(f"Combined data validation failed for {symbol} {timeframe}")
                    return False
            
            # Save combined data
            return self.save_to_parquet(combined_data, symbol, timeframe, validate=False)
            
        except Exception as e:
            logger.error(f"Failed to append to {symbol} {timeframe}: {e}")
            return False
    
    def load_from_parquet(self, symbol: str, timeframe: str, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load data from Parquet file with optional date filtering"""
        try:
            file_path = self.get_file_path(symbol, timeframe)
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return pd.DataFrame()
            
            # Load data
            data = pd.read_parquet(file_path)
            
            # Apply date filtering if specified
            if start_date or end_date:
                if start_date:
                    data = data[data.index >= start_date]
                if end_date:
                    data = data[data.index <= end_date]
            
            logger.info(f"📂 Loaded {symbol} {timeframe}: {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def _validate_and_clean_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """Validate and clean financial data"""
        original_rows = len(data)
        
        # Required columns for financial data
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns for {symbol} {timeframe}: {missing_cols}")
            return pd.DataFrame()
        
        # Remove rows with invalid OHLC data
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Remove rows where High < Low (impossible)
        invalid_hl = data['High'] < data['Low']
        if invalid_hl.any():
            logger.warning(f"Removing {invalid_hl.sum()} rows with High < Low for {symbol} {timeframe}")
            data = data[~invalid_hl]
        
        # Remove rows where Close is outside High/Low range
        invalid_close = (data['Close'] > data['High']) | (data['Close'] < data['Low'])
        if invalid_close.any():
            logger.warning(f"Removing {invalid_close.sum()} rows with invalid Close for {symbol} {timeframe}")
            data = data[~invalid_close]
        
        # Remove rows with zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        invalid_prices = (data[price_cols] <= 0).any(axis=1)
        if invalid_prices.any():
            logger.warning(f"Removing {invalid_prices.sum()} rows with invalid prices for {symbol} {timeframe}")
            data = data[~invalid_prices]
        
        # Remove rows with negative volume
        if 'Volume' in data.columns:
            invalid_volume = data['Volume'] < 0
            if invalid_volume.any():
                logger.warning(f"Removing {invalid_volume.sum()} rows with negative volume for {symbol} {timeframe}")
                data = data[~invalid_volume]
        
        # Check for extreme price movements (>50% in one period) - likely data errors
        if len(data) > 1:
            price_change = data['Close'].pct_change().abs()
            extreme_moves = price_change > 0.5
            if extreme_moves.any():
                logger.warning(f"Found {extreme_moves.sum()} extreme price movements (>50%) for {symbol} {timeframe}")
                # Don't remove these automatically - might be legitimate (stock splits, etc.)
        
        cleaned_rows = len(data)
        if cleaned_rows < original_rows:
            logger.info(f"🧹 Cleaned {symbol} {timeframe}: {original_rows} → {cleaned_rows} rows")
        
        return data
    
    def validate_data_quality(self, symbol: str, timeframe: str) -> Dict[str, float]:
        """Validate data quality and return quality metrics"""
        try:
            data = self.load_from_parquet(symbol, timeframe)
            
            if data.empty:
                return {"quality_score": 0.0, "completeness": 0.0, "consistency": 0.0}
            
            # Calculate quality metrics
            metrics = {}
            
            # Completeness: ratio of non-null values
            total_cells = len(data) * len(data.columns)
            non_null_cells = data.count().sum()
            metrics["completeness"] = non_null_cells / total_cells if total_cells > 0 else 0.0
            
            # Consistency: check for logical consistency in OHLC data
            consistent_hl = (data['High'] >= data['Low']).mean()
            consistent_close = ((data['Close'] >= data['Low']) & (data['Close'] <= data['High'])).mean()
            consistent_open = ((data['Open'] >= data['Low']) & (data['Open'] <= data['High'])).mean()
            metrics["consistency"] = (consistent_hl + consistent_close + consistent_open) / 3
            
            # Continuity: check for gaps in time series
            if len(data) > 1:
                expected_frequency = self._get_expected_frequency(timeframe)
                if expected_frequency:
                    expected_periods = pd.date_range(start=data.index.min(), 
                                                   end=data.index.max(), 
                                                   freq=expected_frequency)
                    # Only count business days for daily data
                    if timeframe == "1d":
                        expected_periods = expected_periods[expected_periods.dayofweek < 5]
                    
                    metrics["continuity"] = len(data) / len(expected_periods) if len(expected_periods) > 0 else 1.0
                else:
                    metrics["continuity"] = 1.0
            else:
                metrics["continuity"] = 1.0
            
            # Overall quality score (weighted average)
            metrics["quality_score"] = (
                metrics["completeness"] * 0.4 + 
                metrics["consistency"] * 0.4 + 
                metrics["continuity"] * 0.2
            )
            
            logger.info(f"📊 Quality metrics for {symbol} {timeframe}: {metrics['quality_score']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to validate quality for {symbol} {timeframe}: {e}")
            return {"quality_score": 0.0, "completeness": 0.0, "consistency": 0.0, "continuity": 0.0}
    
    def _get_expected_frequency(self, timeframe: str) -> Optional[str]:
        """Get expected pandas frequency for timeframe"""
        frequency_map = {
            "1min": "1T",
            "5min": "5T", 
            "15min": "15T",
            "30min": "30T",
            "1d": "1D",
            "1wk": "1W",
            "1mo": "1M",
            "3mo": "3M",
            "1y": "1Y"
        }
        return frequency_map.get(timeframe)
    
    def get_storage_summary(self) -> Dict[str, any]:
        """Get summary of stored data"""
        summary = {
            "total_files": 0,
            "total_size_mb": 0.0,
            "symbols": set(),
            "timeframes": {},
            "file_details": []
        }
        
        # Scan all directories
        for timeframe, directory in self.timeframe_dirs.items():
            if not directory.exists():
                continue
                
            timeframe_files = 0
            timeframe_size = 0.0
            
            if timeframe in ["1min", "5min", "15min", "30min"]:
                # Intraday: scan symbol subdirectories
                intraday_dir = self.base_path / "intraday"
                if intraday_dir.exists():
                    for symbol_dir in intraday_dir.iterdir():
                        if symbol_dir.is_dir():
                            symbol = symbol_dir.name
                            summary["symbols"].add(symbol)
                            
                            timeframe_file = symbol_dir / f"{timeframe}.parquet"
                            if timeframe_file.exists():
                                file_size = timeframe_file.stat().st_size / (1024 * 1024)
                                timeframe_files += 1
                                timeframe_size += file_size
                                
                                summary["file_details"].append({
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "size_mb": file_size,
                                    "path": str(timeframe_file)
                                })
            else:
                # Higher timeframes: scan direct files
                for file_path in directory.glob("*.parquet"):
                    symbol = file_path.stem
                    summary["symbols"].add(symbol)
                    
                    file_size = file_path.stat().st_size / (1024 * 1024)
                    timeframe_files += 1
                    timeframe_size += file_size
                    
                    summary["file_details"].append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "size_mb": file_size,
                        "path": str(file_path)
                    })
            
            if timeframe_files > 0:
                summary["timeframes"][timeframe] = {
                    "files": timeframe_files,
                    "size_mb": timeframe_size
                }
                
                summary["total_files"] += timeframe_files
                summary["total_size_mb"] += timeframe_size
        
        summary["symbols"] = list(summary["symbols"])
        
        logger.info(f"Storage summary: {summary['total_files']} files, {summary['total_size_mb']:.2f} MB")
        return summary
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old intraday data to save space"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_files = 0
        
        intraday_dir = self.base_path / "intraday"
        if not intraday_dir.exists():
            return
        
        for symbol_dir in intraday_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
                
            for timeframe in ["1min", "5min", "15min", "30min"]:
                file_path = symbol_dir / f"{timeframe}.parquet"
                if not file_path.exists():
                    continue
                
                try:
                    data = pd.read_parquet(file_path)
                    if data.empty:
                        continue
                    
                    # Keep only recent data
                    recent_data = data[data.index >= cutoff_date]
                    
                    if len(recent_data) < len(data):
                        recent_data.to_parquet(file_path, compression='snappy', index=True)
                        cleaned_files += 1
                        logger.info(f"🧹 Cleaned {symbol_dir.name} {timeframe}: {len(data)} → {len(recent_data)} rows")
                
                except Exception as e:
                    logger.error(f"Failed to clean {file_path}: {e}")
        
        logger.info(f"🧹 Cleanup complete: {cleaned_files} files cleaned")

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    storage = StorageManager()
    
    # Test with sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1D')
    sample_data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(95, 115, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Test save and load
    print("🧪 Testing storage operations...")
    
    success = storage.save_to_parquet(sample_data, "TEST.TO", "1d")
    print(f"Save test: {'✅ Success' if success else '❌ Failed'}")
    
    loaded_data = storage.load_from_parquet("TEST.TO", "1d")
    print(f"Load test: {'✅ Success' if not loaded_data.empty else '❌ Failed'}")
    
    # Test quality validation
    quality = storage.validate_data_quality("TEST.TO", "1d")
    print(f"Quality score: {quality['quality_score']:.3f}")
    
    # Show storage summary
    summary = storage.get_storage_summary()
    print(f"Storage: {summary['total_files']} files, {summary['total_size_mb']:.2f} MB")