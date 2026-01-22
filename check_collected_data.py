"""
Check Collected Data

Simple script to analyze what data was actually collected and assess quality.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('src')

def analyze_data_directory():
    """Analyze the collected data directory"""
    print("Data Collection Analysis")
    print("=" * 60)
    
    data_dir = Path('TrainingData')
    if not data_dir.exists():
        print("ERROR: TrainingData directory not found!")
        return
    
    print(f"Data Directory: {data_dir.absolute()}")
    print()
    
    # Count files by type
    file_counts = {}
    total_size = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = Path(root) / file
            file_size = file_path.stat().st_size
            total_size += file_size
            
            # Categorize by file type/location
            relative_path = file_path.relative_to(data_dir)
            category = str(relative_path.parts[0]) if relative_path.parts else 'root'
            
            if category not in file_counts:
                file_counts[category] = {'count': 0, 'size': 0}
            
            file_counts[category]['count'] += 1
            file_counts[category]['size'] += file_size
    
    # Display results
    print("File Summary by Category:")
    print("-" * 40)
    for category, stats in sorted(file_counts.items()):
        size_mb = stats['size'] / (1024 * 1024)
        print(f"  {category:20} {stats['count']:4d} files  {size_mb:8.2f} MB")
    
    total_mb = total_size / (1024 * 1024)
    total_files = sum(stats['count'] for stats in file_counts.values())
    
    print("-" * 40)
    print(f"  {'TOTAL':20} {total_files:4d} files  {total_mb:8.2f} MB")
    print()
    
    # Check for specific data types
    print("Data Type Analysis:")
    print("-" * 40)
    
    # Look for daily data
    daily_files = list(data_dir.glob('daily/*_daily.parquet'))
    print(f"  Daily data files: {len(daily_files)}")
    
    # Look for intraday data  
    intraday_files = list(data_dir.glob('intraday/*_1min.parquet')) + list(data_dir.glob('intraday/*_5min.parquet'))
    print(f"  Intraday data files: {len(intraday_files)}")
    
    # Look for technical indicators
    indicator_files = list(data_dir.glob('indicators/*_*.parquet'))
    print(f"  Technical indicator files: {len(indicator_files)}")
    
    # Sample a few files to check data quality
    print("\nData Quality Sample:")
    print("-" * 40)
    
    sample_files = daily_files[:3] if daily_files else []
    for file_path in sample_files:
        try:
            df = pd.read_parquet(file_path)
            symbol = file_path.stem.replace('daily_', '')
            print(f"  {symbol}: {len(df)} rows, {len(df.columns)} columns")
            
            # Check date range
            if 'date' in df.columns:
                date_col = df['date']
                if len(date_col) > 0:
                    min_date = date_col.min()
                    max_date = date_col.max()
                    print(f"    Date range: {min_date} to {max_date}")
            
        except Exception as e:
            print(f"  {file_path.name}: ERROR - {e}")
    
    return {
        'total_files': total_files,
        'total_size_mb': total_mb,
        'daily_files': len(daily_files),
        'intraday_files': len(intraday_files),
        'indicator_files': len(indicator_files),
        'categories': file_counts
    }

def check_symbol_coverage():
    """Check which symbols have complete data"""
    print("\nSymbol Coverage Analysis:")
    print("-" * 40)
    
    data_dir = Path('TrainingData')
    if not data_dir.exists():
        return
    
    # Get all symbols from daily files (this will show us what we actually collected)
    daily_files = list(data_dir.glob('daily/*_daily.parquet'))
    collected_symbols = [f.stem.replace('_daily', '') for f in daily_files]
    collected_symbols.sort()
    
    print(f"Found {len(collected_symbols)} symbols with daily data")
    
    # For display, show first 20 and summary stats
    display_symbols = collected_symbols[:20]
    
    # Analyze all collected symbols
    all_symbol_status = {}
    display_symbol_status = {}
    
    for symbol in collected_symbols:
        status = {
            'daily': False,
            'intraday': False,
            'indicators': False
        }
        
        # Check for daily data
        daily_file = data_dir / 'daily' / f'{symbol}_daily.parquet'
        if daily_file.exists():
            status['daily'] = True
        
        # Check for intraday data
        intraday_1min = data_dir / 'intraday' / f'{symbol}_1min.parquet'
        intraday_5min = data_dir / 'intraday' / f'{symbol}_5min.parquet'
        if intraday_1min.exists() or intraday_5min.exists():
            status['intraday'] = True
        
        # Check for indicators (need at least 5 indicators to be considered complete)
        indicator_files = list(data_dir.glob(f'indicators/{symbol}_*.parquet'))
        if len(indicator_files) >= 5:
            status['indicators'] = True
        
        all_symbol_status[symbol] = status
        if symbol in display_symbols:
            display_symbol_status[symbol] = status
    
    # Calculate overall stats
    total_complete = sum(1 for status in all_symbol_status.values() if all(status.values()))
    total_symbols = len(all_symbol_status)
    
    # Display sample results (first 20)
    print(f"\nSample Results (first 20 of {total_symbols} symbols):")
    print(f"{'Symbol':<8} {'Daily':<6} {'Intraday':<9} {'Indicators':<11} {'Complete'}")
    print("-" * 50)
    
    for symbol, status in display_symbol_status.items():
        daily = "✓" if status['daily'] else "✗"
        intraday = "✓" if status['intraday'] else "✗"
        indicators = "✓" if status['indicators'] else "✗"
        complete = "✓" if all(status.values()) else "✗"
        
        print(f"{symbol:<8} {daily:<6} {intraday:<9} {indicators:<11} {complete}")
    
    print("-" * 50)
    print(f"TOTAL Complete symbols: {total_complete}/{total_symbols} ({total_complete/total_symbols*100:.1f}%)")
    
    # Show breakdown by category
    daily_count = sum(1 for status in all_symbol_status.values() if status['daily'])
    intraday_count = sum(1 for status in all_symbol_status.values() if status['intraday'])
    indicators_count = sum(1 for status in all_symbol_status.values() if status['indicators'])
    
    print(f"\nBreakdown:")
    print(f"  Daily data: {daily_count}/{total_symbols} ({daily_count/total_symbols*100:.1f}%)")
    print(f"  Intraday data: {intraday_count}/{total_symbols} ({intraday_count/total_symbols*100:.1f}%)")
    print(f"  Technical indicators: {indicators_count}/{total_symbols} ({indicators_count/total_symbols*100:.1f}%)")
    
    return all_symbol_status

def main():
    """Main analysis function"""
    try:
        # Analyze data directory
        data_stats = analyze_data_directory()
        
        # Check symbol coverage
        symbol_coverage = check_symbol_coverage()
        
        # Summary assessment
        print("\nCollection Assessment:")
        print("=" * 60)
        
        if data_stats and data_stats['total_files'] > 0:
            print("✓ Data collection has produced files")
            print(f"  Total: {data_stats['total_files']} files ({data_stats['total_size_mb']:.1f} MB)")
            
            if data_stats['daily_files'] > 150:
                print("✓ Good daily data coverage")
            else:
                print("⚠ Limited daily data coverage")
            
            if data_stats['indicator_files'] > 500:
                print("✓ Good technical indicator coverage")
            else:
                print("⚠ Limited technical indicator coverage")
            
            print("\nRecommendations:")
            if data_stats['daily_files'] < 200:
                print("- Consider re-running collection for missing daily data")
            if data_stats['indicator_files'] < 1000:
                print("- Consider re-running technical indicator collection")
            
            print("- Dataset appears suitable for initial AI training")
            print("- Quality can be improved with additional collection cycles")
        else:
            print("✗ No data files found - collection may have failed")
            print("- Run the collection process again")
            print("- Check API keys and network connectivity")
    
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == '__main__':
    main()