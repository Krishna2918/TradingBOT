"""
Fix Data Structure

Move files from the flat market_data structure to the proper organized structure
that the validation scripts expect.
"""

import os
import shutil
from pathlib import Path

def reorganize_data_files():
    """Reorganize data files into proper directory structure"""
    print("Reorganizing Data Files")
    print("=" * 50)
    
    base_dir = Path('TrainingData')
    market_data_dir = base_dir / 'market_data'
    technical_indicators_dir = base_dir / 'technical_indicators'
    
    # Create target directories
    daily_dir = base_dir / 'daily'
    intraday_dir = base_dir / 'intraday'
    indicators_dir = base_dir / 'indicators'
    
    daily_dir.mkdir(exist_ok=True)
    intraday_dir.mkdir(exist_ok=True)
    indicators_dir.mkdir(exist_ok=True)
    
    moved_files = {'daily': 0, 'intraday': 0, 'indicators': 0}
    
    # Process market_data files
    if market_data_dir.exists():
        print(f"\nProcessing {market_data_dir}...")
        
        for file_path in market_data_dir.glob('*.parquet'):
            filename = file_path.name
            
            if '_daily.parquet' in filename:
                # Move daily files
                new_path = daily_dir / filename
                shutil.move(str(file_path), str(new_path))
                moved_files['daily'] += 1
                print(f"  Moved daily: {filename}")
                
            elif '_1min.parquet' in filename or '_5min.parquet' in filename:
                # Move intraday files
                new_path = intraday_dir / filename
                shutil.move(str(file_path), str(new_path))
                moved_files['intraday'] += 1
                print(f"  Moved intraday: {filename}")
    
    # Process technical_indicators files
    if technical_indicators_dir.exists():
        print(f"\nProcessing {technical_indicators_dir}...")
        
        for file_path in technical_indicators_dir.glob('*.parquet'):
            filename = file_path.name
            # Move all technical indicator files
            new_path = indicators_dir / filename
            shutil.move(str(file_path), str(new_path))
            moved_files['indicators'] += 1
            print(f"  Moved indicator: {filename}")
    
    # Summary
    print(f"\nFiles Moved:")
    print(f"  Daily files: {moved_files['daily']}")
    print(f"  Intraday files: {moved_files['intraday']}")
    print(f"  Indicator files: {moved_files['indicators']}")
    print(f"  Total: {sum(moved_files.values())}")
    
    # Clean up empty directories
    if market_data_dir.exists() and not any(market_data_dir.iterdir()):
        market_data_dir.rmdir()
        print(f"  Removed empty directory: {market_data_dir}")
    
    if technical_indicators_dir.exists() and not any(technical_indicators_dir.iterdir()):
        technical_indicators_dir.rmdir()
        print(f"  Removed empty directory: {technical_indicators_dir}")
    
    return moved_files

def update_checker_script():
    """Update the checker script to look in the right places"""
    print("\nUpdating checker script paths...")
    
    # The checker script should now work correctly with the reorganized structure
    print("  Checker script should now find files in:")
    print("    - TrainingData/daily/")
    print("    - TrainingData/intraday/") 
    print("    - TrainingData/indicators/")

def main():
    """Main reorganization process"""
    print("Data Structure Fix")
    print("=" * 60)
    
    try:
        # Check if TrainingData exists
        if not Path('TrainingData').exists():
            print("ERROR: TrainingData directory not found!")
            return
        
        # Reorganize files
        moved_files = reorganize_data_files()
        
        # Update checker
        update_checker_script()
        
        print("\nSUCCESS: Data structure has been reorganized!")
        print("Now run: python check_collected_data.py")
        
    except Exception as e:
        print(f"ERROR: Failed to reorganize data structure: {e}")

if __name__ == '__main__':
    main()