"""
Start Subset Data Collection

Collect data for a smaller subset (top 20 stocks) to validate the complete pipeline
before running the full 200-symbol collection.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.data_collection.comprehensive_data_collector import ComprehensiveDataCollector

def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        log_dir / f'subset_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

class SubsetDataCollector(ComprehensiveDataCollector):
    """Data collector for subset validation"""
    
    def __init__(self, base_path: str = "SubsetTrainingData"):
        super().__init__(base_path)
        
        # Override with top 20 stocks for validation
        self.us_symbols = [
            # Top 20 by market cap - diverse sectors
            'AAPL',  # Technology
            'MSFT',  # Technology  
            'GOOGL', # Technology
            'AMZN',  # Consumer Discretionary
            'NVDA',  # Technology
            'TSLA',  # Consumer Discretionary
            'META',  # Technology
            'BRK.B', # Financial Services
            'UNH',   # Healthcare
            'JNJ',   # Healthcare
            'V',     # Financial Services
            'PG',    # Consumer Staples
            'JPM',   # Financial Services
            'HD',    # Consumer Discretionary
            'MA',    # Financial Services
            'CVX',   # Energy
            'ABBV',  # Healthcare
            'PFE',   # Healthcare
            'KO',    # Consumer Staples
            'AVGO'   # Technology
        ]
        
        logging.info(f"Subset Data Collector initialized with {len(self.us_symbols)} symbols")
        logging.info(f"Symbols: {', '.join(self.us_symbols)}")

async def main():
    """Main subset collection process"""
    print("🧪 Starting SUBSET Data Collection (Top 20 Stocks)")
    print("=" * 60)
    print("This is a validation run before the full 200-symbol collection")
    print("=" * 60)
    
    try:
        # Initialize subset collector
        collector = SubsetDataCollector()
        
        # Show what we're collecting
        print(f"\n📊 Collecting data for {len(collector.us_symbols)} stocks:")
        for i, symbol in enumerate(collector.us_symbols, 1):
            print(f"  {i:2d}. {symbol}")
        
        print(f"\n📈 Data types to collect:")
        print(f"  • Daily OHLCV (20+ years)")
        print(f"  • Intraday 1min & 5min (1 year)")
        print(f"  • Technical indicators")
        print(f"  • Fundamentals (if API quota allows)")
        print(f"  • Macro economics data")
        print(f"  • Sentiment data")
        
        # Confirm start
        print(f"\n⚠️  This will use API quotas but much less than full collection")
        print(f"   Estimated time: 30-60 minutes for market data")
        
        response = input("\nDo you want to continue with subset collection? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Subset collection cancelled by user")
            return
        
        print(f"\n🚀 Starting subset data collection...")
        print(f"   Monitor progress in logs directory")
        
        # Start market data collection (most important for validation)
        print(f"\n📈 Phase 1: Collecting market data...")
        market_results = await collector.collect_market_data()
        
        print(f"\n📊 Market Data Results:")
        print(f"  Symbols processed: {market_results['symbols_processed']}/{len(collector.us_symbols)}")
        print(f"  Daily data collected: {market_results['data_collected']['daily']}")
        print(f"  1min intraday collected: {market_results['data_collected']['intraday_1min']}")
        print(f"  5min intraday collected: {market_results['data_collected']['intraday_5min']}")
        print(f"  Technical indicators: {market_results['data_collected']['technical_indicators']}")
        
        if market_results['symbols_failed']:
            print(f"  Failed symbols: {', '.join(market_results['symbols_failed'])}")
        
        # Quick validation
        print(f"\n✅ Performing data validation...")
        validation_results = await collector.validate_collected_data()
        
        print(f"\n📋 Validation Results:")
        print(f"  Overall quality score: {validation_results['overall_quality_score']:.3f}")
        print(f"  Training ready: {'✅ Yes' if validation_results['training_ready'] else '❌ No'}")
        
        # Show what files were created
        market_data_path = Path(collector.base_path) / "market_data"
        if market_data_path.exists():
            files = list(market_data_path.glob("*.parquet"))
            print(f"\n📁 Files created: {len(files)} market data files")
            
            # Show sample of files
            for file in files[:5]:
                print(f"    - {file.name}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more files")
        
        # Success summary
        success_rate = market_results['symbols_processed'] / len(collector.us_symbols)
        
        print(f"\n" + "=" * 60)
        print(f"SUBSET COLLECTION RESULTS")
        print(f"=" * 60)
        
        if success_rate >= 0.8:
            print(f"✅ SUBSET COLLECTION SUCCESSFUL!")
            print(f"   Success rate: {success_rate:.1%}")
            print(f"   Quality score: {validation_results['overall_quality_score']:.3f}")
            print(f"\n🎯 PIPELINE VALIDATED - Ready for full collection!")
            
            print(f"\n📋 NEXT STEPS:")
            print(f"   1. Review the collected data quality")
            print(f"   2. Check API key usage remaining")
            print(f"   3. Start full 200-symbol collection when ready")
            print(f"\n⚠️  REMINDER: FULL COLLECTION (200 symbols) IS NOT DONE YET!")
            print(f"   This was only a validation run with {len(collector.us_symbols)} symbols.")
            print(f"   Run 'python start_data_collection.py' for full collection.")
            
        else:
            print(f"⚠️  SUBSET COLLECTION PARTIAL SUCCESS")
            print(f"   Success rate: {success_rate:.1%}")
            print(f"   Some issues detected - review logs before full collection")
        
        # Show API usage
        usage = collector.key_manager.get_usage_summary()
        print(f"\n🔑 API Key Usage After Subset Collection:")
        for key in usage['keys']:
            print(f"   {key['name']}: {key['used']}/{key['limit']} ({key['remaining']} remaining)")
    
    except KeyboardInterrupt:
        print(f"\n❌ Subset collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Subset collection failed: {e}")
        logging.error(f"Subset collection failed: {e}", exc_info=True)

if __name__ == '__main__':
    # Setup logging
    setup_logging()
    
    # Run subset collection
    asyncio.run(main())