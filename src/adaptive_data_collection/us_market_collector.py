"""
US market data collector with retry logic and progress tracking.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from .config import CollectionConfig
from .alpha_vantage_client import AlphaVantageClient
from .progress_tracker import JSONProgressTracker
from .interfaces import DataCollector


class USMarketDataCollector:
    """Main collector for US market data with comprehensive error handling."""
    
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.alpha_vantage_client = AlphaVantageClient(config)
        self.progress_tracker = JSONProgressTracker("logs/collection_progress.json")
        
        # Collection state
        self.is_running = False
        self.should_stop = False
        
        self.logger.info("Initialized US Market Data Collector")
    
    def collect_all_symbols(self, resume: bool = True) -> Dict[str, Any]:
        """
        Collect data for all US symbols with progress tracking.
        
        Args:
            resume: Whether to resume from previous progress or start fresh
            
        Returns:
            Collection summary with statistics
        """
        self.logger.info("Starting US market data collection")
        self.is_running = True
        self.should_stop = False
        
        try:
            # Load symbols
            symbols = self._load_symbols()
            self.logger.info(f"Loaded {len(symbols)} symbols for collection")
            
            # Initialize or resume progress
            if not resume:
                self.progress_tracker.reset_progress()
            
            self.progress_tracker.initialize_symbols(symbols)
            
            # Get symbols to process
            if resume:
                pending_symbols = self.progress_tracker.get_pending_symbols()
                self.logger.info(f"Resuming collection: {len(pending_symbols)} symbols remaining")
            else:
                pending_symbols = symbols
                self.logger.info(f"Starting fresh collection: {len(pending_symbols)} symbols to process")
            
            # Collect data for each symbol
            collection_stats = {
                "start_time": time.time(),
                "symbols_processed": 0,
                "symbols_succeeded": 0,
                "symbols_failed": 0,
                "total_data_points": 0,
                "errors": []
            }
            
            for i, symbol in enumerate(pending_symbols):
                if self.should_stop:
                    self.logger.info("Collection stopped by user request")
                    break
                
                self.logger.info(f"Processing {symbol} ({i+1}/{len(pending_symbols)})")
                
                try:
                    # Collect data for symbol
                    success = self._collect_symbol_data(symbol)
                    
                    collection_stats["symbols_processed"] += 1
                    if success:
                        collection_stats["symbols_succeeded"] += 1
                    else:
                        collection_stats["symbols_failed"] += 1
                    
                    # Log progress periodically
                    if (i + 1) % 10 == 0:
                        progress = self.progress_tracker.get_progress_summary()
                        self.logger.info(f"Progress: {progress['progress_percentage']:.1f}% "
                                       f"({progress['completed_symbols']}/{progress['total_symbols']} completed)")
                
                except KeyboardInterrupt:
                    self.logger.info("Collection interrupted by user")
                    self.should_stop = True
                    break
                except Exception as e:
                    self.logger.error(f"Unexpected error processing {symbol}: {e}")
                    collection_stats["errors"].append(f"{symbol}: {str(e)}")
                    collection_stats["symbols_failed"] += 1
            
            # Final statistics
            collection_stats["end_time"] = time.time()
            collection_stats["duration_seconds"] = collection_stats["end_time"] - collection_stats["start_time"]
            
            # Get final progress summary
            final_progress = self.progress_tracker.get_progress_summary()
            collection_stats.update(final_progress)
            
            self.logger.info(f"Collection completed: {collection_stats['symbols_succeeded']} succeeded, "
                           f"{collection_stats['symbols_failed']} failed, "
                           f"{collection_stats['duration_seconds']:.1f}s total")
            
            return collection_stats
            
        finally:
            self.is_running = False
    
    def _collect_symbol_data(self, symbol: str) -> bool:
        """
        Collect data for a single symbol with error handling.
        
        Returns:
            True if successful, False if failed
        """
        try:
            # Mark symbol as started
            self.progress_tracker.mark_symbol_started(symbol)
            
            # Collect data using Alpha Vantage client (with built-in retry logic)
            df = self.alpha_vantage_client.collect_ticker_data(symbol)
            
            if df is None or len(df) == 0:
                raise ValueError(f"No data returned for {symbol}")
            
            # TODO: This will be implemented in Task 7 (Parquet storage)
            # For now, we'll simulate saving and just track the data
            data_points = len(df)
            file_size_bytes = df.memory_usage(deep=True).sum()  # Approximate size
            
            # Update progress with data statistics
            self.progress_tracker.update_symbol_data(symbol, data_points, file_size_bytes)
            
            # Mark as completed
            self.progress_tracker.mark_symbol_complete(symbol, success=True)
            
            self.logger.info(f"Successfully collected {data_points} data points for {symbol}")
            return True
            
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"Failed to collect data for {symbol}: {error_message}")
            
            # Mark as failed
            self.progress_tracker.mark_symbol_complete(symbol, success=False, error=error_message)
            
            return False
    
    def _load_symbols(self) -> List[str]:
        """Load symbols from the configured symbols file."""
        symbols_file = Path(self.config.us_symbols_file)
        
        if not symbols_file.exists():
            raise FileNotFoundError(f"Symbols file not found: {symbols_file}")
        
        try:
            with open(symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            
            if not symbols:
                raise ValueError(f"No symbols found in {symbols_file}")
            
            self.logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
            return symbols
            
        except IOError as e:
            self.logger.error(f"Failed to read symbols file {symbols_file}: {e}")
            raise
    
    def stop_collection(self) -> None:
        """Stop the collection process gracefully."""
        self.logger.info("Stopping collection...")
        self.should_stop = True
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status."""
        progress = self.progress_tracker.get_progress_summary()
        
        status = {
            "is_running": self.is_running,
            "should_stop": self.should_stop,
            "progress": progress,
            "alpha_vantage_status": self.alpha_vantage_client.get_api_status(),
            "retry_statistics": self.alpha_vantage_client.get_retry_statistics()
        }
        
        return status
    
    def get_failed_symbols(self) -> List[Dict[str, Any]]:
        """Get detailed information about failed symbols."""
        failed_symbols = []
        progress_data = self.progress_tracker.load_progress()
        
        for symbol, data in progress_data["symbols"].items():
            if data["status"] == "failed":
                failed_symbols.append({
                    "symbol": symbol,
                    "error": data.get("error", "Unknown error"),
                    "attempts": data.get("attempts", 0),
                    "last_attempt": data.get("end_time", "Unknown")
                })
        
        return failed_symbols
    
    def retry_failed_symbols(self) -> Dict[str, Any]:
        """Retry collection for all failed symbols."""
        failed_symbols = self.progress_tracker.get_failed_symbols()
        
        if not failed_symbols:
            self.logger.info("No failed symbols to retry")
            return {"message": "No failed symbols to retry", "retried_count": 0}
        
        self.logger.info(f"Retrying {len(failed_symbols)} failed symbols")
        
        # Reset failed symbols to pending
        progress_data = self.progress_tracker.load_progress()
        for symbol in failed_symbols:
            if symbol in progress_data["symbols"]:
                progress_data["symbols"][symbol]["status"] = "not_started"
                progress_data["symbols"][symbol]["error"] = None
        
        # Update failed count
        progress_data["failed_symbols"] -= len(failed_symbols)
        self.progress_tracker.save_progress(progress_data)
        
        # Collect the failed symbols
        return self.collect_all_symbols(resume=True)
    
    def export_collection_report(self, output_file: str = "logs/collection_report.json") -> None:
        """Export detailed collection report."""
        self.progress_tracker.export_progress_report(output_file)
        self.logger.info(f"Collection report exported to {output_file}")
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate that all components are properly configured."""
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check Alpha Vantage connection
        try:
            if not self.alpha_vantage_client.test_connection():
                validation_results["valid"] = False
                validation_results["issues"].append("Alpha Vantage API connection failed")
        except Exception as e:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Alpha Vantage connection error: {e}")
        
        # Check symbols file
        try:
            symbols = self._load_symbols()
            if len(symbols) == 0:
                validation_results["valid"] = False
                validation_results["issues"].append("No symbols found in symbols file")
            elif len(symbols) < 50:
                validation_results["warnings"].append(f"Only {len(symbols)} symbols found, expected ~100")
        except Exception as e:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Symbols file error: {e}")
        
        # Check output directories
        output_dir = Path(self.config.output_base_dir)
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                validation_results["warnings"].append(f"Created output directory: {output_dir}")
            except Exception as e:
                validation_results["valid"] = False
                validation_results["issues"].append(f"Cannot create output directory: {e}")
        
        # Check logs directory
        logs_dir = Path("logs")
        if not logs_dir.exists():
            try:
                logs_dir.mkdir(parents=True, exist_ok=True)
                validation_results["warnings"].append(f"Created logs directory: {logs_dir}")
            except Exception as e:
                validation_results["warnings"].append(f"Cannot create logs directory: {e}")
        
        return validation_results