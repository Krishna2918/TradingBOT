"""
Start Training Data Collection

Simple script to begin the comprehensive 20-year data collection process
for AI training using the 4-key Alpha Vantage system.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.data_collection.training_data_orchestrator import TrainingDataOrchestrator

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
        log_dir / f'data_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Specific logger for data collection
    data_logger = logging.getLogger('data_collection')
    data_file_handler = logging.FileHandler(log_dir / 'data_collection.log', encoding='utf-8')
    data_file_handler.setLevel(logging.INFO)
    data_file_handler.setFormatter(detailed_formatter)
    data_logger.addHandler(data_file_handler)

async def main():
    """Main data collection process"""
    print("Starting Training Data Collection")
    print("=" * 60)
    
    try:
        # Initialize orchestrator
        orchestrator = TrainingDataOrchestrator()
        
        # Show initial status
        print("\nInitial Status:")
        print(orchestrator.generate_collection_report())
        
        # Confirm start
        print("\nWARNING: This will collect 20+ years of data for 200 US stocks")
        print("   This process may take several hours and use API quotas")
        
        response = input("\nDo you want to continue? (y/N): ").strip().lower()
        if response != 'y':
            print("Collection cancelled by user")
            return
        
        print("\nStarting comprehensive data collection...")
        print("   You can monitor progress in the logs directory")
        
        # Start collection
        results = await orchestrator.start_full_collection()
        
        # Show results
        print("\n" + "=" * 60)
        print("COLLECTION RESULTS")
        print("=" * 60)
        
        if results['success']:
            print("SUCCESS: Data collection completed successfully!")
        else:
            print("WARNING: Data collection completed with issues")
        
        print(f"Duration: {results.get('duration', 0):.0f} seconds")
        print(f"Dataset ready for training: {'Yes' if results.get('dataset_ready', False) else 'No'}")
        
        # Show phase results
        print("\nPhase Results:")
        for phase, result in results.get('phases', {}).items():
            status = "SUCCESS" if result.get('success', False) else "FAILED"
            duration = result.get('duration', 0)
            print(f"  {status} {phase}: {duration:.0f}s")
        
        # Show final validation
        final_validation = results.get('final_validation', {})
        if final_validation:
            print(f"\nFinal Validation:")
            print(f"  Overall Score: {final_validation.get('overall_score', 0):.3f}")
            print(f"  Training Ready: {'Yes' if final_validation.get('training_ready', False) else 'No'}")
            
            recommendations = final_validation.get('recommendations', [])
            if recommendations:
                print("  Recommendations:")
                for rec in recommendations:
                    print(f"    - {rec}")
        
        # Final status report
        print("\nFinal Status Report:")
        print(orchestrator.generate_collection_report())
        
        if results.get('dataset_ready', False):
            print("\nSUCCESS: Dataset is ready for AI training!")
            print("   Next step: Run the AI training pipeline")
        else:
            print("\nWARNING: Dataset may not be complete")
            print("   Review the logs and consider re-running failed collections")
    
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        print(f"\nCollection failed: {e}")
        logging.error(f"Collection failed: {e}", exc_info=True)

def show_status():
    """Show current collection status without starting new collection"""
    print("Current Collection Status")
    print("=" * 60)
    
    try:
        orchestrator = TrainingDataOrchestrator()
        print(orchestrator.generate_collection_report())
    except Exception as e:
        print(f"Error getting status: {e}")

if __name__ == '__main__':
    # Setup logging
    setup_logging()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        show_status()
    else:
        # Run main collection
        asyncio.run(main())