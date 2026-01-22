#!/usr/bin/env python3
"""
System initialization script for the continuous data collection system.

This script initializes and starts the 24/7 continuous data collection system
with proper configuration loading, component initialization, and error handling.

Usage:
    python start_collection.py [--config CONFIG_FILE] [--log-level LEVEL]

Requirements: 1.1, 1.5, 8.4
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add the continuous_data_collection package to the path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core.config import ConfigLoader
from continuous_data_collection.core.logging_config import setup_logging
from continuous_data_collection.core.system_factory import SystemFactory
from continuous_data_collection.core.exceptions import (
    SystemInitializationError, ConfigurationError
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start the continuous data collection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start_collection.py
    python start_collection.py --config config/production.yaml
    python start_collection.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/development.yaml",
        help="Configuration file path (default: config/development.yaml)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--validate-only", "-v",
        action="store_true",
        help="Only validate configuration and exit"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Perform dry run without starting collection"
    )
    
    return parser.parse_args()


def validate_environment():
    """Validate system environment and requirements."""
    errors = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        errors.append("Python 3.8 or higher is required")
    
    # Check required directories exist
    required_dirs = ["continuous_data_collection", "config", "logs"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            errors.append(f"Required directory '{dir_name}' not found")
    
    # Check for required configuration files
    config_dir = Path("config")
    if config_dir.exists():
        required_configs = ["development.yaml", "production.yaml", "testing.yaml"]
        existing_configs = [f.name for f in config_dir.glob("*.yaml")]
        if not any(config in existing_configs for config in required_configs):
            errors.append("No valid configuration files found in config directory")
    
    return errors


async def initialize_system(config_path: str, log_level: str) -> Optional[object]:
    """Initialize the continuous collection system."""
    try:
        # Setup logging
        setup_logging(level=log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("CONTINUOUS DATA COLLECTION SYSTEM - STARTUP")
        logger.info("=" * 60)
        
        # Load configuration
        logger.info(f"Loading configuration from: {config_path}")
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        # Validate configuration
        logger.info("Validating system configuration...")
        validation_errors = config_loader.validate_config(config)
        if validation_errors:
            logger.error("Configuration validation failed:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            raise ConfigurationError(f"Configuration validation failed: {validation_errors}")
        
        logger.info("Configuration validation passed")
        
        # Create system components
        logger.info("Initializing system components...")
        system_factory = SystemFactory(config)
        collector = await system_factory.create_continuous_collector()
        
        logger.info("System initialization completed successfully")
        return collector
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"System initialization failed: {e}")
        raise SystemInitializationError(f"Failed to initialize system: {e}")


async def start_collection_system(collector):
    """Start the continuous collection system."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting continuous data collection...")
        logger.info("Press Ctrl+C to stop the system gracefully")
        
        # Start the collection process
        await collector.start_collection()
        
        logger.info("Collection completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Shutdown signal received, stopping collection...")
        await collector.stop_collection()
        logger.info("System stopped gracefully")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        try:
            await collector.stop_collection()
        except Exception as stop_error:
            logger.error(f"Error during shutdown: {stop_error}")
        raise


def print_system_info():
    """Print system information and startup banner."""
    print("\n" + "=" * 60)
    print("CONTINUOUS DATA COLLECTION SYSTEM")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {Path.cwd()}")
    print(f"Script Location: {Path(__file__).parent}")
    print("=" * 60 + "\n")


async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print system info
        print_system_info()
        
        # Validate environment
        print("Validating system environment...")
        env_errors = validate_environment()
        if env_errors:
            print("Environment validation failed:")
            for error in env_errors:
                print(f"  ❌ {error}")
            sys.exit(1)
        print("✅ Environment validation passed")
        
        # Check if config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
        print(f"✅ Configuration file found: {config_path}")
        
        # Initialize system
        print(f"Initializing system with log level: {args.log_level}")
        collector = await initialize_system(str(config_path), args.log_level)
        
        if args.validate_only:
            print("✅ Configuration validation completed successfully")
            return
        
        if args.dry_run:
            print("✅ Dry run completed - system would start normally")
            return
        
        # Start collection
        await start_collection_system(collector)
        
    except KeyboardInterrupt:
        print("\n⚠️  Startup interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())