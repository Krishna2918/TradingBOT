#!/usr/bin/env python3
"""
Graceful shutdown script for the continuous data collection system.

This script provides a safe way to stop the running collection system,
ensuring all data is saved and workers are properly terminated.

Usage:
    python stop_collection.py [--force] [--timeout SECONDS]

Requirements: 1.1, 1.5, 8.4
"""

import asyncio
import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Add the continuous_data_collection package to the path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core.config import ConfigLoader
from continuous_data_collection.core.logging_config import setup_logging
from continuous_data_collection.core.state_manager import StateManager
from continuous_data_collection.core.exceptions import StateError


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stop the continuous data collection system gracefully",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stop_collection.py
    python stop_collection.py --force
    python stop_collection.py --timeout 300
        """
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force immediate shutdown without waiting for current tasks"
    )
    
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=120,
        help="Maximum time to wait for graceful shutdown (seconds, default: 120)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/development.yaml",
        help="Configuration file path (default: config/development.yaml)"
    )
    
    parser.add_argument(
        "--save-state", "-s",
        action="store_true",
        help="Force save current state before shutdown"
    )
    
    return parser.parse_args()


def find_collection_process():
    """Find running collection process."""
    import psutil
    
    collection_processes = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('start_collection.py' in arg for arg in cmdline):
                    collection_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': ' '.join(cmdline)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
    except Exception as e:
        logging.warning(f"Error searching for processes: {e}")
    
    return collection_processes


def send_shutdown_signal(pid: int, force: bool = False) -> bool:
    """Send shutdown signal to process."""
    import psutil
    
    try:
        process = psutil.Process(pid)
        
        if force:
            # Send SIGKILL for immediate termination
            logging.info(f"Sending SIGKILL to process {pid}")
            process.kill()
        else:
            # Send SIGTERM for graceful shutdown
            logging.info(f"Sending SIGTERM to process {pid}")
            process.terminate()
        
        return True
        
    except psutil.NoSuchProcess:
        logging.warning(f"Process {pid} not found")
        return False
    except psutil.AccessDenied:
        logging.error(f"Access denied when trying to terminate process {pid}")
        return False
    except Exception as e:
        logging.error(f"Error terminating process {pid}: {e}")
        return False


def wait_for_process_termination(pid: int, timeout: int) -> bool:
    """Wait for process to terminate."""
    import psutil
    
    try:
        process = psutil.Process(pid)
        
        # Wait for process to terminate
        process.wait(timeout=timeout)
        return True
        
    except psutil.TimeoutExpired:
        logging.warning(f"Process {pid} did not terminate within {timeout} seconds")
        return False
    except psutil.NoSuchProcess:
        # Process already terminated
        return True
    except Exception as e:
        logging.error(f"Error waiting for process termination: {e}")
        return False


async def save_current_state(config_path: str):
    """Save current system state before shutdown."""
    try:
        logging.info("Attempting to save current system state...")
        
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        # Create state manager
        state_manager = StateManager(config)
        
        # Try to load current state to verify it exists
        current_state = await state_manager.load_state()
        if current_state:
            # Create a backup
            backup_id = await state_manager.backup_state()
            logging.info(f"Created state backup: {backup_id}")
        else:
            logging.warning("No current state found to save")
            
    except Exception as e:
        logging.error(f"Failed to save state: {e}")
        raise StateError(f"State saving failed: {e}")


def print_shutdown_banner():
    """Print shutdown banner."""
    print("\n" + "=" * 60)
    print("CONTINUOUS DATA COLLECTION SYSTEM - SHUTDOWN")
    print("=" * 60)


async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(level="INFO")
        logger = logging.getLogger(__name__)
        
        print_shutdown_banner()
        
        # Check if psutil is available
        try:
            import psutil
        except ImportError:
            print("❌ psutil package is required for process management")
            print("Install with: pip install psutil")
            sys.exit(1)
        
        # Save state if requested
        if args.save_state:
            try:
                await save_current_state(args.config)
                print("✅ System state saved successfully")
            except Exception as e:
                print(f"⚠️  Failed to save state: {e}")
                if not args.force:
                    response = input("Continue with shutdown anyway? (y/N): ")
                    if response.lower() != 'y':
                        sys.exit(1)
        
        # Find running collection processes
        print("Searching for running collection processes...")
        processes = find_collection_process()
        
        if not processes:
            print("✅ No running collection processes found")
            return
        
        print(f"Found {len(processes)} collection process(es):")
        for proc in processes:
            print(f"  PID {proc['pid']}: {proc['name']}")
            print(f"    Command: {proc['cmdline']}")
        
        # Confirm shutdown unless force is specified
        if not args.force:
            response = input(f"\nShutdown {len(processes)} process(es)? (y/N): ")
            if response.lower() != 'y':
                print("Shutdown cancelled")
                return
        
        # Shutdown processes
        for proc in processes:
            pid = proc['pid']
            print(f"\nShutting down process {pid}...")
            
            # Send shutdown signal
            if send_shutdown_signal(pid, args.force):
                if not args.force:
                    # Wait for graceful shutdown
                    print(f"Waiting up to {args.timeout} seconds for graceful shutdown...")
                    if wait_for_process_termination(pid, args.timeout):
                        print(f"✅ Process {pid} terminated gracefully")
                    else:
                        print(f"⚠️  Process {pid} did not terminate gracefully, forcing...")
                        if send_shutdown_signal(pid, force=True):
                            if wait_for_process_termination(pid, 10):
                                print(f"✅ Process {pid} terminated forcefully")
                            else:
                                print(f"❌ Failed to terminate process {pid}")
                        else:
                            print(f"❌ Failed to force terminate process {pid}")
                else:
                    # Force shutdown - wait briefly
                    time.sleep(2)
                    if wait_for_process_termination(pid, 5):
                        print(f"✅ Process {pid} terminated")
                    else:
                        print(f"❌ Process {pid} may still be running")
            else:
                print(f"❌ Failed to send shutdown signal to process {pid}")
        
        print("\n" + "=" * 60)
        print("SHUTDOWN COMPLETE")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Shutdown interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Shutdown failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())