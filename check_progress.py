#!/usr/bin/env python3
"""
Real-time monitoring script for the continuous data collection system.

This script provides real-time progress monitoring, statistics display,
and system health information for the running collection system.

Usage:
    python check_progress.py [--refresh SECONDS] [--detailed] [--export FORMAT]

Requirements: 4.1, 4.2, 10.1, 10.5
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Add the continuous_data_collection package to the path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core.config import ConfigLoader
from continuous_data_collection.core.logging_config import setup_logging
from continuous_data_collection.core.state_manager import StateManager
from continuous_data_collection.core.exceptions import StateError


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor continuous data collection system progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python check_progress.py
    python check_progress.py --refresh 5 --detailed
    python check_progress.py --export json --output progress_report.json
        """
    )
    
    parser.add_argument(
        "--refresh", "-r",
        type=int,
        default=10,
        help="Refresh interval in seconds (default: 10, 0 for single check)"
    )
    
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed statistics and worker information"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/development.yaml",
        help="Configuration file path (default: config/development.yaml)"
    )
    
    parser.add_argument(
        "--export", "-e",
        choices=["json", "csv", "txt"],
        help="Export progress data to file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for export (default: auto-generated)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - minimal output"
    )
    
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Watch mode - continuous monitoring with screen clearing"
    )
    
    return parser.parse_args()


def clear_screen():
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_eta(eta_datetime: Optional[datetime]) -> str:
    """Format ETA in human-readable format."""
    if not eta_datetime:
        return "Unknown"
    
    now = datetime.utcnow()
    if eta_datetime <= now:
        return "Complete"
    
    delta = eta_datetime - now
    total_seconds = delta.total_seconds()
    
    if total_seconds < 3600:
        minutes = total_seconds / 60
        return f"{minutes:.0f} minutes"
    elif total_seconds < 86400:
        hours = total_seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = total_seconds / 86400
        return f"{days:.1f} days"


def create_progress_bar(completed: int, total: int, width: int = 50) -> str:
    """Create a text-based progress bar."""
    if total == 0:
        return "[" + " " * width + "] 0%"
    
    percentage = completed / total
    filled = int(width * percentage)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percentage:.1%}"


async def load_system_state(config_path: str) -> Optional[Dict[str, Any]]:
    """Load current system state."""
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        # Create state manager
        state_manager = StateManager(config)
        
        # Load current state
        state = await state_manager.load_state()
        if not state:
            return None
        
        # Convert to dictionary for easier handling
        return {
            "completed_stocks": len(state.completed_stocks),
            "failed_stocks": len(state.failed_stocks),
            "pending_stocks": len(state.pending_stocks),
            "in_progress_stocks": len(state.in_progress_stocks),
            "total_target_stocks": state.total_target_stocks,
            "collection_start_time": state.collection_start_time,
            "last_save_time": state.last_save_time,
            "completion_percentage": state.get_completion_percentage(),
            "total_processed": state.get_total_processed(),
            "remaining_count": state.get_remaining_count()
        }
        
    except Exception as e:
        logging.error(f"Failed to load system state: {e}")
        return None


def calculate_statistics(state_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate additional statistics from state data."""
    stats = {}
    
    # Basic counts
    completed = state_data.get("completed_stocks", 0)
    failed = state_data.get("failed_stocks", 0)
    pending = state_data.get("pending_stocks", 0)
    in_progress = state_data.get("in_progress_stocks", 0)
    total_target = state_data.get("total_target_stocks", 0)
    
    # Success rate
    total_processed = completed + failed
    success_rate = (completed / total_processed) if total_processed > 0 else 0
    
    # Runtime calculation
    start_time = state_data.get("collection_start_time")
    runtime_seconds = 0
    if start_time:
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        runtime_seconds = (datetime.utcnow() - start_time.replace(tzinfo=None)).total_seconds()
    
    # Throughput calculation
    throughput = (completed / (runtime_seconds / 60)) if runtime_seconds > 0 else 0
    
    # ETA calculation
    eta = None
    if throughput > 0 and pending > 0:
        remaining_minutes = pending / throughput
        eta = datetime.utcnow() + timedelta(minutes=remaining_minutes)
    
    stats.update({
        "success_rate": success_rate,
        "runtime_seconds": runtime_seconds,
        "throughput_per_minute": throughput,
        "eta": eta,
        "total_processed": total_processed
    })
    
    return stats


def display_progress_summary(state_data: Dict[str, Any], stats: Dict[str, Any], quiet: bool = False):
    """Display progress summary."""
    if quiet:
        # Minimal output for quiet mode
        completed = state_data.get("completed_stocks", 0)
        total = state_data.get("total_target_stocks", 0)
        percentage = state_data.get("completion_percentage", 0)
        print(f"{completed}/{total} ({percentage:.1f}%) - {stats.get('throughput_per_minute', 0):.1f}/min")
        return
    
    print("\n" + "=" * 70)
    print("CONTINUOUS DATA COLLECTION - PROGRESS REPORT")
    print("=" * 70)
    
    # Progress overview
    completed = state_data.get("completed_stocks", 0)
    failed = state_data.get("failed_stocks", 0)
    pending = state_data.get("pending_stocks", 0)
    in_progress = state_data.get("in_progress_stocks", 0)
    total = state_data.get("total_target_stocks", 0)
    
    print(f"\n📊 PROGRESS OVERVIEW")
    print(f"   Total Target:     {total:,}")
    print(f"   Completed:        {completed:,}")
    print(f"   Failed:           {failed:,}")
    print(f"   In Progress:      {in_progress:,}")
    print(f"   Pending:          {pending:,}")
    
    # Progress bar
    progress_bar = create_progress_bar(completed, total)
    print(f"\n   {progress_bar}")
    print(f"   Completion:       {state_data.get('completion_percentage', 0):.2f}%")
    
    # Performance metrics
    print(f"\n⚡ PERFORMANCE METRICS")
    print(f"   Success Rate:     {stats.get('success_rate', 0):.1%}")
    print(f"   Runtime:          {format_duration(stats.get('runtime_seconds', 0))}")
    print(f"   Throughput:       {stats.get('throughput_per_minute', 0):.1f} stocks/minute")
    
    # ETA
    eta = stats.get('eta')
    eta_str = format_eta(eta)
    print(f"   ETA:              {eta_str}")
    
    # Timestamps
    start_time = state_data.get("collection_start_time")
    last_save = state_data.get("last_save_time")
    
    print(f"\n🕐 TIMESTAMPS")
    if start_time:
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        print(f"   Started:          {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    if last_save:
        if isinstance(last_save, str):
            last_save = datetime.fromisoformat(last_save.replace('Z', '+00:00'))
        print(f"   Last Save:        {last_save.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    print(f"   Current Time:     {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")


def display_detailed_info(state_data: Dict[str, Any], stats: Dict[str, Any]):
    """Display detailed system information."""
    print(f"\n📈 DETAILED STATISTICS")
    
    # Collection efficiency
    total_processed = stats.get('total_processed', 0)
    runtime_hours = stats.get('runtime_seconds', 0) / 3600
    
    if runtime_hours > 0:
        hourly_rate = total_processed / runtime_hours
        print(f"   Hourly Rate:      {hourly_rate:.1f} stocks/hour")
    
    # Failure analysis
    failed = state_data.get("failed_stocks", 0)
    if failed > 0:
        failure_rate = failed / total_processed if total_processed > 0 else 0
        print(f"   Failure Rate:     {failure_rate:.1%} ({failed} stocks)")
    
    # Remaining work estimation
    pending = state_data.get("pending_stocks", 0)
    throughput = stats.get('throughput_per_minute', 0)
    
    if throughput > 0 and pending > 0:
        remaining_hours = (pending / throughput) / 60
        print(f"   Remaining Work:   ~{remaining_hours:.1f} hours")


async def export_progress_data(state_data: Dict[str, Any], stats: Dict[str, Any], 
                              format_type: str, output_path: Optional[str]):
    """Export progress data to file."""
    # Combine all data
    export_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "state": state_data,
        "statistics": stats
    }
    
    # Generate output path if not provided
    if not output_path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"progress_report_{timestamp}.{format_type}"
    
    try:
        if format_type == "json":
            # Convert datetime objects to strings for JSON serialization
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=json_serializer)
        
        elif format_type == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                
                # Write state data
                for key, value in state_data.items():
                    writer.writerow([key, value])
                
                # Write statistics
                for key, value in stats.items():
                    if key != 'eta' or value is None:
                        writer.writerow([f"stats_{key}", value])
                    else:
                        writer.writerow([f"stats_{key}", value.isoformat()])
        
        elif format_type == "txt":
            with open(output_path, 'w') as f:
                f.write("CONTINUOUS DATA COLLECTION - PROGRESS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
                
                # Write state data
                f.write("STATE DATA:\n")
                for key, value in state_data.items():
                    f.write(f"  {key}: {value}\n")
                
                f.write("\nSTATISTICS:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
        
        print(f"✅ Progress data exported to: {output_path}")
        
    except Exception as e:
        print(f"❌ Failed to export data: {e}")


async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging (quiet mode)
        setup_logging(level="ERROR" if args.quiet else "WARNING")
        
        if not args.quiet:
            print("CONTINUOUS DATA COLLECTION - PROGRESS MONITOR")
            print("=" * 50)
        
        # Check if config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
        
        # Single check or continuous monitoring
        if args.refresh == 0:
            # Single check
            state_data = await load_system_state(str(config_path))
            if not state_data:
                print("❌ No system state found - collection may not be running")
                sys.exit(1)
            
            stats = calculate_statistics(state_data)
            display_progress_summary(state_data, stats, args.quiet)
            
            if args.detailed and not args.quiet:
                display_detailed_info(state_data, stats)
            
            if args.export:
                await export_progress_data(state_data, stats, args.export, args.output)
        
        else:
            # Continuous monitoring
            if not args.quiet:
                print(f"Monitoring every {args.refresh} seconds (Press Ctrl+C to stop)")
            
            try:
                while True:
                    if args.watch and not args.quiet:
                        clear_screen()
                    
                    state_data = await load_system_state(str(config_path))
                    if not state_data:
                        if not args.quiet:
                            print("⚠️  No system state found - collection may not be running")
                    else:
                        stats = calculate_statistics(state_data)
                        display_progress_summary(state_data, stats, args.quiet)
                        
                        if args.detailed and not args.quiet:
                            display_detailed_info(state_data, stats)
                    
                    if not args.quiet:
                        print(f"\nNext update in {args.refresh} seconds...")
                    
                    await asyncio.sleep(args.refresh)
                    
            except KeyboardInterrupt:
                if not args.quiet:
                    print("\n⚠️  Monitoring stopped by user")
        
    except Exception as e:
        print(f"\n❌ Progress check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())