#!/usr/bin/env python3
"""
Emergency recovery script for the continuous data collection system.

This script provides disaster recovery capabilities including state restoration,
data integrity checks, and system repair functions.

Usage:
    python emergency_recovery.py [--action ACTION] [--backup-id ID]

Requirements: 1.1, 1.5, 8.4
"""

import asyncio
import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the continuous_data_collection package to the path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core.config import ConfigLoader
from continuous_data_collection.core.logging_config import setup_logging
from continuous_data_collection.core.state_manager import StateManager
from continuous_data_collection.core.disaster_recovery import DisasterRecovery
from continuous_data_collection.storage.parquet_storage import ParquetStorage
from continuous_data_collection.core.exceptions import (
    StateError, RecoveryError, ConfigurationError
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Emergency recovery for continuous data collection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Actions:
    diagnose        - Diagnose system issues and data integrity
    list-backups    - List available state backups
    restore-state   - Restore system state from backup
    repair-data     - Repair corrupted data files
    reset-system    - Reset system to initial state (DESTRUCTIVE)
    validate-config - Validate system configuration
    cleanup-temp    - Clean up temporary files and corrupted data

Examples:
    python emergency_recovery.py --action diagnose
    python emergency_recovery.py --action list-backups
    python emergency_recovery.py --action restore-state --backup-id 20241028_143022
    python emergency_recovery.py --action repair-data --force
        """
    )
    
    parser.add_argument(
        "--action", "-a",
        choices=[
            "diagnose", "list-backups", "restore-state", "repair-data",
            "reset-system", "validate-config", "cleanup-temp"
        ],
        required=True,
        help="Recovery action to perform"
    )
    
    parser.add_argument(
        "--backup-id", "-b",
        type=str,
        help="Backup ID for restore operations"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/development.yaml",
        help="Configuration file path (default: config/development.yaml)"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force operation without confirmation prompts"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for recovery reports"
    )
    
    return parser.parse_args()


def print_recovery_banner():
    """Print emergency recovery banner."""
    print("\n" + "🚨" * 20)
    print("EMERGENCY RECOVERY SYSTEM")
    print("🚨" * 20)
    print("⚠️  Use with caution - some operations are destructive!")
    print("=" * 60 + "\n")


async def diagnose_system(config_path: str, output_dir: Optional[str]) -> Dict[str, any]:
    """Diagnose system issues and generate report."""
    print("🔍 SYSTEM DIAGNOSIS")
    print("=" * 40)
    
    diagnosis = {
        "timestamp": datetime.utcnow().isoformat(),
        "issues": [],
        "warnings": [],
        "recommendations": []
    }
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        # Check configuration validity
        print("Checking configuration...")
        config_errors = config_loader.validate_config(config)
        if config_errors:
            diagnosis["issues"].extend([f"Config: {error}" for error in config_errors])
            print(f"  ❌ Configuration errors found: {len(config_errors)}")
        else:
            print("  ✅ Configuration is valid")
        
        # Check directory structure
        print("Checking directory structure...")
        required_dirs = [
            config.storage.data_directory,
            config.storage.state_directory,
            config.storage.backup_directory,
            "logs"
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                diagnosis["issues"].append(f"Missing directory: {dir_path}")
                print(f"  ❌ Missing: {dir_path}")
            elif not path.is_dir():
                diagnosis["issues"].append(f"Not a directory: {dir_path}")
                print(f"  ❌ Not a directory: {dir_path}")
            else:
                print(f"  ✅ Found: {dir_path}")
        
        # Check state files
        print("Checking state files...")
        state_manager = StateManager(config)
        
        try:
            current_state = await state_manager.load_state()
            if current_state:
                print("  ✅ Current state file is valid")
                
                # Check state consistency
                total_stocks = (len(current_state.completed_stocks) + 
                              len(current_state.failed_stocks) + 
                              len(current_state.pending_stocks) + 
                              len(current_state.in_progress_stocks))
                
                if total_stocks != current_state.total_target_stocks:
                    diagnosis["warnings"].append(
                        f"State inconsistency: counted {total_stocks} stocks, "
                        f"expected {current_state.total_target_stocks}"
                    )
                    print(f"  ⚠️  State count mismatch")
                
            else:
                diagnosis["warnings"].append("No current state file found")
                print("  ⚠️  No current state file")
                
        except Exception as e:
            diagnosis["issues"].append(f"State file error: {e}")
            print(f"  ❌ State file error: {e}")
        
        # Check data files
        print("Checking data files...")
        storage = ParquetStorage(config)
        
        try:
            storage_stats = await storage.get_storage_stats()
            print(f"  ✅ Found {storage_stats.get('file_count', 0)} data files")
            
            # Check for corrupted files (sample check)
            data_dir = Path(config.storage.data_directory)
            if data_dir.exists():
                parquet_files = list(data_dir.glob("*.parquet"))
                corrupted_count = 0
                
                # Check first 10 files for corruption
                for file_path in parquet_files[:10]:
                    try:
                        import pandas as pd
                        pd.read_parquet(file_path, nrows=1)
                    except Exception:
                        corrupted_count += 1
                
                if corrupted_count > 0:
                    diagnosis["issues"].append(f"Found {corrupted_count} corrupted data files (sample)")
                    print(f"  ❌ Found corrupted files")
                
        except Exception as e:
            diagnosis["issues"].append(f"Data storage error: {e}")
            print(f"  ❌ Data storage error: {e}")
        
        # Check system resources
        print("Checking system resources...")
        try:
            import psutil
            
            # Disk space
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 1:  # Less than 1GB free
                diagnosis["issues"].append(f"Low disk space: {free_gb:.1f}GB free")
                print(f"  ❌ Low disk space: {free_gb:.1f}GB")
            else:
                print(f"  ✅ Disk space: {free_gb:.1f}GB free")
            
            # Memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                diagnosis["warnings"].append(f"High memory usage: {memory.percent:.1f}%")
                print(f"  ⚠️  High memory usage: {memory.percent:.1f}%")
            else:
                print(f"  ✅ Memory usage: {memory.percent:.1f}%")
                
        except ImportError:
            diagnosis["warnings"].append("psutil not available for resource checking")
            print("  ⚠️  Cannot check system resources (psutil not installed)")
        
        # Generate recommendations
        if diagnosis["issues"]:
            diagnosis["recommendations"].append("Fix configuration and directory issues first")
            diagnosis["recommendations"].append("Consider running 'repair-data' action")
        
        if diagnosis["warnings"]:
            diagnosis["recommendations"].append("Review warnings and consider preventive actions")
        
        if not diagnosis["issues"] and not diagnosis["warnings"]:
            diagnosis["recommendations"].append("System appears healthy")
        
        # Save diagnosis report
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            report_file = output_path / f"diagnosis_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            
            import json
            with open(report_file, 'w') as f:
                json.dump(diagnosis, f, indent=2)
            
            print(f"\n📄 Diagnosis report saved to: {report_file}")
        
        return diagnosis
        
    except Exception as e:
        diagnosis["issues"].append(f"Diagnosis failed: {e}")
        print(f"❌ Diagnosis failed: {e}")
        return diagnosis


async def list_backups(config_path: str) -> List[Dict[str, any]]:
    """List available state backups."""
    print("📋 AVAILABLE BACKUPS")
    print("=" * 40)
    
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        disaster_recovery = DisasterRecovery(config)
        backups = await disaster_recovery.list_available_backups()
        
        if not backups:
            print("No backups found")
            return []
        
        print(f"Found {len(backups)} backup(s):\n")
        
        for backup in backups:
            print(f"  ID: {backup['backup_id']}")
            print(f"  Created: {backup['created_at']}")
            print(f"  Size: {backup.get('size_mb', 'Unknown')} MB")
            print(f"  Valid: {'✅' if backup.get('is_valid', False) else '❌'}")
            print()
        
        return backups
        
    except Exception as e:
        print(f"❌ Failed to list backups: {e}")
        return []


async def restore_state(config_path: str, backup_id: str, force: bool, dry_run: bool) -> bool:
    """Restore system state from backup."""
    print(f"🔄 RESTORING STATE FROM BACKUP: {backup_id}")
    print("=" * 50)
    
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        disaster_recovery = DisasterRecovery(config)
        
        # Validate backup exists
        backups = await disaster_recovery.list_available_backups()
        backup_info = next((b for b in backups if b['backup_id'] == backup_id), None)
        
        if not backup_info:
            print(f"❌ Backup {backup_id} not found")
            return False
        
        print(f"Backup found:")
        print(f"  Created: {backup_info['created_at']}")
        print(f"  Size: {backup_info.get('size_mb', 'Unknown')} MB")
        print(f"  Valid: {'✅' if backup_info.get('is_valid', False) else '❌'}")
        
        if not backup_info.get('is_valid', False):
            print("⚠️  Warning: Backup may be corrupted")
            if not force:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    return False
        
        if dry_run:
            print("🔍 DRY RUN - Would restore state from this backup")
            return True
        
        # Confirm restoration
        if not force:
            print("\n⚠️  WARNING: This will overwrite the current system state!")
            response = input("Continue with restoration? (y/N): ")
            if response.lower() != 'y':
                print("Restoration cancelled")
                return False
        
        # Perform restoration
        print("Restoring state...")
        success = await disaster_recovery.restore_from_backup(backup_id)
        
        if success:
            print("✅ State restored successfully")
            return True
        else:
            print("❌ State restoration failed")
            return False
            
    except Exception as e:
        print(f"❌ Restoration failed: {e}")
        return False


async def repair_data(config_path: str, force: bool, dry_run: bool) -> bool:
    """Repair corrupted data files."""
    print("🔧 DATA REPAIR")
    print("=" * 40)
    
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        disaster_recovery = DisasterRecovery(config)
        
        # Find corrupted files
        print("Scanning for corrupted data files...")
        corrupted_files = await disaster_recovery.find_corrupted_files()
        
        if not corrupted_files:
            print("✅ No corrupted files found")
            return True
        
        print(f"Found {len(corrupted_files)} corrupted file(s):")
        for file_path in corrupted_files:
            print(f"  - {file_path}")
        
        if dry_run:
            print("🔍 DRY RUN - Would repair these files")
            return True
        
        # Confirm repair
        if not force:
            print(f"\n⚠️  This will delete {len(corrupted_files)} corrupted files!")
            response = input("Continue with repair? (y/N): ")
            if response.lower() != 'y':
                print("Repair cancelled")
                return False
        
        # Perform repair
        print("Repairing data files...")
        repaired_count = await disaster_recovery.repair_corrupted_files(corrupted_files)
        
        print(f"✅ Repaired {repaired_count} files")
        return True
        
    except Exception as e:
        print(f"❌ Data repair failed: {e}")
        return False


async def reset_system(config_path: str, force: bool, dry_run: bool) -> bool:
    """Reset system to initial state (DESTRUCTIVE)."""
    print("💥 SYSTEM RESET (DESTRUCTIVE)")
    print("=" * 40)
    
    if dry_run:
        print("🔍 DRY RUN - Would reset system to initial state")
        print("This would:")
        print("  - Delete all state files")
        print("  - Clear all progress data")
        print("  - Remove temporary files")
        print("  - Keep collected data files")
        return True
    
    # Multiple confirmations for destructive operation
    if not force:
        print("⚠️  WARNING: This will PERMANENTLY delete all progress!")
        print("This operation will:")
        print("  - Delete all state files")
        print("  - Clear all progress data")
        print("  - Remove temporary files")
        print("  - Keep collected data files (safe)")
        
        response = input("\nType 'RESET' to confirm: ")
        if response != 'RESET':
            print("Reset cancelled")
            return False
        
        response = input("Are you absolutely sure? (yes/no): ")
        if response.lower() != 'yes':
            print("Reset cancelled")
            return False
    
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        disaster_recovery = DisasterRecovery(config)
        
        print("Resetting system...")
        success = await disaster_recovery.reset_system_state()
        
        if success:
            print("✅ System reset completed")
            return True
        else:
            print("❌ System reset failed")
            return False
            
    except Exception as e:
        print(f"❌ System reset failed: {e}")
        return False


async def cleanup_temp_files(config_path: str, dry_run: bool) -> bool:
    """Clean up temporary files and corrupted data."""
    print("🧹 CLEANUP TEMPORARY FILES")
    print("=" * 40)
    
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        # Find temporary files
        temp_files = []
        
        # Look for common temporary file patterns
        for pattern in ["*.tmp", "*.temp", "*.lock", "*.partial"]:
            temp_files.extend(Path(".").rglob(pattern))
        
        # Look for old log files
        log_dir = Path("logs")
        if log_dir.exists():
            old_logs = [f for f in log_dir.glob("*.log.*") if f.stat().st_mtime < (datetime.now().timestamp() - 7*24*3600)]
            temp_files.extend(old_logs)
        
        if not temp_files:
            print("✅ No temporary files found")
            return True
        
        print(f"Found {len(temp_files)} temporary file(s):")
        for file_path in temp_files:
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"  - {file_path} ({size_mb:.1f} MB)")
        
        if dry_run:
            print("🔍 DRY RUN - Would clean up these files")
            return True
        
        # Clean up files
        print("Cleaning up temporary files...")
        cleaned_count = 0
        
        for file_path in temp_files:
            try:
                file_path.unlink()
                cleaned_count += 1
            except Exception as e:
                print(f"  ⚠️  Failed to delete {file_path}: {e}")
        
        print(f"✅ Cleaned up {cleaned_count} files")
        return True
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False


async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(level="INFO")
        
        print_recovery_banner()
        
        # Check if config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
        
        # Execute requested action
        success = True
        
        if args.action == "diagnose":
            diagnosis = await diagnose_system(str(config_path), args.output_dir)
            if diagnosis["issues"]:
                print(f"\n❌ Found {len(diagnosis['issues'])} issue(s)")
                success = False
            elif diagnosis["warnings"]:
                print(f"\n⚠️  Found {len(diagnosis['warnings'])} warning(s)")
            else:
                print("\n✅ System appears healthy")
        
        elif args.action == "list-backups":
            backups = await list_backups(str(config_path))
            success = len(backups) > 0
        
        elif args.action == "restore-state":
            if not args.backup_id:
                print("❌ --backup-id is required for restore-state action")
                sys.exit(1)
            success = await restore_state(str(config_path), args.backup_id, args.force, args.dry_run)
        
        elif args.action == "repair-data":
            success = await repair_data(str(config_path), args.force, args.dry_run)
        
        elif args.action == "reset-system":
            success = await reset_system(str(config_path), args.force, args.dry_run)
        
        elif args.action == "cleanup-temp":
            success = await cleanup_temp_files(str(config_path), args.dry_run)
        
        elif args.action == "validate-config":
            config_loader = ConfigLoader()
            config = config_loader.load_config(str(config_path))
            errors = config_loader.validate_config(config)
            if errors:
                print("❌ Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
                success = False
            else:
                print("✅ Configuration is valid")
        
        if not success:
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Recovery interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Recovery failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())