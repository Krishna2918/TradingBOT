#!/usr/bin/env python3
"""
Configuration migration and upgrade tool for continuous data collection system.

This script handles configuration file migrations, upgrades, and compatibility
checks when updating the system to newer versions.

Usage:
    python config_migration_tool.py [--source CONFIG] [--target-version VERSION]

Requirements: 1.1, 8.1
"""

import argparse
import json
import logging
import shutil
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the continuous_data_collection package to the path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core.logging_config import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Configuration migration tool for continuous data collection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Migration Operations:
    migrate         - Migrate configuration to newer version
    validate        - Validate configuration compatibility
    backup          - Create configuration backup
    restore         - Restore from configuration backup
    upgrade         - Upgrade configuration with new features

Examples:
    python config_migration_tool.py --source config/development.yaml --target-version 2.0
    python config_migration_tool.py --operation validate --source config/production.yaml
    python config_migration_tool.py --operation backup --source config/development.yaml
        """
    )
    
    parser.add_argument(
        "--operation", "-o",
        choices=["migrate", "validate", "backup", "restore", "upgrade"],
        default="migrate",
        help="Migration operation to perform (default: migrate)"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Source configuration file path"
    )
    
    parser.add_argument(
        "--target-version", "-t",
        type=str,
        default="2.0",
        help="Target configuration version (default: 2.0)"
    )
    
    parser.add_argument(
        "--output", "-out",
        type=str,
        help="Output file for migrated configuration (default: auto-generated)"
    )
    
    parser.add_argument(
        "--backup-dir", "-b",
        type=str,
        default="config/backups",
        help="Backup directory (default: config/backups)"
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force migration even if validation fails"
    )
    
    return parser.parse_args()


def detect_config_version(config_data: Dict[str, Any]) -> str:
    """Detect the version of a configuration file."""
    # Check for explicit version field
    if "version" in config_data:
        return str(config_data["version"])
    
    # Detect version based on structure and fields
    if "api" in config_data and "alpha_vantage" in config_data["api"]:
        if "rate_limiting" in config_data["api"]["alpha_vantage"]:
            return "2.0"  # Version 2.0 has rate_limiting section
        elif "requests_per_minute" in config_data["api"]["alpha_vantage"]:
            return "1.5"  # Version 1.5 has requests_per_minute
        else:
            return "1.0"  # Basic version
    
    # Check for monitoring section structure
    if "monitoring" in config_data:
        if "alerting" in config_data["monitoring"]:
            return "2.0"  # Version 2.0 has alerting
        elif "health_check_interval" in config_data["monitoring"]:
            return "1.5"  # Version 1.5 has health checks
        else:
            return "1.0"  # Basic monitoring
    
    # Default to version 1.0 if cannot determine
    return "1.0"


def validate_config_compatibility(config_data: Dict[str, Any], target_version: str) -> Dict[str, Any]:
    """Validate configuration compatibility with target version."""
    validation = {
        "compatible": True,
        "issues": [],
        "warnings": [],
        "required_changes": [],
        "deprecated_fields": [],
        "new_fields": []
    }
    
    current_version = detect_config_version(config_data)
    
    # Version-specific validation
    if target_version == "2.0":
        # Check for deprecated fields in 2.0
        deprecated_fields = [
            ("api.alpha_vantage.requests_per_minute", "Use api.alpha_vantage.rate_limiting.requests_per_minute instead"),
            ("collection.simple_retry", "Use collection.retry_strategy instead"),
            ("monitoring.basic_logging", "Use monitoring.logging_config instead")
        ]
        
        for field_path, message in deprecated_fields:
            if _get_nested_value(config_data, field_path) is not None:
                validation["deprecated_fields"].append({
                    "field": field_path,
                    "message": message
                })
                validation["warnings"].append(f"Deprecated field '{field_path}': {message}")
        
        # Check for required new fields in 2.0
        required_new_fields = [
            "api.alpha_vantage.rate_limiting",
            "monitoring.alerting",
            "collection.retry_strategy"
        ]
        
        for field_path in required_new_fields:
            if _get_nested_value(config_data, field_path) is None:
                validation["new_fields"].append(field_path)
                validation["required_changes"].append(f"Add required field: {field_path}")
        
        # Check for structural changes
        if "storage" in config_data:
            if "backup_retention" not in config_data["storage"]:
                validation["required_changes"].append("Add storage.backup_retention configuration")
        
        if "collection" in config_data:
            if "performance_tuning" not in config_data["collection"]:
                validation["required_changes"].append("Add collection.performance_tuning configuration")
    
    elif target_version == "1.5":
        # Validation for 1.5 upgrade
        if current_version == "1.0":
            required_fields = [
                "monitoring.health_check_interval",
                "api.alpha_vantage.requests_per_minute"
            ]
            
            for field_path in required_fields:
                if _get_nested_value(config_data, field_path) is None:
                    validation["required_changes"].append(f"Add field: {field_path}")
    
    # Set compatibility based on issues
    if validation["required_changes"] or validation["issues"]:
        validation["compatible"] = False
    
    return validation


def migrate_config_1_0_to_1_5(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate configuration from version 1.0 to 1.5."""
    migrated = config_data.copy()
    
    # Add version field
    migrated["version"] = "1.5"
    
    # Add health check interval to monitoring
    if "monitoring" not in migrated:
        migrated["monitoring"] = {}
    
    if "health_check_interval" not in migrated["monitoring"]:
        migrated["monitoring"]["health_check_interval"] = 30
    
    # Add requests per minute to API configuration
    if "api" in migrated and "alpha_vantage" in migrated["api"]:
        if "requests_per_minute" not in migrated["api"]["alpha_vantage"]:
            # Calculate based on number of API keys (75 RPM per key)
            api_keys = migrated["api"]["alpha_vantage"].get("api_keys", [])
            rpm = len(api_keys) * 75 if api_keys else 75
            migrated["api"]["alpha_vantage"]["requests_per_minute"] = min(rpm, 300)  # Cap at 300
    
    # Add retry configuration
    if "collection" not in migrated:
        migrated["collection"] = {}
    
    if "retry_delay" not in migrated["collection"]:
        migrated["collection"]["retry_delay"] = 1.0
    
    if "max_retries" not in migrated["collection"]:
        migrated["collection"]["max_retries"] = 3
    
    return migrated


def migrate_config_1_5_to_2_0(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate configuration from version 1.5 to 2.0."""
    migrated = config_data.copy()
    
    # Update version
    migrated["version"] = "2.0"
    
    # Migrate API configuration
    if "api" in migrated and "alpha_vantage" in migrated["api"]:
        av_config = migrated["api"]["alpha_vantage"]
        
        # Move requests_per_minute to rate_limiting section
        if "requests_per_minute" in av_config:
            rpm = av_config.pop("requests_per_minute")
            av_config["rate_limiting"] = {
                "requests_per_minute": rpm,
                "burst_limit": rpm // 4,  # Allow 25% burst
                "backoff_strategy": "exponential"
            }
        
        # Add circuit breaker configuration
        if "circuit_breaker" not in av_config:
            av_config["circuit_breaker"] = {
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "half_open_max_calls": 3
            }
    
    # Migrate collection configuration
    if "collection" in migrated:
        collection = migrated["collection"]
        
        # Create retry strategy from simple retry settings
        if "retry_strategy" not in collection:
            collection["retry_strategy"] = {
                "max_retries": collection.get("max_retries", 3),
                "base_delay": collection.get("retry_delay", 1.0),
                "max_delay": 60.0,
                "backoff_multiplier": 2.0,
                "jitter": True
            }
        
        # Add performance tuning
        if "performance_tuning" not in collection:
            collection["performance_tuning"] = {
                "adaptive_batch_size": True,
                "dynamic_worker_scaling": False,
                "memory_optimization": True,
                "connection_pooling": True
            }
    
    # Migrate monitoring configuration
    if "monitoring" in migrated:
        monitoring = migrated["monitoring"]
        
        # Add alerting configuration
        if "alerting" not in monitoring:
            monitoring["alerting"] = {
                "enabled": True,
                "channels": ["log"],
                "thresholds": {
                    "error_rate": 0.1,
                    "throughput_min": 10.0,
                    "memory_usage": 0.9,
                    "disk_usage": 0.95
                }
            }
        
        # Add logging configuration
        if "logging_config" not in monitoring:
            monitoring["logging_config"] = {
                "structured_logging": True,
                "log_rotation": True,
                "max_log_size": "100MB",
                "backup_count": 5
            }
    
    # Add storage enhancements
    if "storage" in migrated:
        storage = migrated["storage"]
        
        # Add backup retention
        if "backup_retention" not in storage:
            storage["backup_retention"] = {
                "max_backups": 10,
                "retention_days": 30,
                "compression": True
            }
        
        # Add data validation settings
        if "data_validation" not in storage:
            storage["data_validation"] = {
                "integrity_checks": True,
                "quality_scoring": True,
                "anomaly_detection": True
            }
    
    return migrated


def migrate_configuration(config_data: Dict[str, Any], target_version: str) -> Dict[str, Any]:
    """Migrate configuration to target version."""
    current_version = detect_config_version(config_data)
    
    if current_version == target_version:
        return config_data
    
    migrated = config_data.copy()
    
    # Migration path: 1.0 -> 1.5 -> 2.0
    if current_version == "1.0":
        if target_version in ["1.5", "2.0"]:
            migrated = migrate_config_1_0_to_1_5(migrated)
            current_version = "1.5"
    
    if current_version == "1.5" and target_version == "2.0":
        migrated = migrate_config_1_5_to_2_0(migrated)
    
    return migrated


def create_config_backup(source_path: str, backup_dir: str) -> str:
    """Create a backup of the configuration file."""
    source = Path(source_path)
    backup_directory = Path(backup_dir)
    
    # Create backup directory if it doesn't exist
    backup_directory.mkdir(parents=True, exist_ok=True)
    
    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{source.stem}_backup_{timestamp}{source.suffix}"
    backup_path = backup_directory / backup_filename
    
    # Copy file
    shutil.copy2(source, backup_path)
    
    return str(backup_path)


def restore_config_backup(backup_path: str, target_path: str) -> bool:
    """Restore configuration from backup."""
    try:
        backup = Path(backup_path)
        target = Path(target_path)
        
        if not backup.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Create target directory if needed
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy backup to target
        shutil.copy2(backup, target)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to restore backup: {e}")
        return False


def upgrade_config_with_defaults(config_data: Dict[str, Any], target_version: str) -> Dict[str, Any]:
    """Upgrade configuration with new default values and features."""
    upgraded = config_data.copy()
    
    # Add version if not present
    if "version" not in upgraded:
        upgraded["version"] = target_version
    
    # Version 2.0 specific upgrades
    if target_version == "2.0":
        # Add new features that weren't in migration
        if "features" not in upgraded:
            upgraded["features"] = {
                "data_quality_monitoring": True,
                "predictive_scaling": False,
                "advanced_retry_logic": True,
                "real_time_alerts": True,
                "performance_analytics": True
            }
        
        # Add security configuration
        if "security" not in upgraded:
            upgraded["security"] = {
                "api_key_rotation": False,
                "encrypted_storage": False,
                "audit_logging": True,
                "rate_limit_enforcement": True
            }
        
        # Add experimental features
        if "experimental" not in upgraded:
            upgraded["experimental"] = {
                "ml_based_optimization": False,
                "distributed_collection": False,
                "real_time_streaming": False
            }
    
    return upgraded


def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """Get nested value from dictionary using dot notation."""
    keys = path.split('.')
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    
    return current


def format_migration_report(migration_results: Dict[str, Any]) -> str:
    """Format migration report for display."""
    lines = [
        "CONFIGURATION MIGRATION REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    # Migration summary
    lines.extend([
        "MIGRATION SUMMARY:",
        f"  Source Version:    {migration_results.get('source_version', 'Unknown')}",
        f"  Target Version:    {migration_results.get('target_version', 'Unknown')}",
        f"  Migration Status:  {'✅ Success' if migration_results.get('success', False) else '❌ Failed'}",
        ""
    ])
    
    # Validation results
    validation = migration_results.get('validation', {})
    if validation:
        lines.extend([
            "VALIDATION RESULTS:",
            f"  Compatible:        {'✅ Yes' if validation.get('compatible', False) else '❌ No'}",
            f"  Issues:            {len(validation.get('issues', []))}",
            f"  Warnings:          {len(validation.get('warnings', []))}",
            f"  Required Changes:  {len(validation.get('required_changes', []))}",
            ""
        ])
        
        # List issues
        if validation.get('issues'):
            lines.append("ISSUES:")
            for issue in validation['issues']:
                lines.append(f"  ❌ {issue}")
            lines.append("")
        
        # List warnings
        if validation.get('warnings'):
            lines.append("WARNINGS:")
            for warning in validation['warnings']:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")
        
        # List required changes
        if validation.get('required_changes'):
            lines.append("REQUIRED CHANGES:")
            for change in validation['required_changes']:
                lines.append(f"  🔧 {change}")
            lines.append("")
    
    # Migration changes
    changes = migration_results.get('changes', {})
    if changes:
        lines.extend([
            "MIGRATION CHANGES:",
            f"  Fields Added:      {len(changes.get('added_fields', []))}",
            f"  Fields Modified:   {len(changes.get('modified_fields', []))}",
            f"  Fields Removed:    {len(changes.get('removed_fields', []))}",
            ""
        ])
        
        if changes.get('added_fields'):
            lines.append("ADDED FIELDS:")
            for field in changes['added_fields']:
                lines.append(f"  ➕ {field}")
            lines.append("")
        
        if changes.get('modified_fields'):
            lines.append("MODIFIED FIELDS:")
            for field in changes['modified_fields']:
                lines.append(f"  🔄 {field}")
            lines.append("")
        
        if changes.get('removed_fields'):
            lines.append("REMOVED FIELDS:")
            for field in changes['removed_fields']:
                lines.append(f"  ➖ {field}")
            lines.append("")
    
    # Backup information
    if migration_results.get('backup_created'):
        lines.extend([
            "BACKUP INFORMATION:",
            f"  Backup Created:    ✅ Yes",
            f"  Backup Location:   {migration_results.get('backup_path', 'Unknown')}",
            ""
        ])
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(level="INFO")
        
        print("🔄 CONFIGURATION MIGRATION TOOL")
        print("=" * 50)
        
        # Check if source file exists
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"❌ Source configuration file not found: {source_path}")
            sys.exit(1)
        
        # Load source configuration
        try:
            with open(source_path, 'r') as f:
                if source_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            print(f"✅ Loaded configuration from: {source_path}")
            
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            sys.exit(1)
        
        # Detect current version
        current_version = detect_config_version(config_data)
        print(f"📋 Detected configuration version: {current_version}")
        
        # Initialize migration results
        migration_results = {
            "source_version": current_version,
            "target_version": args.target_version,
            "success": False,
            "validation": {},
            "changes": {},
            "backup_created": False
        }
        
        # Perform operation
        if args.operation == "validate":
            print(f"🔍 Validating compatibility with version {args.target_version}...")
            validation = validate_config_compatibility(config_data, args.target_version)
            migration_results["validation"] = validation
            migration_results["success"] = validation["compatible"]
            
            if validation["compatible"]:
                print("✅ Configuration is compatible")
            else:
                print("❌ Configuration requires changes for compatibility")
        
        elif args.operation == "backup":
            print("💾 Creating configuration backup...")
            backup_path = create_config_backup(str(source_path), args.backup_dir)
            migration_results["backup_created"] = True
            migration_results["backup_path"] = backup_path
            migration_results["success"] = True
            print(f"✅ Backup created: {backup_path}")
        
        elif args.operation == "restore":
            if not args.output:
                print("❌ --output is required for restore operation")
                sys.exit(1)
            
            print(f"🔄 Restoring configuration from backup...")
            success = restore_config_backup(str(source_path), args.output)
            migration_results["success"] = success
            
            if success:
                print(f"✅ Configuration restored to: {args.output}")
            else:
                print("❌ Failed to restore configuration")
        
        elif args.operation in ["migrate", "upgrade"]:
            # Validate first
            print(f"🔍 Validating compatibility with version {args.target_version}...")
            validation = validate_config_compatibility(config_data, args.target_version)
            migration_results["validation"] = validation
            
            if not validation["compatible"] and not args.force:
                print("❌ Configuration is not compatible. Use --force to proceed anyway.")
                sys.exit(1)
            
            # Create backup
            print("💾 Creating backup before migration...")
            backup_path = create_config_backup(str(source_path), args.backup_dir)
            migration_results["backup_created"] = True
            migration_results["backup_path"] = backup_path
            
            # Perform migration
            print(f"🔄 Migrating configuration to version {args.target_version}...")
            
            if args.operation == "migrate":
                migrated_config = migrate_configuration(config_data, args.target_version)
            else:  # upgrade
                migrated_config = upgrade_config_with_defaults(
                    migrate_configuration(config_data, args.target_version),
                    args.target_version
                )
            
            # Determine output path
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = source_path.parent / f"{source_path.stem}_v{args.target_version}{source_path.suffix}"
            
            if args.dry_run:
                print(f"🔍 DRY RUN - Would save migrated configuration to: {output_path}")
                migration_results["success"] = True
            else:
                # Save migrated configuration
                try:
                    with open(output_path, 'w') as f:
                        if output_path.suffix.lower() in ['.yaml', '.yml']:
                            yaml.dump(migrated_config, f, default_flow_style=False, indent=2)
                        else:
                            json.dump(migrated_config, f, indent=2)
                    
                    print(f"✅ Migrated configuration saved to: {output_path}")
                    migration_results["success"] = True
                    
                except Exception as e:
                    print(f"❌ Failed to save migrated configuration: {e}")
                    migration_results["success"] = False
        
        # Generate and display report
        report = format_migration_report(migration_results)
        print("\n" + report)
        
        # Exit with appropriate code
        if not migration_results["success"]:
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()