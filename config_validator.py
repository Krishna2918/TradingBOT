#!/usr/bin/env python3
"""
Configuration validation and testing utility for continuous data collection system.

This script validates system configuration files, tests configuration settings,
and provides detailed feedback on configuration issues.

Usage:
    python config_validator.py [--config CONFIG_FILE] [--test-connections]

Requirements: 1.1, 8.1
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the continuous_data_collection package to the path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core.config import ConfigLoader, SystemConfig
from continuous_data_collection.core.logging_config import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Configuration validator for continuous data collection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validation Types:
    basic           - Basic configuration structure validation
    comprehensive   - Full validation including file system checks
    connections     - Test API connections and network connectivity
    performance     - Validate performance-related settings

Examples:
    python config_validator.py --config config/development.yaml
    python config_validator.py --config config/production.yaml --test-connections
    python config_validator.py --validation comprehensive --output validation_report.json
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/development.yaml",
        help="Configuration file path to validate (default: config/development.yaml)"
    )
    
    parser.add_argument(
        "--validation", "-v",
        choices=["basic", "comprehensive", "connections", "performance"],
        default="comprehensive",
        help="Type of validation to perform (default: comprehensive)"
    )
    
    parser.add_argument(
        "--test-connections", "-t",
        action="store_true",
        help="Test API connections and network connectivity"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for validation results"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json", "txt"],
        default="txt",
        help="Output format (default: txt)"
    )
    
    parser.add_argument(
        "--fix-issues", "-x",
        action="store_true",
        help="Attempt to automatically fix configuration issues"
    )
    
    return parser.parse_args()


def validate_basic_structure(config: SystemConfig) -> Dict[str, Any]:
    """Perform basic configuration structure validation."""
    validation = {
        "validation_type": "basic",
        "issues": [],
        "warnings": [],
        "passed_checks": [],
        "overall_valid": True
    }
    
    # Check required sections
    required_sections = ["storage", "collection", "monitoring", "api"]
    
    for section in required_sections:
        if hasattr(config, section):
            validation["passed_checks"].append(f"Required section '{section}' present")
        else:
            validation["issues"].append(f"Missing required section: {section}")
            validation["overall_valid"] = False
    
    # Storage configuration validation
    if hasattr(config, 'storage'):
        storage = config.storage
        
        # Required storage fields
        required_storage_fields = ["data_directory", "state_directory", "backup_directory"]
        for field in required_storage_fields:
            if hasattr(storage, field) and getattr(storage, field):
                validation["passed_checks"].append(f"Storage field '{field}' configured")
            else:
                validation["issues"].append(f"Missing or empty storage field: {field}")
                validation["overall_valid"] = False
        
        # Validate file format
        if hasattr(storage, 'file_format'):
            if storage.file_format.lower() in ['parquet', 'csv', 'json']:
                validation["passed_checks"].append(f"Valid file format: {storage.file_format}")
            else:
                validation["warnings"].append(f"Unusual file format: {storage.file_format}")
    
    # Collection configuration validation
    if hasattr(config, 'collection'):
        collection = config.collection
        
        # Worker count validation
        if hasattr(collection, 'max_workers'):
            if 1 <= collection.max_workers <= 20:
                validation["passed_checks"].append(f"Valid worker count: {collection.max_workers}")
            else:
                validation["warnings"].append(f"Worker count may be suboptimal: {collection.max_workers}")
        
        # Batch size validation
        if hasattr(collection, 'batch_size'):
            if 1 <= collection.batch_size <= 100:
                validation["passed_checks"].append(f"Valid batch size: {collection.batch_size}")
            else:
                validation["warnings"].append(f"Batch size may be suboptimal: {collection.batch_size}")
    
    # API configuration validation
    if hasattr(config, 'api'):
        api = config.api
        
        # Alpha Vantage configuration
        if hasattr(api, 'alpha_vantage'):
            av = api.alpha_vantage
            if hasattr(av, 'api_keys') and av.api_keys:
                validation["passed_checks"].append(f"Alpha Vantage API keys configured: {len(av.api_keys)} keys")
                
                # Check API key format (basic validation)
                for i, key in enumerate(av.api_keys):
                    if len(key) < 10:
                        validation["warnings"].append(f"API key {i+1} appears to be too short")
            else:
                validation["issues"].append("No Alpha Vantage API keys configured")
                validation["overall_valid"] = False
    
    return validation


def validate_comprehensive(config: SystemConfig, config_path: str) -> Dict[str, Any]:
    """Perform comprehensive configuration validation including file system checks."""
    validation = {
        "validation_type": "comprehensive",
        "issues": [],
        "warnings": [],
        "passed_checks": [],
        "overall_valid": True,
        "file_system_checks": {},
        "permission_checks": {}
    }
    
    # Start with basic validation
    basic_validation = validate_basic_structure(config)
    validation["issues"].extend(basic_validation["issues"])
    validation["warnings"].extend(basic_validation["warnings"])
    validation["passed_checks"].extend(basic_validation["passed_checks"])
    validation["overall_valid"] = basic_validation["overall_valid"]
    
    # File system validation
    if hasattr(config, 'storage'):
        storage = config.storage
        
        directories_to_check = [
            ("data_directory", getattr(storage, 'data_directory', None)),
            ("state_directory", getattr(storage, 'state_directory', None)),
            ("backup_directory", getattr(storage, 'backup_directory', None))
        ]
        
        for dir_name, dir_path in directories_to_check:
            if dir_path:
                path = Path(dir_path)
                
                validation["file_system_checks"][dir_name] = {
                    "path": str(path),
                    "exists": path.exists(),
                    "is_directory": path.is_dir() if path.exists() else False,
                    "is_writable": False,
                    "is_readable": False
                }
                
                if path.exists():
                    if path.is_dir():
                        validation["passed_checks"].append(f"Directory exists: {dir_path}")
                        
                        # Check permissions
                        try:
                            # Test write permission
                            test_file = path / ".test_write"
                            test_file.touch()
                            test_file.unlink()
                            validation["file_system_checks"][dir_name]["is_writable"] = True
                            validation["passed_checks"].append(f"Directory writable: {dir_path}")
                        except Exception:
                            validation["issues"].append(f"Directory not writable: {dir_path}")
                            validation["overall_valid"] = False
                        
                        try:
                            # Test read permission
                            list(path.iterdir())
                            validation["file_system_checks"][dir_name]["is_readable"] = True
                            validation["passed_checks"].append(f"Directory readable: {dir_path}")
                        except Exception:
                            validation["issues"].append(f"Directory not readable: {dir_path}")
                            validation["overall_valid"] = False
                    else:
                        validation["issues"].append(f"Path exists but is not a directory: {dir_path}")
                        validation["overall_valid"] = False
                else:
                    validation["warnings"].append(f"Directory does not exist (will be created): {dir_path}")
    
    # Configuration file validation
    config_file = Path(config_path)
    validation["file_system_checks"]["config_file"] = {
        "path": str(config_file),
        "exists": config_file.exists(),
        "is_readable": False,
        "size_bytes": config_file.stat().st_size if config_file.exists() else 0
    }
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                f.read()
            validation["file_system_checks"]["config_file"]["is_readable"] = True
            validation["passed_checks"].append(f"Configuration file readable: {config_path}")
        except Exception as e:
            validation["issues"].append(f"Cannot read configuration file: {e}")
            validation["overall_valid"] = False
    
    # Validate numeric ranges and constraints
    if hasattr(config, 'collection'):
        collection = config.collection
        
        # Timeout validations
        if hasattr(collection, 'request_timeout'):
            if collection.request_timeout < 1 or collection.request_timeout > 300:
                validation["warnings"].append(f"Request timeout may be suboptimal: {collection.request_timeout}s")
        
        if hasattr(collection, 'retry_delay'):
            if collection.retry_delay < 0.1 or collection.retry_delay > 60:
                validation["warnings"].append(f"Retry delay may be suboptimal: {collection.retry_delay}s")
    
    # Validate monitoring configuration
    if hasattr(config, 'monitoring'):
        monitoring = config.monitoring
        
        if hasattr(monitoring, 'progress_update_interval'):
            if monitoring.progress_update_interval < 1 or monitoring.progress_update_interval > 1000:
                validation["warnings"].append(f"Progress update interval may be suboptimal: {monitoring.progress_update_interval}")
        
        if hasattr(monitoring, 'health_check_interval'):
            if monitoring.health_check_interval < 1 or monitoring.health_check_interval > 300:
                validation["warnings"].append(f"Health check interval may be suboptimal: {monitoring.health_check_interval}s")
    
    return validation


async def test_api_connections(config: SystemConfig) -> Dict[str, Any]:
    """Test API connections and network connectivity."""
    validation = {
        "validation_type": "connections",
        "issues": [],
        "warnings": [],
        "passed_checks": [],
        "overall_valid": True,
        "connection_tests": {}
    }
    
    # Test Alpha Vantage API
    if hasattr(config, 'api') and hasattr(config.api, 'alpha_vantage'):
        av_config = config.api.alpha_vantage
        
        if hasattr(av_config, 'api_keys') and av_config.api_keys:
            import urllib.request
            import json
            import time
            
            for i, api_key in enumerate(av_config.api_keys[:2]):  # Test first 2 keys
                test_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=1min&apikey={api_key}"
                
                try:
                    start_time = time.time()
                    response = urllib.request.urlopen(test_url, timeout=30)
                    response_time = time.time() - start_time
                    
                    if response.getcode() == 200:
                        data = json.loads(response.read().decode())
                        
                        if "Error Message" in data:
                            validation["issues"].append(f"Alpha Vantage API key {i+1} returned error: {data['Error Message']}")
                            validation["overall_valid"] = False
                        elif "Note" in data:
                            validation["warnings"].append(f"Alpha Vantage API key {i+1} rate limited: {data['Note']}")
                        else:
                            validation["passed_checks"].append(f"Alpha Vantage API key {i+1} working (response: {response_time:.1f}s)")
                        
                        validation["connection_tests"][f"alpha_vantage_key_{i+1}"] = {
                            "success": "Error Message" not in data,
                            "response_time": response_time,
                            "status_code": response.getcode()
                        }
                    else:
                        validation["issues"].append(f"Alpha Vantage API key {i+1} returned status: {response.getcode()}")
                        validation["overall_valid"] = False
                        
                except Exception as e:
                    validation["issues"].append(f"Alpha Vantage API key {i+1} connection failed: {e}")
                    validation["overall_valid"] = False
                    validation["connection_tests"][f"alpha_vantage_key_{i+1}"] = {
                        "success": False,
                        "error": str(e)
                    }
    
    # Test yfinance connectivity
    try:
        import yfinance as yf
        
        # Test with a simple stock
        ticker = yf.Ticker("AAPL")
        start_time = time.time()
        hist = ticker.history(period="1d")
        response_time = time.time() - start_time
        
        if not hist.empty:
            validation["passed_checks"].append(f"yfinance connectivity working (response: {response_time:.1f}s)")
            validation["connection_tests"]["yfinance"] = {
                "success": True,
                "response_time": response_time,
                "records_returned": len(hist)
            }
        else:
            validation["warnings"].append("yfinance returned empty data")
            validation["connection_tests"]["yfinance"] = {
                "success": False,
                "error": "Empty data returned"
            }
            
    except ImportError:
        validation["warnings"].append("yfinance package not installed")
        validation["connection_tests"]["yfinance"] = {
            "success": False,
            "error": "Package not installed"
        }
    except Exception as e:
        validation["issues"].append(f"yfinance connection failed: {e}")
        validation["connection_tests"]["yfinance"] = {
            "success": False,
            "error": str(e)
        }
    
    # Test general internet connectivity
    test_hosts = ["8.8.8.8", "1.1.1.1", "www.google.com"]
    
    for host in test_hosts:
        try:
            import socket
            start_time = time.time()
            
            if host.replace(".", "").isdigit():  # IP address
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                result = sock.connect_ex((host, 53))  # DNS port
                sock.close()
                success = result == 0
            else:  # Hostname
                socket.gethostbyname(host)
                success = True
            
            response_time = time.time() - start_time
            
            if success:
                validation["passed_checks"].append(f"Network connectivity to {host} working")
                validation["connection_tests"][f"network_{host}"] = {
                    "success": True,
                    "response_time": response_time
                }
            else:
                validation["issues"].append(f"Cannot connect to {host}")
                validation["overall_valid"] = False
                
        except Exception as e:
            validation["issues"].append(f"Network test to {host} failed: {e}")
            validation["overall_valid"] = False
            validation["connection_tests"][f"network_{host}"] = {
                "success": False,
                "error": str(e)
            }
    
    return validation


def validate_performance_settings(config: SystemConfig) -> Dict[str, Any]:
    """Validate performance-related configuration settings."""
    validation = {
        "validation_type": "performance",
        "issues": [],
        "warnings": [],
        "passed_checks": [],
        "overall_valid": True,
        "performance_analysis": {}
    }
    
    # Analyze worker configuration
    if hasattr(config, 'collection'):
        collection = config.collection
        
        # Get system info for recommendations
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            validation["performance_analysis"]["system_resources"] = {
                "cpu_count": cpu_count,
                "memory_gb": memory_gb
            }
            
            # Worker count analysis
            if hasattr(collection, 'max_workers'):
                workers = collection.max_workers
                
                if workers > cpu_count * 2:
                    validation["warnings"].append(f"Worker count ({workers}) is high for CPU count ({cpu_count})")
                elif workers < max(1, cpu_count // 2):
                    validation["warnings"].append(f"Worker count ({workers}) may be too low for CPU count ({cpu_count})")
                else:
                    validation["passed_checks"].append(f"Worker count ({workers}) appropriate for system")
                
                validation["performance_analysis"]["worker_analysis"] = {
                    "configured_workers": workers,
                    "recommended_min": max(1, cpu_count // 2),
                    "recommended_max": cpu_count * 2
                }
            
            # Memory usage estimation
            if hasattr(collection, 'max_workers') and hasattr(collection, 'batch_size'):
                estimated_memory_per_worker = 0.1  # GB per worker (rough estimate)
                total_estimated_memory = collection.max_workers * estimated_memory_per_worker
                
                if total_estimated_memory > memory_gb * 0.8:
                    validation["warnings"].append(f"Estimated memory usage ({total_estimated_memory:.1f}GB) may exceed available memory")
                
                validation["performance_analysis"]["memory_analysis"] = {
                    "estimated_usage_gb": total_estimated_memory,
                    "available_gb": memory_gb,
                    "usage_percentage": (total_estimated_memory / memory_gb) * 100
                }
                
        except ImportError:
            validation["warnings"].append("psutil not available for system resource analysis")
    
    # Rate limiting analysis
    if hasattr(config, 'api') and hasattr(config.api, 'alpha_vantage'):
        av = config.api.alpha_vantage
        
        if hasattr(av, 'requests_per_minute') and hasattr(av, 'api_keys'):
            rpm = av.requests_per_minute
            key_count = len(av.api_keys) if av.api_keys else 1
            
            # Alpha Vantage limit is 75 requests per minute per key
            max_theoretical_rpm = key_count * 75
            
            if rpm > max_theoretical_rpm:
                validation["issues"].append(f"Configured RPM ({rpm}) exceeds API limits ({max_theoretical_rpm})")
                validation["overall_valid"] = False
            elif rpm > max_theoretical_rpm * 0.9:
                validation["warnings"].append(f"Configured RPM ({rpm}) is close to API limits")
            else:
                validation["passed_checks"].append(f"Rate limiting configuration within API limits")
            
            validation["performance_analysis"]["rate_limiting"] = {
                "configured_rpm": rpm,
                "max_theoretical_rpm": max_theoretical_rpm,
                "utilization_percentage": (rpm / max_theoretical_rpm) * 100
            }
    
    # Timeout analysis
    if hasattr(config, 'collection'):
        collection = config.collection
        
        if hasattr(collection, 'request_timeout'):
            timeout = collection.request_timeout
            
            if timeout < 5:
                validation["warnings"].append(f"Request timeout ({timeout}s) may be too short for API calls")
            elif timeout > 60:
                validation["warnings"].append(f"Request timeout ({timeout}s) may be too long")
            else:
                validation["passed_checks"].append(f"Request timeout ({timeout}s) is reasonable")
    
    return validation


def attempt_config_fixes(validation_results: List[Dict[str, Any]], config_path: str) -> Dict[str, Any]:
    """Attempt to automatically fix configuration issues."""
    fix_results = {
        "fixes_attempted": [],
        "fixes_successful": [],
        "fixes_failed": [],
        "manual_intervention_required": []
    }
    
    # Create missing directories
    for validation in validation_results:
        if validation.get("validation_type") == "comprehensive":
            file_system_checks = validation.get("file_system_checks", {})
            
            for dir_name, dir_info in file_system_checks.items():
                if dir_name.endswith("_directory") and not dir_info.get("exists", True):
                    try:
                        dir_path = Path(dir_info["path"])
                        dir_path.mkdir(parents=True, exist_ok=True)
                        
                        fix_results["fixes_attempted"].append(f"Create directory: {dir_path}")
                        
                        if dir_path.exists():
                            fix_results["fixes_successful"].append(f"Successfully created: {dir_path}")
                        else:
                            fix_results["fixes_failed"].append(f"Failed to create: {dir_path}")
                            
                    except Exception as e:
                        fix_results["fixes_failed"].append(f"Error creating {dir_info['path']}: {e}")
    
    # Issues that require manual intervention
    manual_issues = [
        "API key configuration",
        "Network connectivity",
        "Permission issues",
        "Invalid configuration values"
    ]
    
    for validation in validation_results:
        for issue in validation.get("issues", []):
            if any(manual_issue in issue for manual_issue in manual_issues):
                fix_results["manual_intervention_required"].append(issue)
    
    return fix_results


def format_validation_output(validation_results: List[Dict[str, Any]], format_type: str) -> str:
    """Format validation output for display or file export."""
    if format_type == "json":
        return json.dumps(validation_results, indent=2, default=str)
    
    else:  # txt format
        lines = []
        
        lines.extend([
            "CONFIGURATION VALIDATION REPORT",
            "=" * 60,
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            ""
        ])
        
        # Summary
        total_issues = sum(len(v.get("issues", [])) for v in validation_results)
        total_warnings = sum(len(v.get("warnings", [])) for v in validation_results)
        total_passed = sum(len(v.get("passed_checks", [])) for v in validation_results)
        overall_valid = all(v.get("overall_valid", False) for v in validation_results)
        
        lines.extend([
            "SUMMARY:",
            f"  Overall Status:   {'✅ VALID' if overall_valid else '❌ INVALID'}",
            f"  Issues Found:     {total_issues}",
            f"  Warnings:         {total_warnings}",
            f"  Checks Passed:    {total_passed}",
            ""
        ])
        
        # Individual validation results
        for validation in validation_results:
            validation_type = validation.get("validation_type", "unknown").upper()
            
            lines.extend([
                f"{validation_type} VALIDATION:",
                "-" * 40
            ])
            
            # Issues
            issues = validation.get("issues", [])
            if issues:
                lines.append("❌ ISSUES:")
                for issue in issues:
                    lines.append(f"  • {issue}")
                lines.append("")
            
            # Warnings
            warnings = validation.get("warnings", [])
            if warnings:
                lines.append("⚠️  WARNINGS:")
                for warning in warnings:
                    lines.append(f"  • {warning}")
                lines.append("")
            
            # Passed checks
            passed = validation.get("passed_checks", [])
            if passed:
                lines.append("✅ PASSED CHECKS:")
                for check in passed[:5]:  # Show first 5
                    lines.append(f"  • {check}")
                if len(passed) > 5:
                    lines.append(f"  ... and {len(passed) - 5} more")
                lines.append("")
            
            # Special sections for specific validation types
            if validation_type == "CONNECTIONS":
                connection_tests = validation.get("connection_tests", {})
                if connection_tests:
                    lines.append("🌐 CONNECTION TEST RESULTS:")
                    for test_name, result in connection_tests.items():
                        status = "✅" if result.get("success", False) else "❌"
                        response_time = result.get("response_time", 0)
                        lines.append(f"  {status} {test_name}: {response_time:.1f}s" if response_time else f"  {status} {test_name}")
                    lines.append("")
            
            elif validation_type == "PERFORMANCE":
                perf_analysis = validation.get("performance_analysis", {})
                if perf_analysis:
                    lines.append("⚡ PERFORMANCE ANALYSIS:")
                    
                    if "system_resources" in perf_analysis:
                        sys_res = perf_analysis["system_resources"]
                        lines.append(f"  System: {sys_res.get('cpu_count', 0)} CPUs, {sys_res.get('memory_gb', 0):.1f}GB RAM")
                    
                    if "worker_analysis" in perf_analysis:
                        worker_analysis = perf_analysis["worker_analysis"]
                        lines.append(f"  Workers: {worker_analysis.get('configured_workers', 0)} configured "
                                   f"(recommended: {worker_analysis.get('recommended_min', 0)}-{worker_analysis.get('recommended_max', 0)})")
                    
                    lines.append("")
        
        return "\n".join(lines)


async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(level="WARNING")
        
        print("🔍 CONFIGURATION VALIDATOR")
        print("=" * 50)
        
        # Check if config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
        
        print(f"Validating configuration: {config_path}")
        
        # Load configuration
        try:
            config_loader = ConfigLoader()
            config = config_loader.load_config(str(config_path))
            print("✅ Configuration loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            sys.exit(1)
        
        # Run validations
        validation_results = []
        
        if args.validation == "basic":
            validation_results.append(validate_basic_structure(config))
        elif args.validation == "comprehensive":
            validation_results.append(validate_comprehensive(config, str(config_path)))
        elif args.validation == "connections" or args.test_connections:
            validation_results.append(await test_api_connections(config))
        elif args.validation == "performance":
            validation_results.append(validate_performance_settings(config))
        
        # If comprehensive or test_connections flag, run additional validations
        if args.validation == "comprehensive":
            if args.test_connections:
                print("Running connection tests...")
                validation_results.append(await test_api_connections(config))
            
            print("Running performance validation...")
            validation_results.append(validate_performance_settings(config))
        
        # Attempt fixes if requested
        if args.fix_issues:
            print("Attempting automatic fixes...")
            fix_results = attempt_config_fixes(validation_results, str(config_path))
            validation_results.append(fix_results)
        
        # Format and display/save results
        output_text = format_validation_output(validation_results, args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"✅ Validation report saved to: {args.output}")
        else:
            print("\n" + output_text)
        
        # Exit with error code if validation failed
        overall_valid = all(v.get("overall_valid", False) for v in validation_results if "overall_valid" in v)
        if not overall_valid:
            print("\n⚠️  Configuration validation failed - please fix the issues above")
            sys.exit(1)
        else:
            print("\n✅ Configuration validation passed")
        
    except Exception as e:
        print(f"\n❌ Configuration validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Import datetime here to avoid issues
    from datetime import datetime
    
    # Run the async main function
    asyncio.run(main())