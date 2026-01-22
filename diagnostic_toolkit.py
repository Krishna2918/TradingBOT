#!/usr/bin/env python3
"""
Comprehensive diagnostic toolkit for continuous data collection system.

This script provides troubleshooting utilities, system diagnostics,
and automated problem detection for the collection system.

Usage:
    python diagnostic_toolkit.py [--diagnostic TYPE] [--fix-issues]

Requirements: 4.1, 4.2, 10.1, 10.5
"""

import asyncio
import argparse
import json
import logging
import psutil
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the continuous_data_collection package to the path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core.config import ConfigLoader
from continuous_data_collection.core.logging_config import setup_logging
from continuous_data_collection.core.state_manager import StateManager
from continuous_data_collection.storage.parquet_storage import ParquetStorage


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnostic toolkit for continuous data collection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Diagnostic Types:
    system          - System health and resource diagnostics
    configuration   - Configuration validation and issues
    storage         - Storage system diagnostics
    network         - Network connectivity diagnostics
    performance     - Performance bottleneck diagnostics
    logs            - Log file analysis and error detection
    comprehensive   - Run all diagnostics

Examples:
    python diagnostic_toolkit.py --diagnostic system
    python diagnostic_toolkit.py --diagnostic comprehensive --fix-issues
    python diagnostic_toolkit.py --diagnostic logs --output diagnostic_report.json
        """
    )
    
    parser.add_argument(
        "--diagnostic", "-d",
        choices=["system", "configuration", "storage", "network", "performance", "logs", "comprehensive"],
        default="system",
        help="Type of diagnostic to run (default: system)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/development.yaml",
        help="Configuration file path (default: config/development.yaml)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for diagnostic results"
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
        help="Attempt to automatically fix detected issues"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with detailed information"
    )
    
    return parser.parse_args()


def run_system_diagnostics() -> Dict[str, Any]:
    """Run system health and resource diagnostics."""
    print("🔍 Running system diagnostics...")
    
    diagnostics = {
        "diagnostic_type": "system",
        "timestamp": datetime.utcnow().isoformat(),
        "system_info": {},
        "resource_usage": {},
        "processes": {},
        "issues": [],
        "recommendations": []
    }
    
    try:
        # System information
        diagnostics["system_info"] = {
            "platform": psutil.WINDOWS if psutil.WINDOWS else "Unix-like",
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "uptime_hours": (datetime.now().timestamp() - psutil.boot_time()) / 3600
        }
        
        # Resource usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        diagnostics["resource_usage"] = {
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
        }
        
        # Check for collection processes
        collection_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('start_collection.py' in arg for arg in cmdline):
                    collection_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'cmdline': ' '.join(cmdline)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        diagnostics["processes"] = {
            "collection_processes": collection_processes,
            "collection_running": len(collection_processes) > 0
        }
        
        # Identify issues
        if cpu_percent > 90:
            diagnostics["issues"].append({
                "severity": "high",
                "category": "performance",
                "message": f"High CPU usage: {cpu_percent:.1f}%",
                "recommendation": "Check for runaway processes or reduce system load"
            })
        
        if memory.percent > 90:
            diagnostics["issues"].append({
                "severity": "high",
                "category": "performance",
                "message": f"High memory usage: {memory.percent:.1f}%",
                "recommendation": "Free up memory or add more RAM"
            })
        
        if (disk.used / disk.total) > 0.95:
            diagnostics["issues"].append({
                "severity": "critical",
                "category": "storage",
                "message": f"Disk space critically low: {(disk.used/disk.total)*100:.1f}% used",
                "recommendation": "Free up disk space immediately"
            })
        
        # Check for multiple collection processes
        if len(collection_processes) > 1:
            diagnostics["issues"].append({
                "severity": "medium",
                "category": "process",
                "message": f"Multiple collection processes detected: {len(collection_processes)}",
                "recommendation": "Stop duplicate processes to avoid conflicts"
            })
        
        # Generate recommendations
        if not diagnostics["issues"]:
            diagnostics["recommendations"].append("System appears healthy")
        else:
            diagnostics["recommendations"].append("Address identified issues to improve system performance")
        
        print(f"✅ System diagnostics complete - found {len(diagnostics['issues'])} issues")
        
    except Exception as e:
        diagnostics["error"] = str(e)
        print(f"❌ System diagnostics failed: {e}")
    
    return diagnostics


def run_configuration_diagnostics(config_path: str) -> Dict[str, Any]:
    """Run configuration validation diagnostics."""
    print("🔍 Running configuration diagnostics...")
    
    diagnostics = {
        "diagnostic_type": "configuration",
        "timestamp": datetime.utcnow().isoformat(),
        "config_file": config_path,
        "validation_results": {},
        "file_checks": {},
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Check if config file exists
        config_file = Path(config_path)
        diagnostics["file_checks"]["config_exists"] = config_file.exists()
        
        if not config_file.exists():
            diagnostics["issues"].append({
                "severity": "critical",
                "category": "configuration",
                "message": f"Configuration file not found: {config_path}",
                "recommendation": "Create configuration file or specify correct path"
            })
            return diagnostics
        
        # Load and validate configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        validation_errors = config_loader.validate_config(config)
        diagnostics["validation_results"] = {
            "is_valid": len(validation_errors) == 0,
            "errors": validation_errors
        }
        
        # Check required directories
        required_dirs = [
            config.storage.data_directory,
            config.storage.state_directory,
            config.storage.backup_directory
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            diagnostics["file_checks"][f"dir_{dir_path}"] = {
                "exists": path.exists(),
                "is_directory": path.is_dir() if path.exists() else False,
                "writable": path.exists() and path.is_dir() and path.stat().st_mode & 0o200
            }
            
            if not path.exists():
                diagnostics["issues"].append({
                    "severity": "medium",
                    "category": "configuration",
                    "message": f"Required directory does not exist: {dir_path}",
                    "recommendation": f"Create directory: {dir_path}"
                })
        
        # Check API configuration
        if hasattr(config, 'api') and hasattr(config.api, 'alpha_vantage'):
            if not config.api.alpha_vantage.api_keys:
                diagnostics["issues"].append({
                    "severity": "high",
                    "category": "configuration",
                    "message": "No Alpha Vantage API keys configured",
                    "recommendation": "Add API keys to configuration"
                })
        
        # Add validation errors as issues
        for error in validation_errors:
            diagnostics["issues"].append({
                "severity": "high",
                "category": "configuration",
                "message": f"Configuration validation error: {error}",
                "recommendation": "Fix configuration file"
            })
        
        print(f"✅ Configuration diagnostics complete - found {len(diagnostics['issues'])} issues")
        
    except Exception as e:
        diagnostics["error"] = str(e)
        print(f"❌ Configuration diagnostics failed: {e}")
    
    return diagnostics


def run_storage_diagnostics(config_path: str) -> Dict[str, Any]:
    """Run storage system diagnostics."""
    print("🔍 Running storage diagnostics...")
    
    diagnostics = {
        "diagnostic_type": "storage",
        "timestamp": datetime.utcnow().isoformat(),
        "storage_info": {},
        "file_analysis": {},
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        # Check storage directories
        data_dir = Path(config.storage.data_directory)
        state_dir = Path(config.storage.state_directory)
        backup_dir = Path(config.storage.backup_directory)
        
        diagnostics["storage_info"] = {
            "data_directory": {
                "path": str(data_dir),
                "exists": data_dir.exists(),
                "file_count": len(list(data_dir.glob("*.parquet"))) if data_dir.exists() else 0,
                "total_size": sum(f.stat().st_size for f in data_dir.glob("*.parquet")) if data_dir.exists() else 0
            },
            "state_directory": {
                "path": str(state_dir),
                "exists": state_dir.exists(),
                "file_count": len(list(state_dir.glob("*"))) if state_dir.exists() else 0
            },
            "backup_directory": {
                "path": str(backup_dir),
                "exists": backup_dir.exists(),
                "file_count": len(list(backup_dir.glob("*"))) if backup_dir.exists() else 0
            }
        }
        
        # Check for corrupted files
        corrupted_files = []
        if data_dir.exists():
            for parquet_file in data_dir.glob("*.parquet"):
                try:
                    import pandas as pd
                    # Try to read just the first row to check if file is readable
                    pd.read_parquet(parquet_file, nrows=1)
                except Exception as e:
                    corrupted_files.append({
                        "file": str(parquet_file),
                        "error": str(e)
                    })
        
        diagnostics["file_analysis"] = {
            "corrupted_files": corrupted_files,
            "corrupted_count": len(corrupted_files)
        }
        
        # Check disk space
        if data_dir.exists():
            disk_usage = shutil.disk_usage(data_dir)
            free_space_gb = disk_usage.free / (1024**3)
            
            if free_space_gb < 1:
                diagnostics["issues"].append({
                    "severity": "critical",
                    "category": "storage",
                    "message": f"Very low disk space: {free_space_gb:.1f}GB free",
                    "recommendation": "Free up disk space immediately"
                })
            elif free_space_gb < 5:
                diagnostics["issues"].append({
                    "severity": "high",
                    "category": "storage",
                    "message": f"Low disk space: {free_space_gb:.1f}GB free",
                    "recommendation": "Monitor disk usage and clean up old files"
                })
        
        # Report corrupted files
        if corrupted_files:
            diagnostics["issues"].append({
                "severity": "medium",
                "category": "storage",
                "message": f"Found {len(corrupted_files)} corrupted data files",
                "recommendation": "Run data repair utility to fix corrupted files"
            })
        
        # Check for missing directories
        for dir_name, dir_info in diagnostics["storage_info"].items():
            if not dir_info["exists"]:
                diagnostics["issues"].append({
                    "severity": "medium",
                    "category": "storage",
                    "message": f"Storage directory missing: {dir_info['path']}",
                    "recommendation": f"Create directory: {dir_info['path']}"
                })
        
        print(f"✅ Storage diagnostics complete - found {len(diagnostics['issues'])} issues")
        
    except Exception as e:
        diagnostics["error"] = str(e)
        print(f"❌ Storage diagnostics failed: {e}")
    
    return diagnostics


def run_network_diagnostics() -> Dict[str, Any]:
    """Run network connectivity diagnostics."""
    print("🔍 Running network diagnostics...")
    
    diagnostics = {
        "diagnostic_type": "network",
        "timestamp": datetime.utcnow().isoformat(),
        "connectivity_tests": {},
        "dns_tests": {},
        "issues": [],
        "recommendations": []
    }
    
    try:
        import socket
        import urllib.request
        import time
        
        # Test connectivity to key services
        test_urls = [
            ("Alpha Vantage", "https://www.alphavantage.co"),
            ("Yahoo Finance", "https://finance.yahoo.com"),
            ("Google DNS", "8.8.8.8"),
            ("Cloudflare DNS", "1.1.1.1")
        ]
        
        for service_name, url in test_urls:
            try:
                start_time = time.time()
                
                if url.startswith("http"):
                    # HTTP connectivity test
                    response = urllib.request.urlopen(url, timeout=10)
                    status_code = response.getcode()
                    success = status_code == 200
                else:
                    # Socket connectivity test (for DNS servers)
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(10)
                    result = sock.connect_ex((url, 53))  # DNS port
                    sock.close()
                    success = result == 0
                    status_code = "Connected" if success else "Failed"
                
                response_time = (time.time() - start_time) * 1000  # ms
                
                diagnostics["connectivity_tests"][service_name] = {
                    "url": url,
                    "success": success,
                    "status_code": status_code,
                    "response_time_ms": response_time
                }
                
                if not success:
                    diagnostics["issues"].append({
                        "severity": "high",
                        "category": "network",
                        "message": f"Cannot connect to {service_name} ({url})",
                        "recommendation": "Check internet connection and firewall settings"
                    })
                elif response_time > 5000:  # 5 seconds
                    diagnostics["issues"].append({
                        "severity": "medium",
                        "category": "network",
                        "message": f"Slow connection to {service_name}: {response_time:.0f}ms",
                        "recommendation": "Check network performance and consider using different DNS"
                    })
                
            except Exception as e:
                diagnostics["connectivity_tests"][service_name] = {
                    "url": url,
                    "success": False,
                    "error": str(e),
                    "response_time_ms": 0
                }
                
                diagnostics["issues"].append({
                    "severity": "high",
                    "category": "network",
                    "message": f"Network test failed for {service_name}: {e}",
                    "recommendation": "Check internet connection and network configuration"
                })
        
        # DNS resolution tests
        dns_hosts = ["www.alphavantage.co", "finance.yahoo.com", "www.google.com"]
        
        for host in dns_hosts:
            try:
                start_time = time.time()
                ip_address = socket.gethostbyname(host)
                resolution_time = (time.time() - start_time) * 1000
                
                diagnostics["dns_tests"][host] = {
                    "success": True,
                    "ip_address": ip_address,
                    "resolution_time_ms": resolution_time
                }
                
                if resolution_time > 1000:  # 1 second
                    diagnostics["issues"].append({
                        "severity": "medium",
                        "category": "network",
                        "message": f"Slow DNS resolution for {host}: {resolution_time:.0f}ms",
                        "recommendation": "Consider using faster DNS servers (8.8.8.8, 1.1.1.1)"
                    })
                
            except Exception as e:
                diagnostics["dns_tests"][host] = {
                    "success": False,
                    "error": str(e),
                    "resolution_time_ms": 0
                }
                
                diagnostics["issues"].append({
                    "severity": "high",
                    "category": "network",
                    "message": f"DNS resolution failed for {host}: {e}",
                    "recommendation": "Check DNS configuration"
                })
        
        print(f"✅ Network diagnostics complete - found {len(diagnostics['issues'])} issues")
        
    except Exception as e:
        diagnostics["error"] = str(e)
        print(f"❌ Network diagnostics failed: {e}")
    
    return diagnostics


def run_log_analysis() -> Dict[str, Any]:
    """Analyze log files for errors and issues."""
    print("🔍 Running log analysis...")
    
    diagnostics = {
        "diagnostic_type": "logs",
        "timestamp": datetime.utcnow().isoformat(),
        "log_files": {},
        "error_analysis": {},
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Find log files
        log_dir = Path("logs")
        log_files = []
        
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
        
        diagnostics["log_files"] = {
            "log_directory_exists": log_dir.exists(),
            "log_files_found": len(log_files),
            "log_files": [str(f) for f in log_files]
        }
        
        if not log_files:
            diagnostics["issues"].append({
                "severity": "medium",
                "category": "logging",
                "message": "No log files found",
                "recommendation": "Check if logging is properly configured"
            })
            return diagnostics
        
        # Analyze recent log files
        error_patterns = [
            "ERROR", "CRITICAL", "FATAL", "Exception", "Traceback",
            "Failed", "Error", "timeout", "connection refused"
        ]
        
        recent_errors = []
        total_lines = 0
        
        for log_file in log_files[-3:]:  # Analyze last 3 log files
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    
                    for line_num, line in enumerate(lines, 1):
                        for pattern in error_patterns:
                            if pattern.lower() in line.lower():
                                recent_errors.append({
                                    "file": str(log_file),
                                    "line_number": line_num,
                                    "pattern": pattern,
                                    "content": line.strip()[:200]  # First 200 chars
                                })
                                break
                                
            except Exception as e:
                diagnostics["issues"].append({
                    "severity": "medium",
                    "category": "logging",
                    "message": f"Could not read log file {log_file}: {e}",
                    "recommendation": "Check log file permissions"
                })
        
        diagnostics["error_analysis"] = {
            "total_lines_analyzed": total_lines,
            "errors_found": len(recent_errors),
            "recent_errors": recent_errors[-20:]  # Last 20 errors
        }
        
        # Categorize errors
        error_categories = {}
        for error in recent_errors:
            pattern = error["pattern"]
            error_categories[pattern] = error_categories.get(pattern, 0) + 1
        
        diagnostics["error_analysis"]["error_categories"] = error_categories
        
        # Generate issues based on error analysis
        if len(recent_errors) > 100:
            diagnostics["issues"].append({
                "severity": "high",
                "category": "logging",
                "message": f"High number of errors in logs: {len(recent_errors)}",
                "recommendation": "Investigate and resolve recurring errors"
            })
        
        # Check for specific error patterns
        critical_patterns = ["CRITICAL", "FATAL", "Exception"]
        critical_errors = [e for e in recent_errors if e["pattern"] in critical_patterns]
        
        if critical_errors:
            diagnostics["issues"].append({
                "severity": "high",
                "category": "logging",
                "message": f"Found {len(critical_errors)} critical errors in logs",
                "recommendation": "Review and fix critical errors immediately"
            })
        
        print(f"✅ Log analysis complete - found {len(diagnostics['issues'])} issues")
        
    except Exception as e:
        diagnostics["error"] = str(e)
        print(f"❌ Log analysis failed: {e}")
    
    return diagnostics


async def attempt_auto_fixes(all_diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Attempt to automatically fix detected issues."""
    print("🔧 Attempting automatic fixes...")
    
    fix_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "fixes_attempted": [],
        "fixes_successful": [],
        "fixes_failed": [],
        "manual_intervention_required": []
    }
    
    # Collect all fixable issues
    fixable_issues = []
    for diagnostic in all_diagnostics:
        for issue in diagnostic.get("issues", []):
            if issue.get("category") in ["storage", "configuration"]:
                fixable_issues.append(issue)
    
    for issue in fixable_issues:
        fix_attempted = {
            "issue": issue["message"],
            "category": issue["category"],
            "attempted_fix": "",
            "success": False
        }
        
        try:
            if "directory does not exist" in issue["message"] or "directory missing" in issue["message"]:
                # Extract directory path from message
                import re
                path_match = re.search(r'([/\\]?[\w/\\.-]+)', issue["message"])
                if path_match:
                    dir_path = Path(path_match.group(1))
                    dir_path.mkdir(parents=True, exist_ok=True)
                    fix_attempted["attempted_fix"] = f"Created directory: {dir_path}"
                    fix_attempted["success"] = dir_path.exists()
            
            elif "corrupted data files" in issue["message"]:
                fix_attempted["attempted_fix"] = "Corrupted files require manual review"
                fix_attempted["success"] = False
                fix_results["manual_intervention_required"].append(issue["message"])
            
            else:
                fix_attempted["attempted_fix"] = "No automatic fix available"
                fix_attempted["success"] = False
                fix_results["manual_intervention_required"].append(issue["message"])
            
        except Exception as e:
            fix_attempted["attempted_fix"] = f"Fix failed: {e}"
            fix_attempted["success"] = False
        
        fix_results["fixes_attempted"].append(fix_attempted)
        
        if fix_attempted["success"]:
            fix_results["fixes_successful"].append(fix_attempted)
        else:
            fix_results["fixes_failed"].append(fix_attempted)
    
    print(f"✅ Auto-fix complete: {len(fix_results['fixes_successful'])} successful, "
          f"{len(fix_results['fixes_failed'])} failed")
    
    return fix_results


def format_diagnostic_output(diagnostics: List[Dict[str, Any]], format_type: str, verbose: bool) -> str:
    """Format diagnostic output for display or file export."""
    if format_type == "json":
        return json.dumps(diagnostics, indent=2, default=str)
    
    else:  # txt format
        lines = []
        
        lines.extend([
            "DIAGNOSTIC TOOLKIT REPORT",
            "=" * 60,
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Diagnostics Run: {len(diagnostics)}",
            ""
        ])
        
        # Summary
        total_issues = sum(len(d.get("issues", [])) for d in diagnostics)
        critical_issues = sum(len([i for i in d.get("issues", []) if i.get("severity") == "critical"]) for d in diagnostics)
        high_issues = sum(len([i for i in d.get("issues", []) if i.get("severity") == "high"]) for d in diagnostics)
        
        lines.extend([
            "SUMMARY:",
            f"  Total Issues:     {total_issues}",
            f"  Critical Issues:  {critical_issues}",
            f"  High Issues:      {high_issues}",
            ""
        ])
        
        # Individual diagnostic results
        for diagnostic in diagnostics:
            diagnostic_type = diagnostic.get("diagnostic_type", "unknown").upper()
            issues = diagnostic.get("issues", [])
            
            lines.extend([
                f"{diagnostic_type} DIAGNOSTICS:",
                "-" * 40
            ])
            
            if "error" in diagnostic:
                lines.append(f"❌ Diagnostic failed: {diagnostic['error']}")
            elif not issues:
                lines.append("✅ No issues found")
            else:
                for issue in issues:
                    severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(issue["severity"], "⚪")
                    lines.append(f"  {severity_icon} [{issue['severity'].upper()}] {issue['message']}")
                    if verbose:
                        lines.append(f"      Recommendation: {issue['recommendation']}")
            
            lines.append("")
        
        # Recommendations summary
        all_recommendations = []
        for diagnostic in diagnostics:
            all_recommendations.extend(diagnostic.get("recommendations", []))
        
        if all_recommendations:
            lines.extend([
                "OVERALL RECOMMENDATIONS:",
                "-" * 40
            ])
            for rec in set(all_recommendations):  # Remove duplicates
                lines.append(f"  • {rec}")
        
        return "\n".join(lines)


async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(level="WARNING")
        
        print("🔧 DIAGNOSTIC TOOLKIT")
        print("=" * 50)
        
        # Check if config file exists
        config_path = Path(args.config)
        if not config_path.exists() and args.diagnostic in ["configuration", "storage", "comprehensive"]:
            print(f"⚠️  Configuration file not found: {config_path}")
            print("Some diagnostics may be limited")
        
        # Run diagnostics
        all_diagnostics = []
        
        if args.diagnostic == "system" or args.diagnostic == "comprehensive":
            all_diagnostics.append(run_system_diagnostics())
        
        if args.diagnostic == "configuration" or args.diagnostic == "comprehensive":
            all_diagnostics.append(run_configuration_diagnostics(str(config_path)))
        
        if args.diagnostic == "storage" or args.diagnostic == "comprehensive":
            all_diagnostics.append(run_storage_diagnostics(str(config_path)))
        
        if args.diagnostic == "network" or args.diagnostic == "comprehensive":
            all_diagnostics.append(run_network_diagnostics())
        
        if args.diagnostic == "logs" or args.diagnostic == "comprehensive":
            all_diagnostics.append(run_log_analysis())
        
        if args.diagnostic == "performance":
            print("⚠️  Performance diagnostics available in performance_analyzer.py")
        
        # Attempt auto-fixes if requested
        if args.fix_issues:
            fix_results = await attempt_auto_fixes(all_diagnostics)
            all_diagnostics.append(fix_results)
        
        # Format and display/save results
        output_text = format_diagnostic_output(all_diagnostics, args.format, args.verbose)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"✅ Diagnostic report saved to: {args.output}")
        else:
            print("\n" + output_text)
        
        # Exit with error code if critical issues found
        critical_issues = sum(len([i for i in d.get("issues", []) if i.get("severity") == "critical"]) for d in all_diagnostics)
        if critical_issues > 0:
            print(f"\n⚠️  Found {critical_issues} critical issues that require immediate attention")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Diagnostic toolkit failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())