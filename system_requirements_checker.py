#!/usr/bin/env python3
"""
System requirements checker for continuous data collection system.

This script validates system requirements, checks dependencies,
and provides recommendations for optimal system configuration.

Usage:
    python system_requirements_checker.py [--detailed] [--fix-issues]

Requirements: 1.1, 8.1
"""

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the continuous_data_collection package to the path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core.logging_config import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="System requirements checker for continuous data collection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Check Types:
    basic       - Basic system requirements (Python, disk, memory)
    detailed    - Detailed system analysis including performance
    packages    - Python package requirements
    network     - Network connectivity requirements
    all         - All requirement checks

Examples:
    python system_requirements_checker.py
    python system_requirements_checker.py --detailed --output requirements_report.json
    python system_requirements_checker.py --check packages --fix-issues
        """
    )
    
    parser.add_argument(
        "--check", "-c",
        choices=["basic", "detailed", "packages", "network", "all"],
        default="all",
        help="Type of requirements check (default: all)"
    )
    
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Include detailed system information"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for requirements report"
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
        help="Attempt to fix detected issues automatically"
    )
    
    return parser.parse_args()


def check_python_requirements() -> Dict[str, Any]:
    """Check Python version and installation requirements."""
    requirements = {
        "check_type": "python",
        "python_version": {
            "current": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "required_minimum": "3.8.0",
            "recommended": "3.9.0+",
            "satisfied": sys.version_info >= (3, 8),
            "recommended_satisfied": sys.version_info >= (3, 9)
        },
        "python_installation": {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "platform": platform.platform(),
            "architecture": platform.architecture()[0]
        },
        "pip_available": False,
        "virtual_env": {
            "in_venv": hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
            "venv_path": os.environ.get('VIRTUAL_ENV', None)
        },
        "issues": [],
        "recommendations": []
    }
    
    # Check pip availability
    try:
        import pip
        requirements["pip_available"] = True
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         capture_output=True, check=True)
            requirements["pip_available"] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            requirements["pip_available"] = False
            requirements["issues"].append("pip is not available")
    
    # Generate recommendations
    if not requirements["python_version"]["satisfied"]:
        requirements["issues"].append(f"Python version {requirements['python_version']['current']} is below minimum required version {requirements['python_version']['required_minimum']}")
        requirements["recommendations"].append("Upgrade Python to version 3.8 or higher")
    
    if not requirements["python_version"]["recommended_satisfied"]:
        requirements["recommendations"].append("Consider upgrading to Python 3.9+ for better performance")
    
    if not requirements["virtual_env"]["in_venv"]:
        requirements["recommendations"].append("Consider using a virtual environment for package isolation")
    
    if not requirements["pip_available"]:
        requirements["recommendations"].append("Install pip for package management")
    
    return requirements


def check_system_resources() -> Dict[str, Any]:
    """Check system resource requirements."""
    requirements = {
        "check_type": "system_resources",
        "cpu": {
            "cores": 0,
            "logical_cores": 0,
            "architecture": platform.machine(),
            "frequency": 0,
            "recommended_cores": 4,
            "satisfied": False
        },
        "memory": {
            "total_gb": 0,
            "available_gb": 0,
            "required_gb": 4,
            "recommended_gb": 8,
            "satisfied": False,
            "recommended_satisfied": False
        },
        "disk": {
            "total_gb": 0,
            "free_gb": 0,
            "required_gb": 10,
            "recommended_gb": 50,
            "satisfied": False,
            "recommended_satisfied": False
        },
        "issues": [],
        "recommendations": []
    }
    
    try:
        import psutil
        
        # CPU information
        requirements["cpu"]["cores"] = psutil.cpu_count(logical=False) or 0
        requirements["cpu"]["logical_cores"] = psutil.cpu_count(logical=True) or 0
        
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                requirements["cpu"]["frequency"] = cpu_freq.current
        except:
            pass
        
        requirements["cpu"]["satisfied"] = requirements["cpu"]["cores"] >= 2
        
        # Memory information
        memory = psutil.virtual_memory()
        requirements["memory"]["total_gb"] = memory.total / (1024**3)
        requirements["memory"]["available_gb"] = memory.available / (1024**3)
        requirements["memory"]["satisfied"] = requirements["memory"]["total_gb"] >= requirements["memory"]["required_gb"]
        requirements["memory"]["recommended_satisfied"] = requirements["memory"]["total_gb"] >= requirements["memory"]["recommended_gb"]
        
        # Disk information
        disk = psutil.disk_usage('.')
        requirements["disk"]["total_gb"] = disk.total / (1024**3)
        requirements["disk"]["free_gb"] = disk.free / (1024**3)
        requirements["disk"]["satisfied"] = requirements["disk"]["free_gb"] >= requirements["disk"]["required_gb"]
        requirements["disk"]["recommended_satisfied"] = requirements["disk"]["free_gb"] >= requirements["disk"]["recommended_gb"]
        
    except ImportError:
        requirements["issues"].append("psutil package not available for detailed system analysis")
        
        # Fallback disk check
        try:
            disk = shutil.disk_usage('.')
            requirements["disk"]["total_gb"] = disk.total / (1024**3)
            requirements["disk"]["free_gb"] = disk.free / (1024**3)
            requirements["disk"]["satisfied"] = requirements["disk"]["free_gb"] >= requirements["disk"]["required_gb"]
            requirements["disk"]["recommended_satisfied"] = requirements["disk"]["free_gb"] >= requirements["disk"]["recommended_gb"]
        except:
            requirements["issues"].append("Cannot determine disk space")
    
    # Generate issues and recommendations
    if not requirements["cpu"]["satisfied"]:
        requirements["issues"].append(f"CPU cores ({requirements['cpu']['cores']}) below recommended minimum (2)")
        requirements["recommendations"].append("Consider upgrading to a system with more CPU cores")
    
    if not requirements["memory"]["satisfied"]:
        requirements["issues"].append(f"Memory ({requirements['memory']['total_gb']:.1f}GB) below required minimum ({requirements['memory']['required_gb']}GB)")
        requirements["recommendations"].append("Add more RAM to meet minimum requirements")
    elif not requirements["memory"]["recommended_satisfied"]:
        requirements["recommendations"].append(f"Consider upgrading to {requirements['memory']['recommended_gb']}GB+ RAM for optimal performance")
    
    if not requirements["disk"]["satisfied"]:
        requirements["issues"].append(f"Free disk space ({requirements['disk']['free_gb']:.1f}GB) below required minimum ({requirements['disk']['required_gb']}GB)")
        requirements["recommendations"].append("Free up disk space or add more storage")
    elif not requirements["disk"]["recommended_satisfied"]:
        requirements["recommendations"].append(f"Consider having {requirements['disk']['recommended_gb']}GB+ free space for large datasets")
    
    return requirements


def check_package_requirements() -> Dict[str, Any]:
    """Check Python package requirements."""
    requirements = {
        "check_type": "packages",
        "required_packages": {},
        "optional_packages": {},
        "package_manager": {
            "pip_version": None,
            "pip_available": False
        },
        "issues": [],
        "recommendations": []
    }
    
    # Define required packages with minimum versions
    required_packages = {
        "pandas": "1.3.0",
        "numpy": "1.21.0",
        "pyyaml": "5.4.0",
        "requests": "2.25.0",
        "aiohttp": "3.8.0",
        "pyarrow": "5.0.0"
    }
    
    # Define optional packages
    optional_packages = {
        "psutil": "5.8.0",
        "yfinance": "0.1.70",
        "asyncio-throttle": "1.0.0",
        "python-json-logger": "2.0.0"
    }
    
    # Check pip version
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, check=True)
        pip_version = result.stdout.strip().split()[1]
        requirements["package_manager"]["pip_version"] = pip_version
        requirements["package_manager"]["pip_available"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        requirements["package_manager"]["pip_available"] = False
        requirements["issues"].append("pip is not available")
    
    # Check required packages
    for package, min_version in required_packages.items():
        package_info = {
            "required": True,
            "min_version": min_version,
            "installed": False,
            "current_version": None,
            "version_satisfied": False
        }
        
        try:
            module = __import__(package)
            package_info["installed"] = True
            
            # Try to get version
            version = None
            for attr in ['__version__', 'version', 'VERSION']:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    break
            
            if version:
                package_info["current_version"] = str(version)
                # Simple version comparison (works for most cases)
                try:
                    from packaging import version as pkg_version
                    package_info["version_satisfied"] = pkg_version.parse(str(version)) >= pkg_version.parse(min_version)
                except ImportError:
                    # Fallback: assume version is satisfied if package is installed
                    package_info["version_satisfied"] = True
            else:
                package_info["version_satisfied"] = True  # Assume OK if we can't check version
                
        except ImportError:
            requirements["issues"].append(f"Required package '{package}' is not installed")
        
        requirements["required_packages"][package] = package_info
    
    # Check optional packages
    for package, min_version in optional_packages.items():
        package_info = {
            "required": False,
            "min_version": min_version,
            "installed": False,
            "current_version": None,
            "version_satisfied": False
        }
        
        try:
            module = __import__(package)
            package_info["installed"] = True
            
            # Try to get version
            version = None
            for attr in ['__version__', 'version', 'VERSION']:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    break
            
            if version:
                package_info["current_version"] = str(version)
                try:
                    from packaging import version as pkg_version
                    package_info["version_satisfied"] = pkg_version.parse(str(version)) >= pkg_version.parse(min_version)
                except ImportError:
                    package_info["version_satisfied"] = True
            else:
                package_info["version_satisfied"] = True
                
        except ImportError:
            pass  # Optional packages don't generate issues
        
        requirements["optional_packages"][package] = package_info
    
    # Generate recommendations
    missing_required = [pkg for pkg, info in requirements["required_packages"].items() if not info["installed"]]
    if missing_required:
        requirements["recommendations"].append(f"Install required packages: {', '.join(missing_required)}")
    
    outdated_required = [pkg for pkg, info in requirements["required_packages"].items() 
                        if info["installed"] and not info["version_satisfied"]]
    if outdated_required:
        requirements["recommendations"].append(f"Update outdated packages: {', '.join(outdated_required)}")
    
    missing_optional = [pkg for pkg, info in requirements["optional_packages"].items() if not info["installed"]]
    if missing_optional:
        requirements["recommendations"].append(f"Consider installing optional packages for enhanced functionality: {', '.join(missing_optional)}")
    
    return requirements


def check_network_requirements() -> Dict[str, Any]:
    """Check network connectivity requirements."""
    requirements = {
        "check_type": "network",
        "connectivity_tests": {},
        "dns_resolution": {},
        "firewall_ports": {},
        "issues": [],
        "recommendations": []
    }
    
    # Test connectivity to required services
    test_hosts = {
        "alpha_vantage": "www.alphavantage.co",
        "yahoo_finance": "finance.yahoo.com",
        "google_dns": "8.8.8.8",
        "cloudflare_dns": "1.1.1.1"
    }
    
    for service, host in test_hosts.items():
        test_result = {
            "host": host,
            "reachable": False,
            "response_time": 0,
            "error": None
        }
        
        try:
            import socket
            import time
            
            start_time = time.time()
            
            if host.replace(".", "").isdigit():  # IP address
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                result = sock.connect_ex((host, 53))  # DNS port
                sock.close()
                test_result["reachable"] = result == 0
            else:  # Hostname
                socket.gethostbyname(host)
                test_result["reachable"] = True
            
            test_result["response_time"] = time.time() - start_time
            
        except Exception as e:
            test_result["error"] = str(e)
            requirements["issues"].append(f"Cannot reach {service} ({host}): {e}")
        
        requirements["connectivity_tests"][service] = test_result
    
    # Test DNS resolution
    dns_hosts = ["www.alphavantage.co", "finance.yahoo.com"]
    
    for host in dns_hosts:
        dns_result = {
            "host": host,
            "resolved": False,
            "ip_address": None,
            "resolution_time": 0,
            "error": None
        }
        
        try:
            import socket
            import time
            
            start_time = time.time()
            ip_address = socket.gethostbyname(host)
            dns_result["resolved"] = True
            dns_result["ip_address"] = ip_address
            dns_result["resolution_time"] = time.time() - start_time
            
        except Exception as e:
            dns_result["error"] = str(e)
            requirements["issues"].append(f"DNS resolution failed for {host}: {e}")
        
        requirements["dns_resolution"][host] = dns_result
    
    # Check common firewall ports
    required_ports = {
        "HTTP": 80,
        "HTTPS": 443,
        "DNS": 53
    }
    
    for port_name, port_num in required_ports.items():
        port_result = {
            "port": port_num,
            "accessible": False,
            "error": None
        }
        
        try:
            import socket
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(("8.8.8.8", port_num))
            sock.close()
            port_result["accessible"] = result == 0
            
        except Exception as e:
            port_result["error"] = str(e)
        
        requirements["firewall_ports"][port_name] = port_result
        
        if not port_result["accessible"]:
            requirements["issues"].append(f"Port {port_num} ({port_name}) may be blocked")
    
    # Generate recommendations
    unreachable_services = [service for service, result in requirements["connectivity_tests"].items() 
                           if not result["reachable"]]
    if unreachable_services:
        requirements["recommendations"].append("Check internet connection and firewall settings")
        requirements["recommendations"].append(f"Ensure access to: {', '.join(unreachable_services)}")
    
    failed_dns = [host for host, result in requirements["dns_resolution"].items() 
                  if not result["resolved"]]
    if failed_dns:
        requirements["recommendations"].append("Check DNS configuration")
        requirements["recommendations"].append("Consider using public DNS servers (8.8.8.8, 1.1.1.1)")
    
    blocked_ports = [port_name for port_name, result in requirements["firewall_ports"].items() 
                     if not result["accessible"]]
    if blocked_ports:
        requirements["recommendations"].append(f"Ensure firewall allows outbound connections on ports: {', '.join(blocked_ports)}")
    
    return requirements


def get_detailed_system_info() -> Dict[str, Any]:
    """Get detailed system information."""
    info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node()
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "build": platform.python_build()
        },
        "environment": {
            "path": os.environ.get("PATH", ""),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
            "user": os.environ.get("USER", os.environ.get("USERNAME", ""))
        }
    }
    
    # Add system-specific information
    try:
        import psutil
        
        # Boot time
        boot_time = psutil.boot_time()
        info["system"] = {
            "boot_time": boot_time,
            "uptime_hours": (psutil.time.time() - boot_time) / 3600
        }
        
        # Network interfaces
        network_interfaces = psutil.net_if_addrs()
        info["network_interfaces"] = {
            name: [addr.address for addr in addrs if addr.family == socket.AF_INET]
            for name, addrs in network_interfaces.items()
        }
        
    except ImportError:
        pass
    except Exception as e:
        info["system_info_error"] = str(e)
    
    return info


def attempt_fixes(all_requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Attempt to automatically fix detected issues."""
    fix_results = {
        "fixes_attempted": [],
        "fixes_successful": [],
        "fixes_failed": [],
        "manual_intervention_required": []
    }
    
    # Try to install missing required packages
    for req in all_requirements:
        if req.get("check_type") == "packages":
            missing_required = [pkg for pkg, info in req["required_packages"].items() 
                              if not info["installed"]]
            
            if missing_required and req["package_manager"]["pip_available"]:
                for package in missing_required:
                    try:
                        min_version = req["required_packages"][package]["min_version"]
                        package_spec = f"{package}>={min_version}"
                        
                        fix_results["fixes_attempted"].append(f"Install {package_spec}")
                        
                        result = subprocess.run([
                            sys.executable, "-m", "pip", "install", package_spec
                        ], capture_output=True, text=True, check=True)
                        
                        fix_results["fixes_successful"].append(f"Successfully installed {package}")
                        
                    except subprocess.CalledProcessError as e:
                        fix_results["fixes_failed"].append(f"Failed to install {package}: {e}")
    
    # Issues that require manual intervention
    manual_issues = [
        "Python version upgrade",
        "System resource upgrades",
        "Network connectivity issues",
        "Firewall configuration"
    ]
    
    for req in all_requirements:
        for issue in req.get("issues", []):
            if any(manual_issue.lower() in issue.lower() for manual_issue in manual_issues):
                fix_results["manual_intervention_required"].append(issue)
    
    return fix_results


def format_requirements_output(all_requirements: List[Dict[str, Any]], format_type: str, detailed: bool) -> str:
    """Format requirements output for display or file export."""
    if format_type == "json":
        return json.dumps(all_requirements, indent=2, default=str)
    
    else:  # txt format
        lines = []
        
        lines.extend([
            "SYSTEM REQUIREMENTS REPORT",
            "=" * 60,
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            ""
        ])
        
        # Overall summary
        total_issues = sum(len(req.get("issues", [])) for req in all_requirements)
        total_recommendations = sum(len(req.get("recommendations", [])) for req in all_requirements)
        
        lines.extend([
            "SUMMARY:",
            f"  Total Issues:         {total_issues}",
            f"  Total Recommendations: {total_recommendations}",
            ""
        ])
        
        # Individual requirement checks
        for req in all_requirements:
            check_type = req.get("check_type", "unknown").upper()
            
            lines.extend([
                f"{check_type} REQUIREMENTS:",
                "-" * 40
            ])
            
            # Specific formatting for each check type
            if check_type == "PYTHON":
                python_info = req["python_version"]
                status = "✅" if python_info["satisfied"] else "❌"
                lines.append(f"  {status} Python Version: {python_info['current']} (required: {python_info['required_minimum']})")
                
                pip_status = "✅" if req["pip_available"] else "❌"
                lines.append(f"  {pip_status} pip Available: {req['pip_available']}")
                
                venv_status = "✅" if req["virtual_env"]["in_venv"] else "⚠️ "
                lines.append(f"  {venv_status} Virtual Environment: {req['virtual_env']['in_venv']}")
            
            elif check_type == "SYSTEM_RESOURCES":
                cpu_info = req["cpu"]
                memory_info = req["memory"]
                disk_info = req["disk"]
                
                lines.extend([
                    f"  CPU: {cpu_info['cores']} cores, {cpu_info['logical_cores']} logical",
                    f"  Memory: {memory_info['total_gb']:.1f}GB total, {memory_info['available_gb']:.1f}GB available",
                    f"  Disk: {disk_info['free_gb']:.1f}GB free of {disk_info['total_gb']:.1f}GB total"
                ])
            
            elif check_type == "PACKAGES":
                required_packages = req["required_packages"]
                installed_count = sum(1 for info in required_packages.values() if info["installed"])
                lines.append(f"  Required Packages: {installed_count}/{len(required_packages)} installed")
                
                if detailed:
                    for pkg, info in required_packages.items():
                        status = "✅" if info["installed"] and info["version_satisfied"] else "❌"
                        version_info = f" (v{info['current_version']})" if info["current_version"] else ""
                        lines.append(f"    {status} {pkg}{version_info}")
            
            elif check_type == "NETWORK":
                connectivity_tests = req["connectivity_tests"]
                reachable_count = sum(1 for result in connectivity_tests.values() if result["reachable"])
                lines.append(f"  Connectivity Tests: {reachable_count}/{len(connectivity_tests)} passed")
                
                if detailed:
                    for service, result in connectivity_tests.items():
                        status = "✅" if result["reachable"] else "❌"
                        time_info = f" ({result['response_time']:.1f}s)" if result["response_time"] > 0 else ""
                        lines.append(f"    {status} {service}{time_info}")
            
            # Issues
            issues = req.get("issues", [])
            if issues:
                lines.append("\n  ❌ ISSUES:")
                for issue in issues:
                    lines.append(f"    • {issue}")
            
            # Recommendations
            recommendations = req.get("recommendations", [])
            if recommendations:
                lines.append("\n  💡 RECOMMENDATIONS:")
                for rec in recommendations:
                    lines.append(f"    • {rec}")
            
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(level="WARNING")
        
        print("🔍 SYSTEM REQUIREMENTS CHECKER")
        print("=" * 50)
        
        # Run requirement checks
        all_requirements = []
        
        if args.check in ["basic", "all"]:
            print("Checking Python requirements...")
            all_requirements.append(check_python_requirements())
            
            print("Checking system resources...")
            all_requirements.append(check_system_resources())
        
        if args.check in ["detailed", "all"]:
            print("Getting detailed system information...")
            detailed_info = get_detailed_system_info()
            all_requirements.append({
                "check_type": "detailed_system_info",
                "system_info": detailed_info,
                "issues": [],
                "recommendations": []
            })
        
        if args.check in ["packages", "all"]:
            print("Checking package requirements...")
            all_requirements.append(check_package_requirements())
        
        if args.check in ["network", "all"]:
            print("Checking network requirements...")
            all_requirements.append(check_network_requirements())
        
        # Attempt fixes if requested
        if args.fix_issues:
            print("Attempting automatic fixes...")
            fix_results = attempt_fixes(all_requirements)
            all_requirements.append({
                "check_type": "fixes",
                "fix_results": fix_results,
                "issues": [],
                "recommendations": []
            })
        
        # Format and display/save results
        output_text = format_requirements_output(all_requirements, args.format, args.detailed)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"✅ Requirements report saved to: {args.output}")
        else:
            print("\n" + output_text)
        
        # Exit with error code if critical issues found
        total_issues = sum(len(req.get("issues", [])) for req in all_requirements)
        if total_issues > 0:
            print(f"\n⚠️  Found {total_issues} issues that may affect system operation")
            sys.exit(1)
        else:
            print("\n✅ All system requirements satisfied")
        
    except Exception as e:
        print(f"\n❌ Requirements check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Import datetime here to avoid issues
    from datetime import datetime
    import socket
    
    main()