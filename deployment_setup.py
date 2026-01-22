#!/usr/bin/env python3
"""
Deployment setup script for continuous data collection system.

This script handles system deployment, environment setup, and initial configuration
for the continuous data collection system.

Usage:
    python deployment_setup.py [--environment ENV] [--install-deps]

Requirements: 1.1, 8.1
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the continuous_data_collection package to the path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core.logging_config import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deployment setup for continuous data collection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Types:
    development     - Development environment setup
    testing         - Testing environment setup
    production      - Production environment setup
    custom          - Custom environment with user-specified settings

Setup Actions:
    --install-deps      Install Python dependencies
    --create-dirs       Create required directories
    --setup-config      Set up configuration files
    --setup-logging     Set up logging configuration
    --setup-systemd     Set up systemd service (Linux only)
    --all              Perform all setup actions

Examples:
    python deployment_setup.py --environment development --all
    python deployment_setup.py --environment production --install-deps --setup-config
    python deployment_setup.py --environment custom --create-dirs --setup-logging
        """
    )
    
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "testing", "production", "custom"],
        default="development",
        help="Target environment (default: development)"
    )
    
    parser.add_argument(
        "--install-deps", "-i",
        action="store_true",
        help="Install Python dependencies"
    )
    
    parser.add_argument(
        "--create-dirs", "-d",
        action="store_true",
        help="Create required directories"
    )
    
    parser.add_argument(
        "--setup-config", "-c",
        action="store_true",
        help="Set up configuration files"
    )
    
    parser.add_argument(
        "--setup-logging", "-l",
        action="store_true",
        help="Set up logging configuration"
    )
    
    parser.add_argument(
        "--setup-systemd", "-s",
        action="store_true",
        help="Set up systemd service (Linux only)"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Perform all setup actions"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force overwrite existing files"
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    return parser.parse_args()


def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements for deployment."""
    requirements = {
        "python_version": {
            "required": "3.8+",
            "current": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "satisfied": sys.version_info >= (3, 8)
        },
        "disk_space": {
            "required_gb": 10,
            "available_gb": 0,
            "satisfied": False
        },
        "memory": {
            "required_gb": 4,
            "available_gb": 0,
            "satisfied": False
        },
        "packages": {},
        "overall_satisfied": False
    }
    
    # Check disk space
    try:
        disk_usage = shutil.disk_usage('.')
        available_gb = disk_usage.free / (1024**3)
        requirements["disk_space"]["available_gb"] = available_gb
        requirements["disk_space"]["satisfied"] = available_gb >= requirements["disk_space"]["required_gb"]
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.total / (1024**3)
        requirements["memory"]["available_gb"] = available_gb
        requirements["memory"]["satisfied"] = available_gb >= requirements["memory"]["required_gb"]
    except ImportError:
        print("⚠️  psutil not available for memory check")
    except Exception as e:
        print(f"⚠️  Could not check memory: {e}")
    
    # Check required packages
    required_packages = [
        "pandas", "numpy", "pyyaml", "requests", "psutil", "yfinance"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            requirements["packages"][package] = {"installed": True, "version": "unknown"}
        except ImportError:
            requirements["packages"][package] = {"installed": False, "version": None}
    
    # Overall satisfaction
    requirements["overall_satisfied"] = (
        requirements["python_version"]["satisfied"] and
        requirements["disk_space"]["satisfied"] and
        requirements["memory"]["satisfied"] and
        all(pkg["installed"] for pkg in requirements["packages"].values())
    )
    
    return requirements


def install_dependencies(dry_run: bool = False) -> bool:
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Define requirements
    requirements = [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "pyyaml>=5.4.0",
        "requests>=2.25.0",
        "psutil>=5.8.0",
        "yfinance>=0.1.70",
        "pyarrow>=5.0.0",  # For Parquet support
        "aiohttp>=3.8.0",  # For async HTTP requests
        "asyncio-throttle>=1.0.0"  # For rate limiting
    ]
    
    if dry_run:
        print("🔍 DRY RUN - Would install:")
        for req in requirements:
            print(f"  - {req}")
        return True
    
    try:
        # Create requirements.txt
        requirements_file = Path("requirements.txt")
        with open(requirements_file, 'w') as f:
            for req in requirements:
                f.write(f"{req}\n")
        
        print(f"Created {requirements_file}")
        
        # Install using pip
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        
        print("Running: " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def create_directory_structure(environment: str, dry_run: bool = False) -> bool:
    """Create required directory structure."""
    print("📁 Creating directory structure...")
    
    # Define directory structure based on environment
    base_dirs = {
        "data": "data",
        "state": "state",
        "backups": "backups",
        "logs": "logs",
        "config": "config",
        "scripts": "scripts"
    }
    
    # Environment-specific subdirectories
    if environment == "production":
        base_dirs.update({
            "data": "data/production",
            "state": "state/production",
            "backups": "backups/production",
            "logs": "logs/production"
        })
    elif environment == "testing":
        base_dirs.update({
            "data": "data/testing",
            "state": "state/testing",
            "backups": "backups/testing",
            "logs": "logs/testing"
        })
    
    success = True
    
    for dir_name, dir_path in base_dirs.items():
        path = Path(dir_path)
        
        if dry_run:
            print(f"🔍 Would create: {path}")
            continue
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {path}")
            
            # Set appropriate permissions
            if os.name != 'nt':  # Unix-like systems
                os.chmod(path, 0o755)
                
        except Exception as e:
            print(f"❌ Failed to create {path}: {e}")
            success = False
    
    return success


def setup_configuration_files(environment: str, force: bool = False, dry_run: bool = False) -> bool:
    """Set up configuration files for the specified environment."""
    print(f"⚙️  Setting up configuration files for {environment} environment...")
    
    # Configuration templates
    config_templates = {
        "development": {
            "storage": {
                "data_directory": "data/development",
                "state_directory": "state/development",
                "backup_directory": "backups/development",
                "file_format": "parquet",
                "compression": "snappy"
            },
            "collection": {
                "max_workers": 4,
                "batch_size": 20,
                "request_timeout": 30,
                "retry_delay": 1.0,
                "max_retries": 3
            },
            "api": {
                "alpha_vantage": {
                    "api_keys": ["YOUR_API_KEY_HERE"],
                    "requests_per_minute": 60,
                    "base_url": "https://www.alphavantage.co/query"
                }
            },
            "monitoring": {
                "progress_update_interval": 10,
                "health_check_interval": 30,
                "log_level": "INFO"
            }
        },
        "testing": {
            "storage": {
                "data_directory": "data/testing",
                "state_directory": "state/testing",
                "backup_directory": "backups/testing",
                "file_format": "parquet",
                "compression": "snappy"
            },
            "collection": {
                "max_workers": 2,
                "batch_size": 5,
                "request_timeout": 15,
                "retry_delay": 0.5,
                "max_retries": 2
            },
            "api": {
                "alpha_vantage": {
                    "api_keys": ["TEST_API_KEY"],
                    "requests_per_minute": 30,
                    "base_url": "https://www.alphavantage.co/query"
                }
            },
            "monitoring": {
                "progress_update_interval": 5,
                "health_check_interval": 15,
                "log_level": "DEBUG"
            }
        },
        "production": {
            "storage": {
                "data_directory": "data/production",
                "state_directory": "state/production",
                "backup_directory": "backups/production",
                "file_format": "parquet",
                "compression": "snappy"
            },
            "collection": {
                "max_workers": 8,
                "batch_size": 50,
                "request_timeout": 60,
                "retry_delay": 2.0,
                "max_retries": 5
            },
            "api": {
                "alpha_vantage": {
                    "api_keys": ["PROD_API_KEY_1", "PROD_API_KEY_2", "PROD_API_KEY_3", "PROD_API_KEY_4"],
                    "requests_per_minute": 280,
                    "base_url": "https://www.alphavantage.co/query"
                }
            },
            "monitoring": {
                "progress_update_interval": 20,
                "health_check_interval": 60,
                "log_level": "INFO"
            }
        }
    }
    
    if environment not in config_templates:
        print(f"❌ Unknown environment: {environment}")
        return False
    
    config_data = config_templates[environment]
    config_file = Path(f"config/{environment}.yaml")
    
    if dry_run:
        print(f"🔍 Would create: {config_file}")
        return True
    
    # Check if file exists
    if config_file.exists() and not force:
        print(f"⚠️  Configuration file already exists: {config_file}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Skipped configuration file creation")
            return True
    
    try:
        # Ensure config directory exists
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write YAML configuration
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        print(f"✅ Created configuration file: {config_file}")
        
        # Create environment-specific note
        note_content = f"""
# Configuration Notes for {environment.upper()} Environment

## Important Setup Steps:

1. **API Keys**: Replace placeholder API keys with real Alpha Vantage API keys
   - Get free API keys from: https://www.alphavantage.co/support/#api-key
   - Update the api_keys list in {config_file}

2. **Directory Permissions**: Ensure the application has write access to:
   - {config_data['storage']['data_directory']}
   - {config_data['storage']['state_directory']}
   - {config_data['storage']['backup_directory']}

3. **Resource Allocation**: Adjust worker count based on your system:
   - Current setting: {config_data['collection']['max_workers']} workers
   - Recommended: 1-2 workers per CPU core

4. **Monitoring**: Check log files in the logs directory for system status

## Security Considerations:
- Keep API keys secure and never commit them to version control
- Use environment variables for sensitive configuration in production
- Regularly rotate API keys
- Monitor API usage to avoid rate limiting

## Performance Tuning:
- Adjust batch_size based on memory availability
- Tune request_timeout based on network conditions
- Monitor system resources and adjust worker count accordingly
"""
        
        note_file = Path(f"config/{environment}_setup_notes.md")
        with open(note_file, 'w') as f:
            f.write(note_content)
        
        print(f"✅ Created setup notes: {note_file}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create configuration: {e}")
        return False


def setup_logging_configuration(environment: str, dry_run: bool = False) -> bool:
    """Set up logging configuration."""
    print("📝 Setting up logging configuration...")
    
    # Logging configuration based on environment
    log_configs = {
        "development": {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "simple": {
                    "format": "%(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/development.log",
                    "maxBytes": 10485760,
                    "backupCount": 5
                }
            },
            "loggers": {
                "continuous_data_collection": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["console"]
            }
        },
        "production": {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
                },
                "json": {
                    "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s"
                }
            },
            "handlers": {
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "filename": "logs/production.log",
                    "maxBytes": 52428800,
                    "backupCount": 10
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": "logs/errors.log",
                    "maxBytes": 10485760,
                    "backupCount": 5
                }
            },
            "loggers": {
                "continuous_data_collection": {
                    "level": "INFO",
                    "handlers": ["file", "error_file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "WARNING",
                "handlers": ["file"]
            }
        }
    }
    
    if environment not in log_configs:
        environment = "development"  # Default fallback
    
    log_config = log_configs[environment]
    log_config_file = Path(f"config/logging_{environment}.json")
    
    if dry_run:
        print(f"🔍 Would create: {log_config_file}")
        return True
    
    try:
        # Ensure config directory exists
        log_config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write logging configuration
        with open(log_config_file, 'w') as f:
            json.dump(log_config, f, indent=2)
        
        print(f"✅ Created logging configuration: {log_config_file}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create logging configuration: {e}")
        return False


def setup_systemd_service(environment: str, dry_run: bool = False) -> bool:
    """Set up systemd service for Linux systems."""
    if os.name == 'nt':
        print("⚠️  Systemd service setup is only available on Linux systems")
        return True
    
    print("🔧 Setting up systemd service...")
    
    # Get current working directory and Python executable
    work_dir = Path.cwd()
    python_exec = sys.executable
    
    # Service file content
    service_content = f"""[Unit]
Description=Continuous Data Collection System ({environment})
After=network.target
Wants=network.target

[Service]
Type=simple
User={os.getenv('USER', 'nobody')}
WorkingDirectory={work_dir}
Environment=PYTHONPATH={work_dir}
ExecStart={python_exec} start_collection.py --config config/{environment}.yaml
ExecStop={python_exec} stop_collection.py --config config/{environment}.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=continuous-data-collection-{environment}

# Resource limits
LimitNOFILE=65536
MemoryMax=4G

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path(f"continuous-data-collection-{environment}.service")
    
    if dry_run:
        print(f"🔍 Would create: {service_file}")
        print("🔍 Would copy to: /etc/systemd/system/")
        return True
    
    try:
        # Write service file
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"✅ Created systemd service file: {service_file}")
        
        # Instructions for manual installation
        print("\n📋 To complete systemd setup, run these commands as root:")
        print(f"   sudo cp {service_file} /etc/systemd/system/")
        print("   sudo systemctl daemon-reload")
        print(f"   sudo systemctl enable continuous-data-collection-{environment}.service")
        print(f"   sudo systemctl start continuous-data-collection-{environment}.service")
        print("\n📋 To check service status:")
        print(f"   sudo systemctl status continuous-data-collection-{environment}.service")
        print(f"   sudo journalctl -u continuous-data-collection-{environment}.service -f")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create systemd service: {e}")
        return False


def create_deployment_scripts(environment: str, dry_run: bool = False) -> bool:
    """Create deployment and management scripts."""
    print("📜 Creating deployment scripts...")
    
    scripts = {
        f"deploy_{environment}.sh": f"""#!/bin/bash
# Deployment script for {environment} environment

set -e

echo "🚀 Deploying Continuous Data Collection System - {environment.upper()}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
python3 -m pip install -r requirements.txt

# Create directories
echo "📁 Creating directories..."
python3 deployment_setup.py --environment {environment} --create-dirs

# Validate configuration
echo "🔍 Validating configuration..."
python3 config_validator.py --config config/{environment}.yaml

# Run system diagnostics
echo "🔧 Running system diagnostics..."
python3 diagnostic_toolkit.py --diagnostic system

echo "✅ Deployment complete!"
echo "📋 Next steps:"
echo "   1. Update API keys in config/{environment}.yaml"
echo "   2. Run: python3 start_collection.py --config config/{environment}.yaml"
echo "   3. Monitor: python3 check_progress.py --config config/{environment}.yaml"
""",
        
        f"start_{environment}.sh": f"""#!/bin/bash
# Start script for {environment} environment

echo "🚀 Starting Continuous Data Collection System - {environment.upper()}"

# Validate configuration first
python3 config_validator.py --config config/{environment}.yaml --validation basic

if [ $? -eq 0 ]; then
    echo "✅ Configuration valid, starting collection..."
    python3 start_collection.py --config config/{environment}.yaml
else
    echo "❌ Configuration validation failed"
    exit 1
fi
""",
        
        f"stop_{environment}.sh": f"""#!/bin/bash
# Stop script for {environment} environment

echo "🛑 Stopping Continuous Data Collection System - {environment.upper()}"
python3 stop_collection.py --config config/{environment}.yaml
""",
        
        f"status_{environment}.sh": f"""#!/bin/bash
# Status script for {environment} environment

echo "📊 Continuous Data Collection System Status - {environment.upper()}"
python3 check_progress.py --config config/{environment}.yaml --refresh 0
"""
    }
    
    success = True
    
    for script_name, script_content in scripts.items():
        script_path = Path(f"scripts/{script_name}")
        
        if dry_run:
            print(f"🔍 Would create: {script_path}")
            continue
        
        try:
            # Ensure scripts directory exists
            script_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write script
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable on Unix-like systems
            if os.name != 'nt':
                os.chmod(script_path, 0o755)
            
            print(f"✅ Created script: {script_path}")
            
        except Exception as e:
            print(f"❌ Failed to create script {script_path}: {e}")
            success = False
    
    return success


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(level="INFO")
        
        print("🚀 DEPLOYMENT SETUP")
        print("=" * 50)
        print(f"Environment: {args.environment}")
        print(f"Dry run: {args.dry_run}")
        print()
        
        # Check system requirements
        print("🔍 Checking system requirements...")
        requirements = check_system_requirements()
        
        print(f"Python version: {requirements['python_version']['current']} "
              f"({'✅' if requirements['python_version']['satisfied'] else '❌'} required: {requirements['python_version']['required']})")
        
        print(f"Disk space: {requirements['disk_space']['available_gb']:.1f}GB "
              f"({'✅' if requirements['disk_space']['satisfied'] else '❌'} required: {requirements['disk_space']['required_gb']}GB)")
        
        if requirements['memory']['available_gb'] > 0:
            print(f"Memory: {requirements['memory']['available_gb']:.1f}GB "
                  f"({'✅' if requirements['memory']['satisfied'] else '❌'} required: {requirements['memory']['required_gb']}GB)")
        
        # Check packages
        missing_packages = [pkg for pkg, info in requirements['packages'].items() if not info['installed']]
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
        else:
            print("✅ All required packages available")
        
        if not requirements['overall_satisfied'] and not args.install_deps:
            print("\n⚠️  System requirements not satisfied. Use --install-deps to install missing packages.")
        
        print()
        
        # Determine actions to perform
        actions = []
        if args.all:
            actions = ['install_deps', 'create_dirs', 'setup_config', 'setup_logging', 'setup_systemd']
        else:
            if args.install_deps:
                actions.append('install_deps')
            if args.create_dirs:
                actions.append('create_dirs')
            if args.setup_config:
                actions.append('setup_config')
            if args.setup_logging:
                actions.append('setup_logging')
            if args.setup_systemd:
                actions.append('setup_systemd')
        
        if not actions:
            print("⚠️  No actions specified. Use --all or specify individual actions.")
            return
        
        # Perform actions
        success = True
        
        if 'install_deps' in actions:
            success &= install_dependencies(args.dry_run)
        
        if 'create_dirs' in actions:
            success &= create_directory_structure(args.environment, args.dry_run)
        
        if 'setup_config' in actions:
            success &= setup_configuration_files(args.environment, args.force, args.dry_run)
        
        if 'setup_logging' in actions:
            success &= setup_logging_configuration(args.environment, args.dry_run)
        
        if 'setup_systemd' in actions:
            success &= setup_systemd_service(args.environment, args.dry_run)
        
        # Always create deployment scripts
        success &= create_deployment_scripts(args.environment, args.dry_run)
        
        # Summary
        print("\n" + "=" * 50)
        if success:
            print("✅ DEPLOYMENT SETUP COMPLETE")
            if not args.dry_run:
                print("\n📋 Next steps:")
                print(f"   1. Update API keys in config/{args.environment}.yaml")
                print("   2. Run configuration validation:")
                print(f"      python config_validator.py --config config/{args.environment}.yaml")
                print("   3. Start the system:")
                print(f"      python start_collection.py --config config/{args.environment}.yaml")
        else:
            print("❌ DEPLOYMENT SETUP FAILED")
            print("Please check the errors above and retry.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()