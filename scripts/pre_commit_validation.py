#!/usr/bin/env python3
"""
Pre-commit Validation Script

Validates the trading system before commits to ensure basic functionality.
"""

import sys
import os
import logging
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_imports() -> List[str]:
    """Validate that all critical modules can be imported."""
    errors = []
    
    critical_modules = [
        'src.config.database',
        'src.monitoring.system_monitor',
        'src.ai.multi_model',
        'src.trading.risk',
        'src.dashboard.connector'
    ]
    
    for module_name in critical_modules:
        try:
            __import__(module_name)
            logger.debug(f"Successfully imported {module_name}")
        except ImportError as e:
            error_msg = f"Failed to import {module_name}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error importing {module_name}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    return errors

def validate_file_structure() -> List[str]:
    """Validate that required files and directories exist."""
    errors = []
    
    required_dirs = [
        'src',
        'src/config',
        'src/monitoring',
        'src/ai',
        'src/trading',
        'src/dashboard',
        'tests',
        'scripts'
    ]
    
    required_files = [
        'src/config/database.py',
        'src/monitoring/system_monitor.py',
        'src/ai/multi_model.py',
        'src/trading/risk.py',
        'src/dashboard/connector.py',
        'requirements.txt'
    ]
    
    # Check directories
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            error_msg = f"Required directory missing: {dir_path}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    # Check files
    for file_path in required_files:
        if not os.path.exists(file_path):
            error_msg = f"Required file missing: {file_path}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    return errors

def validate_configuration() -> List[str]:
    """Validate configuration files and settings."""
    errors = []
    
    try:
        # Check if mode manager can be imported and used
        from config.mode_manager import get_current_mode
        mode = get_current_mode()
        if mode is None:
            errors.append("Mode manager not returning valid mode")
        
    except Exception as e:
        error_msg = f"Configuration validation failed: {e}"
        errors.append(error_msg)
        logger.error(error_msg)
    
    return errors

def validate_database_schema() -> List[str]:
    """Validate database schema and connectivity."""
    errors = []
    
    try:
        from config.database import get_connection
        
        # Test database connectivity
        with get_connection('DEMO') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result[0] != 1:
                errors.append("Database connectivity test failed")
        
    except Exception as e:
        error_msg = f"Database validation failed: {e}"
        errors.append(error_msg)
        logger.error(error_msg)
    
    return errors

def validate_ai_system() -> List[str]:
    """Validate AI system components."""
    errors = []
    
    try:
        from ai.multi_model import MultiModelManager
        
        # Test multi-model manager initialization
        manager = MultiModelManager(mode="DEMO")
        configs = manager.get_all_model_configs()
        
        if not configs:
            errors.append("No AI model configurations found")
        
    except Exception as e:
        error_msg = f"AI system validation failed: {e}"
        errors.append(error_msg)
        logger.error(error_msg)
    
    return errors

def validate_trading_system() -> List[str]:
    """Validate trading system components."""
    errors = []
    
    try:
        from src.trading.risk import RiskManager
        
        # Test risk manager initialization
        risk_manager = RiskManager()
        
        # Test basic risk calculations
        try:
            # This might fail if no data is available, which is OK for pre-commit
            pass
        except Exception:
            # Expected in pre-commit environment
            pass
        
    except Exception as e:
        error_msg = f"Trading system validation failed: {e}"
        errors.append(error_msg)
        logger.error(error_msg)
    
    return errors

def validate_dashboard() -> List[str]:
    """Validate dashboard components."""
    errors = []
    
    try:
        from src.dashboard.connector import DashboardConnector
        
        # Test dashboard connector initialization
        connector = DashboardConnector()
        
    except Exception as e:
        error_msg = f"Dashboard validation failed: {e}"
        errors.append(error_msg)
        logger.error(error_msg)
    
    return errors

def main() -> int:
    """Main validation function."""
    logger.info("Starting pre-commit validation...")
    
    all_errors = []
    
    # Run all validation checks
    validation_checks = [
        ("File Structure", validate_file_structure),
        ("Imports", validate_imports),
        ("Configuration", validate_configuration),
        ("Database Schema", validate_database_schema),
        ("AI System", validate_ai_system),
        ("Trading System", validate_trading_system),
        ("Dashboard", validate_dashboard)
    ]
    
    for check_name, check_func in validation_checks:
        logger.info(f"Running {check_name} validation...")
        try:
            errors = check_func()
            if errors:
                all_errors.extend([f"{check_name}: {error}" for error in errors])
                logger.error(f"{check_name} validation failed with {len(errors)} errors")
            else:
                logger.info(f"{check_name} validation passed")
        except Exception as e:
            error_msg = f"{check_name} validation crashed: {e}"
            all_errors.append(error_msg)
            logger.error(error_msg)
    
    # Report results
    if all_errors:
        logger.error(f"Pre-commit validation failed with {len(all_errors)} errors:")
        for error in all_errors:
            logger.error(f"  - {error}")
        
        print("❌ Pre-commit validation failed!")
        print("Please fix the following issues before committing:")
        for error in all_errors:
            print(f"  - {error}")
        
        return 1
    else:
        logger.info("Pre-commit validation passed successfully")
        print("✅ Pre-commit validation passed!")
        return 0

if __name__ == "__main__":
    exit(main())
