#!/usr/bin/env python3
"""
Quick Readiness Check for Phase 12-13
Fast validation of critical components
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists"""
    if Path(file_path).exists():
        print(f"[OK] {description}: {file_path}")
        return True
    else:
        print(f"[MISSING] {description}: {file_path}")
        return False

def check_import(module_path: str, description: str) -> bool:
    """Check if a module can be imported"""
    try:
        __import__(module_path)
        print(f"[OK] {description}: {module_path}")
        return True
    except ImportError as e:
        print(f"[FAIL] {description}: {module_path} - {e}")
        return False

def main():
    """Quick readiness check"""
    print("Quick Readiness Check for Phase 12-13")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 0
    
    # Critical files
    critical_files = [
        ("src/config/database.py", "Database Configuration"),
        ("src/monitoring/system_monitor.py", "System Monitor"),
        ("src/data_pipeline/api_budget_manager.py", "API Budget Manager"),
        ("src/validation/data_quality.py", "Data Quality Validator"),
        ("src/adaptive/confidence_calibration.py", "Confidence Calibration"),
        ("src/ai/adaptive_weights.py", "Adaptive Weights"),
        ("src/trading/risk.py", "Risk Manager"),
        ("src/trading/atr_brackets.py", "ATR Brackets"),
        ("src/ai/regime_detection.py", "Regime Detection"),
        ("src/dashboard/connector.py", "Dashboard Connector"),
        ("src/ai/ollama_lifecycle.py", "Ollama Lifecycle"),
        ("src/config/feature_flags.py", "Feature Flags"),
        ("src/dashboard/safety_controls.py", "Safety Controls"),
        ("docs/ROLLOUT_PLAN.md", "Rollout Plan"),
        (".github/workflows/nightly-validation.yml", "GitHub Workflow"),
        (".pre-commit-config.yaml", "Pre-commit Config")
    ]
    
    print("\nCritical Files:")
    for file_path, description in critical_files:
        total_checks += 1
        if check_file_exists(file_path, description):
            checks_passed += 1
    
    # Critical imports
    critical_imports = [
        ("config.database", "Database Module"),
        ("monitoring.system_monitor", "System Monitor Module"),
        ("data_pipeline.api_budget_manager", "API Budget Manager Module"),
        ("validation.data_quality", "Data Quality Module"),
        ("adaptive.confidence_calibration", "Confidence Calibration Module"),
        ("ai.adaptive_weights", "Adaptive Weights Module"),
        ("trading.risk", "Risk Manager Module"),
        ("trading.atr_brackets", "ATR Brackets Module"),
        ("ai.regime_detection", "Regime Detection Module"),
        ("dashboard.connector", "Dashboard Connector Module"),
        ("ai.ollama_lifecycle", "Ollama Lifecycle Module"),
        ("config.feature_flags", "Feature Flags Module"),
        ("dashboard.safety_controls", "Safety Controls Module")
    ]
    
    print("\nCritical Imports:")
    for module_path, description in critical_imports:
        total_checks += 1
        if check_import(module_path, description):
            checks_passed += 1
    
    # Database tables
    print("\nDatabase Schema:")
    try:
        from config.database import get_connection
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
        required_tables = [
            'api_validation_log',
            'phase_execution_tracking', 
            'api_usage_metrics',
            'data_provenance',
            'data_quality_violations',
            'confidence_calibration',
            'model_performance',
            'bracket_parameters',
            'portfolio_snapshots',
            'regime_state'
        ]
        
        for table in required_tables:
            total_checks += 1
            if table in tables:
                print(f"[OK] Database Table: {table}")
                checks_passed += 1
            else:
                print(f"[MISSING] Database Table: {table}")
                
    except Exception as e:
        print(f"[FAIL] Database Check Failed: {e}")
        total_checks += 1
    
    # Test scripts
    print("\nTest Scripts:")
    test_scripts = [
        ("scripts/phase0_smoke_test.py", "Phase 0 Smoke Test"),
        ("scripts/phase1_smoke_test.py", "Phase 1 Smoke Test"),
        ("scripts/phase2_smoke_test.py", "Phase 2 Smoke Test"),
        ("scripts/phase3_smoke_test.py", "Phase 3 Smoke Test"),
        ("scripts/phase4_smoke_test.py", "Phase 4 Smoke Test"),
        ("scripts/phase5_smoke_test.py", "Phase 5 Smoke Test"),
        ("scripts/phase6_smoke_test.py", "Phase 6 Smoke Test"),
        ("scripts/phase7_smoke_test.py", "Phase 7 Smoke Test"),
        ("scripts/phase8_smoke_test.py", "Phase 8 Smoke Test"),
        ("scripts/phase9_smoke_test.py", "Phase 9 Smoke Test"),
        ("scripts/phase10_smoke_test.py", "Phase 10 Smoke Test"),
        ("scripts/phase11_smoke_test.py", "Phase 11 Smoke Test")
    ]
    
    for script_path, description in test_scripts:
        total_checks += 1
        if check_file_exists(script_path, description):
            checks_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("READINESS SUMMARY")
    print("=" * 50)
    
    success_rate = (checks_passed / total_checks) * 100 if total_checks > 0 else 0
    
    print(f"Checks Passed: {checks_passed}")
    print(f"Checks Failed: {total_checks - checks_passed}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("\nSYSTEM READY FOR PHASE 12-13!")
        print("All critical components are in place.")
        return 0
    elif success_rate >= 75:
        print("\nSYSTEM MOSTLY READY")
        print("Some components missing but core functionality available.")
        return 0
    else:
        print("\nSYSTEM NOT READY")
        print("Too many critical components missing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
