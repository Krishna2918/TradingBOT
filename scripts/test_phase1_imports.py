#!/usr/bin/env python3
"""Phase 1 Validation: Import & Dependency Check"""

import sys
import io
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        pass  # Already wrapped or not needed

def test_all_imports():
    """Test that all critical imports work."""
    errors = []
    
    try:
        from src.config.mode_manager import get_mode_manager
        from src.config.database import DatabaseManager
        from src.trading.positions import PositionManager
        from src.trading.risk import RiskManager
        from src.ai.advanced_models.feature_pipeline import AdvancedFeaturePipeline
        print("[PASS] All critical imports successful")
    except Exception as e:
        errors.append(f"Import error: {e}")
    
    try:
        from collections import defaultdict
        print("[PASS] defaultdict import successful")
    except Exception as e:
        errors.append(f"defaultdict import error: {e}")
    
    # Test gymnasium (optional)
    try:
        import gymnasium
        print("[PASS] gymnasium installed")
    except ImportError:
        print("[WARN] gymnasium not installed (RL features limited)")
    
    return len(errors) == 0, errors

if __name__ == "__main__":
    success, errors = test_all_imports()
    if success:
        print("\n[PASS] PHASE 1 VALIDATION: PASSED")
        exit(0)
    else:
        print("\n[FAIL] PHASE 1 VALIDATION: FAILED")
        for error in errors:
            print(f"  - {error}")
        exit(1)
