#!/usr/bin/env python3
"""
Phase 8 Simple Smoke Test - Dashboard Enhancements
Tests core functionality without complex database dependencies
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_chart_generation():
    """Test chart generation functions."""
    print("\n[TEST] Chart Generation Functions")
    print("=" * 50)
    
    try:
        from dashboard.charts import (
            generate_api_budget_chart,
            generate_phase_duration_timeline,
            generate_confidence_calibration_chart,
            generate_ensemble_weights_chart,
            generate_drawdown_regime_chart,
            generate_system_health_chart,
            generate_rationale_trace_chart
        )
        
        # Test data as DataFrames
        api_data = pd.DataFrame({
            'api_name': ['Questrade', 'Yahoo Finance', 'Alpha Vantage'],
            'requests_made': [100, 200, 50],
            'rate_limit_hits': [5, 2, 1]
        })
        
        phase_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(6)],
            'phase_name': ['data_collection', 'analysis', 'execution', 'data_collection', 'analysis', 'execution'],
            'duration_ms': [1000, 2000, 1500, 1200, 1800, 1600]
        })
        
        confidence_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(5)],
            'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
            'raw_confidence': [0.7, 0.8, 0.6, 0.9, 0.75],
            'calibrated_confidence': [0.65, 0.75, 0.55, 0.85, 0.7],
            'outcome': [1, 0, 1, 1, 0]  # 1 for win, 0 for loss
        })
        
        ensemble_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(6)],
            'model': ['model1', 'model2', 'model3', 'model1', 'model2', 'model3'],
            'weight': [0.4, 0.3, 0.3, 0.35, 0.35, 0.3]
        })
        
        # Test each chart function
        charts = [
            ("API Budget Chart", lambda: generate_api_budget_chart(api_data)),
            ("Phase Duration Timeline", lambda: generate_phase_duration_timeline(phase_data)),
            ("Confidence Calibration Chart", lambda: generate_confidence_calibration_chart(confidence_data)),
            ("Ensemble Weights Chart", lambda: generate_ensemble_weights_chart(ensemble_data)),
            ("System Health Chart", lambda: generate_system_health_chart({})),
        ]
        
        for chart_name, chart_func in charts:
            try:
                chart = chart_func()
                print(f"[OK] {chart_name} generated successfully")
            except Exception as e:
                print(f"[FAIL] {chart_name} failed: {e}")
                return False
        
        print("[PASS] Chart generation test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Chart generation test failed: {e}")
        return False

def test_enhanced_layout():
    """Test enhanced layout creation."""
    print("\n[TEST] Enhanced Layout Creation")
    print("=" * 50)
    
    try:
        from dashboard.enhanced_layout import (
            create_api_budget_panel,
            create_phase_duration_panel,
            create_confidence_calibration_panel,
            create_ensemble_weights_panel,
            create_drawdown_regime_panel,
            create_system_health_panel,
            create_rationale_trace_panel
        )
        
        # Mock the connector to avoid database dependencies
        with patch('dashboard.enhanced_layout.DashboardConnector') as mock_connector:
            mock_connector.return_value.api_budget_status.return_value = pd.DataFrame()
            mock_connector.return_value.phase_duration_timeline.return_value = pd.DataFrame()
            mock_connector.return_value.confidence_calibration_data.return_value = pd.DataFrame()
            mock_connector.return_value.ensemble_weights_history.return_value = pd.DataFrame()
            mock_connector.return_value.drawdown_and_regime_data.return_value = pd.DataFrame()
            mock_connector.return_value.system_health_metrics.return_value = pd.DataFrame()
            mock_connector.return_value.model_rationale_trace.return_value = []
            
            # Test each panel creation function
            panels = [
                ("API Budget Panel", lambda: create_api_budget_panel()),
                ("Phase Duration Panel", lambda: create_phase_duration_panel()),
                ("Confidence Calibration Panel", lambda: create_confidence_calibration_panel()),
                ("Ensemble Weights Panel", lambda: create_ensemble_weights_panel()),
                ("Drawdown Regime Panel", lambda: create_drawdown_regime_panel()),
                ("System Health Panel", lambda: create_system_health_panel()),
                ("Rationale Trace Panel", lambda: create_rationale_trace_panel()),
            ]
            
            for panel_name, panel_func in panels:
                try:
                    panel = panel_func()
                    print(f"[OK] {panel_name} created successfully")
                except Exception as e:
                    print(f"[FAIL] {panel_name} failed: {e}")
                    return False
            
            print("[PASS] Enhanced layout test completed successfully")
            return True
        
    except Exception as e:
        print(f"[FAIL] Enhanced layout test failed: {e}")
        return False

def test_connector_functions():
    """Test connector functions with mocked data."""
    print("\n[TEST] Connector Functions")
    print("=" * 50)
    
    try:
        from dashboard.connector import DashboardConnector
        
        # Mock database connection
        with patch('config.database.get_connection') as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=None)
            mock_get_conn.return_value = mock_conn
            
            # Mock pandas read_sql_query to return empty DataFrame
            with patch('pandas.read_sql_query') as mock_read_sql:
                mock_read_sql.return_value = pd.DataFrame()
                
                connector = DashboardConnector()
                
                # Test each connector function
                functions = [
                    ("api_budget_status", lambda: connector.api_budget_status()),
                    ("phase_duration_timeline", lambda: connector.phase_duration_timeline()),
                    ("confidence_calibration_data", lambda: connector.confidence_calibration_data()),
                    ("ensemble_weights_history", lambda: connector.ensemble_weights_history()),
                    ("drawdown_and_regime_data", lambda: connector.drawdown_and_regime_data()),
                    ("model_rationale_trace", lambda: connector.model_rationale_trace()),
                    ("system_health_metrics", lambda: connector.system_health_metrics()),
                ]
                
                for func_name, func in functions:
                    try:
                        result = func()
                        print(f"[OK] {func_name} executed successfully")
                    except Exception as e:
                        print(f"[FAIL] {func_name} failed: {e}")
                        return False
                
                print("[PASS] Connector functions test completed successfully")
                return True
        
    except Exception as e:
        print(f"[FAIL] Connector functions test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Phase 8 Simple Smoke Test - Dashboard Enhancements")
    print("=" * 60)
    
    tests = [
        ("Chart Generation", test_chart_generation),
        ("Enhanced Layout", test_enhanced_layout),
        ("Connector Functions", test_connector_functions),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[READY] Starting {test_name} test...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"[FAIL] {test_name} test failed")
        except Exception as e:
            print(f"[FAIL] {test_name} test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Phase 8 Simple Smoke Test Summary")
    print("=" * 60)
    
    for test_name, _ in tests:
        status = "[PASS]" if test_name in [tests[i][0] for i in range(passed)] else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Phase 8 dashboard enhancements are working correctly.")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main())
