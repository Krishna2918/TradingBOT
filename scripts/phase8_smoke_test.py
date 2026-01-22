#!/usr/bin/env python3
"""
Phase 8 Smoke Test - Dashboard Enhancements

This script performs a comprehensive smoke test of Phase 8 components:
1. Enhanced dashboard connector data providers
2. New chart generation functions
3. Dashboard panel creation
4. Layout integration
5. Data provider integration
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dashboard_connector():
    """Test enhanced dashboard connector data providers."""
    print("\n[TEST] Dashboard Connector Data Providers")
    print("=" * 50)
    
    try:
        from dashboard.connector import DashboardConnector
        
        # Create connector
        connector = DashboardConnector()
        print("[OK] DashboardConnector created successfully")
        
        # Test API budget status
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock API budget data
            api_df = pd.DataFrame({
                'api_name': ['Questrade', 'Alpha Vantage', 'Finnhub'],
                'requests_made': [100, 25, 60],
                'requests_limit': [1000, 25, 60],
                'daily_requests': [500, 25, 60],
                'daily_limit': [1000, 25, 60],
                'rate_limit_hits': [0, 0, 1],
                'last_request_time': [datetime.now()] * 3,
                'window_start': [datetime.now()] * 3,
                'window_end': [datetime.now()] * 3,
                'mode': ['DEMO'] * 3
            })
            mock_conn.cursor.return_value.execute.return_value = None
            pd.read_sql_query.return_value = api_df
            
            api_data = connector.api_budget_status()
            print(f"[OK] API budget status: {len(api_data)} APIs tracked")
        
        # Test phase duration timeline
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock phase duration data
            phase_df = pd.DataFrame({
                'timestamp': [datetime.now()] * 5,
                'phase_name': ['ingest', 'features', 'scoring', 'ensemble', 'sizing'],
                'step_name': ['data_collection', 'feature_calc', 'ai_analysis', 'ensemble', 'risk_calc'],
                'duration_ms': [100, 200, 300, 150, 250],
                'status': ['success'] * 5,
                'detail': ['Completed successfully'] * 5,
                'mode': ['DEMO'] * 5
            })
            mock_conn.cursor.return_value.execute.return_value = None
            pd.read_sql_query.return_value = phase_df
            
            phase_data = connector.phase_duration_timeline()
            print(f"[OK] Phase duration timeline: {len(phase_data)} phase executions")
        
        # Test confidence calibration data
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock confidence calibration data
            confidence_df = pd.DataFrame({
                'trade_date': [datetime.now()] * 3,
                'model': ['technical_analyst', 'sentiment_analyst', 'ensemble'],
                'symbol': ['AAPL', 'MSFT', 'GOOGL'],
                'raw_confidence': [0.8, 0.7, 0.75],
                'calibrated_confidence': [0.75, 0.65, 0.70],
                'outcome': [1, 0, 1],
                'window_id': ['20241013_20241113'] * 3,
                'created_at': [datetime.now()] * 3
            })
            mock_conn.cursor.return_value.execute.return_value = None
            pd.read_sql_query.return_value = confidence_df
            
            confidence_data = connector.confidence_calibration_data()
            print(f"[OK] Confidence calibration: {len(confidence_data)} trades")
        
        # Test ensemble weights history
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock ensemble weights data
            weights_df = pd.DataFrame({
                'timestamp': [datetime.now()] * 5,
                'model': ['technical_analyst', 'sentiment_analyst', 'fundamental_analyst', 'risk_analyst', 'market_regime_analyst'],
                'weight': [0.3, 0.25, 0.2, 0.15, 0.1],
                'brier_score': [0.2, 0.25, 0.3, 0.15, 0.35],
                'accuracy': [0.8, 0.75, 0.7, 0.85, 0.65],
                'sample_count': [100, 100, 100, 100, 100],
                'created_at': [datetime.now()] * 5
            })
            mock_conn.cursor.return_value.execute.return_value = None
            pd.read_sql_query.return_value = weights_df
            
            weights_data = connector.ensemble_weights_history()
            print(f"[OK] Ensemble weights history: {len(weights_data)} model records")
        
        # Test drawdown and regime data
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock portfolio data
            portfolio_df = pd.DataFrame({
                'timestamp': [datetime.now()] * 3,
                'portfolio_value': [10000, 10100, 9950],
                'daily_pnl': [0, 100, -150],
                'total_pnl': [0, 100, -50],
                'mode': ['DEMO'] * 3
            })
            
            # Mock regime data
            regime_df = pd.DataFrame({
                'timestamp': [datetime.now()] * 3,
                'regime': ['TRENDING_LOW_VOL', 'TRENDING_LOW_VOL', 'CHOPPY_HIGH_VOL'],
                'trend_direction': ['UPTREND', 'UPTREND', 'SIDEWAYS'],
                'volatility_level': ['LOW', 'LOW', 'HIGH'],
                'trend_strength': [0.8, 0.7, 0.3],
                'volatility_ratio': [1.0, 1.1, 1.5],
                'regime_confidence': [0.9, 0.8, 0.6],
                'transition_probability': [0.1, 0.2, 0.4],
                'mode': ['DEMO'] * 3
            })
            
            # Mock the two separate queries
            def mock_read_sql_query(query, conn, params=None):
                if 'portfolio_snapshots' in query:
                    return portfolio_df
                elif 'regime_state' in query:
                    return regime_df
                return pd.DataFrame()
            
            pd.read_sql_query.side_effect = mock_read_sql_query
            
            regime_data = connector.drawdown_and_regime_data()
            print(f"[OK] Drawdown and regime data: {len(regime_data)} records")
        
        # Test model rationale trace
        with patch('duckdb.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock rationale trace data
            rationale_df = pd.DataFrame({
                'ts': [datetime.now()] * 3,
                'symbol': ['AAPL', 'MSFT', 'GOOGL'],
                'score': [0.8, 0.7, 0.75],
                'details_json': [
                    '{"technical": {"rsi": 65, "macd": 0.1}, "sentiment": {"score": 0.7}}',
                    '{"technical": {"rsi": 45, "macd": -0.1}, "sentiment": {"score": 0.6}}',
                    '{"technical": {"rsi": 55, "macd": 0.05}, "sentiment": {"score": 0.8}}'
                ],
                'model_name': ['ensemble', 'ensemble', 'ensemble']
            })
            mock_conn.execute.return_value.fetch_df.return_value = rationale_df
            
            rationale_data = connector.model_rationale_trace()
            print(f"[OK] Model rationale trace: {len(rationale_data)} traces")
        
        # Test system health metrics
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock system health data
            api_df = pd.DataFrame({
                'api_name': ['Questrade', 'Alpha Vantage'],
                'status': ['healthy', 'healthy'],
                'response_time_ms': [150, 200],
                'last_validated': [datetime.now()] * 2
            })
            
            phase_df = pd.DataFrame({
                'phase_name': ['ingest', 'scoring'],
                'avg_duration_ms': [100, 300],
                'execution_count': [10, 10],
                'success_count': [10, 9]
            })
            
            risk_df = pd.DataFrame({
                'event_type': ['position_limit', 'drawdown_warning'],
                'event_count': [2, 1]
            })
            
            # Mock the queries
            def mock_read_sql_query(query, conn):
                if 'api_validation_log' in query:
                    return api_df
                elif 'phase_execution_tracking' in query:
                    return phase_df
                elif 'risk_events' in query:
                    return risk_df
                return pd.DataFrame()
            
            pd.read_sql_query.side_effect = mock_read_sql_query
            
            health_data = connector.system_health_metrics()
            print(f"[OK] System health metrics: {len(health_data)} metric categories")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Dashboard connector test failed: {e}")
        logger.exception("Dashboard connector test failed")
        return False

def test_enhanced_charts():
    """Test enhanced chart generation functions."""
    print("\n[TEST] Enhanced Chart Generation")
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
        
        # Test API budget chart
        api_data = pd.DataFrame({
            'api_name': ['Questrade', 'Alpha Vantage', 'Finnhub'],
            'requests_made': [100, 25, 60],
            'rate_limit_hits': [0, 0, 1]
        })
        api_chart = generate_api_budget_chart(api_data)
        print("[OK] API budget chart generated")
        
        # Test phase duration timeline chart
        phase_data = pd.DataFrame({
            'timestamp': [datetime.now()] * 5,
            'phase_name': ['ingest', 'features', 'scoring', 'ensemble', 'sizing'],
            'duration_ms': [100, 200, 300, 150, 250]
        })
        phase_chart = generate_phase_duration_timeline(phase_data)
        print("[OK] Phase duration timeline chart generated")
        
        # Test confidence calibration chart
        confidence_data = pd.DataFrame({
            'raw_confidence': [0.8, 0.7, 0.75],
            'calibrated_confidence': [0.75, 0.65, 0.70],
            'outcome': [1, 0, 1],
            'symbol': ['AAPL', 'MSFT', 'GOOGL']
        })
        confidence_chart = generate_confidence_calibration_chart(confidence_data)
        print("[OK] Confidence calibration chart generated")
        
        # Test ensemble weights chart
        weights_data = pd.DataFrame({
            'timestamp': [datetime.now()] * 5,
            'model': ['technical_analyst', 'sentiment_analyst', 'fundamental_analyst', 'risk_analyst', 'market_regime_analyst'],
            'weight': [0.3, 0.25, 0.2, 0.15, 0.1]
        })
        weights_chart = generate_ensemble_weights_chart(weights_data)
        print("[OK] Ensemble weights chart generated")
        
        # Test drawdown and regime chart
        regime_data = pd.DataFrame({
            'timestamp': [datetime.now()] * 3,
            'portfolio_value': [10000, 10100, 9950],
            'daily_pnl': [0, 100, -150],
            'regime': ['TRENDING_LOW_VOL', 'TRENDING_LOW_VOL', 'CHOPPY_HIGH_VOL']
        })
        regime_chart = generate_drawdown_regime_chart(regime_data)
        print("[OK] Drawdown and regime chart generated")
        
        # Test system health chart
        health_data = {
            'api_status': [
                {'api_name': 'Questrade', 'response_time_ms': 150},
                {'api_name': 'Alpha Vantage', 'response_time_ms': 200}
            ],
            'phase_performance': [
                {'phase_name': 'ingest', 'avg_duration_ms': 100},
                {'phase_name': 'scoring', 'avg_duration_ms': 300}
            ]
        }
        health_chart = generate_system_health_chart(health_data)
        print("[OK] System health chart generated")
        
        # Test rationale trace chart
        rationale_data = [
            {
                'timestamp': datetime.now(),
                'symbol': 'AAPL',
                'score': 0.8,
                'model_name': 'ensemble'
            },
            {
                'timestamp': datetime.now(),
                'symbol': 'MSFT',
                'score': 0.7,
                'model_name': 'ensemble'
            }
        ]
        rationale_chart = generate_rationale_trace_chart(rationale_data)
        print("[OK] Rationale trace chart generated")
        
        # Test empty data handling
        empty_charts = [
            generate_api_budget_chart(pd.DataFrame()),
            generate_phase_duration_timeline(pd.DataFrame()),
            generate_confidence_calibration_chart(pd.DataFrame()),
            generate_ensemble_weights_chart(pd.DataFrame()),
            generate_drawdown_regime_chart(pd.DataFrame()),
            generate_system_health_chart({}),
            generate_rationale_trace_chart([])
        ]
        print("[OK] Empty data handling: All charts handle empty data gracefully")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Enhanced charts test failed: {e}")
        logger.exception("Enhanced charts test failed")
        return False

def test_dashboard_panels():
    """Test dashboard panel creation functions."""
    print("\n[TEST] Dashboard Panel Creation")
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
        
        # Mock the connector and chart functions
        with patch('dashboard.enhanced_layout.DashboardConnector') as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector
            
            # Mock data for all panels
            mock_connector.api_budget_status.return_value = pd.DataFrame({
                'api_name': ['Questrade'],
                'requests_made': [100],
                'rate_limit_hits': [0]
            })
            
            mock_connector.phase_duration_timeline.return_value = pd.DataFrame({
                'phase_name': ['ingest'],
                'duration_ms': [100]
            })
            
            mock_connector.confidence_calibration_data.return_value = pd.DataFrame({
                'raw_confidence': [0.8],
                'calibrated_confidence': [0.75]
            })
            
            mock_connector.ensemble_weights_history.return_value = pd.DataFrame({
                'model': ['technical_analyst'],
                'weight': [0.3]
            })
            
            mock_connector.drawdown_and_regime_data.return_value = pd.DataFrame({
                'portfolio_value': [10000],
                'regime': ['TRENDING_LOW_VOL']
            })
            
            mock_connector.system_health_metrics.return_value = {
                'api_status': [],
                'phase_performance': []
            }
            
            mock_connector.model_rationale_trace.return_value = [
                {'timestamp': datetime.now(), 'symbol': 'AAPL', 'score': 0.8}
            ]
            
            # Mock chart generation functions
            with patch('dashboard.enhanced_layout.generate_api_budget_chart') as mock_api_chart, \
                 patch('dashboard.enhanced_layout.generate_phase_duration_timeline') as mock_phase_chart, \
                 patch('dashboard.enhanced_layout.generate_confidence_calibration_chart') as mock_confidence_chart, \
                 patch('dashboard.enhanced_layout.generate_ensemble_weights_chart') as mock_weights_chart, \
                 patch('dashboard.enhanced_layout.generate_drawdown_regime_chart') as mock_regime_chart, \
                 patch('dashboard.enhanced_layout.generate_system_health_chart') as mock_health_chart, \
                 patch('dashboard.enhanced_layout.generate_rationale_trace_chart') as mock_rationale_chart:
                
                mock_api_chart.return_value = Mock()
                mock_phase_chart.return_value = Mock()
                mock_confidence_chart.return_value = Mock()
                mock_weights_chart.return_value = Mock()
                mock_regime_chart.return_value = Mock()
                mock_health_chart.return_value = Mock()
                mock_rationale_chart.return_value = Mock()
                
                # Test all panel creation functions
                panels = [
                    create_api_budget_panel(),
                    create_phase_duration_panel(),
                    create_confidence_calibration_panel(),
                    create_ensemble_weights_panel(),
                    create_drawdown_regime_panel(),
                    create_system_health_panel(),
                    create_rationale_trace_panel()
                ]
                
                for i, panel in enumerate(panels):
                    assert panel is not None
                    assert hasattr(panel, 'children')
                    print(f"[OK] Panel {i+1} created successfully")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Dashboard panels test failed: {e}")
        logger.exception("Dashboard panels test failed")
        return False

def test_layout_integration():
    """Test dashboard layout integration."""
    print("\n[TEST] Dashboard Layout Integration")
    print("=" * 50)
    
    try:
        from dashboard.enhanced_layout import (
            create_enhanced_dashboard_layout,
            create_legacy_compatibility_layout,
            create_minimal_enhanced_layout,
            get_enhanced_layout
        )
        
        # Test enhanced dashboard layout
        enhanced_layout = create_enhanced_dashboard_layout()
        assert enhanced_layout is not None
        assert hasattr(enhanced_layout, 'children')
        print("[OK] Enhanced dashboard layout created")
        
        # Test legacy compatibility layout
        legacy_layout = create_legacy_compatibility_layout()
        assert legacy_layout is not None
        assert hasattr(legacy_layout, 'children')
        print("[OK] Legacy compatibility layout created")
        
        # Test minimal enhanced layout
        minimal_layout = create_minimal_enhanced_layout()
        assert minimal_layout is not None
        assert hasattr(minimal_layout, 'children')
        print("[OK] Minimal enhanced layout created")
        
        # Test main layout function
        main_layout = get_enhanced_layout()
        assert main_layout is not None
        assert hasattr(main_layout, 'children')
        print("[OK] Main enhanced layout function working")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Layout integration test failed: {e}")
        logger.exception("Layout integration test failed")
        return False

def test_data_provider_integration():
    """Test integration between data providers and charts."""
    print("\n[TEST] Data Provider Integration")
    print("=" * 50)
    
    try:
        from dashboard.connector import DashboardConnector
        from dashboard.charts import (
            generate_api_budget_chart,
            generate_phase_duration_timeline,
            generate_confidence_calibration_chart
        )
        
        connector = DashboardConnector()
        
        # Test API budget integration
        with patch.object(connector, 'api_budget_status') as mock_api:
            mock_api.return_value = pd.DataFrame({
                'api_name': ['Questrade', 'Alpha Vantage'],
                'requests_made': [100, 25],
                'rate_limit_hits': [0, 0]
            })
            
            api_data = connector.api_budget_status()
            chart = generate_api_budget_chart(api_data)
            
            assert chart is not None
            assert len(chart.data) == 2
            print("[OK] API budget data provider integration working")
        
        # Test phase duration integration
        with patch.object(connector, 'phase_duration_timeline') as mock_phase:
            mock_phase.return_value = pd.DataFrame({
                'timestamp': [datetime.now()] * 3,
                'phase_name': ['ingest', 'features', 'scoring'],
                'duration_ms': [100, 200, 300]
            })
            
            phase_data = connector.phase_duration_timeline()
            chart = generate_phase_duration_timeline(phase_data)
            
            assert chart is not None
            assert len(chart.data) == 3
            print("[OK] Phase duration data provider integration working")
        
        # Test confidence calibration integration
        with patch.object(connector, 'confidence_calibration_data') as mock_confidence:
            mock_confidence.return_value = pd.DataFrame({
                'raw_confidence': [0.8, 0.7],
                'calibrated_confidence': [0.75, 0.65],
                'outcome': [1, 0],
                'symbol': ['AAPL', 'MSFT']
            })
            
            confidence_data = connector.confidence_calibration_data()
            chart = generate_confidence_calibration_chart(confidence_data)
            
            assert chart is not None
            assert len(chart.data) == 2
            print("[OK] Confidence calibration data provider integration working")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Data provider integration test failed: {e}")
        logger.exception("Data provider integration test failed")
        return False

def main():
    """Run all Phase 8 smoke tests."""
    print("Phase 8 Smoke Test - Dashboard Enhancements")
    print("=" * 60)
    
    tests = [
        ("Dashboard Connector Data Providers", test_dashboard_connector),
        ("Enhanced Chart Generation", test_enhanced_charts),
        ("Dashboard Panel Creation", test_dashboard_panels),
        ("Layout Integration", test_layout_integration),
        ("Data Provider Integration", test_data_provider_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[READY] Starting {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"[PASS] {test_name} test completed successfully")
            else:
                print(f"[FAIL] {test_name} test failed")
        except Exception as e:
            print(f"[ERROR] {test_name} test encountered an error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Phase 8 Smoke Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All Phase 8 dashboard enhancements are working correctly!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
