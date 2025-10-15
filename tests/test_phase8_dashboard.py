"""
Phase 8 Dashboard Enhancement Tests

Tests the enhanced dashboard functionality including:
1. New data providers in connector
2. Enhanced chart generation
3. New dashboard panels
4. Layout integration
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dashboard.connector import DashboardConnector
from dashboard.charts import (
    generate_api_budget_chart,
    generate_phase_duration_timeline,
    generate_confidence_calibration_chart,
    generate_ensemble_weights_chart,
    generate_drawdown_regime_chart,
    generate_system_health_chart,
    generate_rationale_trace_chart
)
from dashboard.enhanced_layout import (
    create_api_budget_panel,
    create_phase_duration_panel,
    create_confidence_calibration_panel,
    create_ensemble_weights_panel,
    create_drawdown_regime_panel,
    create_system_health_panel,
    create_rationale_trace_panel
)


class TestDashboardConnector:
    """Test enhanced dashboard connector data providers."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.connector = DashboardConnector()
    
    def test_api_budget_status(self):
        """Test API budget status data provider."""
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock query result
            mock_df = pd.DataFrame({
                'api_name': ['Questrade', 'Alpha Vantage', 'Finnhub'],
                'requests_made': [100, 25, 60],
                'requests_limit': [1000, 25, 60],
                'daily_requests': [500, 25, 60],
                'daily_limit': [1000, 25, 60],
                'rate_limit_hits': [0, 0, 0],
                'last_request_time': [datetime.now()] * 3,
                'window_start': [datetime.now()] * 3,
                'window_end': [datetime.now()] * 3,
                'mode': ['DEMO'] * 3
            })
            mock_conn.cursor.return_value.execute.return_value = None
            pd.read_sql_query.return_value = mock_df
            
            result = self.connector.api_budget_status()
            
            assert isinstance(result, pd.DataFrame)
            assert 'api_name' in result.columns
            assert 'requests_made' in result.columns
            assert 'rate_limit_hits' in result.columns
    
    def test_phase_duration_timeline(self):
        """Test phase duration timeline data provider."""
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock query result
            mock_df = pd.DataFrame({
                'timestamp': [datetime.now()] * 5,
                'phase_name': ['ingest', 'features', 'scoring', 'ensemble', 'sizing'],
                'step_name': ['data_collection', 'feature_calc', 'ai_analysis', 'ensemble', 'risk_calc'],
                'duration_ms': [100, 200, 300, 150, 250],
                'status': ['success'] * 5,
                'detail': ['Completed successfully'] * 5,
                'mode': ['DEMO'] * 5
            })
            mock_conn.cursor.return_value.execute.return_value = None
            pd.read_sql_query.return_value = mock_df
            
            result = self.connector.phase_duration_timeline()
            
            assert isinstance(result, pd.DataFrame)
            assert 'phase_name' in result.columns
            assert 'duration_ms' in result.columns
            assert 'status' in result.columns
    
    def test_confidence_calibration_data(self):
        """Test confidence calibration data provider."""
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock query result
            mock_df = pd.DataFrame({
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
            pd.read_sql_query.return_value = mock_df
            
            result = self.connector.confidence_calibration_data()
            
            assert isinstance(result, pd.DataFrame)
            assert 'raw_confidence' in result.columns
            assert 'calibrated_confidence' in result.columns
            assert 'outcome' in result.columns
    
    def test_ensemble_weights_history(self):
        """Test ensemble weights history data provider."""
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock query result
            mock_df = pd.DataFrame({
                'timestamp': [datetime.now()] * 5,
                'model': ['technical_analyst', 'sentiment_analyst', 'fundamental_analyst', 'risk_analyst', 'market_regime_analyst'],
                'weight': [0.3, 0.25, 0.2, 0.15, 0.1],
                'brier_score': [0.2, 0.25, 0.3, 0.15, 0.35],
                'accuracy': [0.8, 0.75, 0.7, 0.85, 0.65],
                'sample_count': [100, 100, 100, 100, 100],
                'created_at': [datetime.now()] * 5
            })
            mock_conn.cursor.return_value.execute.return_value = None
            pd.read_sql_query.return_value = mock_df
            
            result = self.connector.ensemble_weights_history()
            
            assert isinstance(result, pd.DataFrame)
            assert 'model' in result.columns
            assert 'weight' in result.columns
            assert 'brier_score' in result.columns
    
    def test_drawdown_and_regime_data(self):
        """Test drawdown and regime data provider."""
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
            
            result = self.connector.drawdown_and_regime_data()
            
            assert isinstance(result, pd.DataFrame)
            # Should have merged data or at least one of the dataframes
            assert len(result) > 0
    
    def test_model_rationale_trace(self):
        """Test model rationale trace data provider."""
        with patch('duckdb.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock query result
            mock_df = pd.DataFrame({
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
            mock_conn.execute.return_value.fetch_df.return_value = mock_df
            
            result = self.connector.model_rationale_trace()
            
            assert isinstance(result, list)
            assert len(result) == 3
            assert 'timestamp' in result[0]
            assert 'symbol' in result[0]
            assert 'technical_analysis' in result[0]
    
    def test_system_health_metrics(self):
        """Test system health metrics data provider."""
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Mock API status data
            api_df = pd.DataFrame({
                'api_name': ['Questrade', 'Alpha Vantage'],
                'status': ['healthy', 'healthy'],
                'response_time_ms': [150, 200],
                'last_validated': [datetime.now()] * 2
            })
            
            # Mock phase performance data
            phase_df = pd.DataFrame({
                'phase_name': ['ingest', 'scoring'],
                'avg_duration_ms': [100, 300],
                'execution_count': [10, 10],
                'success_count': [10, 9]
            })
            
            # Mock risk events data
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
            
            result = self.connector.system_health_metrics()
            
            assert isinstance(result, dict)
            assert 'api_status' in result
            assert 'phase_performance' in result
            assert 'risk_events' in result
            assert 'timestamp' in result


class TestEnhancedCharts:
    """Test enhanced chart generation functions."""
    
    def test_api_budget_chart(self):
        """Test API budget chart generation."""
        # Test with data
        api_data = pd.DataFrame({
            'api_name': ['Questrade', 'Alpha Vantage', 'Finnhub'],
            'requests_made': [100, 25, 60],
            'rate_limit_hits': [0, 0, 1]
        })
        
        chart = generate_api_budget_chart(api_data)
        
        assert chart is not None
        assert chart.layout.title.text == "API Budget Status"
        assert len(chart.data) == 2  # Two traces: requests made and rate limit hits
        
        # Test with empty data
        empty_chart = generate_api_budget_chart(pd.DataFrame())
        assert empty_chart is not None
        assert "No Data" in empty_chart.layout.title.text
    
    def test_phase_duration_timeline(self):
        """Test phase duration timeline chart generation."""
        # Test with data
        phase_data = pd.DataFrame({
            'timestamp': [datetime.now()] * 5,
            'phase_name': ['ingest', 'features', 'scoring', 'ensemble', 'sizing'],
            'duration_ms': [100, 200, 300, 150, 250]
        })
        
        chart = generate_phase_duration_timeline(phase_data)
        
        assert chart is not None
        assert chart.layout.title.text == "Phase Duration Timeline"
        assert len(chart.data) == 5  # One trace per phase
        
        # Test with empty data
        empty_chart = generate_phase_duration_timeline(pd.DataFrame())
        assert empty_chart is not None
        assert "No Data" in empty_chart.layout.title.text
    
    def test_confidence_calibration_chart(self):
        """Test confidence calibration chart generation."""
        # Test with data
        confidence_data = pd.DataFrame({
            'raw_confidence': [0.8, 0.7, 0.75],
            'calibrated_confidence': [0.75, 0.65, 0.70],
            'outcome': [1, 0, 1],
            'symbol': ['AAPL', 'MSFT', 'GOOGL']
        })
        
        chart = generate_confidence_calibration_chart(confidence_data)
        
        assert chart is not None
        assert "Confidence Calibration" in chart.layout.title.text
        assert len(chart.data) == 2  # Scatter plot + perfect calibration line
        
        # Test with empty data
        empty_chart = generate_confidence_calibration_chart(pd.DataFrame())
        assert empty_chart is not None
        assert "No Data" in empty_chart.layout.title.text
    
    def test_ensemble_weights_chart(self):
        """Test ensemble weights chart generation."""
        # Test with data
        weights_data = pd.DataFrame({
            'timestamp': [datetime.now()] * 5,
            'model': ['technical_analyst', 'sentiment_analyst', 'fundamental_analyst', 'risk_analyst', 'market_regime_analyst'],
            'weight': [0.3, 0.25, 0.2, 0.15, 0.1]
        })
        
        chart = generate_ensemble_weights_chart(weights_data)
        
        assert chart is not None
        assert "Ensemble Weights History" in chart.layout.title.text
        assert len(chart.data) == 5  # One trace per model
        
        # Test with empty data
        empty_chart = generate_ensemble_weights_chart(pd.DataFrame())
        assert empty_chart is not None
        assert "No Data" in empty_chart.layout.title.text
    
    def test_drawdown_regime_chart(self):
        """Test drawdown and regime chart generation."""
        # Test with data
        regime_data = pd.DataFrame({
            'timestamp': [datetime.now()] * 3,
            'portfolio_value': [10000, 10100, 9950],
            'daily_pnl': [0, 100, -150],
            'regime': ['TRENDING_LOW_VOL', 'TRENDING_LOW_VOL', 'CHOPPY_HIGH_VOL']
        })
        
        chart = generate_drawdown_regime_chart(regime_data)
        
        assert chart is not None
        assert "Portfolio Performance & Market Regime" in chart.layout.title.text
        assert len(chart.data) >= 2  # Portfolio value + daily P&L + regime markers
        
        # Test with empty data
        empty_chart = generate_drawdown_regime_chart(pd.DataFrame())
        assert empty_chart is not None
        assert "No Data" in empty_chart.layout.title.text
    
    def test_system_health_chart(self):
        """Test system health chart generation."""
        # Test with data
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
        
        chart = generate_system_health_chart(health_data)
        
        assert chart is not None
        assert chart.layout.title.text == "System Health Metrics"
        assert len(chart.data) == 2  # API response time + phase duration
        
        # Test with empty data
        empty_chart = generate_system_health_chart({})
        assert empty_chart is not None
        assert chart.layout.title.text == "System Health Metrics"
    
    def test_rationale_trace_chart(self):
        """Test rationale trace chart generation."""
        # Test with data
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
        
        chart = generate_rationale_trace_chart(rationale_data)
        
        assert chart is not None
        assert "Model Rationale Trace" in chart.layout.title.text
        assert len(chart.data) == 1  # One trace for ensemble model
        
        # Test with empty data
        empty_chart = generate_rationale_trace_chart([])
        assert empty_chart is not None
        assert "No Data" in empty_chart.layout.title.text


class TestDashboardPanels:
    """Test dashboard panel creation functions."""
    
    def test_api_budget_panel(self):
        """Test API budget panel creation."""
        with patch('dashboard.enhanced_layout.DashboardConnector') as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector
            mock_connector.api_budget_status.return_value = pd.DataFrame({
                'api_name': ['Questrade'],
                'requests_made': [100],
                'rate_limit_hits': [0]
            })
            
            with patch('dashboard.enhanced_layout.generate_api_budget_chart') as mock_chart:
                mock_chart.return_value = Mock()
                
                panel = create_api_budget_panel()
                
                assert panel is not None
                assert hasattr(panel, 'children')
                mock_connector.api_budget_status.assert_called_once()
                mock_chart.assert_called_once()
    
    def test_phase_duration_panel(self):
        """Test phase duration panel creation."""
        with patch('dashboard.enhanced_layout.DashboardConnector') as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector
            mock_connector.phase_duration_timeline.return_value = pd.DataFrame({
                'phase_name': ['ingest'],
                'duration_ms': [100]
            })
            
            with patch('dashboard.enhanced_layout.generate_phase_duration_timeline') as mock_chart:
                mock_chart.return_value = Mock()
                
                panel = create_phase_duration_panel()
                
                assert panel is not None
                assert hasattr(panel, 'children')
                mock_connector.phase_duration_timeline.assert_called_once()
                mock_chart.assert_called_once()
    
    def test_confidence_calibration_panel(self):
        """Test confidence calibration panel creation."""
        with patch('dashboard.enhanced_layout.DashboardConnector') as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector
            mock_connector.confidence_calibration_data.return_value = pd.DataFrame({
                'raw_confidence': [0.8],
                'calibrated_confidence': [0.75]
            })
            
            with patch('dashboard.enhanced_layout.generate_confidence_calibration_chart') as mock_chart:
                mock_chart.return_value = Mock()
                
                panel = create_confidence_calibration_panel()
                
                assert panel is not None
                assert hasattr(panel, 'children')
                mock_connector.confidence_calibration_data.assert_called_once()
                mock_chart.assert_called_once()
    
    def test_ensemble_weights_panel(self):
        """Test ensemble weights panel creation."""
        with patch('dashboard.enhanced_layout.DashboardConnector') as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector
            mock_connector.ensemble_weights_history.return_value = pd.DataFrame({
                'model': ['technical_analyst'],
                'weight': [0.3]
            })
            
            with patch('dashboard.enhanced_layout.generate_ensemble_weights_chart') as mock_chart:
                mock_chart.return_value = Mock()
                
                panel = create_ensemble_weights_panel()
                
                assert panel is not None
                assert hasattr(panel, 'children')
                mock_connector.ensemble_weights_history.assert_called_once()
                mock_chart.assert_called_once()
    
    def test_drawdown_regime_panel(self):
        """Test drawdown and regime panel creation."""
        with patch('dashboard.enhanced_layout.DashboardConnector') as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector
            mock_connector.drawdown_and_regime_data.return_value = pd.DataFrame({
                'portfolio_value': [10000],
                'regime': ['TRENDING_LOW_VOL']
            })
            
            with patch('dashboard.enhanced_layout.generate_drawdown_regime_chart') as mock_chart:
                mock_chart.return_value = Mock()
                
                panel = create_drawdown_regime_panel()
                
                assert panel is not None
                assert hasattr(panel, 'children')
                mock_connector.drawdown_and_regime_data.assert_called_once()
                mock_chart.assert_called_once()
    
    def test_system_health_panel(self):
        """Test system health panel creation."""
        with patch('dashboard.enhanced_layout.DashboardConnector') as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector
            mock_connector.system_health_metrics.return_value = {
                'api_status': [],
                'phase_performance': []
            }
            
            with patch('dashboard.enhanced_layout.generate_system_health_chart') as mock_chart:
                mock_chart.return_value = Mock()
                
                panel = create_system_health_panel()
                
                assert panel is not None
                assert hasattr(panel, 'children')
                mock_connector.system_health_metrics.assert_called_once()
                mock_chart.assert_called_once()
    
    def test_rationale_trace_panel(self):
        """Test rationale trace panel creation."""
        with patch('dashboard.enhanced_layout.DashboardConnector') as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector
            mock_connector.model_rationale_trace.return_value = [
                {'timestamp': datetime.now(), 'symbol': 'AAPL', 'score': 0.8}
            ]
            
            with patch('dashboard.enhanced_layout.generate_rationale_trace_chart') as mock_chart:
                mock_chart.return_value = Mock()
                
                panel = create_rationale_trace_panel()
                
                assert panel is not None
                assert hasattr(panel, 'children')
                mock_connector.model_rationale_trace.assert_called_once()
                mock_chart.assert_called_once()


class TestDataProviderIntegration:
    """Test integration between data providers and charts."""
    
    def test_connector_chart_integration(self):
        """Test that connector data works with chart generation."""
        connector = DashboardConnector()
        
        # Test API budget integration
        with patch.object(connector, 'api_budget_status') as mock_api:
            mock_api.return_value = pd.DataFrame({
                'api_name': ['Questrade'],
                'requests_made': [100],
                'rate_limit_hits': [0]
            })
            
            api_data = connector.api_budget_status()
            chart = generate_api_budget_chart(api_data)
            
            assert chart is not None
            assert len(chart.data) == 2
        
        # Test phase duration integration
        with patch.object(connector, 'phase_duration_timeline') as mock_phase:
            mock_phase.return_value = pd.DataFrame({
                'timestamp': [datetime.now()],
                'phase_name': ['ingest'],
                'duration_ms': [100]
            })
            
            phase_data = connector.phase_duration_timeline()
            chart = generate_phase_duration_timeline(phase_data)
            
            assert chart is not None
            assert len(chart.data) == 1
    
    def test_empty_data_handling(self):
        """Test that empty data is handled gracefully."""
        # Test all chart functions with empty data
        empty_df = pd.DataFrame()
        
        charts = [
            generate_api_budget_chart(empty_df),
            generate_phase_duration_timeline(empty_df),
            generate_confidence_calibration_chart(empty_df),
            generate_ensemble_weights_chart(empty_df),
            generate_drawdown_regime_chart(empty_df),
            generate_system_health_chart({}),
            generate_rationale_trace_chart([])
        ]
        
        for chart in charts:
            assert chart is not None
            assert "No Data" in chart.layout.title.text or chart.layout.title.text == "System Health Metrics"
    
    def test_data_type_validation(self):
        """Test that data type validation works correctly."""
        # Test with correct data types
        api_data = pd.DataFrame({
            'api_name': ['Questrade'],
            'requests_made': [100],
            'rate_limit_hits': [0]
        })
        
        chart = generate_api_budget_chart(api_data)
        assert chart is not None
        
        # Test with missing columns (should handle gracefully)
        incomplete_data = pd.DataFrame({
            'api_name': ['Questrade']
        })
        
        # Should not crash, but may show "No Data" or handle missing columns
        try:
            chart = generate_api_budget_chart(incomplete_data)
            assert chart is not None
        except KeyError:
            # Expected if required columns are missing
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
