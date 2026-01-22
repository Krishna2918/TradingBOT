"""
Phase 7 Integration Tests - Regime Awareness

Tests the integration of:
1. Regime detection and classification
2. Regime-aware ensemble weights
3. Regime-aware Kelly adjustments
4. Regime-aware ATR brackets
5. Regime policy management
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

from ai.regime_detection import RegimeDetector, MarketRegime, TrendDirection, RegimeState
from config.regime_policy_manager import RegimePolicyManager, get_regime_policy_manager
from ai.enhanced_ensemble import EnhancedEnsemble
from trading.risk import RiskManager
from trading.atr_brackets import ATRBracketManager
from config.database import get_database_manager


class TestRegimeDetection:
    """Test regime detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.regime_detector = RegimeDetector()
    
    def test_regime_classification(self):
        """Test regime classification logic."""
        # Test trending low volatility
        metrics = Mock()
        metrics.atr_percentile = 0.3  # Low volatility
        metrics.trend_consistency = 0.8  # Strong trend
        
        regime = self.regime_detector._classify_regime(metrics)
        assert regime == MarketRegime.TRENDING_LOW_VOL
        
        # Test choppy high volatility
        metrics.atr_percentile = 0.8  # High volatility
        metrics.trend_consistency = 0.2  # Weak trend
        
        regime = self.regime_detector._classify_regime(metrics)
        assert regime == MarketRegime.CHOPPY_HIGH_VOL
    
    def test_trend_direction_detection(self):
        """Test trend direction detection."""
        metrics = Mock()
        
        # Test uptrend
        metrics.price_trend_slope = 0.005
        direction = self.regime_detector._determine_trend_direction(metrics)
        assert direction == TrendDirection.UPTREND
        
        # Test downtrend
        metrics.price_trend_slope = -0.005
        direction = self.regime_detector._determine_trend_direction(metrics)
        assert direction == TrendDirection.DOWNTREND
        
        # Test sideways
        metrics.price_trend_slope = 0.0005
        direction = self.regime_detector._determine_trend_direction(metrics)
        assert direction == TrendDirection.SIDEWAYS
    
    def test_regime_confidence_calculation(self):
        """Test regime confidence calculation."""
        metrics = Mock()
        metrics.atr_percentile = 0.9  # High volatility
        metrics.trend_consistency = 0.8  # Strong trend
        
        confidence = self.regime_detector._calculate_regime_confidence(metrics, MarketRegime.TRENDING_HIGH_VOL)
        assert 0.1 <= confidence <= 0.95
        
        # Test transition regime (lower confidence)
        confidence = self.regime_detector._calculate_regime_confidence(metrics, MarketRegime.TRANSITION)
        assert confidence < 0.5
    
    def test_transition_probability_calculation(self):
        """Test transition probability calculation."""
        metrics = Mock()
        metrics.atr_percentile = 0.7  # Near threshold
        metrics.trend_consistency = 0.3  # Near threshold
        
        prob = self.regime_detector._calculate_transition_probability(metrics, MarketRegime.TRANSITION)
        assert 0.0 <= prob <= 0.8
        
        # Test non-transition regime
        prob = self.regime_detector._calculate_transition_probability(metrics, MarketRegime.TRENDING_LOW_VOL)
        assert prob >= 0.0
    
    def test_atr_calculation(self):
        """Test ATR calculation."""
        # Create sample price data
        data = pd.DataFrame({
            'high': [100, 102, 101, 103, 105],
            'low': [98, 99, 100, 101, 102],
            'close': [99, 101, 100, 102, 104]
        })
        
        atr = self.regime_detector._calculate_atr(data)
        assert atr > 0
        assert isinstance(atr, float)
    
    def test_trend_slope_calculation(self):
        """Test trend slope calculation."""
        # Create sample price data
        prices = pd.Series([100, 101, 102, 103, 104, 105])
        
        slope = self.regime_detector._calculate_trend_slope(prices)
        assert slope > 0  # Upward trend
        
        # Test downward trend
        prices = pd.Series([105, 104, 103, 102, 101, 100])
        slope = self.regime_detector._calculate_trend_slope(prices)
        assert slope < 0  # Downward trend
    
    def test_trend_consistency_calculation(self):
        """Test trend consistency calculation."""
        # Create consistent upward trend
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        consistency = self.regime_detector._calculate_trend_consistency(prices)
        assert consistency > 0.5  # Should be consistent
        
        # Create choppy prices
        prices = pd.Series([100, 102, 98, 104, 96, 108, 94, 110, 92, 112])
        consistency = self.regime_detector._calculate_trend_consistency(prices)
        assert consistency < 0.5  # Should be less consistent


class TestRegimePolicyManager:
    """Test regime policy management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.policy_manager = RegimePolicyManager()
    
    def test_ensemble_weights_retrieval(self):
        """Test ensemble weights retrieval for different regimes."""
        # Test trending low volatility regime
        weights = self.policy_manager.get_ensemble_weights(MarketRegime.TRENDING_LOW_VOL)
        assert isinstance(weights, dict)
        assert 'technical_analyst' in weights
        assert 'sentiment_analyst' in weights
        assert 'fundamental_analyst' in weights
        assert 'risk_analyst' in weights
        assert 'market_regime_analyst' in weights
        
        # Weights should sum to approximately 1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # Test choppy high volatility regime
        weights = self.policy_manager.get_ensemble_weights(MarketRegime.CHOPPY_HIGH_VOL)
        assert weights['risk_analyst'] > weights['technical_analyst']  # Higher risk weight in choppy volatile markets
    
    def test_kelly_adjustments_retrieval(self):
        """Test Kelly adjustments retrieval for different regimes."""
        # Test trending low volatility regime
        adjustments = self.policy_manager.get_kelly_adjustments(MarketRegime.TRENDING_LOW_VOL)
        assert isinstance(adjustments, dict)
        assert 'kelly_multiplier' in adjustments
        assert 'max_position_risk' in adjustments
        assert 'confidence_threshold' in adjustments
        
        # Test choppy high volatility regime
        adjustments = self.policy_manager.get_kelly_adjustments(MarketRegime.CHOPPY_HIGH_VOL)
        assert adjustments['kelly_multiplier'] < 1.0  # Should reduce position sizes
        assert adjustments['max_position_risk'] < 0.02  # Should have lower risk limits
        assert adjustments['confidence_threshold'] > 0.7  # Should require higher confidence
    
    def test_atr_brackets_retrieval(self):
        """Test ATR brackets retrieval for different regimes."""
        # Test trending low volatility regime
        brackets = self.policy_manager.get_atr_brackets(MarketRegime.TRENDING_LOW_VOL)
        assert isinstance(brackets, dict)
        assert 'atr_multiplier' in brackets
        assert 'r_multiple' in brackets
        assert 'min_stop_loss_percent' in brackets
        assert 'max_stop_loss_percent' in brackets
        
        # Test choppy high volatility regime
        brackets = self.policy_manager.get_atr_brackets(MarketRegime.CHOPPY_HIGH_VOL)
        assert brackets['atr_multiplier'] > 2.0  # Should have wider stops
        assert brackets['r_multiple'] < 1.5  # Should have lower reward targets
    
    def test_lookback_windows_retrieval(self):
        """Test lookback windows retrieval for different regimes."""
        # Test trending low volatility regime
        windows = self.policy_manager.get_lookback_windows(MarketRegime.TRENDING_LOW_VOL)
        assert isinstance(windows, dict)
        assert 'price_lookback' in windows
        assert 'volume_lookback' in windows
        assert 'volatility_lookback' in windows
        assert 'sentiment_lookback' in windows
        
        # Test transition regime
        windows = self.policy_manager.get_lookback_windows(MarketRegime.TRANSITION)
        assert windows['price_lookback'] < 20  # Should have shorter lookbacks during transitions
    
    def test_position_management_retrieval(self):
        """Test position management parameters retrieval for different regimes."""
        # Test trending low volatility regime
        management = self.policy_manager.get_position_management(MarketRegime.TRENDING_LOW_VOL)
        assert isinstance(management, dict)
        assert 'max_positions' in management
        assert 'max_holding_days' in management
        assert 'rebalance_frequency' in management
        assert 'stop_loss_trailing' in management
        
        # Test choppy high volatility regime
        management = self.policy_manager.get_position_management(MarketRegime.CHOPPY_HIGH_VOL)
        assert management['max_positions'] < 10  # Should have fewer positions
        assert management['max_holding_days'] < 25  # Should have shorter holding periods
    
    def test_risk_management_retrieval(self):
        """Test risk management parameters retrieval for different regimes."""
        # Test trending low volatility regime
        risk = self.policy_manager.get_risk_management(MarketRegime.TRENDING_LOW_VOL)
        assert isinstance(risk, dict)
        assert 'max_portfolio_risk' in risk
        assert 'daily_loss_limit' in risk
        assert 'drawdown_threshold' in risk
        assert 'correlation_limit' in risk
        
        # Test transition regime
        risk = self.policy_manager.get_risk_management(MarketRegime.TRANSITION)
        assert risk['max_portfolio_risk'] < 0.15  # Should have lower portfolio risk
        assert risk['daily_loss_limit'] < 0.02  # Should have lower daily loss limit
    
    def test_feature_flags(self):
        """Test feature flag functionality."""
        # Test default feature flags
        assert self.policy_manager.is_feature_enabled('enable_regime_weights')
        assert self.policy_manager.is_feature_enabled('enable_regime_kelly')
        assert self.policy_manager.is_feature_enabled('enable_regime_brackets')
        
        # Test non-existent feature flag
        assert not self.policy_manager.is_feature_enabled('non_existent_feature')
    
    def test_transition_rules(self):
        """Test transition rules retrieval."""
        rules = self.policy_manager.get_transition_rules()
        assert isinstance(rules, dict)
        assert 'min_regime_duration' in rules
        assert 'regime_change_threshold' in rules
        assert 'transition_smoothing' in rules
        assert 'regime_persistence' in rules


class TestRegimeAwareEnsemble:
    """Test regime-aware ensemble functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ensemble = EnhancedEnsemble()
    
    def test_regime_aware_weight_blending(self):
        """Test that ensemble weights are blended with regime weights."""
        with patch('ai.regime_detection.detect_current_regime') as mock_regime:
            with patch('config.regime_policy_manager.get_ensemble_weights') as mock_regime_weights:
                with patch('ai.adaptive_weights.get_ensemble_weights') as mock_adaptive_weights:
                    # Mock regime detection
                    mock_regime.return_value = Mock(regime=MarketRegime.TRENDING_LOW_VOL)
                    
                    # Mock regime weights
                    mock_regime_weights.return_value = {
                        'technical_analyst': 0.35,
                        'sentiment_analyst': 0.20,
                        'fundamental_analyst': 0.25,
                        'risk_analyst': 0.10,
                        'market_regime_analyst': 0.10
                    }
                    
                    # Mock adaptive weights
                    mock_adaptive_weights.return_value = {
                        'technical_analyst': 0.30,
                        'sentiment_analyst': 0.25,
                        'fundamental_analyst': 0.20,
                        'risk_analyst': 0.15,
                        'market_regime_analyst': 0.10
                    }
                    
                    # Mock market analysis
                    market_analysis = Mock()
                    market_analysis.technical_indicators = {'overall_score': 0.7}
                    market_analysis.sentiment_score = 0.6
                    market_analysis.fundamental_score = 0.8
                    market_analysis.market_regime = "TRENDING"
                    market_analysis.volatility = 0.02
                    
                    # Test ensemble analysis
                    result = self.ensemble._run_ensemble_analysis(market_analysis, "ENTRY")
                    
                    # Verify regime weights were called
                    mock_regime_weights.assert_called_once_with(MarketRegime.TRENDING_LOW_VOL)
                    
                    # Verify result contains weights
                    assert 'weights' in result or 'confidence' in result


class TestRegimeAwareRiskManagement:
    """Test regime-aware risk management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager()
    
    def test_regime_aware_kelly_adjustments(self):
        """Test that Kelly sizing uses regime adjustments."""
        with patch('ai.regime_detection.detect_current_regime') as mock_regime:
            with patch('config.regime_policy_manager.get_kelly_adjustments') as mock_adjustments:
                with patch('adaptive.confidence_calibration.get_confidence_calibrator') as mock_calibrator:
                    # Mock regime detection
                    mock_regime.return_value = Mock(regime=MarketRegime.CHOPPY_HIGH_VOL)
                    
                    # Mock Kelly adjustments
                    mock_adjustments.return_value = {
                        'kelly_multiplier': 0.6,
                        'max_position_risk': 0.01,
                        'confidence_threshold': 0.8
                    }
                    
                    # Mock confidence calibrator
                    mock_cal = Mock()
                    mock_cal.calibrate_confidence.return_value = 0.75
                    mock_calibrator.return_value = mock_cal
                    
                    # Mock database calls
                    with patch('config.database.execute_query') as mock_query:
                        mock_query.return_value = []
                        
                        # Test position sizing
                        metrics = self.risk_manager.calculate_position_size(
                            signal_confidence=0.75,
                            account_balance=10000.0,
                            volatility=0.02,
                            entry_price=100.0,
                            stop_loss=98.0,
                            model_name="test_model",
                            symbol="AAPL",
                            trade_date=datetime.now()
                        )
                        
                        # Verify regime adjustments were called
                        mock_adjustments.assert_called_once_with(MarketRegime.CHOPPY_HIGH_VOL)
                        
                        # Verify metrics contain regime information
                        assert hasattr(metrics, 'calibrated_confidence')
                        assert hasattr(metrics, 'drawdown_scale')
                        assert hasattr(metrics, 'kelly_fraction')
    
    def test_regime_confidence_threshold(self):
        """Test that confidence threshold is enforced by regime."""
        with patch('ai.regime_detection.detect_current_regime') as mock_regime:
            with patch('config.regime_policy_manager.get_kelly_adjustments') as mock_adjustments:
                with patch('adaptive.confidence_calibration.get_confidence_calibrator') as mock_calibrator:
                    # Mock regime detection
                    mock_regime.return_value = Mock(regime=MarketRegime.TRANSITION)
                    
                    # Mock Kelly adjustments with high confidence threshold
                    mock_adjustments.return_value = {
                        'kelly_multiplier': 0.5,
                        'max_position_risk': 0.008,
                        'confidence_threshold': 0.85
                    }
                    
                    # Mock confidence calibrator with low confidence
                    mock_cal = Mock()
                    mock_cal.calibrate_confidence.return_value = 0.7  # Below threshold
                    mock_calibrator.return_value = mock_cal
                    
                    # Mock database calls
                    with patch('config.database.execute_query') as mock_query:
                        mock_query.return_value = []
                        
                        # Test position sizing with low confidence
                        metrics = self.risk_manager.calculate_position_size(
                            signal_confidence=0.7,
                            account_balance=10000.0,
                            volatility=0.02,
                            entry_price=100.0,
                            stop_loss=98.0,
                            model_name="test_model",
                            symbol="AAPL",
                            trade_date=datetime.now()
                        )
                        
                        # Should return minimal position size due to low confidence
                        assert metrics.position_size == 0
                        assert metrics.risk_amount == 0


class TestRegimeAwareATRBrackets:
    """Test regime-aware ATR brackets."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.atr_manager = ATRBracketManager()
    
    def test_regime_aware_bracket_parameters(self):
        """Test that ATR brackets use regime-specific parameters."""
        with patch('ai.regime_detection.detect_current_regime') as mock_regime:
            with patch('config.regime_policy_manager.get_atr_brackets') as mock_brackets:
                # Mock regime detection
                mock_regime.return_value = Mock(regime=MarketRegime.TRENDING_HIGH_VOL)
                
                # Mock bracket parameters
                mock_brackets.return_value = {
                    'atr_multiplier': 2.5,
                    'r_multiple': 1.5,
                    'min_stop_loss_percent': 0.02,
                    'max_stop_loss_percent': 0.12
                }
                
                # Test bracket calculation
                params = self.atr_manager.calculate_atr_brackets(
                    symbol="AAPL",
                    entry_price=150.0,
                    atr=3.0
                )
                
                # Verify regime brackets were called
                mock_brackets.assert_called_once_with(MarketRegime.TRENDING_HIGH_VOL)
                
                # Verify bracket parameters
                assert params.symbol == "AAPL"
                assert params.entry_price == 150.0
                assert params.atr == 3.0
                assert params.atr_multiplier == 2.5
                assert params.r_multiple == 1.5
    
    def test_regime_specific_stop_loss_bounds(self):
        """Test that stop loss bounds are regime-specific."""
        with patch('ai.regime_detection.detect_current_regime') as mock_regime:
            with patch('config.regime_policy_manager.get_atr_brackets') as mock_brackets:
                # Mock regime detection for choppy high volatility
                mock_regime.return_value = Mock(regime=MarketRegime.CHOPPY_HIGH_VOL)
                
                # Mock bracket parameters with wide stops
                mock_brackets.return_value = {
                    'atr_multiplier': 3.0,
                    'r_multiple': 1.2,
                    'min_stop_loss_percent': 0.025,
                    'max_stop_loss_percent': 0.15
                }
                
                # Test bracket calculation
                params = self.atr_manager.calculate_atr_brackets(
                    symbol="AAPL",
                    entry_price=150.0,
                    atr=3.0
                )
                
                # Verify stop loss is within regime-specific bounds
                stop_loss_percent = (150.0 - params.stop_loss) / 150.0
                assert 0.025 <= stop_loss_percent <= 0.15


class TestDatabaseIntegration:
    """Test database integration for regime state."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = get_database_manager()
    
    def test_regime_state_table_exists(self):
        """Test that regime_state table exists."""
        with self.db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='regime_state'")
            result = cursor.fetchone()
            assert result is not None
    
    def test_regime_state_table_schema(self):
        """Test that regime_state table has correct schema."""
        with self.db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(regime_state)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Check for required columns
            required_columns = [
                'id', 'timestamp', 'symbol', 'regime', 'trend_direction',
                'volatility_level', 'trend_strength', 'volatility_ratio',
                'atr_percentile', 'regime_confidence', 'transition_probability',
                'mode', 'created_at'
            ]
            
            for column in required_columns:
                assert column in columns
    
    def test_regime_state_indexes(self):
        """Test that regime_state table has proper indexes."""
        with self.db_manager.get_connection("DEMO") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='regime_state'")
            indexes = [row[0] for row in cursor.fetchall()]
            
            # Check for required indexes
            required_indexes = [
                'idx_regime_state_symbol',
                'idx_regime_state_timestamp',
                'idx_regime_state_regime',
                'idx_regime_state_mode'
            ]
            
            for index in required_indexes:
                assert index in indexes


class TestRegimeTransitionSmoothing:
    """Test regime transition smoothing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.regime_detector = RegimeDetector()
    
    def test_regime_transition_detection(self):
        """Test detection of regime transitions."""
        # Test metrics near transition thresholds
        metrics = Mock()
        metrics.atr_percentile = 0.7  # Near volatility threshold
        metrics.trend_consistency = 0.3  # Near trend threshold
        
        # Test transition probability calculation
        prob = self.regime_detector._calculate_transition_probability(metrics, MarketRegime.TRANSITION)
        assert prob > 0.0
        
        # Test non-transition regime
        prob = self.regime_detector._calculate_transition_probability(metrics, MarketRegime.TRENDING_LOW_VOL)
        assert prob >= 0.0
    
    def test_regime_confidence_variation(self):
        """Test that regime confidence varies appropriately."""
        metrics = Mock()
        
        # Test high confidence scenario
        metrics.atr_percentile = 0.9  # Clear high volatility
        metrics.trend_consistency = 0.8  # Clear strong trend
        confidence = self.regime_detector._calculate_regime_confidence(metrics, MarketRegime.TRENDING_HIGH_VOL)
        assert confidence > 0.5
        
        # Test low confidence scenario
        metrics.atr_percentile = 0.5  # Neutral volatility
        metrics.trend_consistency = 0.3  # Weak trend
        confidence = self.regime_detector._calculate_regime_confidence(metrics, MarketRegime.TRANSITION)
        assert confidence < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
