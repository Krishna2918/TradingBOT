"""
Comprehensive Unit Tests for Advanced ML Predictive Models

This module contains comprehensive unit tests for all advanced ML models
implemented in Phase 14A: Advanced ML Predictive Models.

Author: AI Trading System
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from enterprise.advanced_ml.crash_detection import CrashDetector, CrashSignal, CrashMetrics
from enterprise.advanced_ml.bubble_detection import BubbleDetector, BubbleSignal, BubbleMetrics
from enterprise.advanced_ml.regime_prediction import RegimePredictor, RegimeSignal, RegimeMetrics
from enterprise.advanced_ml.volatility_forecasting import VolatilityForecaster, VolatilityForecast, VolatilityMetrics
from enterprise.advanced_ml.correlation_analysis import CorrelationAnalyzer, CorrelationSignal, CorrelationMetrics

class TestCrashDetector:
    """Test cases for CrashDetector class."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        # Generate realistic market data with some crash patterns
        prices = [100]
        for i in range(299):
            if i in [150, 200]:  # Simulate crashes
                change = -0.15 + np.random.normal(0, 0.02)
            else:
                change = np.random.normal(0.001, 0.02)
            prices.append(prices[-1] * (1 + change))
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 300)
        })
        
        return data
    
    @pytest.fixture
    def crash_detector(self):
        """Create CrashDetector instance for testing."""
        return CrashDetector(lookback_days=252, min_confidence=0.6)
    
    def test_crash_detector_initialization(self, crash_detector):
        """Test CrashDetector initialization."""
        assert crash_detector.lookback_days == 252
        assert crash_detector.min_confidence == 0.6
        assert not crash_detector.is_trained
        assert crash_detector.performance_metrics is None
    
    def test_calculate_crash_indicators(self, crash_detector, sample_market_data):
        """Test crash indicator calculation."""
        indicators = crash_detector._calculate_crash_indicators(sample_market_data)
        
        # Check that indicators are calculated
        expected_columns = [
            'volatility_ratio', 'momentum_5', 'momentum_20', 'momentum_divergence',
            'atr_ratio', 'volatility_clustering', 'volume_ratio', 'price_volume_trend',
            'rsi', 'rsi_divergence', 'support_resistance', 'trend_strength',
            'crash_momentum', 'panic_volume', 'liquidity_stress'
        ]
        
        for col in expected_columns:
            assert col in indicators.columns
        
        # Check that indicators are numeric
        for col in expected_columns:
            assert pd.api.types.is_numeric_dtype(indicators[col])
    
    def test_create_crash_labels(self, crash_detector, sample_market_data):
        """Test crash label creation."""
        labels = crash_detector._create_crash_labels(sample_market_data)
        
        # Check that labels are binary
        assert labels.dtype in ['int64', 'int32']
        assert set(labels.dropna().unique()).issubset({0, 1})
        
        # Check that we have some crash labels
        assert labels.sum() > 0
    
    def test_train_crash_detector(self, crash_detector, sample_market_data):
        """Test crash detector training."""
        result = crash_detector.train(sample_market_data)
        
        # Check training results
        assert crash_detector.is_trained
        assert 'performance_metrics' in result
        assert 'feature_importance' in result
        assert 'training_samples' in result
        assert 'crash_samples' in result
        
        # Check performance metrics
        metrics = result['performance_metrics']
        assert isinstance(metrics, CrashMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
    
    def test_predict_crash_probability(self, crash_detector, sample_market_data):
        """Test crash probability prediction."""
        # Train the model first
        crash_detector.train(sample_market_data)
        
        # Test prediction
        probability = crash_detector.predict_crash_probability(sample_market_data)
        
        # Check that probability is valid
        assert 0 <= probability <= 1
        assert isinstance(probability, float)
    
    def test_generate_crash_signal(self, crash_detector, sample_market_data):
        """Test crash signal generation."""
        # Train the model first
        crash_detector.train(sample_market_data)
        
        # Test signal generation
        signal = crash_detector.generate_crash_signal(sample_market_data)
        
        if signal is not None:
            assert isinstance(signal, CrashSignal)
            assert 0 <= signal.confidence <= 1
            assert signal.severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            assert isinstance(signal.explanation, str)
            assert signal.time_horizon > 0
    
    def test_crash_detector_validation(self, crash_detector, sample_market_data):
        """Test crash detector validation."""
        # Train the model first
        crash_detector.train(sample_market_data)
        
        # Create test data
        test_data = sample_market_data.iloc[-100:].copy()
        
        # Test validation
        validation_result = crash_detector.validate_model(test_data)
        
        # Check validation results
        assert 'accuracy' in validation_result
        assert 'precision' in validation_result
        assert 'recall' in validation_result
        assert 'f1_score' in validation_result
        assert 'test_samples' in validation_result


class TestBubbleDetector:
    """Test cases for BubbleDetector class."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data with bubble patterns."""
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        # Generate market data with bubble patterns
        prices = [100]
        for i in range(299):
            if 100 <= i <= 150:  # Bubble period
                change = 0.05 + np.random.normal(0, 0.02)  # High growth
            elif 151 <= i <= 160:  # Bubble burst
                change = -0.2 + np.random.normal(0, 0.05)  # Crash
            else:
                change = np.random.normal(0.001, 0.02)  # Normal
            prices.append(prices[-1] * (1 + change))
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 300)
        })
        
        return data
    
    @pytest.fixture
    def bubble_detector(self):
        """Create BubbleDetector instance for testing."""
        return BubbleDetector(lookback_days=252, min_confidence=0.6)
    
    def test_bubble_detector_initialization(self, bubble_detector):
        """Test BubbleDetector initialization."""
        assert bubble_detector.lookback_days == 252
        assert bubble_detector.min_confidence == 0.6
        assert not bubble_detector.is_trained
    
    def test_calculate_price_indicators(self, bubble_detector, sample_market_data):
        """Test price indicator calculation."""
        indicators = bubble_detector._calculate_price_indicators(sample_market_data)
        
        # Check that indicators are calculated
        expected_columns = [
            'price_momentum_1m', 'price_momentum_3m', 'price_momentum_6m', 'price_momentum_1y',
            'price_acceleration', 'volatility_1m', 'volatility_3m', 'volatility_ratio',
            'price_deviation_50', 'price_deviation_200', 'price_bubble_score',
            'exponential_growth', 'price_parabolic'
        ]
        
        for col in expected_columns:
            assert col in indicators.columns
    
    def test_calculate_volume_indicators(self, bubble_detector, sample_market_data):
        """Test volume indicator calculation."""
        indicators = bubble_detector._calculate_volume_indicators(sample_market_data)
        
        # Check that indicators are calculated
        expected_columns = [
            'volume_momentum_1m', 'volume_momentum_3m', 'volume_acceleration',
            'volume_price_trend', 'volume_spike', 'volume_bubble_score', 'volume_anomaly'
        ]
        
        for col in expected_columns:
            assert col in indicators.columns
    
    def test_calculate_sentiment_indicators(self, bubble_detector, sample_market_data):
        """Test sentiment indicator calculation."""
        indicators = bubble_detector._calculate_sentiment_indicators(sample_market_data)
        
        # Check that indicators are calculated
        expected_columns = [
            'rsi', 'rsi_extreme', 'volatility_sentiment', 'momentum_sentiment',
            'sentiment_bubble_score', 'euphoria_indicator'
        ]
        
        for col in expected_columns:
            assert col in indicators.columns
    
    def test_train_bubble_detector(self, bubble_detector, sample_market_data):
        """Test bubble detector training."""
        result = bubble_detector.train(sample_market_data)
        
        # Check training results
        assert bubble_detector.is_trained
        assert 'performance_metrics' in result
        assert 'price_mse' in result
        assert 'sentiment_mse' in result
        assert 'composite_mse' in result
    
    def test_predict_bubble_probability(self, bubble_detector, sample_market_data):
        """Test bubble probability prediction."""
        # Train the model first
        bubble_detector.train(sample_market_data)
        
        # Test prediction
        probabilities = bubble_detector.predict_bubble_probability(sample_market_data)
        
        # Check that probabilities are valid
        assert 'price_bubble' in probabilities
        assert 'volume_bubble' in probabilities
        assert 'sentiment_bubble' in probabilities
        assert 'composite_bubble' in probabilities
        
        for prob in probabilities.values():
            assert 0 <= prob <= 1
    
    def test_generate_bubble_signal(self, bubble_detector, sample_market_data):
        """Test bubble signal generation."""
        # Train the model first
        bubble_detector.train(sample_market_data)
        
        # Test signal generation
        signal = bubble_detector.generate_bubble_signal(sample_market_data, 'TEST_ASSET')
        
        if signal is not None:
            assert isinstance(signal, BubbleSignal)
            assert 0 <= signal.confidence <= 1
            assert signal.severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            assert signal.bubble_type in ['PRICE', 'VOLUME', 'SENTIMENT', 'COMPOSITE']
            assert signal.asset == 'TEST_ASSET'


class TestRegimePredictor:
    """Test cases for RegimePredictor class."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data with regime changes."""
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        # Generate market data with regime changes
        prices = [100]
        for i in range(299):
            if i < 100:  # Bull market
                change = 0.002 + np.random.normal(0, 0.015)
            elif i < 200:  # Bear market
                change = -0.001 + np.random.normal(0, 0.025)
            else:  # Sideways market
                change = np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + change))
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 300)
        })
        
        return data
    
    @pytest.fixture
    def regime_predictor(self):
        """Create RegimePredictor instance for testing."""
        return RegimePredictor(lookback_days=252, min_confidence=0.6)
    
    def test_regime_predictor_initialization(self, regime_predictor):
        """Test RegimePredictor initialization."""
        assert regime_predictor.lookback_days == 252
        assert regime_predictor.min_confidence == 0.6
        assert not regime_predictor.is_trained
    
    def test_calculate_regime_indicators(self, regime_predictor, sample_market_data):
        """Test regime indicator calculation."""
        indicators = regime_predictor._calculate_regime_indicators(sample_market_data)
        
        # Check that indicators are calculated
        expected_columns = [
            'volatility_20', 'volatility_50', 'volatility_ratio', 'volatility_percentile',
            'trend_strength', 'trend_direction', 'momentum_1m', 'momentum_3m', 'momentum_6m',
            'momentum_divergence', 'volume_ratio', 'volume_trend', 'rsi', 'rsi_divergence',
            'support_resistance', 'market_breadth', 'correlation_stability',
            'bull_bear_score', 'volatility_regime_score', 'trend_regime_score'
        ]
        
        for col in expected_columns:
            assert col in indicators.columns
    
    def test_create_regime_labels(self, regime_predictor, sample_market_data):
        """Test regime label creation."""
        labels = regime_predictor._create_regime_labels(sample_market_data)
        
        # Check that labels are created for all regime types
        assert 'bull_bear' in labels
        assert 'volatility' in labels
        assert 'trend' in labels
        
        # Check that labels have valid values
        for regime_type, regime_labels in labels.items():
            valid_values = regime_predictor.regime_definitions[regime_type]
            assert set(regime_labels.dropna().unique()).issubset(set(valid_values))
    
    def test_train_regime_predictor(self, regime_predictor, sample_market_data):
        """Test regime predictor training."""
        result = regime_predictor.train(sample_market_data)
        
        # Check training results
        assert regime_predictor.is_trained
        assert 'performance_metrics' in result
        assert 'bull_bear_accuracy' in result
        assert 'volatility_accuracy' in result
        assert 'trend_accuracy' in result
        assert 'composite_accuracy' in result
    
    def test_predict_regime_probabilities(self, regime_predictor, sample_market_data):
        """Test regime probability prediction."""
        # Train the model first
        regime_predictor.train(sample_market_data)
        
        # Test prediction
        probabilities = regime_predictor.predict_regime_probabilities(sample_market_data)
        
        # Check that probabilities are valid
        assert 'bull_bear' in probabilities
        assert 'volatility' in probabilities
        assert 'trend' in probabilities
        assert 'composite' in probabilities
        
        for regime_type, probs in probabilities.items():
            if regime_type != 'composite':
                for prob in probs.values():
                    assert 0 <= prob <= 1
    
    def test_get_current_regime(self, regime_predictor, sample_market_data):
        """Test current regime classification."""
        # Train the model first
        regime_predictor.train(sample_market_data)
        
        # Test current regime
        current_regime = regime_predictor.get_current_regime(sample_market_data)
        
        # Check that regime is classified
        assert 'bull_bear' in current_regime
        assert 'volatility' in current_regime
        assert 'trend' in current_regime
        
        # Check that values are valid
        for regime_type, regime in current_regime.items():
            valid_values = regime_predictor.regime_definitions[regime_type]
            assert regime in valid_values


class TestVolatilityForecaster:
    """Test cases for VolatilityForecaster class."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for volatility forecasting."""
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        # Generate market data with varying volatility
        prices = [100]
        for i in range(299):
            if i < 100:  # Low volatility period
                change = np.random.normal(0, 0.01)
            elif i < 200:  # High volatility period
                change = np.random.normal(0, 0.03)
            else:  # Normal volatility period
                change = np.random.normal(0, 0.02)
            prices.append(prices[-1] * (1 + change))
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 300)
        })
        
        return data
    
    @pytest.fixture
    def volatility_forecaster(self):
        """Create VolatilityForecaster instance for testing."""
        return VolatilityForecaster(lookback_days=252, forecast_horizons=[1, 5, 10, 20])
    
    def test_volatility_forecaster_initialization(self, volatility_forecaster):
        """Test VolatilityForecaster initialization."""
        assert volatility_forecaster.lookback_days == 252
        assert volatility_forecaster.forecast_horizons == [1, 5, 10, 20]
        assert not volatility_forecaster.is_trained
    
    def test_calculate_volatility_indicators(self, volatility_forecaster, sample_market_data):
        """Test volatility indicator calculation."""
        indicators = volatility_forecaster._calculate_volatility_indicators(sample_market_data)
        
        # Check that indicators are calculated
        expected_columns = [
            'rv_1d', 'rv_5d', 'rv_10d', 'rv_20d', 'vol_of_vol',
            'vol_ratio_5_20', 'vol_ratio_10_20', 'vol_percentile_20', 'vol_percentile_50',
            'garch_alpha', 'garch_beta', 'vol_clustering', 'volume_ratio',
            'vol_volume_corr', 'price_vol_corr', 'vol_momentum', 'vol_acceleration',
            'vol_mean_reversion', 'vol_regime_score'
        ]
        
        for col in expected_columns:
            assert col in indicators.columns
    
    def test_train_volatility_forecaster(self, volatility_forecaster, sample_market_data):
        """Test volatility forecaster training."""
        result = volatility_forecaster.train(sample_market_data)
        
        # Check training results
        assert volatility_forecaster.is_trained
        assert 'horizon_results' in result
        assert 'regime_training_samples' in result
        assert 'regime_distribution' in result
        
        # Check that all horizons are trained
        for horizon in volatility_forecaster.forecast_horizons:
            assert horizon in result['horizon_results']
    
    def test_predict_volatility(self, volatility_forecaster, sample_market_data):
        """Test volatility prediction."""
        # Train the model first
        volatility_forecaster.train(sample_market_data)
        
        # Test prediction for each horizon
        for horizon in volatility_forecaster.forecast_horizons:
            forecast = volatility_forecaster.predict_volatility(sample_market_data, horizon)
            
            assert isinstance(forecast, VolatilityForecast)
            assert forecast.forecast_horizon == horizon
            assert forecast.point_forecast >= 0
            assert forecast.confidence_interval_lower <= forecast.point_forecast
            assert forecast.confidence_interval_upper >= forecast.point_forecast
            assert forecast.volatility_regime in volatility_forecaster.volatility_regimes
    
    def test_predict_volatility_ensemble(self, volatility_forecaster, sample_market_data):
        """Test ensemble volatility prediction."""
        # Train the model first
        volatility_forecaster.train(sample_market_data)
        
        # Test ensemble prediction
        forecasts = volatility_forecaster.predict_volatility_ensemble(sample_market_data)
        
        # Check that forecasts are generated for all horizons
        for horizon in volatility_forecaster.forecast_horizons:
            assert horizon in forecasts
            assert isinstance(forecasts[horizon], VolatilityForecast)


class TestCorrelationAnalyzer:
    """Test cases for CorrelationAnalyzer class."""
    
    @pytest.fixture
    def sample_multi_asset_data(self):
        """Create sample multi-asset market data."""
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        # Generate correlated asset data
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        data = {}
        
        for asset in assets:
            prices = [100]
            for i in range(299):
                if i < 100:  # High correlation period
                    change = 0.001 + np.random.normal(0, 0.02)
                elif i < 200:  # Low correlation period
                    change = np.random.normal(0, 0.02)
                else:  # Normal correlation period
                    change = 0.0005 + np.random.normal(0, 0.02)
                prices.append(prices[-1] * (1 + change))
            
            data[asset] = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, 300)
            })
        
        return data
    
    @pytest.fixture
    def correlation_analyzer(self):
        """Create CorrelationAnalyzer instance for testing."""
        return CorrelationAnalyzer(lookback_days=252, min_confidence=0.6)
    
    def test_correlation_analyzer_initialization(self, correlation_analyzer):
        """Test CorrelationAnalyzer initialization."""
        assert correlation_analyzer.lookback_days == 252
        assert correlation_analyzer.min_confidence == 0.6
        assert not correlation_analyzer.is_trained
    
    def test_calculate_correlation_indicators(self, correlation_analyzer, sample_multi_asset_data):
        """Test correlation indicator calculation."""
        indicators = correlation_analyzer._calculate_correlation_indicators(sample_multi_asset_data)
        
        # Check that indicators are calculated
        expected_columns = [
            'avg_correlation_20', 'avg_correlation_50', 'max_correlation_20', 'min_correlation_20',
            'correlation_volatility', 'correlation_regime', 'correlation_trend', 'correlation_momentum',
            'market_stress', 'flight_to_quality'
        ]
        
        for col in expected_columns:
            assert col in indicators.columns
    
    def test_train_correlation_analyzer(self, correlation_analyzer, sample_multi_asset_data):
        """Test correlation analyzer training."""
        result = correlation_analyzer.train(sample_multi_asset_data)
        
        # Check training results
        assert correlation_analyzer.is_trained
        assert 'performance_metrics' in result
        assert 'breakdown_accuracy' in result
        assert 'regime_mse' in result
        assert 'forecast_mse' in result
    
    def test_predict_correlation_breakdown(self, correlation_analyzer, sample_multi_asset_data):
        """Test correlation breakdown prediction."""
        # Train the model first
        correlation_analyzer.train(sample_multi_asset_data)
        
        # Test prediction
        probability = correlation_analyzer.predict_correlation_breakdown(sample_multi_asset_data)
        
        # Check that probability is valid
        assert 0 <= probability <= 1
        assert isinstance(probability, float)
    
    def test_predict_correlation_regime_change(self, correlation_analyzer, sample_multi_asset_data):
        """Test correlation regime change prediction."""
        # Train the model first
        correlation_analyzer.train(sample_multi_asset_data)
        
        # Test prediction
        probability = correlation_analyzer.predict_correlation_regime_change(sample_multi_asset_data)
        
        # Check that probability is valid
        assert 0 <= probability <= 1
        assert isinstance(probability, float)
    
    def test_generate_correlation_signal(self, correlation_analyzer, sample_multi_asset_data):
        """Test correlation signal generation."""
        # Train the model first
        correlation_analyzer.train(sample_multi_asset_data)
        
        # Test signal generation
        signal = correlation_analyzer.generate_correlation_signal(sample_multi_asset_data)
        
        if signal is not None:
            assert isinstance(signal, CorrelationSignal)
            assert 0 <= signal.confidence <= 1
            assert signal.signal_type in ['BREAKDOWN', 'REGIME_CHANGE', 'ANOMALY']
            assert signal.severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            assert len(signal.affected_assets) > 0
    
    def test_cluster_assets(self, correlation_analyzer, sample_multi_asset_data):
        """Test asset clustering."""
        # Test clustering
        clusters = correlation_analyzer.cluster_assets(sample_multi_asset_data, n_clusters=2)
        
        # Check that clustering is performed
        assert len(clusters) == len(sample_multi_asset_data)
        assert all(isinstance(cluster_id, int) for cluster_id in clusters.values())
        assert set(clusters.values()).issubset({1, 2})


class TestAdvancedMLIntegration:
    """Integration tests for all advanced ML models."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create comprehensive sample market data."""
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        # Generate realistic market data with various patterns
        prices = [100]
        for i in range(299):
            if i in [50, 150, 250]:  # Crash periods
                change = -0.12 + np.random.normal(0, 0.03)
            elif 100 <= i <= 120:  # Bubble period
                change = 0.04 + np.random.normal(0, 0.02)
            elif i < 100:  # Bull market
                change = 0.002 + np.random.normal(0, 0.015)
            elif i < 200:  # Bear market
                change = -0.001 + np.random.normal(0, 0.025)
            else:  # Sideways market
                change = np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + change))
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 300)
        })
        
        return data
    
    def test_all_models_training(self, sample_market_data):
        """Test training of all advanced ML models."""
        # Initialize all models
        crash_detector = CrashDetector()
        bubble_detector = BubbleDetector()
        regime_predictor = RegimePredictor()
        volatility_forecaster = VolatilityForecaster()
        
        # Create multi-asset data for correlation analyzer
        multi_asset_data = {
            'AAPL': sample_market_data,
            'GOOGL': sample_market_data.copy(),
            'MSFT': sample_market_data.copy()
        }
        correlation_analyzer = CorrelationAnalyzer()
        
        # Train all models
        crash_result = crash_detector.train(sample_market_data)
        bubble_result = bubble_detector.train(sample_market_data)
        regime_result = regime_predictor.train(sample_market_data)
        volatility_result = volatility_forecaster.train(sample_market_data)
        correlation_result = correlation_analyzer.train(multi_asset_data)
        
        # Check that all models are trained
        assert crash_detector.is_trained
        assert bubble_detector.is_trained
        assert regime_predictor.is_trained
        assert volatility_forecaster.is_trained
        assert correlation_analyzer.is_trained
        
        # Check that training results are valid
        assert 'performance_metrics' in crash_result
        assert 'performance_metrics' in bubble_result
        assert 'performance_metrics' in regime_result
        assert 'horizon_results' in volatility_result
        assert 'performance_metrics' in correlation_result
    
    def test_all_models_prediction(self, sample_market_data):
        """Test prediction capabilities of all advanced ML models."""
        # Initialize and train all models
        crash_detector = CrashDetector()
        bubble_detector = BubbleDetector()
        regime_predictor = RegimePredictor()
        volatility_forecaster = VolatilityForecaster()
        
        multi_asset_data = {
            'AAPL': sample_market_data,
            'GOOGL': sample_market_data.copy(),
            'MSFT': sample_market_data.copy()
        }
        correlation_analyzer = CorrelationAnalyzer()
        
        # Train all models
        crash_detector.train(sample_market_data)
        bubble_detector.train(sample_market_data)
        regime_predictor.train(sample_market_data)
        volatility_forecaster.train(sample_market_data)
        correlation_analyzer.train(multi_asset_data)
        
        # Test predictions
        crash_prob = crash_detector.predict_crash_probability(sample_market_data)
        bubble_probs = bubble_detector.predict_bubble_probability(sample_market_data)
        regime_probs = regime_predictor.predict_regime_probabilities(sample_market_data)
        volatility_forecast = volatility_forecaster.predict_volatility(sample_market_data, 5)
        correlation_breakdown = correlation_analyzer.predict_correlation_breakdown(multi_asset_data)
        
        # Check that all predictions are valid
        assert 0 <= crash_prob <= 1
        assert all(0 <= prob <= 1 for prob in bubble_probs.values())
        assert all(0 <= prob <= 1 for probs in regime_probs.values() for prob in probs.values() if isinstance(probs, dict))
        assert volatility_forecast.point_forecast >= 0
        assert 0 <= correlation_breakdown <= 1
    
    def test_all_models_signal_generation(self, sample_market_data):
        """Test signal generation capabilities of all advanced ML models."""
        # Initialize and train all models
        crash_detector = CrashDetector()
        bubble_detector = BubbleDetector()
        regime_predictor = RegimePredictor()
        volatility_forecaster = VolatilityForecaster()
        
        multi_asset_data = {
            'AAPL': sample_market_data,
            'GOOGL': sample_market_data.copy(),
            'MSFT': sample_market_data.copy()
        }
        correlation_analyzer = CorrelationAnalyzer()
        
        # Train all models
        crash_detector.train(sample_market_data)
        bubble_detector.train(sample_market_data)
        regime_predictor.train(sample_market_data)
        volatility_forecaster.train(sample_market_data)
        correlation_analyzer.train(multi_asset_data)
        
        # Test signal generation
        crash_signal = crash_detector.generate_crash_signal(sample_market_data)
        bubble_signal = bubble_detector.generate_bubble_signal(sample_market_data, 'TEST_ASSET')
        regime_signal = regime_predictor.generate_regime_signal(sample_market_data)
        correlation_signal = correlation_analyzer.generate_correlation_signal(multi_asset_data)
        
        # Check that signals are generated (may be None if conditions not met)
        if crash_signal is not None:
            assert isinstance(crash_signal, CrashSignal)
        if bubble_signal is not None:
            assert isinstance(bubble_signal, BubbleSignal)
        if regime_signal is not None:
            assert isinstance(regime_signal, RegimeSignal)
        if correlation_signal is not None:
            assert isinstance(correlation_signal, CorrelationSignal)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
