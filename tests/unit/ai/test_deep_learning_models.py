"""
Unit tests for deep learning models.

This module contains comprehensive tests for all deep learning models
including LSTM, CNN-LSTM, Transformer, and Autoencoder models.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from ai.deep_learning.lstm_models import LSTMPricePredictor, LSTMTrendPredictor
from ai.deep_learning.cnn_lstm_models import CNNLSTMPredictor, MultiScaleCNNLSTM
from ai.deep_learning.transformer_models import TransformerPredictor, FinancialTransformer
from ai.deep_learning.autoencoder_models import MarketAutoencoder, LSTMAnomalyDetector, AnomalyDetector
from ai.deep_learning.model_manager import DeepLearningModelManager
from ai.deep_learning.feature_engineering import DeepLearningFeatureEngineer

class TestLSTMPricePredictor:
    """Test LSTM price prediction model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 105 + np.random.randn(100).cumsum(),
            'low': 95 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': 1000000 + np.random.randint(-100000, 100000, 100),
            'sma_20': 100 + np.random.randn(100).cumsum(),
            'ema_12': 100 + np.random.randn(100).cumsum(),
            'rsi': 50 + np.random.randn(100) * 10,
            'macd': np.random.randn(100),
            'bb_upper': 105 + np.random.randn(100).cumsum()
        })
        return data
    
    @pytest.fixture
    def lstm_model(self):
        """Create LSTM model instance."""
        return LSTMPricePredictor(
            sequence_length=30,
            features=10,
            lstm_units=[32, 16],
            dropout_rate=0.2,
            learning_rate=0.001
        )
    
    def test_lstm_initialization(self, lstm_model):
        """Test LSTM model initialization."""
        assert lstm_model.sequence_length == 30
        assert lstm_model.features == 10
        assert lstm_model.lstm_units == [32, 16]
        assert lstm_model.dropout_rate == 0.2
        assert lstm_model.learning_rate == 0.001
        assert not lstm_model.is_trained
    
    def test_lstm_build_model(self, lstm_model):
        """Test LSTM model building."""
        model = lstm_model.build_model()
        assert model is not None
        assert lstm_model.model is not None
        assert model.count_params() > 0
    
    def test_lstm_prepare_data(self, lstm_model, sample_data):
        """Test LSTM data preparation."""
        X, y = lstm_model.prepare_data(sample_data, 'close')
        
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == lstm_model.sequence_length
        assert X.shape[2] == lstm_model.features
        assert len(y) > 0
    
    @patch('tensorflow.keras.models.Sequential')
    def test_lstm_training(self, mock_sequential, lstm_model, sample_data):
        """Test LSTM model training."""
        # Mock the model
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        mock_model.count_params.return_value = 1000
        
        # Mock training
        mock_history = Mock()
        mock_history.history = {'loss': [0.5, 0.4, 0.3], 'val_loss': [0.6, 0.5, 0.4]}
        mock_model.fit.return_value = mock_history
        
        # Prepare data
        X, y = lstm_model.prepare_data(sample_data, 'close')
        
        # Train model
        history = lstm_model.train(X, y, epochs=3, verbose=0)
        
        assert 'loss' in history
        assert 'val_loss' in history
        assert lstm_model.is_trained
    
    def test_lstm_prediction(self, lstm_model, sample_data):
        """Test LSTM model prediction."""
        # Mock trained model
        lstm_model.is_trained = True
        lstm_model.model = Mock()
        lstm_model.model.predict.return_value = np.array([[100.5]])
        lstm_model.scaler = Mock()
        lstm_model.scaler.transform.return_value = np.random.rand(30, 10)
        lstm_model.feature_names = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper']
        
        # Make prediction
        result = lstm_model.predict_next_price(sample_data.tail(30))
        
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'steps_ahead' in result
        assert 'timestamp' in result
        assert result['prediction'] == 100.5


class TestLSTMTrendPredictor:
    """Test LSTM trend prediction model."""
    
    @pytest.fixture
    def trend_model(self):
        """Create LSTM trend model instance."""
        return LSTMTrendPredictor(
            sequence_length=20,
            features=8,
            lstm_units=[16, 8],
            dropout_rate=0.3
        )
    
    def test_trend_model_initialization(self, trend_model):
        """Test trend model initialization."""
        assert trend_model.sequence_length == 20
        assert trend_model.features == 8
        assert trend_model.lstm_units == [16, 8]
        assert not trend_model.is_trained
    
    def test_trend_model_build(self, trend_model):
        """Test trend model building."""
        model = trend_model.build_model()
        assert model is not None
        assert trend_model.model is not None
    
    def test_trend_prediction(self, trend_model, sample_data):
        """Test trend prediction."""
        # Mock trained model
        trend_model.is_trained = True
        trend_model.model = Mock()
        trend_model.model.predict.return_value = np.array([[0.7, 0.2, 0.1]])  # UP trend
        trend_model.scaler = Mock()
        trend_model.scaler.transform.return_value = np.random.rand(20, 8)
        trend_model.feature_names = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_position']
        
        # Make prediction
        result = trend_model.predict_trend(sample_data.tail(20))
        
        assert 'trend' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert result['trend'] == 'UP'
        assert result['confidence'] == 0.7


class TestCNNLSTMPredictor:
    """Test CNN-LSTM hybrid model."""
    
    @pytest.fixture
    def cnn_lstm_model(self):
        """Create CNN-LSTM model instance."""
        return CNNLSTMPredictor(
            sequence_length=40,
            features=10,
            cnn_filters=[32, 16],
            cnn_kernel_size=3,
            lstm_units=[32, 16]
        )
    
    def test_cnn_lstm_initialization(self, cnn_lstm_model):
        """Test CNN-LSTM initialization."""
        assert cnn_lstm_model.sequence_length == 40
        assert cnn_lstm_model.cnn_filters == [32, 16]
        assert cnn_lstm_model.cnn_kernel_size == 3
        assert cnn_lstm_model.lstm_units == [32, 16]
    
    def test_cnn_lstm_build_model(self, cnn_lstm_model):
        """Test CNN-LSTM model building."""
        model = cnn_lstm_model.build_model()
        assert model is not None
        assert cnn_lstm_model.model is not None
    
    def test_cnn_lstm_prediction(self, cnn_lstm_model, sample_data):
        """Test CNN-LSTM prediction."""
        # Mock trained model
        cnn_lstm_model.is_trained = True
        cnn_lstm_model.model = Mock()
        cnn_lstm_model.model.predict.return_value = np.array([[101.2]])
        cnn_lstm_model.scaler = Mock()
        cnn_lstm_model.scaler.transform.return_value = np.random.rand(40, 10)
        cnn_lstm_model.feature_names = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper']
        
        # Make prediction
        result = cnn_lstm_model.predict_next_price(sample_data.tail(40))
        
        assert 'prediction' in result
        assert 'model_type' in result
        assert result['model_type'] == 'CNN-LSTM'
        assert result['prediction'] == 101.2


class TestMultiScaleCNNLSTM:
    """Test Multi-scale CNN-LSTM model."""
    
    @pytest.fixture
    def multi_scale_model(self):
        """Create multi-scale CNN-LSTM model."""
        return MultiScaleCNNLSTM(
            sequence_lengths=[15, 30, 45],
            features=10,
            cnn_filters=[16, 8],
            lstm_units=[16, 8]
        )
    
    def test_multi_scale_initialization(self, multi_scale_model):
        """Test multi-scale model initialization."""
        assert multi_scale_model.sequence_lengths == [15, 30, 45]
        assert multi_scale_model.features == 10
        assert multi_scale_model.cnn_filters == [16, 8]
    
    def test_multi_scale_build_model(self, multi_scale_model):
        """Test multi-scale model building."""
        model = multi_scale_model.build_model()
        assert model is not None
        assert multi_scale_model.model is not None
    
    def test_multi_scale_prediction(self, multi_scale_model, sample_data):
        """Test multi-scale prediction."""
        # Mock trained model
        multi_scale_model.is_trained = True
        multi_scale_model.model = Mock()
        multi_scale_model.model.predict.return_value = np.array([[102.5]])
        multi_scale_model.scaler = Mock()
        multi_scale_model.scaler.transform.return_value = np.random.rand(45, 10)
        multi_scale_model.feature_names = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper']
        
        # Make prediction
        result = multi_scale_model.predict_next_price(sample_data.tail(45))
        
        assert 'prediction' in result
        assert 'model_type' in result
        assert 'scales_used' in result
        assert result['model_type'] == 'Multi-Scale CNN-LSTM'
        assert result['scales_used'] == [15, 30, 45]


class TestTransformerPredictor:
    """Test Transformer prediction model."""
    
    @pytest.fixture
    def transformer_model(self):
        """Create Transformer model instance."""
        return TransformerPredictor(
            sequence_length=50,
            features=10,
            d_model=64,
            num_heads=8,
            num_layers=4
        )
    
    def test_transformer_initialization(self, transformer_model):
        """Test Transformer initialization."""
        assert transformer_model.sequence_length == 50
        assert transformer_model.d_model == 64
        assert transformer_model.num_heads == 8
        assert transformer_model.num_layers == 4
    
    def test_transformer_build_model(self, transformer_model):
        """Test Transformer model building."""
        model = transformer_model.build_model()
        assert model is not None
        assert transformer_model.model is not None
    
    def test_transformer_prediction(self, transformer_model, sample_data):
        """Test Transformer prediction."""
        # Mock trained model
        transformer_model.is_trained = True
        transformer_model.model = Mock()
        transformer_model.model.predict.return_value = np.array([[103.8]])
        transformer_model.scaler = Mock()
        transformer_model.scaler.transform.return_value = np.random.rand(50, 10)
        transformer_model.feature_names = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper']
        
        # Make prediction
        result = transformer_model.predict_next_price(sample_data.tail(50))
        
        assert 'prediction' in result
        assert 'model_type' in result
        assert 'attention_heads' in result
        assert result['model_type'] == 'Transformer'
        assert result['attention_heads'] == 8


class TestFinancialTransformer:
    """Test Financial Transformer model."""
    
    @pytest.fixture
    def financial_transformer(self):
        """Create Financial Transformer model."""
        return FinancialTransformer(
            sequence_length=60,
            features=10,
            d_model=128,
            num_heads=8,
            num_layers=6
        )
    
    def test_financial_transformer_initialization(self, financial_transformer):
        """Test Financial Transformer initialization."""
        assert financial_transformer.sequence_length == 60
        assert financial_transformer.d_model == 128
        assert financial_transformer.num_layers == 6
    
    def test_financial_transformer_prediction(self, financial_transformer, sample_data):
        """Test Financial Transformer prediction."""
        # Mock trained model
        financial_transformer.is_trained = True
        financial_transformer.model = Mock()
        financial_transformer.model.predict.return_value = np.array([[104.2]])
        financial_transformer.scaler = Mock()
        financial_transformer.scaler.transform.return_value = np.random.rand(60, 10)
        financial_transformer.feature_names = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper']
        
        # Make prediction
        result = financial_transformer.predict_next_price(sample_data.tail(60))
        
        assert 'prediction' in result
        assert 'model_type' in result
        assert 'layers' in result
        assert result['model_type'] == 'Financial Transformer'
        assert result['layers'] == 6


class TestMarketAutoencoder:
    """Test Market Autoencoder model."""
    
    @pytest.fixture
    def autoencoder_model(self):
        """Create Market Autoencoder model."""
        return MarketAutoencoder(
            input_dim=10,
            encoding_dim=5,
            hidden_layers=[8, 6],
            dropout_rate=0.2
        )
    
    def test_autoencoder_initialization(self, autoencoder_model):
        """Test autoencoder initialization."""
        assert autoencoder_model.input_dim == 10
        assert autoencoder_model.encoding_dim == 5
        assert autoencoder_model.hidden_layers == [8, 6]
        assert autoencoder_model.reconstruction_threshold is None
    
    def test_autoencoder_build_model(self, autoencoder_model):
        """Test autoencoder model building."""
        autoencoder, encoder, decoder = autoencoder_model.build_model()
        assert autoencoder is not None
        assert encoder is not None
        assert decoder is not None
        assert autoencoder_model.autoencoder is not None
    
    def test_autoencoder_anomaly_detection(self, autoencoder_model, sample_data):
        """Test autoencoder anomaly detection."""
        # Mock trained model
        autoencoder_model.is_trained = True
        autoencoder_model.autoencoder = Mock()
        autoencoder_model.autoencoder.predict.return_value = np.random.rand(100, 10)
        autoencoder_model.scaler = Mock()
        autoencoder_model.scaler.transform.return_value = np.random.rand(100, 10)
        autoencoder_model.feature_names = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper']
        autoencoder_model.reconstruction_threshold = 0.1
        
        # Detect anomalies
        result = autoencoder_model.detect_anomalies(sample_data)
        
        assert 'anomalies_detected' in result
        assert 'total_samples' in result
        assert 'anomaly_rate' in result
        assert 'threshold' in result
        assert 'anomaly_indices' in result
        assert 'anomaly_scores' in result


class TestLSTMAnomalyDetector:
    """Test LSTM Anomaly Detector model."""
    
    @pytest.fixture
    def lstm_anomaly_detector(self):
        """Create LSTM Anomaly Detector model."""
        return LSTMAnomalyDetector(
            sequence_length=25,
            features=10,
            lstm_units=[16, 8],
            dropout_rate=0.2
        )
    
    def test_lstm_anomaly_detector_initialization(self, lstm_anomaly_detector):
        """Test LSTM anomaly detector initialization."""
        assert lstm_anomaly_detector.sequence_length == 25
        assert lstm_anomaly_detector.features == 10
        assert lstm_anomaly_detector.lstm_units == [16, 8]
    
    def test_lstm_anomaly_detection(self, lstm_anomaly_detector, sample_data):
        """Test LSTM anomaly detection."""
        # Mock trained model
        lstm_anomaly_detector.is_trained = True
        lstm_anomaly_detector.autoencoder = Mock()
        lstm_anomaly_detector.autoencoder.predict.return_value = np.random.rand(75, 25, 10)
        lstm_anomaly_detector.scaler = Mock()
        lstm_anomaly_detector.scaler.transform.return_value = np.random.rand(100, 10)
        lstm_anomaly_detector.feature_names = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper']
        lstm_anomaly_detector.reconstruction_threshold = 0.15
        
        # Detect anomalies
        result = lstm_anomaly_detector.detect_anomalies(sample_data)
        
        assert 'anomalies_detected' in result
        assert 'total_samples' in result
        assert 'anomaly_rate' in result
        assert 'threshold' in result


class TestAnomalyDetector:
    """Test unified Anomaly Detector."""
    
    @pytest.fixture
    def anomaly_detector(self):
        """Create unified anomaly detector."""
        return AnomalyDetector(
            model_types=['standard', 'lstm'],
            model_configs={
                'standard': {'input_dim': 10, 'encoding_dim': 5},
                'lstm': {'sequence_length': 20, 'features': 10}
            }
        )
    
    def test_anomaly_detector_initialization(self, anomaly_detector):
        """Test anomaly detector initialization."""
        assert 'standard' in anomaly_detector.models
        assert 'lstm' in anomaly_detector.models
        assert len(anomaly_detector.models) == 2
    
    def test_ensemble_anomaly_detection(self, anomaly_detector, sample_data):
        """Test ensemble anomaly detection."""
        # Mock trained models
        anomaly_detector.is_trained = True
        
        # Mock standard model
        standard_model = Mock()
        standard_model.detect_anomalies.return_value = {
            'anomalies_detected': 5,
            'total_samples': 100,
            'anomaly_rate': 0.05,
            'threshold': 0.1,
            'anomaly_indices': [10, 20, 30, 40, 50],
            'anomaly_scores': [0.15, 0.12, 0.18, 0.11, 0.13]
        }
        anomaly_detector.models['standard'] = standard_model
        
        # Mock LSTM model
        lstm_model = Mock()
        lstm_model.detect_anomalies.return_value = {
            'anomalies_detected': 3,
            'total_samples': 100,
            'anomaly_rate': 0.03,
            'threshold': 0.12,
            'anomaly_indices': [15, 25, 35],
            'anomaly_scores': [0.16, 0.14, 0.17]
        }
        anomaly_detector.models['lstm'] = lstm_model
        
        # Detect anomalies
        result = anomaly_detector.detect_anomalies(sample_data, ensemble_method='majority')
        
        assert 'ensemble_anomalies' in result
        assert 'anomalies_detected' in result
        assert 'ensemble_method' in result
        assert 'model_results' in result
        assert result['ensemble_method'] == 'majority'


class TestDeepLearningModelManager:
    """Test Deep Learning Model Manager."""
    
    @pytest.fixture
    def model_manager(self):
        """Create model manager instance."""
        return DeepLearningModelManager()
    
    def test_model_manager_initialization(self, model_manager):
        """Test model manager initialization."""
        assert model_manager.model_dir is not None
        assert len(model_manager.model_configs) > 0
        assert len(model_manager.models) == 0
    
    def test_create_model(self, model_manager):
        """Test model creation."""
        model = model_manager.create_model('lstm_price')
        assert model is not None
        assert 'lstm_price' in model_manager.models
        assert 'lstm_price' in model_manager.model_metadata
    
    def test_list_models(self, model_manager):
        """Test model listing."""
        models_info = model_manager.list_models()
        assert len(models_info) > 0
        assert all('name' in info for info in models_info)
        assert all('class' in info for info in models_info)
        assert all('parameters' in info for info in models_info)
    
    def test_get_model_status(self, model_manager):
        """Test model status retrieval."""
        status = model_manager.get_model_status()
        assert 'total_models' in status
        assert 'created_models' in status
        assert 'trained_models' in status
        assert 'models' in status
    
    def test_ensemble_prediction(self, model_manager, sample_data):
        """Test ensemble prediction."""
        # Create and mock models
        model1 = model_manager.create_model('lstm_price')
        model2 = model_manager.create_model('lstm_trend')
        
        # Mock trained models
        model1.is_trained = True
        model1.predict_next_price = Mock(return_value={'prediction': 100.5, 'confidence': 0.8})
        
        model2.is_trained = True
        model2.predict_trend = Mock(return_value={'trend': 'UP', 'confidence': 0.7})
        
        # Update metadata
        model_manager.model_metadata['lstm_price']['is_trained'] = True
        model_manager.model_metadata['lstm_trend']['is_trained'] = True
        
        # Make ensemble prediction
        result = model_manager.create_ensemble_prediction(sample_data, ['lstm_price', 'lstm_trend'])
        
        assert 'ensemble_prediction' in result
        assert 'ensemble_confidence' in result
        assert 'individual_predictions' in result
        assert 'individual_confidences' in result
        assert 'weights' in result


class TestDeepLearningFeatureEngineer:
    """Test Deep Learning Feature Engineer."""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance."""
        return DeepLearningFeatureEngineer(
            lookback_periods=[5, 10, 20],
            feature_scaling='minmax',
            include_technical_indicators=True,
            include_statistical_features=True
        )
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 105 + np.random.randn(100).cumsum(),
            'low': 95 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': 1000000 + np.random.randint(-100000, 100000, 100)
        })
        return data
    
    def test_feature_engineer_initialization(self, feature_engineer):
        """Test feature engineer initialization."""
        assert feature_engineer.lookback_periods == [5, 10, 20]
        assert feature_engineer.feature_scaling == 'minmax'
        assert feature_engineer.include_technical_indicators is True
        assert feature_engineer.include_statistical_features is True
    
    @patch('talib.SMA')
    @patch('talib.RSI')
    @patch('talib.MACD')
    def test_create_features(self, mock_macd, mock_rsi, mock_sma, feature_engineer, sample_ohlcv_data):
        """Test feature creation."""
        # Mock TA-Lib functions
        mock_sma.return_value = np.random.rand(100)
        mock_rsi.return_value = np.random.rand(100) * 100
        mock_macd.return_value = (np.random.rand(100), np.random.rand(100), np.random.rand(100))
        
        # Create features
        features_df = feature_engineer.create_features(sample_ohlcv_data)
        
        assert len(features_df.columns) > 5  # Should have more than base columns
        assert len(features_df) == len(sample_ohlcv_data)
        assert not features_df.isnull().all().any()  # No all-null columns
    
    def test_scale_features(self, feature_engineer, sample_ohlcv_data):
        """Test feature scaling."""
        # Create features first
        features_df = feature_engineer.create_features(sample_ohlcv_data)
        
        # Scale features
        scaled_df = feature_engineer.scale_features(features_df, fit_scaler=True)
        
        assert scaled_df.shape == features_df.shape
        assert feature_engineer.scaler is not None
        
        # Test inference scaling
        scaled_df_inference = feature_engineer.scale_features(features_df, fit_scaler=False)
        assert scaled_df_inference.shape == features_df.shape
    
    def test_create_sequences(self, feature_engineer, sample_ohlcv_data):
        """Test sequence creation."""
        # Create features
        features_df = feature_engineer.create_features(sample_ohlcv_data)
        
        # Create sequences
        X, y = feature_engineer.create_sequences(features_df, sequence_length=20, target_column='close')
        
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 20  # sequence length
        assert X.shape[2] == len(features_df.columns)
        assert len(y) > 0
    
    def test_get_feature_names(self, feature_engineer, sample_ohlcv_data):
        """Test feature name retrieval."""
        # Create features
        features_df = feature_engineer.create_features(sample_ohlcv_data)
        
        # Get feature names
        feature_names = feature_engineer.get_feature_names()
        
        assert len(feature_names) > 0
        assert all(name in features_df.columns for name in feature_names)


class TestIntegration:
    """Integration tests for deep learning models."""
    
    @pytest.fixture
    def integration_data(self):
        """Create integration test data."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(200).cumsum() * 0.1,
            'high': 105 + np.random.randn(200).cumsum() * 0.1,
            'low': 95 + np.random.randn(200).cumsum() * 0.1,
            'close': 100 + np.random.randn(200).cumsum() * 0.1,
            'volume': 1000000 + np.random.randint(-100000, 100000, 200)
        })
        return data
    
    def test_end_to_end_prediction_pipeline(self, integration_data):
        """Test end-to-end prediction pipeline."""
        # Create feature engineer
        feature_engineer = DeepLearningFeatureEngineer()
        
        # Create features
        features_df = feature_engineer.create_features(integration_data)
        scaled_features = feature_engineer.scale_features(features_df, fit_scaler=True)
        
        # Create sequences
        X, y = feature_engineer.create_sequences(scaled_features, sequence_length=30, target_column='close')
        
        # Create and train model
        model = LSTMPricePredictor(sequence_length=30, features=len(scaled_features.columns))
        
        # Mock training
        model.is_trained = True
        model.model = Mock()
        model.model.predict.return_value = np.array([[101.5]])
        model.scaler = feature_engineer.scaler
        model.feature_names = feature_engineer.get_feature_names()
        
        # Make prediction
        recent_data = integration_data.tail(30)
        result = model.predict_next_price(recent_data)
        
        assert 'prediction' in result
        assert 'confidence' in result
        assert result['prediction'] == 101.5
    
    def test_model_manager_integration(self, integration_data):
        """Test model manager integration."""
        # Create model manager
        manager = DeepLearningModelManager()
        
        # Create models
        lstm_model = manager.create_model('lstm_price')
        trend_model = manager.create_model('lstm_trend')
        
        # Mock training
        lstm_model.is_trained = True
        lstm_model.predict_next_price = Mock(return_value={'prediction': 102.0, 'confidence': 0.85})
        
        trend_model.is_trained = True
        trend_model.predict_trend = Mock(return_value={'trend': 'UP', 'confidence': 0.75})
        
        # Update metadata
        manager.model_metadata['lstm_price']['is_trained'] = True
        manager.model_metadata['lstm_trend']['is_trained'] = True
        
        # Test ensemble prediction
        result = manager.create_ensemble_prediction(integration_data.tail(50))
        
        assert 'ensemble_prediction' in result
        assert 'ensemble_confidence' in result
        assert 'model_count' in result
        assert result['model_count'] == 2

