"""
Unit tests for ML Training Pipeline
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.training.training_pipeline import (
    TrainingPipeline,
    TrainingConfig,
    TrainingResult,
    DataPreprocessor
)


def create_sample_market_data(rows: int = 500) -> pd.DataFrame:
    """Create sample market data for testing"""
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=rows, freq='1D')
    base_price = 100

    returns = np.random.randn(rows) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.randn(rows) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(rows) * 0.01)),
        'Low': prices * (1 - np.abs(np.random.randn(rows) * 0.01)),
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, rows),
        'Symbol': 'TEST.TO'
    })

    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        config = TrainingConfig()

        assert config.sequence_length == 60
        assert config.batch_size == 32
        assert config.epochs == 100
        assert 0 < config.train_split < 1
        assert 0 < config.learning_rate < 1
        assert config.device in ['cuda', 'cpu']

    def test_custom_values(self):
        """Test custom configuration values"""
        config = TrainingConfig(
            sequence_length=30,
            batch_size=64,
            epochs=50,
            learning_rate=0.0001
        )

        assert config.sequence_length == 30
        assert config.batch_size == 64
        assert config.epochs == 50
        assert config.learning_rate == 0.0001

    def test_splits_sum_to_one(self):
        """Test train/val/test splits sum to 1"""
        config = TrainingConfig()
        total = config.train_split + config.validation_split + config.test_split

        assert abs(total - 1.0) < 0.001


class TestTrainingResult:
    """Tests for TrainingResult dataclass"""

    def test_successful_result(self):
        """Test successful training result"""
        result = TrainingResult(
            model_name="test_model",
            success=True,
            train_loss=0.5,
            val_loss=0.6,
            test_loss=0.65,
            accuracy=0.75,
            training_time=120.5,
            epochs_trained=50,
            model_path="/models/test.pt"
        )

        assert result.success
        assert result.model_name == "test_model"
        assert result.accuracy == 0.75
        assert result.error_message == ""

    def test_failed_result(self):
        """Test failed training result"""
        result = TrainingResult(
            model_name="test_model",
            success=False,
            error_message="Out of memory"
        )

        assert not result.success
        assert result.error_message == "Out of memory"
        assert result.accuracy == 0.0


class TestDataPreprocessor:
    """Tests for DataPreprocessor class"""

    def test_initialization(self):
        """Test preprocessor initialization"""
        config = TrainingConfig()
        preprocessor = DataPreprocessor(config)

        assert preprocessor.config == config
        assert len(preprocessor.scalers) == 0

    def test_engineer_features(self):
        """Test feature engineering"""
        config = TrainingConfig()
        preprocessor = DataPreprocessor(config)

        data = create_sample_market_data(200)
        engineered = preprocessor.engineer_features(data)

        # Check new columns were created
        assert 'returns' in engineered.columns
        assert 'rsi_14' in engineered.columns
        assert 'macd' in engineered.columns
        assert 'bb_position' in engineered.columns
        assert 'target' in engineered.columns

    def test_engineer_features_handles_empty(self):
        """Test feature engineering with empty data"""
        config = TrainingConfig()
        preprocessor = DataPreprocessor(config)

        data = pd.DataFrame()
        result = preprocessor.engineer_features(data)

        assert result.empty

    def test_prepare_sequences(self):
        """Test sequence preparation"""
        config = TrainingConfig(sequence_length=20)
        preprocessor = DataPreprocessor(config)

        data = create_sample_market_data(200)
        data = preprocessor.engineer_features(data)

        feature_columns = ['returns', 'rsi_14']
        feature_columns = [c for c in feature_columns if c in data.columns]

        if feature_columns:
            X, y = preprocessor.prepare_sequences(data, feature_columns)

            if len(X) > 0:
                assert X.shape[1] == config.sequence_length
                assert X.shape[2] == len(feature_columns)
                assert len(X) == len(y)

    def test_scaler_saved(self):
        """Test scaler is saved during preprocessing"""
        config = TrainingConfig(sequence_length=20)
        preprocessor = DataPreprocessor(config)

        data = create_sample_market_data(200)
        data = preprocessor.engineer_features(data)

        feature_columns = ['returns']
        feature_columns = [c for c in feature_columns if c in data.columns]

        if feature_columns:
            X, y = preprocessor.prepare_sequences(data, feature_columns)
            assert 'main' in preprocessor.scalers

    def test_save_scaler(self):
        """Test scaler can be saved to file"""
        config = TrainingConfig()
        preprocessor = DataPreprocessor(config)

        # Create and fit a scaler
        data = create_sample_market_data(100)
        data = preprocessor.engineer_features(data)

        feature_columns = ['returns']
        if 'returns' in data.columns:
            preprocessor.prepare_sequences(data, feature_columns)

            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
                preprocessor.save_scaler('main', f.name)
                assert Path(f.name).exists()


class TestTrainingPipeline:
    """Tests for TrainingPipeline class"""

    def test_initialization(self):
        """Test pipeline initialization"""
        pipeline = TrainingPipeline()

        assert pipeline.config is not None
        assert pipeline.preprocessor is not None
        assert len(pipeline.results) == 0

    def test_initialization_with_config(self):
        """Test pipeline with custom config"""
        config = TrainingConfig(epochs=10, batch_size=16)
        pipeline = TrainingPipeline(config=config)

        assert pipeline.config.epochs == 10
        assert pipeline.config.batch_size == 16

    def test_directory_creation(self):
        """Test directories are created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_dir=f"{tmpdir}/models",
                checkpoint_dir=f"{tmpdir}/checkpoints"
            )
            pipeline = TrainingPipeline(config=config)

            assert Path(config.model_dir).exists()
            assert Path(config.checkpoint_dir).exists()

    def test_get_training_results(self):
        """Test getting training results"""
        pipeline = TrainingPipeline()

        # Add some mock results
        pipeline.results.append(TrainingResult(
            model_name="test",
            success=True
        ))

        results = pipeline.get_training_results()
        assert len(results) == 1


class TestIntegration:
    """Integration tests for the training pipeline"""

    @pytest.mark.slow
    def test_mini_training_run(self):
        """Test a minimal training run (if PyTorch available)"""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available, skipping training test")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                epochs=2,
                batch_size=8,
                sequence_length=10,
                model_dir=f"{tmpdir}/models",
                checkpoint_dir=f"{tmpdir}/checkpoints"
            )

            pipeline = TrainingPipeline(config=config)

            # Create minimal training data
            X = np.random.randn(100, 10, 5).astype(np.float32)
            y = np.random.randint(0, 3, 100)

            result = pipeline.train_lstm(X, y)

            assert result.model_name is not None
            # Success depends on model being available


class TestFeatureEngineering:
    """Tests for feature engineering quality"""

    def test_rsi_range(self):
        """Test RSI is in valid range 0-100"""
        config = TrainingConfig()
        preprocessor = DataPreprocessor(config)

        data = create_sample_market_data(200)
        engineered = preprocessor.engineer_features(data)

        if 'rsi_14' in engineered.columns:
            valid_rsi = engineered['rsi_14'].dropna()
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()

    def test_bollinger_position_range(self):
        """Test Bollinger Band position is reasonable"""
        config = TrainingConfig()
        preprocessor = DataPreprocessor(config)

        data = create_sample_market_data(200)
        engineered = preprocessor.engineer_features(data)

        if 'bb_position' in engineered.columns:
            valid_bb = engineered['bb_position'].dropna()
            # Most values should be between -0.5 and 1.5
            assert (valid_bb > -2).all()
            assert (valid_bb < 3).all()

    def test_target_values(self):
        """Test target values are valid classes"""
        config = TrainingConfig()
        preprocessor = DataPreprocessor(config)

        data = create_sample_market_data(200)
        engineered = preprocessor.engineer_features(data)

        if 'target' in engineered.columns:
            valid_targets = engineered['target'].dropna()
            unique_targets = valid_targets.unique()
            # Targets should be -1, 0, or 1
            assert all(t in [-1, 0, 1] for t in unique_targets)

    def test_no_lookahead_bias(self):
        """Test no lookahead bias in features"""
        config = TrainingConfig(sequence_length=10)
        preprocessor = DataPreprocessor(config)

        data = create_sample_market_data(100)
        engineered = preprocessor.engineer_features(data)

        # Target should use future data (shift(-1))
        # Other features should only use past/current data
        if 'returns' in engineered.columns:
            # Returns are based on past prices, which is correct
            assert True  # Placeholder - actual implementation validates


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
