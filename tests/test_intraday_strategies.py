"""
Unit tests for Intraday Trading Strategies
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading.strategies.base_strategy import BaseStrategy, StrategySignal, SignalType
from src.trading.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.trading.strategies.mean_reversion import MeanReversionStrategy
from src.trading.strategies.vwap_crossover import VWAPCrossoverStrategy
from src.trading.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from src.trading.strategies.rsi_divergence import RSIDivergenceStrategy


def create_sample_data(rows: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)

    dates = pd.date_range(start='2024-01-01', periods=rows, freq='1D')
    base_price = 100

    # Generate random walk price data
    returns = np.random.randn(rows) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(rows) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(rows) * 0.01)),
        'Low': prices * (1 - np.abs(np.random.randn(rows) * 0.01)),
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, rows)
    }, index=dates)

    # Ensure High >= Open, Close and Low <= Open, Close
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


def create_breakout_data() -> pd.DataFrame:
    """Create data with a clear breakout pattern"""
    np.random.seed(42)
    rows = 100

    dates = pd.date_range(start='2024-01-01', periods=rows, freq='1D')

    # Consolidation phase (first 80 rows)
    consolidation_prices = 100 + np.random.randn(80) * 0.5

    # Breakout phase (last 20 rows)
    breakout_prices = np.linspace(101, 110, 20) + np.random.randn(20) * 0.3

    prices = np.concatenate([consolidation_prices, breakout_prices])

    data = pd.DataFrame({
        'Open': prices * 0.999,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.concatenate([
            np.random.randint(100000, 300000, 80),  # Low volume consolidation
            np.random.randint(500000, 1000000, 20)  # High volume breakout
        ])
    }, index=dates)

    return data


def create_oversold_data() -> pd.DataFrame:
    """Create data with oversold conditions"""
    np.random.seed(42)
    rows = 50

    dates = pd.date_range(start='2024-01-01', periods=rows, freq='1D')

    # Steady decline followed by bounce
    decline = np.linspace(100, 80, 40)  # Decline
    bounce = np.linspace(80, 85, 10)  # Bounce
    prices = np.concatenate([decline, bounce])

    data = pd.DataFrame({
        'Open': prices * 1.002,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(200000, 500000, rows)
    }, index=dates)

    return data


class TestBaseStrategy:
    """Tests for BaseStrategy"""

    def test_sma_calculation(self):
        """Test SMA calculation"""
        data = create_sample_data()
        strategy = MomentumBreakoutStrategy()

        sma = strategy.calculate_sma(data['Close'], 20)

        assert len(sma) == len(data)
        assert pd.notna(sma.iloc[-1])
        # SMA should be close to the average of last 20 closes
        expected = data['Close'].iloc[-20:].mean()
        assert abs(sma.iloc[-1] - expected) < 0.01

    def test_ema_calculation(self):
        """Test EMA calculation"""
        data = create_sample_data()
        strategy = MomentumBreakoutStrategy()

        ema = strategy.calculate_ema(data['Close'], 20)

        assert len(ema) == len(data)
        assert pd.notna(ema.iloc[-1])

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        data = create_sample_data()
        strategy = MomentumBreakoutStrategy()

        rsi = strategy.calculate_rsi(data['Close'], 14)

        assert len(rsi) == len(data)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_macd_calculation(self):
        """Test MACD calculation"""
        data = create_sample_data()
        strategy = MomentumBreakoutStrategy()

        macd = strategy.calculate_macd(data['Close'])

        assert 'macd' in macd
        assert 'signal' in macd
        assert 'histogram' in macd
        assert len(macd['macd']) == len(data)

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        data = create_sample_data()
        strategy = MeanReversionStrategy()

        bb = strategy.calculate_bollinger_bands(data['Close'], 20, 2.0)

        assert 'upper' in bb
        assert 'middle' in bb
        assert 'lower' in bb

        # Upper should be above middle, middle above lower
        valid_idx = bb['middle'].dropna().index
        assert (bb['upper'].loc[valid_idx] >= bb['middle'].loc[valid_idx]).all()
        assert (bb['middle'].loc[valid_idx] >= bb['lower'].loc[valid_idx]).all()

    def test_atr_calculation(self):
        """Test ATR calculation"""
        data = create_sample_data()
        strategy = MomentumBreakoutStrategy()

        atr = strategy.calculate_atr(data['High'], data['Low'], data['Close'], 14)

        assert len(atr) == len(data)
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()  # ATR should be positive

    def test_vwap_calculation(self):
        """Test VWAP calculation"""
        data = create_sample_data()
        strategy = VWAPCrossoverStrategy()

        vwap = strategy.calculate_vwap(
            data['High'], data['Low'], data['Close'], data['Volume']
        )

        assert len(vwap) == len(data)
        assert pd.notna(vwap.iloc[-1])


class TestMomentumBreakoutStrategy:
    """Tests for Momentum Breakout Strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = MomentumBreakoutStrategy()

        assert strategy.name == "Momentum Breakout"
        assert strategy.lookback_period > 0
        assert strategy.volume_multiplier > 0

    def test_required_lookback(self):
        """Test required lookback period"""
        strategy = MomentumBreakoutStrategy()
        lookback = strategy.get_required_lookback()

        assert lookback > 0
        assert isinstance(lookback, int)

    def test_signal_generation_with_valid_data(self):
        """Test signal generation with valid market data"""
        strategy = MomentumBreakoutStrategy({'min_confidence': 0.3})
        data = create_breakout_data()

        signal = strategy.generate_signal(data, 'TEST.TO')

        # May or may not generate signal depending on data
        if signal:
            assert isinstance(signal, StrategySignal)
            assert signal.symbol == 'TEST.TO'
            assert signal.strategy_name == "Momentum Breakout"
            assert 0 <= signal.confidence <= 1

    def test_signal_generation_with_insufficient_data(self):
        """Test with insufficient data"""
        strategy = MomentumBreakoutStrategy()
        data = create_sample_data(10)  # Too little data

        signal = strategy.generate_signal(data, 'TEST.TO')
        assert signal is None

    def test_stop_loss_take_profit(self):
        """Test stop loss and take profit calculation"""
        strategy = MomentumBreakoutStrategy()

        stop_loss_long = strategy.calculate_stop_loss(100.0, 'long', atr=2.0)
        stop_loss_short = strategy.calculate_stop_loss(100.0, 'short', atr=2.0)

        assert stop_loss_long < 100.0  # Below entry for long
        assert stop_loss_short > 100.0  # Above entry for short

        take_profit_long = strategy.calculate_take_profit(100.0, 'long', atr=2.0)
        take_profit_short = strategy.calculate_take_profit(100.0, 'short', atr=2.0)

        assert take_profit_long > 100.0  # Above entry for long
        assert take_profit_short < 100.0  # Below entry for short


class TestMeanReversionStrategy:
    """Tests for Mean Reversion Strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = MeanReversionStrategy()

        assert strategy.name == "Mean Reversion"
        assert strategy.bb_period > 0
        assert 0 < strategy.rsi_oversold < strategy.rsi_overbought < 100

    def test_signal_with_oversold_data(self):
        """Test signal generation with oversold conditions"""
        strategy = MeanReversionStrategy({'min_confidence': 0.3})
        data = create_oversold_data()

        signal = strategy.generate_signal(data, 'TEST.TO')

        # May generate buy signal due to oversold conditions
        if signal:
            assert isinstance(signal, StrategySignal)
            assert signal.strategy_name == "Mean Reversion"


class TestVWAPCrossoverStrategy:
    """Tests for VWAP Crossover Strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = VWAPCrossoverStrategy()

        assert strategy.name == "VWAP Crossover"
        assert strategy.max_distance_from_vwap > 0

    def test_signal_generation(self):
        """Test signal generation"""
        strategy = VWAPCrossoverStrategy({'min_confidence': 0.3})
        data = create_sample_data()

        signal = strategy.generate_signal(data, 'TEST.TO')

        # May or may not generate signal
        if signal:
            assert isinstance(signal, StrategySignal)
            assert 'vwap' in signal.indicators


class TestOpeningRangeBreakoutStrategy:
    """Tests for Opening Range Breakout Strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = OpeningRangeBreakoutStrategy()

        assert strategy.name == "Opening Range Breakout"
        assert strategy.opening_range_minutes > 0

    def test_opening_range_setting(self):
        """Test opening range setting"""
        strategy = OpeningRangeBreakoutStrategy()

        strategy.set_opening_range(105.0, 95.0, datetime.now())

        assert strategy.opening_range_high == 105.0
        assert strategy.opening_range_low == 95.0
        assert strategy.opening_range_set

    def test_opening_range_reset(self):
        """Test opening range reset"""
        strategy = OpeningRangeBreakoutStrategy()

        strategy.set_opening_range(105.0, 95.0, datetime.now())
        strategy.reset_opening_range()

        assert strategy.opening_range_high is None
        assert strategy.opening_range_low is None
        assert not strategy.opening_range_set


class TestRSIDivergenceStrategy:
    """Tests for RSI Divergence Strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = RSIDivergenceStrategy()

        assert strategy.name == "RSI Divergence"
        assert strategy.rsi_period > 0
        assert strategy.oversold_zone < 50 < strategy.overbought_zone

    def test_signal_generation(self):
        """Test signal generation"""
        strategy = RSIDivergenceStrategy({'min_confidence': 0.3})
        data = create_sample_data()

        signal = strategy.generate_signal(data, 'TEST.TO')

        if signal:
            assert isinstance(signal, StrategySignal)
            assert 'divergence_type' in signal.indicators


class TestStrategySignal:
    """Tests for StrategySignal dataclass"""

    def test_signal_creation(self):
        """Test signal creation"""
        signal = StrategySignal(
            signal_type=SignalType.BUY,
            symbol='TEST.TO',
            price=100.0,
            timestamp=datetime.now(),
            confidence=0.75,
            strategy_name="Test Strategy"
        )

        assert signal.signal_type == SignalType.BUY
        assert signal.symbol == 'TEST.TO'
        assert signal.price == 100.0
        assert signal.confidence == 0.75

    def test_confidence_clamping(self):
        """Test confidence is clamped to 0-1"""
        signal = StrategySignal(
            signal_type=SignalType.BUY,
            symbol='TEST.TO',
            price=100.0,
            timestamp=datetime.now(),
            confidence=1.5,  # Over 1.0
            strategy_name="Test"
        )

        assert signal.confidence == 1.0  # Should be clamped

        signal2 = StrategySignal(
            signal_type=SignalType.BUY,
            symbol='TEST.TO',
            price=100.0,
            timestamp=datetime.now(),
            confidence=-0.5,  # Under 0.0
            strategy_name="Test"
        )

        assert signal2.confidence == 0.0  # Should be clamped


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
