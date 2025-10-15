import pytest

from src.ai.features import FeatureEngineer


@pytest.mark.integration
def test_feature_engineer_generates_indicators():
    engineer = FeatureEngineer()
    frame = engineer.build_for_symbol("AAPL", min_history=120)
    assert not frame.empty
    for column in [
        "rsi",
        "macd_line",
        "macd_signal",
        "bollinger_upper",
        "atr",
        "adx",
        "volume_zscore",
        "momentum",
    ]:
        assert column in frame.columns
        assert frame[column].notna().sum() > 0
