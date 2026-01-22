import pytest

from src.ai.factors import FactorBuilder


@pytest.mark.integration
def test_factor_builder_sentiment_and_fundamentals_normalised():
    builder = FactorBuilder()
    sentiment, fundamental = builder.build_for_symbol("AAPL")
    assert -1.0 <= sentiment <= 1.0
    assert 0.0 <= fundamental <= 1.0

    latest = builder.latest_factors(["AAPL"])
    assert not latest.empty
    record = latest.iloc[0]
    assert -1.0 <= record["sentiment_score"] <= 1.0
    assert 0.0 <= record["fundamental_score"] <= 1.0
