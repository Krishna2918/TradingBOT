import json

import pytest

from src.ai.scoring import ScoreEngine


@pytest.mark.integration
def test_score_engine_combines_components():
    engine = ScoreEngine()
    scores = engine.score_universe(["AAPL"])
    assert not scores.empty

    record = scores.iloc[0]
    assert 0.0 <= record["score"] <= 1.0

    details = json.loads(record["details_json"])
    weights = details["shape"]
    assert pytest.approx(sum(weights.values()), 1e-6) == 1.0
    for key in ["technical", "sentiment", "fundamental", "momentum", "volume"]:
        assert key in details
        assert 0.0 <= details[key] <= 1.0
