import pytest

from src.agents.ai_selector import AISelectorAgent


@pytest.mark.integration
def test_kelly_sizing_respects_risk_bounds():
    agent = AISelectorAgent()
    config = agent.risk_config
    outcome = agent._kelly_sizing(confidence=0.7, price=150.0)
    assert outcome["risk_fraction"] <= config["max_position_risk_pct"]
    assert outcome["stop_price"] < 150.0
    assert outcome["take_profit"] > 150.0
    assert outcome["shares"] >= 0
