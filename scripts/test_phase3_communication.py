#!/usr/bin/env python3
"""Phase 3 Validation: Enhanced Model Communications + Intelligent Model Selection"""

import sys
import os
import io

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        pass  # Already wrapped or not needed

from datetime import datetime, timedelta
from src.ai.model_communication_hub import (
    get_communication_hub, MarketContext, ModelInsight, ModelPerformance
)
from src.ai.intelligent_model_selector import (
    get_model_selector, SelectionStrategy, SelectionCriteria
)

def test_communication_hub():
    """Test model communication hub functionality."""
    hub = get_communication_hub()
    
    # Test insight sharing
    success = hub.share_insight(
        model_name="test_model",
        insight_type="prediction",
        data={"prediction": 0.75, "confidence": 0.8},
        confidence=0.8,
        reasoning="Technical analysis suggests upward trend"
    )
    assert success, "Failed to share insight"
    print("[PASS] Insight sharing works")
    
    # Test market context update
    context = MarketContext(
        timestamp=datetime.now(),
        regime="trending",
        volatility_zscore=1.2,
        correlation=0.7,
        sector_dispersion={"tech": 0.8, "finance": 0.6},
        liquidity_score=0.9,
        news_sentiment=0.3,
        market_phase="mid_day"
    )
    hub.update_market_context(context)
    retrieved_context = hub.get_market_context()
    assert retrieved_context is not None, "Failed to retrieve market context"
    assert retrieved_context.regime == "trending", "Market context not updated correctly"
    print("[PASS] Market context update works")
    
    # Test performance tracking
    performance = ModelPerformance(
        model_name="test_model",
        accuracy=0.85,
        precision=0.82,
        recall=0.88,
        f1_score=0.85,
        sharpe_ratio=1.5,
        max_drawdown=0.05,
        win_rate=0.75,
        avg_return=0.12,
        last_updated=datetime.now()
    )
    hub.update_model_performance(performance)
    retrieved_performance = hub.get_model_performance("test_model")
    assert retrieved_performance is not None, "Failed to retrieve model performance"
    assert retrieved_performance.accuracy == 0.85, "Performance not updated correctly"
    print("[PASS] Performance tracking works")
    
    # Test consensus building
    # Add more insights for consensus
    for i in range(3):
        hub.share_insight(
            model_name=f"model_{i}",
            insight_type="prediction",
            data={"prediction": 0.7 + i * 0.05},
            confidence=0.8,
            reasoning=f"Model {i} analysis"
        )
    
    consensus = hub.build_consensus("prediction", timedelta(minutes=5))
    assert consensus['participant_count'] >= 4, "Consensus not built correctly"
    assert consensus['agreement_level'] >= 0.0, "Agreement level not calculated"
    print("[PASS] Consensus building works")
    
    return True

def test_intelligent_model_selector():
    """Test intelligent model selector functionality."""
    selector = get_model_selector()
    
    # Register test models
    models_registered = 0
    test_models = [
        ("model_a", ["technical_analysis", "sentiment"], "trending"),
        ("model_b", ["technical_analysis", "fundamental"], "ranging"),
        ("model_c", ["technical_analysis", "risk"], "volatile"),
        ("model_d", ["technical_analysis", "sentiment", "fundamental"], "general")
    ]
    
    for model_name, capabilities, specialization in test_models:
        success = selector.register_model(model_name, capabilities, specialization)
        assert success, f"Failed to register model {model_name}"
        models_registered += 1
    
    assert models_registered == 4, "Not all models registered successfully"
    print("[PASS] Model registration works")
    
    # Update model performance
    performance_updates = 0
    for i, (model_name, _, _) in enumerate(test_models):
        success = selector.update_model_performance(
            model_name, 
            performance_score=0.6 + i * 0.1,
            reliability_score=0.7 + i * 0.05,
            speed_score=0.8 - i * 0.1
        )
        assert success, f"Failed to update performance for {model_name}"
        performance_updates += 1
    
    assert performance_updates == 4, "Not all performance updates successful"
    print("[PASS] Performance updates work")
    
    # Test model selection
    available_models = ["model_a", "model_b", "model_c", "model_d"]
    selected_models = selector.select_models_for_conditions(available_models)
    assert len(selected_models) > 0, "No models selected"
    assert len(selected_models) <= 3, "Too many models selected"
    assert all(model in available_models for model in selected_models), "Invalid models selected"
    print("[PASS] Model selection works")
    
    # Test criteria-based selection
    criteria = SelectionCriteria(
        required_capabilities=["technical_analysis", "sentiment"],
        min_performance_threshold=0.6,
        max_models=2,
        strategy=SelectionStrategy.PERFORMANCE_BASED,
        market_regime="trending"
    )
    
    selection_result = selector.select_models(criteria)
    assert selection_result.criteria_met, "Criteria not met"
    assert len(selection_result.selected_models) <= 2, "Too many models selected"
    assert selection_result.confidence > 0.0, "No confidence in selection"
    print("[PASS] Criteria-based selection works")
    
    # Test selection statistics
    stats = selector.get_selection_statistics()
    assert stats['total_selections'] > 0, "No selection history"
    assert stats['registered_models'] == 4, "Incorrect model count"
    print("[PASS] Selection statistics work")
    
    return True

def test_integration():
    """Test integration between communication hub and model selector."""
    hub = get_communication_hub()
    selector = get_model_selector()
    
    # Register models in selector
    selector.register_model("integration_model", ["technical_analysis"], "general")
    selector.update_model_performance("integration_model", 0.8, 0.9, 0.7)
    
    # Share performance insight in hub
    performance = ModelPerformance(
        model_name="integration_model",
        accuracy=0.8,
        precision=0.85,
        recall=0.75,
        f1_score=0.8,
        sharpe_ratio=1.2,
        max_drawdown=0.03,
        win_rate=0.8,
        avg_return=0.15,
        last_updated=datetime.now()
    )
    hub.update_model_performance(performance)
    
    # Test that both systems can access the same model
    hub_performance = hub.get_model_performance("integration_model")
    selector_capability = selector.model_capabilities.get("integration_model")
    
    assert hub_performance is not None, "Hub doesn't have model performance"
    assert selector_capability is not None, "Selector doesn't have model capability"
    assert hub_performance.accuracy == 0.8, "Performance mismatch between systems"
    assert selector_capability.performance_score == 0.8, "Performance score mismatch"
    print("[PASS] Integration between systems works")
    
    return True

if __name__ == "__main__":
    try:
        test_communication_hub()
        test_intelligent_model_selector()
        test_integration()
        print("\n[PASS] PHASE 3 VALIDATION: PASSED")
        exit(0)
    except AssertionError as e:
        print(f"\n[FAIL] PHASE 3 VALIDATION: FAILED - {e}")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] PHASE 3 VALIDATION: ERROR - {e}")
        exit(1)