"""
Test AI Logging System
Quick test to verify the AI activity logger is working
"""

from src.logging.ai_activity_logger import (
    log_ai_activity, log_ai_signal, log_ai_trade, log_ai_decision,
    get_ai_activity_summary
)

def test_ai_logging():
    """Test the AI logging system"""
    print("Testing AI Logging System...")
    
    # Test 1: General activity logging
    print("1. Testing general activity logging...")
    log_ai_activity('test', 'Testing AI logging system', {
        'test_type': 'general_activity',
        'timestamp': '2025-10-06T10:35:00'
    })
    
    # Test 2: Signal generation logging
    print("2. Testing signal generation logging...")
    log_ai_signal(
        symbol='TD.TO',
        signal_type='BUY',
        confidence=0.75,
        score=0.78,
        sources={'lstm': 0.8, 'gru': 0.7, 'ppo': 0.75},
        reasoning=['Strong technical indicators', 'Positive sentiment']
    )
    
    # Test 3: Decision making logging
    print("3. Testing decision making logging...")
    log_ai_decision(
        decision_type='meta_ensemble',
        symbol='TD.TO',
        decision='BUY 0.15 shares',
        reasoning=['Meta-ensemble score: 0.78', 'Risk-adjusted position size'],
        risk_factors={'confidence': 0.75, 'risk_score': 0.25}
    )
    
    # Test 4: Trade execution logging
    print("4. Testing trade execution logging...")
    log_ai_trade(
        symbol='TD.TO',
        action='BUY',
        quantity=0.15,
        price=113.26,
        pnl=0.0,
        confidence=0.75,
        reasoning=['Meta-ensemble decision', 'Risk-adjusted sizing']
    )
    
    # Test 5: Get activity summary
    print("5. Getting activity summary...")
    summary = get_ai_activity_summary()
    print(f"   Activity counters: {summary.get('counters', {})}")
    print(f"   Background logging: {summary.get('background_logging_active', False)}")
    print(f"   Queue size: {summary.get('queue_size', 0)}")
    
    print("\nAI Logging System Test Complete!")
    print("Check the following log files:")
    print("  logs/ai_activity.log")
    print("  logs/ai_trades.log")
    print("  logs/ai_signals.log")
    print("  logs/ai_decisions.log")
    print("  logs/ai_activity.json")

if __name__ == "__main__":
    test_ai_logging()
