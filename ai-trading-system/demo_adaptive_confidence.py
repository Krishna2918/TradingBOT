#!/usr/bin/env python3
"""
Demonstration of Adaptive Confidence System
"""

import sys
import os
sys.path.append('.')

from src.ai.adaptive_confidence import (
    AdaptiveConfidenceManager, 
    get_confidence_threshold, 
    adjust_confidence_threshold,
    get_confidence_info,
    record_trade_result
)
import random
from datetime import datetime, timedelta

def demo_adaptive_confidence():
    """Demonstrate the adaptive confidence system."""
    print("=" * 60)
    print("ADAPTIVE CONFIDENCE SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Initialize manager
    manager = AdaptiveConfidenceManager()
    
    # Show initial state
    print("\n1. INITIAL STATE")
    print("-" * 30)
    initial_threshold = get_confidence_threshold()
    print(f"Initial confidence threshold: {initial_threshold:.3f}")
    
    info = get_confidence_info()
    print(f"Threshold range: {info['min_threshold']:.2f} - {info['max_threshold']:.2f}")
    print(f"Recent performance: {info['recent_performance']['total_trades']} trades")
    
    # Simulate poor performance scenario
    print("\n2. SIMULATING POOR PERFORMANCE")
    print("-" * 30)
    print("Adding 20 losing trades to simulate poor performance...")
    
    for i in range(20):
        symbol = f"POOR{i:02d}"
        confidence = random.uniform(0.3, 0.7)
        entry_price = random.uniform(50, 150)
        exit_price = entry_price * random.uniform(0.90, 0.98)  # 2-10% loss
        quantity = random.randint(50, 200)
        
        record_trade_result(
            symbol=symbol,
            action="BUY",
            confidence=confidence,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity
        )
    
    # Check performance after poor trades
    stats = manager.calculate_performance_stats()
    print(f"Performance after poor trades:")
    print(f"  Total trades: {stats.total_trades}")
    print(f"  Win rate: {stats.win_rate:.1%}")
    print(f"  Profit factor: {stats.profit_factor:.2f}")
    print(f"  Average confidence: {stats.avg_confidence:.3f}")
    
    # Check if threshold should be adjusted
    should_adjust = manager.should_adjust_threshold()
    print(f"  Should adjust threshold: {should_adjust}")
    
    if should_adjust:
        new_threshold, reason = manager.calculate_optimal_threshold()
        print(f"  Recommended threshold: {new_threshold:.3f}")
        print(f"  Reason: {reason}")
        
        # Adjust threshold
        adjusted = adjust_confidence_threshold()
        if adjusted:
            print("  ✅ Threshold automatically adjusted!")
        else:
            print("  ❌ Threshold adjustment failed")
    
    # Show updated state
    print("\n3. AFTER POOR PERFORMANCE ADJUSTMENT")
    print("-" * 30)
    current_threshold = get_confidence_threshold()
    print(f"Current confidence threshold: {current_threshold:.3f}")
    print(f"Change from initial: {current_threshold - initial_threshold:+.3f}")
    
    # Simulate excellent performance scenario
    print("\n4. SIMULATING EXCELLENT PERFORMANCE")
    print("-" * 30)
    print("Adding 25 winning trades to simulate excellent performance...")
    
    for i in range(25):
        symbol = f"EXCELLENT{i:02d}"
        confidence = random.uniform(0.4, 0.8)
        entry_price = random.uniform(50, 150)
        exit_price = entry_price * random.uniform(1.05, 1.15)  # 5-15% gain
        quantity = random.randint(50, 200)
        
        record_trade_result(
            symbol=symbol,
            action="BUY",
            confidence=confidence,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity
        )
    
    # Check performance after excellent trades
    stats = manager.calculate_performance_stats()
    print(f"Performance after excellent trades:")
    print(f"  Total trades: {stats.total_trades}")
    print(f"  Win rate: {stats.win_rate:.1%}")
    print(f"  Profit factor: {stats.profit_factor:.2f}")
    print(f"  Sharpe ratio: {stats.sharpe_ratio:.2f}")
    print(f"  Average confidence: {stats.avg_confidence:.3f}")
    
    # Check if threshold should be adjusted again
    should_adjust = manager.should_adjust_threshold()
    print(f"  Should adjust threshold: {should_adjust}")
    
    if should_adjust:
        new_threshold, reason = manager.calculate_optimal_threshold()
        print(f"  Recommended threshold: {new_threshold:.3f}")
        print(f"  Reason: {reason}")
        
        # Adjust threshold
        adjusted = adjust_confidence_threshold()
        if adjusted:
            print("  ✅ Threshold automatically adjusted!")
        else:
            print("  ❌ Threshold adjustment failed")
    
    # Show final state
    print("\n5. FINAL STATE")
    print("-" * 30)
    final_threshold = get_confidence_threshold()
    print(f"Final confidence threshold: {final_threshold:.3f}")
    print(f"Total change from initial: {final_threshold - initial_threshold:+.3f}")
    
    # Show adjustment history
    info = get_confidence_info()
    print(f"Total adjustments made: {info['adjustment_count']}")
    print(f"Last adjustment: {info['last_adjustment']}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("=" * 60)
    
    print("\nKEY FEATURES DEMONSTRATED:")
    print("✅ Automatic confidence threshold adjustment")
    print("✅ Performance-based learning")
    print("✅ Risk management through threshold control")
    print("✅ Self-optimizing trading system")
    print("✅ No human intervention required")

def show_confidence_rules():
    """Show the rules for confidence adjustment."""
    print("\n" + "=" * 60)
    print("ADAPTIVE CONFIDENCE RULES")
    print("=" * 60)
    
    print("\nThe AI automatically adjusts confidence threshold based on:")
    print("\n1. WIN RATE ANALYSIS:")
    print("   • Win rate < 40% → Increase threshold (be more selective)")
    print("   • Win rate > 70% + good profit factor → Decrease threshold (be more aggressive)")
    
    print("\n2. PROFIT FACTOR ANALYSIS:")
    print("   • High win rate but low profit factor → Increase threshold")
    print("   • Good profit factor + high win rate → Decrease threshold")
    
    print("\n3. RISK MANAGEMENT:")
    print("   • Max drawdown > 15% → Increase threshold")
    print("   • Low trade volume + good performance → Decrease threshold")
    
    print("\n4. CONSERVATIVE BEHAVIOR:")
    print("   • High threshold + low volume + good performance → Decrease threshold")
    print("   • System becomes more aggressive when performing well")
    
    print("\n5. ADJUSTMENT CONSTRAINTS:")
    print("   • Minimum 6 hours between adjustments")
    print("   • Minimum 10 trades required for adjustment")
    print("   • Threshold range: 0.10 - 0.90")
    print("   • Step size: 0.05 per adjustment")

if __name__ == "__main__":
    demo_adaptive_confidence()
    show_confidence_rules()
