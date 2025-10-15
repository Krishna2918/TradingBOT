#!/usr/bin/env python3
"""
Test script for Adaptive Confidence System
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

def test_adaptive_confidence():
    """Test the adaptive confidence system."""
    print("Testing Adaptive Confidence System...")
    
    # Initialize manager
    manager = AdaptiveConfidenceManager()
    
    # Get initial threshold
    initial_threshold = get_confidence_threshold()
    print(f"Initial confidence threshold: {initial_threshold:.3f}")
    
    # Simulate some trade results to test the system
    print("\nSimulating trade results...")
    
    # Simulate some winning trades
    for i in range(15):
        symbol = f"STOCK{i:02d}"
        confidence = random.uniform(0.3, 0.8)
        entry_price = random.uniform(50, 150)
        exit_price = entry_price * random.uniform(1.02, 1.08)  # 2-8% gain
        quantity = random.randint(50, 200)
        
        record_trade_result(
            symbol=symbol,
            action="BUY",
            confidence=confidence,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity
        )
        print(f"  Recorded winning trade: {symbol} (+{((exit_price/entry_price)-1)*100:.1f}%)")
    
    # Simulate some losing trades
    for i in range(5):
        symbol = f"LOSS{i:02d}"
        confidence = random.uniform(0.3, 0.8)
        entry_price = random.uniform(50, 150)
        exit_price = entry_price * random.uniform(0.92, 0.98)  # 2-8% loss
        quantity = random.randint(50, 200)
        
        record_trade_result(
            symbol=symbol,
            action="BUY",
            confidence=confidence,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity
        )
        print(f"  Recorded losing trade: {symbol} ({((exit_price/entry_price)-1)*100:.1f}%)")
    
    # Get performance stats
    print("\nPerformance Statistics:")
    stats = manager.calculate_performance_stats()
    print(f"  Total trades: {stats.total_trades}")
    print(f"  Win rate: {stats.win_rate:.1%}")
    print(f"  Profit factor: {stats.profit_factor:.2f}")
    print(f"  Sharpe ratio: {stats.sharpe_ratio:.2f}")
    print(f"  Max drawdown: {stats.max_drawdown:.1%}")
    print(f"  Average confidence: {stats.avg_confidence:.3f}")
    
    # Check if threshold should be adjusted
    should_adjust = manager.should_adjust_threshold()
    print(f"\nShould adjust threshold: {should_adjust}")
    
    if should_adjust:
        # Calculate optimal threshold
        new_threshold, reason = manager.calculate_optimal_threshold()
        print(f"Optimal threshold: {new_threshold:.3f}")
        print(f"Reason: {reason}")
        
        # Adjust threshold
        adjusted = adjust_confidence_threshold()
        if adjusted:
            print("✅ Threshold successfully adjusted!")
        else:
            print("❌ Threshold adjustment failed")
    
    # Get final threshold info
    print("\nFinal Confidence Information:")
    info = get_confidence_info()
    print(f"  Current threshold: {info['current_threshold']:.3f}")
    print(f"  Range: {info['min_threshold']:.2f} - {info['max_threshold']:.2f}")
    print(f"  Last adjustment: {info['last_adjustment']}")
    print(f"  Adjustment count: {info['adjustment_count']}")
    print(f"  Should adjust: {info['should_adjust']}")
    
    print("\nTest completed!")

def test_confidence_scenarios():
    """Test different performance scenarios."""
    print("\n" + "="*50)
    print("Testing Different Performance Scenarios")
    print("="*50)
    
    manager = AdaptiveConfidenceManager()
    
    # Scenario 1: Poor performance (should increase threshold)
    print("\nScenario 1: Poor Performance")
    for i in range(20):
        record_trade_result(
            symbol=f"POOR{i:02d}",
            action="BUY",
            confidence=0.4,
            entry_price=100,
            exit_price=95,  # 5% loss
            quantity=100
        )
    
    stats = manager.calculate_performance_stats()
    print(f"  Win rate: {stats.win_rate:.1%}")
    print(f"  Profit factor: {stats.profit_factor:.2f}")
    
    new_threshold, reason = manager.calculate_optimal_threshold()
    print(f"  Recommended threshold: {new_threshold:.3f}")
    print(f"  Reason: {reason}")
    
    # Scenario 2: Excellent performance (should decrease threshold)
    print("\nScenario 2: Excellent Performance")
    for i in range(20):
        record_trade_result(
            symbol=f"EXCELLENT{i:02d}",
            action="BUY",
            confidence=0.6,
            entry_price=100,
            exit_price=110,  # 10% gain
            quantity=100
        )
    
    stats = manager.calculate_performance_stats()
    print(f"  Win rate: {stats.win_rate:.1%}")
    print(f"  Profit factor: {stats.profit_factor:.2f}")
    
    new_threshold, reason = manager.calculate_optimal_threshold()
    print(f"  Recommended threshold: {new_threshold:.3f}")
    print(f"  Reason: {reason}")

if __name__ == "__main__":
    test_adaptive_confidence()
    test_confidence_scenarios()
