#!/usr/bin/env python3
"""Phase 8 Validation: Master Orchestrator + Production Optimizer"""

import sys
import os
import io
import asyncio

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

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.integration.master_orchestrator import get_orchestrator
from src.optimization.production_optimizer import get_production_optimizer

def test_master_orchestrator():
    """Test master orchestrator functionality."""
    orchestrator = get_orchestrator()
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    
    base_price = 100.0
    prices = [base_price]
    volumes = [1000000]
    
    for i in range(99):
        price_change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        volume_change = np.random.normal(0, 0.1)
        new_volume = volumes[-1] * (1 + volume_change)
        volumes.append(max(100000, new_volume))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Test orchestrator statistics
    stats = orchestrator.get_orchestrator_statistics()
    assert 'performance_metrics' in stats, "Missing performance_metrics in stats"
    assert 'decision_pipeline' in stats, "Missing decision_pipeline in stats"
    assert 'component_status' in stats, "Missing component_status in stats"
    assert len(stats['decision_pipeline']) == 10, "Incorrect decision pipeline length"
    print("[PASS] Orchestrator statistics work")
    
    return True

async def test_orchestrator_decision_pipeline():
    """Test orchestrator decision pipeline."""
    orchestrator = get_orchestrator()
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    
    base_price = 100.0
    prices = [base_price]
    volumes = [1000000]
    
    for i in range(99):
        price_change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        volume_change = np.random.normal(0, 0.1)
        new_volume = volumes[-1] * (1 + volume_change)
        volumes.append(max(100000, new_volume))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Create additional data
    additional_data = {
        'order_book': pd.DataFrame({
            'spread': np.random.uniform(0.001, 0.01, 50),
            'depth': np.random.uniform(100000, 1000000, 50),
            'price_impact': np.random.uniform(0.0001, 0.005, 50)
        }),
        'market_regime': {
            'regime_score': 0.7,
            'confidence': 0.8
        }
    }
    
    # Test decision pipeline
    decision = await orchestrator.run_decision_pipeline(data, additional_data)
    
    assert decision is not None, "No decision returned from pipeline"
    assert decision.action in ['buy', 'sell', 'hold'], "Invalid action"
    assert 0.0 <= decision.confidence <= 1.0, "Invalid confidence"
    assert decision.position_size >= 0.0, "Invalid position size"
    assert len(decision.reasoning) > 0, "No reasoning provided"
    assert len(decision.model_consensus) > 0, "No model consensus"
    assert len(decision.risk_assessment) > 0, "No risk assessment"
    assert len(decision.execution_recommendations) > 0, "No execution recommendations"
    print("[PASS] Decision pipeline works")
    
    # Test multiple decisions
    decisions = []
    for i in range(5):
        # Vary the data slightly
        data_variant = data.copy()
        data_variant['close'] = data_variant['close'] * (1 + i * 0.01)
        
        decision = await orchestrator.run_decision_pipeline(data_variant, additional_data)
        decisions.append(decision)
    
    assert len(decisions) == 5, "Not all decisions generated"
    assert all(d.action in ['buy', 'sell', 'hold'] for d in decisions), "Invalid actions in decisions"
    print("[PASS] Multiple decision generation works")
    
    return True

async def test_production_optimizer():
    """Test production optimizer functionality."""
    optimizer = get_production_optimizer()
    
    # Start optimizer to enable monitoring
    await optimizer.start()
    
    # Wait a bit for monitoring to collect data
    await asyncio.sleep(1)
    
    # Test optimizer statistics
    stats = optimizer.get_optimization_statistics()
    assert 'is_running' in stats, "Missing is_running in stats"
    assert 'monitoring_active' in stats, "Missing monitoring_active in stats"
    assert 'total_optimizations' in stats, "Missing total_optimizations in stats"
    assert 'successful_optimizations' in stats, "Missing successful_optimizations in stats"
    assert 'failed_optimizations' in stats, "Missing failed_optimizations in stats"
    assert 'system_metrics' in stats, "Missing system_metrics in stats"
    print("[PASS] Optimizer statistics work")
    
    # Test system health
    health = optimizer.get_system_health()
    assert 'status' in health, "Missing status in health"
    assert 'issues' in health, "Missing issues in health"
    assert 'metrics' in health, "Missing metrics in health"
    assert health['status'] in ['healthy', 'warning', 'critical', 'unknown'], "Invalid health status"
    assert isinstance(health['issues'], list), "Issues should be a list"
    print("[PASS] System health monitoring works")
    
    # Stop optimizer
    await optimizer.stop()
    
    return True

async def test_optimizer_lifecycle():
    """Test optimizer start/stop lifecycle."""
    optimizer = get_production_optimizer()
    
    # Test starting optimizer
    await optimizer.start()
    assert optimizer.is_running, "Optimizer not running after start"
    assert optimizer.system_monitor.is_monitoring, "System monitor not active after start"
    print("[PASS] Optimizer start works")
    
    # Wait a bit for monitoring to collect some data
    await asyncio.sleep(2)
    
    # Test system metrics collection
    current_metrics = optimizer.system_monitor.get_current_metrics()
    assert current_metrics is not None, "No current metrics collected"
    assert current_metrics.cpu_usage >= 0.0, "Invalid CPU usage"
    assert current_metrics.memory_usage >= 0.0, "Invalid memory usage"
    assert current_metrics.memory_available >= 0.0, "Invalid available memory"
    print("[PASS] System metrics collection works")
    
    # Test metrics summary
    summary = optimizer.system_monitor.get_metrics_summary()
    assert 'avg_cpu_usage' in summary, "Missing avg_cpu_usage in summary"
    assert 'avg_memory_usage' in summary, "Missing avg_memory_usage in summary"
    assert 'total_samples' in summary, "Missing total_samples in summary"
    assert summary['total_samples'] > 0, "No metrics samples collected"
    print("[PASS] Metrics summary works")
    
    # Test stopping optimizer
    await optimizer.stop()
    assert not optimizer.is_running, "Optimizer still running after stop"
    assert not optimizer.system_monitor.is_monitoring, "System monitor still active after stop"
    print("[PASS] Optimizer stop works")
    
    return True

def test_optimization_triggers():
    """Test optimization trigger detection."""
    optimizer = get_production_optimizer()
    
    # Create mock system metrics
    from src.optimization.production_optimizer import SystemMetrics
    
    # Test high CPU usage trigger
    high_cpu_metrics = SystemMetrics(
        timestamp=datetime.now(),
        cpu_usage=85.0,  # Above threshold
        memory_usage=50.0,
        memory_available=2.0,
        disk_usage=50.0,
        network_io={'bytes_sent': 1000, 'bytes_recv': 1000, 'packets_sent': 10, 'packets_recv': 10},
        process_count=100,
        load_average=1.0
    )
    
    triggered = optimizer.performance_optimizer.check_optimization_triggers(high_cpu_metrics)
    assert 'high_cpu_usage' in triggered, "High CPU usage trigger not detected"
    print("[PASS] High CPU usage trigger detection works")
    
    # Test high memory usage trigger
    high_memory_metrics = SystemMetrics(
        timestamp=datetime.now(),
        cpu_usage=50.0,
        memory_usage=90.0,  # Above threshold
        memory_available=2.0,
        disk_usage=50.0,
        network_io={'bytes_sent': 1000, 'bytes_recv': 1000, 'packets_sent': 10, 'packets_recv': 10},
        process_count=100,
        load_average=1.0
    )
    
    triggered = optimizer.performance_optimizer.check_optimization_triggers(high_memory_metrics)
    assert 'high_memory_usage' in triggered, "High memory usage trigger not detected"
    print("[PASS] High memory usage trigger detection works")
    
    # Test low memory available trigger
    low_memory_metrics = SystemMetrics(
        timestamp=datetime.now(),
        cpu_usage=50.0,
        memory_usage=50.0,
        memory_available=0.5,  # Below threshold
        disk_usage=50.0,
        network_io={'bytes_sent': 1000, 'bytes_recv': 1000, 'packets_sent': 10, 'packets_recv': 10},
        process_count=100,
        load_average=1.0
    )
    
    triggered = optimizer.performance_optimizer.check_optimization_triggers(low_memory_metrics)
    assert 'low_memory_available' in triggered, "Low memory available trigger not detected"
    print("[PASS] Low memory available trigger detection works")
    
    return True

async def test_optimization_actions():
    """Test optimization action generation and execution."""
    optimizer = get_production_optimizer()
    
    # Generate optimization actions for high CPU usage
    triggered_optimizations = ['high_cpu_usage']
    actions = optimizer.performance_optimizer.generate_optimization_actions(triggered_optimizations)
    
    assert len(actions) > 0, "No optimization actions generated"
    assert all(action.action_type == 'cpu_optimization' for action in actions), "Wrong action type generated"
    assert all(action.expected_improvement > 0.0 for action in actions), "Invalid expected improvement"
    print("[PASS] Optimization action generation works")
    
    # Test action execution
    if actions:
        action = actions[0]
        result = await optimizer.performance_optimizer.execute_optimization_action(action)
        
        assert result is not None, "No optimization result returned"
        assert result.action == action, "Action mismatch in result"
        assert isinstance(result.success, bool), "Invalid success flag"
        assert result.actual_improvement >= 0.0, "Invalid actual improvement"
        assert result.execution_time >= 0.0, "Invalid execution time"
        print("[PASS] Optimization action execution works")
    
    return True

async def test_integration():
    """Test integration between orchestrator and optimizer."""
    orchestrator = get_orchestrator()
    optimizer = get_production_optimizer()
    
    # Start optimizer
    await optimizer.start()
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    
    base_price = 100.0
    prices = [base_price]
    volumes = [1000000]
    
    for i in range(99):
        price_change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        volume_change = np.random.normal(0, 0.1)
        new_volume = volumes[-1] * (1 + volume_change)
        volumes.append(max(100000, new_volume))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Run orchestrator decision pipeline
    decision = await orchestrator.run_decision_pipeline(data)
    
    # Check that both systems are working
    assert decision is not None, "No decision from orchestrator"
    assert optimizer.is_running, "Optimizer not running"
    
    # Get statistics from both systems
    orchestrator_stats = orchestrator.get_orchestrator_statistics()
    optimizer_stats = optimizer.get_optimization_statistics()
    
    assert 'performance_metrics' in orchestrator_stats, "Missing orchestrator performance metrics"
    assert 'system_metrics' in optimizer_stats, "Missing optimizer system metrics"
    
    # Stop optimizer
    await optimizer.stop()
    
    print("[PASS] Integration between systems works")
    
    return True

async def main():
    """Run all tests."""
    try:
        test_master_orchestrator()
        await test_orchestrator_decision_pipeline()
        await test_production_optimizer()
        await test_optimizer_lifecycle()
        test_optimization_triggers()
        await test_optimization_actions()
        await test_integration()
        print("\n[PASS] PHASE 8 VALIDATION: PASSED")
        return True
    except AssertionError as e:
        print(f"\n[FAIL] PHASE 8 VALIDATION: FAILED - {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] PHASE 8 VALIDATION: ERROR - {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)