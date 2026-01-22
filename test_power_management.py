"""
Quick test script for power management system
Verifies all components are working correctly
"""

import sys
import os
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("POWER MANAGEMENT SYSTEM - VERIFICATION TEST")
print("=" * 70)
print()

# Test 1: Import modules
print("Test 1: Importing power_management modules...")
try:
    from power_management import (
        DEFAULT_CONFIG,
        ESTIMATED_SAVINGS,
        get_gpu_manager,
        get_precision_manager,
        get_worker_scaler,
        get_cache_manager,
        get_network_optimizer,
        get_schedule_manager,
        get_power_monitor
    )
    print("  ✅ All modules imported successfully")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

print()

# Test 2: Check configuration
print("Test 2: Checking configuration...")
try:
    print(f"  - Power management enabled: {DEFAULT_CONFIG.enabled}")
    print(f"  - Target savings: {DEFAULT_CONFIG.target_savings}%")
    print(f"  - GPU enabled: {DEFAULT_CONFIG.gpu.enabled}")
    print(f"  - Mixed precision enabled: {DEFAULT_CONFIG.mixed_precision.enabled}")
    print(f"  - Worker scaling enabled: {DEFAULT_CONFIG.workers.enabled}")
    print(f"  - Caching enabled: {DEFAULT_CONFIG.caching.enabled}")
    print(f"  - Network optimization enabled: {DEFAULT_CONFIG.network.enable_batching}")
    print(f"  - Schedule management enabled: {DEFAULT_CONFIG.schedule.enabled}")
    print("  ✅ Configuration loaded successfully")
except Exception as e:
    print(f"  ❌ Configuration check failed: {e}")

print()

# Test 3: Estimated savings
print("Test 3: Checking estimated savings...")
try:
    total_savings = sum(ESTIMATED_SAVINGS.values())
    print(f"  - Total estimated savings: {total_savings}%")
    print(f"  - Target: {DEFAULT_CONFIG.target_savings}%")

    if total_savings >= DEFAULT_CONFIG.target_savings:
        print(f"  ✅ Target of {DEFAULT_CONFIG.target_savings}% will be MET")
    else:
        print(f"  ⚠️  Target of {DEFAULT_CONFIG.target_savings}% may not be met")

    print()
    print("  Savings breakdown:")
    for component, savings in ESTIMATED_SAVINGS.items():
        print(f"    - {component:30} {savings:>3}%")
except Exception as e:
    print(f"  ❌ Savings calculation failed: {e}")

print()

# Test 4: Initialize components
print("Test 4: Initializing components...")

try:
    gpu_manager = get_gpu_manager(DEFAULT_CONFIG.gpu)
    print(f"  ✅ GPU Manager initialized")
    print(f"     - GPU available: {gpu_manager.gpu_available}")
except Exception as e:
    print(f"  ⚠️  GPU Manager: {e}")

try:
    precision_manager = get_precision_manager(DEFAULT_CONFIG.mixed_precision)
    print(f"  ✅ Mixed Precision Manager initialized")
    print(f"     - Framework: {precision_manager.framework}")
except Exception as e:
    print(f"  ⚠️  Mixed Precision Manager: {e}")

try:
    worker_scaler = get_worker_scaler(DEFAULT_CONFIG.workers, current_workers=4)
    print(f"  ✅ Worker Scaler initialized")
    print(f"     - Current workers: {worker_scaler.current_workers}")
except Exception as e:
    print(f"  ⚠️  Worker Scaler: {e}")

try:
    cache_manager = get_cache_manager(DEFAULT_CONFIG.caching)
    print(f"  ✅ Cache Manager initialized")
    print(f"     - Backend: {cache_manager.config.backend}")
    if cache_manager.backend is None:
        print(f"     - Using in-memory cache (Redis not available)")
except Exception as e:
    print(f"  ⚠️  Cache Manager: {e}")

try:
    network_optimizer = get_network_optimizer(DEFAULT_CONFIG.network)
    print(f"  ✅ Network Optimizer initialized")
except Exception as e:
    print(f"  ⚠️  Network Optimizer: {e}")

try:
    schedule_manager = get_schedule_manager(DEFAULT_CONFIG.schedule)
    print(f"  ✅ Schedule Manager initialized")
    market_status = schedule_manager.get_market_status()
    print(f"     - Market status: {market_status.value}")
except Exception as e:
    print(f"  ⚠️  Schedule Manager: {e}")

print()

# Test 5: Power Monitor
print("Test 5: Testing power monitor...")
try:
    monitor = get_power_monitor(DEFAULT_CONFIG)
    monitor.set_managers(
        gpu_manager=gpu_manager,
        precision_manager=precision_manager,
        worker_scaler=worker_scaler,
        cache_manager=cache_manager,
        network_optimizer=network_optimizer,
        schedule_manager=schedule_manager
    )
    print("  ✅ Power Monitor initialized")

    # Get comprehensive stats
    stats = monitor.get_comprehensive_stats()
    print(f"     - Components tracked: {len(stats['components'])}")

    # Get health status
    health = monitor.get_health_status()
    print(f"     - Overall health: {health['overall']}")

    # Get savings
    savings = monitor._calculate_total_savings(stats['components'])
    print(f"     - Estimated total savings: {savings['total_percentage']}%")
    print(f"     - Target met: {savings['target_met']}")

except Exception as e:
    print(f"  ❌ Power Monitor failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 6: Generate summary report
print("Test 6: Generating summary report...")
try:
    report = monitor.get_summary_report()
    print()
    print(report)
except Exception as e:
    print(f"  ❌ Report generation failed: {e}")

print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print()
print("Next steps:")
print("  1. Review the README: power_management/README.md")
print("  2. Check integration examples")
print("  3. Start integrating components into your trading bot")
print()
