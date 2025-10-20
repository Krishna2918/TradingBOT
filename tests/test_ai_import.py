#!/usr/bin/env python3
"""Test AI system import"""

try:
    from src.ai.autonomous_trading_ai import AutonomousTradingAI
    print("✅ AI import successful - AutonomousTradingAI available")
except Exception as e:
    print(f"❌ AI import failed: {e}")
    import traceback
    traceback.print_exc()
