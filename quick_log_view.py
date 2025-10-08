"""
Quick AI Log Viewer - One Command
Shows all AI logs in one simple view
"""

import os
from pathlib import Path

def show_all_logs():
    """Show all AI logs in one view"""
    print("AI TRADING LOGS - QUICK VIEW")
    print("=" * 60)
    
    log_files = [
        ("AI ACTIVITY", "logs/ai_activity.log"),
        ("AI TRADES", "logs/ai_trades.log"),
        ("AI SIGNALS", "logs/ai_signals.log"),
        ("AI DECISIONS", "logs/ai_decisions.log")
    ]
    
    for log_name, log_path in log_files:
        print(f"\n{log_name}:")
        print("-" * 40)
        
        if Path(log_path).exists():
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        # Show last 3 lines
                        for line in lines[-3:]:
                            print(f"  {line.strip()}")
                    else:
                        print("  No entries yet...")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("  File not found")
    
    print("\n" + "=" * 60)
    print("Run this script again to refresh")

if __name__ == "__main__":
    show_all_logs()
