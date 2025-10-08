"""
Simple AI Log Monitor
Quick way to see what the AI is doing in real-time
"""

import os
import time
from pathlib import Path

def tail_log_file(file_path, lines=10):
    """Get last N lines from a log file"""
    try:
        if not Path(file_path).exists():
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            return all_lines[-lines:] if len(all_lines) > lines else all_lines
    except Exception as e:
        return [f"Error reading {file_path}: {e}\n"]

def main():
    """Simple AI log monitor"""
    print("AI Activity Monitor - Simple View")
    print("=" * 60)
    print("Monitoring AI trading activities...")
    print("Press Ctrl+C to exit")
    print("=" * 60)
    
    log_files = {
        "AI Activity": "logs/ai_activity.log",
        "AI Trades": "logs/ai_trades.log",
        "AI Signals": "logs/ai_signals.log"
    }
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"{time.strftime('%H:%M:%S')} - AI Activity Monitor")
            print("=" * 60)
            
            for log_name, log_path in log_files.items():
                print(f"\n{log_name}:")
                print("-" * 40)
                
                lines = tail_log_file(log_path, 5)
                if lines:
                    for line in lines:
                        print(f"  {line.strip()}")
                else:
                    print("  No activity yet...")
            
            print("\n" + "=" * 60)
            print("Refreshing in 3 seconds...")
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        print("AI is still running and logging to files.")

if __name__ == "__main__":
    main()
