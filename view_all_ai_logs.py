"""
Unified AI Log Viewer
Combines all AI log files into one comprehensive view
"""

import os
import time
from pathlib import Path
from datetime import datetime

def get_all_log_entries():
    """Get all log entries from all AI log files, sorted by timestamp"""
    all_entries = []
    
    log_files = {
        "ACTIVITY": "logs/ai_activity.log",
        "TRADES": "logs/ai_trades.log", 
        "SIGNALS": "logs/ai_signals.log",
        "DECISIONS": "logs/ai_decisions.log"
    }
    
    for log_type, log_path in log_files.items():
        if Path(log_path).exists():
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            # Add log type prefix
                            entry = f"[{log_type}] {line.strip()}"
                            all_entries.append(entry)
            except Exception as e:
                all_entries.append(f"[{log_type}] Error reading file: {e}")
    
    # Sort by timestamp (if available)
    try:
        all_entries.sort(key=lambda x: extract_timestamp(x))
    except:
        pass  # If sorting fails, just return unsorted
    
    return all_entries

def extract_timestamp(line):
    """Extract timestamp from log line for sorting"""
    try:
        # Look for timestamp pattern: 2025-10-06 10:30:00
        parts = line.split('|')
        if len(parts) > 1:
            timestamp_str = parts[0].strip()
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    except:
        pass
    return datetime.min

def display_log_summary():
    """Display summary of log files"""
    print("=" * 80)
    print("AI TRADING LOGS - UNIFIED VIEW")
    print("=" * 80)
    
    log_files = {
        "Activity": "logs/ai_activity.log",
        "Trades": "logs/ai_trades.log",
        "Signals": "logs/ai_signals.log", 
        "Decisions": "logs/ai_decisions.log"
    }
    
    for log_name, log_path in log_files.items():
        if Path(log_path).exists():
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    line_count = len([l for l in lines if l.strip()])
                    file_size = Path(log_path).stat().st_size
                    last_modified = datetime.fromtimestamp(Path(log_path).stat().st_mtime)
                    print(f"{log_name:12} | {line_count:3} entries | {file_size:6} bytes | {last_modified.strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"{log_name:12} | Error: {e}")
        else:
            print(f"{log_name:12} | File not found")

def main():
    """Main function for unified log viewer"""
    print("AI Trading Logs - Unified Viewer")
    print("Press Ctrl+C to exit")
    print("=" * 80)
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AI Trading Logs")
            print("=" * 80)
            
            # Display summary
            display_log_summary()
            
            print("\nRECENT ACTIVITIES (Last 20 entries):")
            print("-" * 80)
            
            # Get all log entries
            all_entries = get_all_log_entries()
            
            if all_entries:
                # Show last 20 entries
                recent_entries = all_entries[-20:] if len(all_entries) > 20 else all_entries
                
                for entry in recent_entries:
                    # Color coding based on log type
                    if "[TRADES]" in entry:
                        print(f"TRADE: {entry}")
                    elif "[SIGNALS]" in entry:
                        print(f"SIGNAL: {entry}")
                    elif "[DECISIONS]" in entry:
                        print(f"DECISION: {entry}")
                    elif "[ACTIVITY]" in entry:
                        print(f"ACTIVITY: {entry}")
                    else:
                        print(f"OTHER: {entry}")
            else:
                print("No log entries found yet.")
                print("AI may still be starting up...")
            
            print("\n" + "=" * 80)
            print("Refreshing in 5 seconds... (Ctrl+C to exit)")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nUnified log viewer stopped.")
        print("AI is still running and logging to files.")
        print("\nLog files location:")
        print("  logs/ai_activity.log")
        print("  logs/ai_trades.log")
        print("  logs/ai_signals.log")
        print("  logs/ai_decisions.log")

if __name__ == "__main__":
    main()
