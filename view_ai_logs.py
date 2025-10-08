"""
AI Activity Log Viewer
View real-time AI trading activities and logs
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from src.logging.ai_activity_logger import get_ai_activity_summary, get_recent_ai_activities

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%H:%M:%S')
    except:
        return timestamp_str

def display_ai_activity_summary():
    """Display AI activity summary"""
    print("=" * 80)
    print("🤖 AI TRADING ACTIVITY SUMMARY")
    print("=" * 80)
    
    try:
        summary = get_ai_activity_summary()
        
        print(f"📊 Activity Counters:")
        counters = summary.get('counters', {})
        for activity, count in counters.items():
            print(f"   {activity.replace('_', ' ').title()}: {count}")
        
        print(f"\n📁 Log Files:")
        log_files = summary.get('log_files', {})
        for log_type, file_path in log_files.items():
            file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
            print(f"   {log_type}: {file_size} bytes")
        
        print(f"\n🔄 Background Logging: {'Active' if summary.get('background_logging_active') else 'Inactive'}")
        print(f"📋 Queue Size: {summary.get('queue_size', 0)}")
        
    except Exception as e:
        print(f"Error getting summary: {e}")

def display_recent_activities(count=20):
    """Display recent AI activities"""
    print(f"\n📈 RECENT AI ACTIVITIES (Last {count})")
    print("-" * 80)
    
    try:
        activities = get_recent_ai_activities(count)
        
        if not activities:
            print("No activities found. AI may not be running yet.")
            return
        
        for activity in reversed(activities[-count:]):  # Show most recent first
            timestamp = format_timestamp(activity.get('timestamp', ''))
            activity_type = activity.get('type', 'unknown').upper()
            message = activity.get('message', 'No message')
            
            # Color coding based on activity type
            if activity_type == 'TRADE':
                print(f"🟢 {timestamp} | {activity_type} | {message}")
            elif activity_type == 'SIGNAL':
                print(f"🔵 {timestamp} | {activity_type} | {message}")
            elif activity_type == 'DECISION':
                print(f"🟡 {timestamp} | {activity_type} | {message}")
            elif activity_type == 'ERROR':
                print(f"🔴 {timestamp} | {activity_type} | {message}")
            else:
                print(f"⚪ {timestamp} | {activity_type} | {message}")
            
            # Show additional details for trades
            if activity_type == 'TRADE':
                symbol = activity.get('symbol', 'N/A')
                action = activity.get('action', 'N/A')
                quantity = activity.get('quantity', 0)
                price = activity.get('price', 0)
                pnl = activity.get('pnl', 0)
                print(f"    📊 {symbol} | {action} {quantity:.4f} @ ${price:.2f} | P&L: ${pnl:+.2f}")
            
            # Show additional details for signals
            elif activity_type == 'SIGNAL':
                symbol = activity.get('symbol', 'N/A')
                signal_type = activity.get('signal_type', 'N/A')
                confidence = activity.get('confidence', 0)
                score = activity.get('score', 0)
                print(f"    📈 {symbol} | {signal_type} | Score: {score:.3f} | Confidence: {confidence:.2f}")
            
            # Show additional details for decisions
            elif activity_type == 'DECISION':
                symbol = activity.get('symbol', 'N/A')
                decision = activity.get('decision', 'N/A')
                print(f"    🎯 {symbol} | {decision}")
            
            print()
            
    except Exception as e:
        print(f"Error getting activities: {e}")

def display_live_logs():
    """Display live log files"""
    print("\n📄 LIVE LOG FILES")
    print("-" * 80)
    
    log_files = [
        ("AI Activity", "logs/ai_activity.log"),
        ("AI Trades", "logs/ai_trades.log"),
        ("AI Signals", "logs/ai_signals.log"),
        ("AI Decisions", "logs/ai_decisions.log")
    ]
    
    for log_name, log_path in log_files:
        if Path(log_path).exists():
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    last_line = lines[-1].strip() if lines else "No entries"
                    print(f"📋 {log_name}: {last_line}")
            except Exception as e:
                print(f"📋 {log_name}: Error reading file - {e}")
        else:
            print(f"📋 {log_name}: File not found")

def main():
    """Main function for AI log viewer"""
    print("🤖 AI Trading Activity Monitor")
    print("Press Ctrl+C to exit")
    print("=" * 80)
    
    try:
        while True:
            clear_screen()
            
            print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("🤖 AI Trading Activity Monitor")
            print("=" * 80)
            
            # Display summary
            display_ai_activity_summary()
            
            # Display recent activities
            display_recent_activities(15)
            
            # Display live logs
            display_live_logs()
            
            print("\n" + "=" * 80)
            print("🔄 Refreshing in 5 seconds... (Ctrl+C to exit)")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n👋 AI Activity Monitor stopped.")
        print("Log files are still being written to:")
        print("  📁 logs/ai_activity.log")
        print("  📁 logs/ai_trades.log") 
        print("  📁 logs/ai_signals.log")
        print("  📁 logs/ai_decisions.log")
        print("  📁 logs/ai_activity.json")

if __name__ == "__main__":
    main()
