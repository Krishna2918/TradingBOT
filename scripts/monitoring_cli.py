#!/usr/bin/env python3
"""
Feature Consistency Monitoring CLI

Command-line interface for managing the feature consistency monitoring system.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.monitoring.feature_consistency_monitor import setup_feature_consistency_monitoring, AlertRule


def show_status(args):
    """Show monitoring system status."""
    print("Feature Consistency Monitoring Status")
    print("=" * 50)
    
    # Load dashboard data
    dashboard_path = Path(args.dashboard_data)
    if dashboard_path.exists():
        try:
            with open(dashboard_path, 'r') as f:
                dashboard_data = json.load(f)
            
            current_run = dashboard_data.get('current_run', {})
            processing = current_run.get('processing_summary', {})
            features = current_run.get('feature_summary', {})
            alerts = current_run.get('alerts', {})
            
            print(f"Last Updated: {dashboard_data.get('last_updated', 'Unknown')}")
            print(f"Symbols Processed: {processing.get('symbols_processed', 'N/A')}")
            print(f"Symbols Included: {processing.get('symbols_included', 'N/A')}")
            print(f"Exclusion Rate: {processing.get('exclusion_rate_pct', 'N/A')}%")
            print(f"Feature Stability: {features.get('stability_rate_pct', 'N/A')}%")
            print(f"Active Alerts: {alerts.get('total_alerts', 'N/A')}")
            
        except Exception as e:
            print(f"Error reading dashboard data: {e}")
    else:
        print("No monitoring data available")
        print(f"Expected dashboard file: {dashboard_path}")


def show_alerts(args):
    """Show recent alerts."""
    print("Recent Alerts")
    print("=" * 50)
    
    alerts_path = Path(args.alerts_data)
    if alerts_path.exists():
        try:
            with open(alerts_path, 'r') as f:
                alerts = json.load(f)
            
            if not alerts:
                print("No alerts found")
                return
            
            # Sort by timestamp (most recent first)
            sorted_alerts = sorted(alerts, key=lambda x: x.get('timestamp', ''), reverse=True)
            recent_alerts = sorted_alerts[:args.limit]
            
            for alert in recent_alerts:
                timestamp = alert.get('timestamp', 'Unknown')
                try:
                    dt = datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_time = timestamp
                
                severity = alert.get('severity', 'unknown').upper()
                description = alert.get('description', 'No description')
                value = alert.get('actual_value', 'N/A')
                threshold = alert.get('threshold', 'N/A')
                
                print(f"[{severity}] {formatted_time}")
                print(f"  {description}")
                print(f"  Value: {value}, Threshold: {threshold}")
                print()
                
        except Exception as e:
            print(f"Error reading alerts data: {e}")
    else:
        print("No alerts data available")
        print(f"Expected alerts file: {alerts_path}")


def show_metrics(args):
    """Show metrics history."""
    print("Metrics History")
    print("=" * 50)
    
    metrics_path = Path(args.metrics_data)
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics_history = json.load(f)
            
            if not metrics_history:
                print("No metrics history found")
                return
            
            # Show recent runs
            recent_metrics = metrics_history[-args.limit:]
            
            print(f"Showing last {len(recent_metrics)} runs:")
            print()
            
            for i, metrics in enumerate(recent_metrics, 1):
                start_time = metrics.get('processing_start_time', 'Unknown')
                try:
                    dt = datetime.fromisoformat(start_time)
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_time = start_time
                
                symbols_processed = metrics.get('total_symbols_processed', 0)
                symbols_included = metrics.get('symbols_included', 0)
                exclusion_rate = metrics.get('exclusion_rate', 0) * 100
                stability_rate = metrics.get('feature_stability_rate', 0) * 100
                processing_time = metrics.get('total_processing_time_seconds', 0)
                alerts_count = len(metrics.get('alerts_triggered', []))
                
                print(f"Run {i}: {formatted_time}")
                print(f"  Symbols: {symbols_included}/{symbols_processed} included ({exclusion_rate:.1f}% excluded)")
                print(f"  Stability: {stability_rate:.1f}%")
                print(f"  Time: {processing_time:.1f}s")
                print(f"  Alerts: {alerts_count}")
                print()
                
        except Exception as e:
            print(f"Error reading metrics data: {e}")
    else:
        print("No metrics data available")
        print(f"Expected metrics file: {metrics_path}")


def generate_dashboard(args):
    """Generate HTML dashboard."""
    print("Generating monitoring dashboard...")
    
    try:
        from scripts.generate_monitoring_dashboard import main as generate_dashboard_main
        
        # Set up arguments for dashboard generator
        original_argv = sys.argv
        sys.argv = [
            'generate_monitoring_dashboard.py',
            '--dashboard-data', args.dashboard_data,
            '--metrics-data', args.metrics_data,
            '--alerts-data', args.alerts_data,
            '--output', args.output
        ]
        
        generate_dashboard_main()
        
        sys.argv = original_argv
        
        print(f"✓ Dashboard generated: {args.output}")
        print(f"Open {Path(args.output).absolute()} in your browser")
        
    except Exception as e:
        print(f"✗ Dashboard generation failed: {e}")


def test_alerts(args):
    """Test alert system with sample data."""
    print("Testing alert system...")
    
    # Create monitoring system
    monitor = setup_feature_consistency_monitoring({
        'dashboard_output_path': args.dashboard_data,
        'metrics_output_path': args.metrics_data,
        'alert_output_path': args.alerts_data
    })
    
    # Start monitoring
    monitor.start_processing_monitoring()
    
    # Simulate high exclusion rate to trigger alert
    for i in range(10):
        included = i < 3  # 70% exclusion rate
        coverage = 0.85 if included else 0.75
        monitor.record_symbol_processing(f'TEST_{i}', included, coverage, 1000, 950, 1.0)
    
    # Record feature analysis
    monitor.record_feature_analysis(100, ['feature_' + str(i) for i in range(50)], 
                                   ['unstable_' + str(i) for i in range(50)])
    
    # End monitoring (this will trigger alerts)
    final_metrics = monitor.end_processing_monitoring()
    
    print(f"✓ Test completed")
    print(f"  Exclusion rate: {final_metrics.exclusion_rate:.1%}")
    print(f"  Alerts triggered: {len(final_metrics.alerts_triggered)}")
    print(f"  Alert types: {', '.join(final_metrics.alerts_triggered)}")


def cleanup_old_data(args):
    """Clean up old monitoring data."""
    print("Cleaning up old monitoring data...")
    
    cutoff_date = datetime.now() - timedelta(days=args.days)
    
    files_to_clean = [
        (args.metrics_data, 'metrics'),
        (args.alerts_data, 'alerts')
    ]
    
    for file_path, data_type in files_to_clean:
        path = Path(file_path)
        if not path.exists():
            continue
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                continue
            
            # Filter out old data
            filtered_data = []
            for item in data:
                timestamp_str = item.get('timestamp') or item.get('processing_start_time')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp > cutoff_date:
                            filtered_data.append(item)
                    except:
                        # Keep items with invalid timestamps
                        filtered_data.append(item)
                else:
                    # Keep items without timestamps
                    filtered_data.append(item)
            
            # Save filtered data
            with open(path, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            
            removed_count = len(data) - len(filtered_data)
            print(f"✓ Cleaned {data_type}: removed {removed_count} old entries")
            
        except Exception as e:
            print(f"✗ Error cleaning {data_type}: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Feature Consistency Monitoring CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current status
  python monitoring_cli.py status
  
  # Show recent alerts
  python monitoring_cli.py alerts --limit 10
  
  # Show metrics history
  python monitoring_cli.py metrics --limit 5
  
  # Generate dashboard
  python monitoring_cli.py dashboard --output monitoring/dashboard.html
  
  # Test alert system
  python monitoring_cli.py test-alerts
  
  # Clean up old data (older than 30 days)
  python monitoring_cli.py cleanup --days 30
        """
    )
    
    # Global arguments
    parser.add_argument('--dashboard-data', default='monitoring/feature_consistency_dashboard.json',
                       help='Path to dashboard data file')
    parser.add_argument('--metrics-data', default='monitoring/feature_consistency_metrics.json',
                       help='Path to metrics data file')
    parser.add_argument('--alerts-data', default='monitoring/alerts.json',
                       help='Path to alerts data file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show monitoring status')
    status_parser.set_defaults(func=show_status)
    
    # Alerts command
    alerts_parser = subparsers.add_parser('alerts', help='Show recent alerts')
    alerts_parser.add_argument('--limit', type=int, default=10, help='Number of alerts to show')
    alerts_parser.set_defaults(func=show_alerts)
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show metrics history')
    metrics_parser.add_argument('--limit', type=int, default=5, help='Number of runs to show')
    metrics_parser.set_defaults(func=show_metrics)
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate HTML dashboard')
    dashboard_parser.add_argument('--output', default='monitoring/dashboard.html',
                                 help='Output HTML file path')
    dashboard_parser.set_defaults(func=generate_dashboard)
    
    # Test alerts command
    test_parser = subparsers.add_parser('test-alerts', help='Test alert system')
    test_parser.set_defaults(func=test_alerts)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old monitoring data')
    cleanup_parser.add_argument('--days', type=int, default=30,
                               help='Remove data older than N days')
    cleanup_parser.set_defaults(func=cleanup_old_data)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()