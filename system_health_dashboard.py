#!/usr/bin/env python3
"""
System health dashboard and reporting tool for continuous data collection system.

This script provides a comprehensive dashboard for monitoring system health,
performance metrics, and generating detailed reports.

Usage:
    python system_health_dashboard.py [--mode MODE] [--refresh SECONDS]

Requirements: 4.1, 4.2, 10.1, 10.5
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the continuous_data_collection package to the path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core.config import ConfigLoader
from continuous_data_collection.core.logging_config import setup_logging
from continuous_data_collection.core.state_manager import StateManager
from continuous_data_collection.monitoring.health_monitor import HealthMonitor
from continuous_data_collection.monitoring.progress_tracker import ProgressTracker
from continuous_data_collection.core.system_factory import SystemFactory


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="System health dashboard for continuous data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dashboard Modes:
    live        - Live dashboard with real-time updates
    report      - Generate comprehensive health report
    metrics     - Show performance metrics only
    alerts      - Show active alerts and issues

Examples:
    python system_health_dashboard.py --mode live
    python system_health_dashboard.py --mode report --output health_report.json
    python system_health_dashboard.py --mode metrics --refresh 5
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["live", "report", "metrics", "alerts"],
        default="live",
        help="Dashboard mode (default: live)"
    )
    
    parser.add_argument(
        "--refresh", "-r",
        type=int,
        default=5,
        help="Refresh interval in seconds for live mode (default: 5)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/development.yaml",
        help="Configuration file path (default: config/development.yaml)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for report mode"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json", "html", "txt"],
        default="json",
        help="Output format for reports (default: json)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=80.0,
        help="Alert threshold for resource usage (default: 80.0)"
    )
    
    return parser.parse_args()


def clear_screen():
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days:.0f}d {hours:.0f}h"


def create_status_indicator(value: float, threshold: float, reverse: bool = False) -> str:
    """Create a status indicator based on value and threshold."""
    if reverse:
        # For metrics where lower is better (like error rates)
        if value <= threshold * 0.5:
            return "🟢"
        elif value <= threshold:
            return "🟡"
        else:
            return "🔴"
    else:
        # For metrics where higher is better (like success rates)
        if value >= threshold:
            return "🟢"
        elif value >= threshold * 0.7:
            return "🟡"
        else:
            return "🔴"


async def collect_system_metrics(config_path: str) -> Dict[str, Any]:
    """Collect comprehensive system metrics."""
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        # Create system components
        system_factory = SystemFactory(config)
        health_monitor = await system_factory.create_health_monitor()
        progress_tracker = await system_factory.create_progress_tracker()
        state_manager = StateManager(config)
        
        # Collect metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": {},
            "progress": {},
            "performance": {},
            "resources": {},
            "alerts": []
        }
        
        # System health
        health_status = await health_monitor.check_system_health()
        metrics["system_health"] = {
            "is_healthy": health_status.is_healthy,
            "cpu_usage": health_status.cpu_usage,
            "memory_usage": health_status.memory_usage,
            "disk_usage": health_status.disk_usage,
            "api_connectivity": health_status.api_connectivity,
            "active_workers": health_status.active_workers,
            "collection_rate": health_status.collection_rate,
            "error_rate": health_status.error_rate,
            "issues": health_status.issues
        }
        
        # Progress metrics
        progress_stats = progress_tracker.get_current_stats()
        metrics["progress"] = {
            "total_target": progress_stats.total_target,
            "completed": progress_stats.completed,
            "failed": progress_stats.failed,
            "pending": progress_stats.pending,
            "in_progress": progress_stats.in_progress,
            "success_rate": progress_stats.success_rate,
            "current_throughput": progress_stats.current_throughput,
            "eta": progress_stats.eta.isoformat() if progress_stats.eta else None,
            "data_quality_avg": progress_stats.data_quality_avg
        }
        
        # Performance metrics
        try:
            import psutil
            
            # CPU details
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            # Memory details
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk details
            disk = psutil.disk_usage('.')
            disk_io = psutil.disk_io_counters()
            
            # Network details
            network_io = psutil.net_io_counters()
            
            metrics["performance"] = {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg": list(load_avg)
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "percent": swap.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                }
            }
            
        except ImportError:
            metrics["performance"] = {"error": "psutil not available"}
        
        # Resource usage alerts
        if metrics["system_health"]["cpu_usage"] > 90:
            metrics["alerts"].append({
                "level": "critical",
                "message": f"High CPU usage: {metrics['system_health']['cpu_usage']:.1f}%"
            })
        
        if metrics["system_health"]["memory_usage"] > 90:
            metrics["alerts"].append({
                "level": "critical",
                "message": f"High memory usage: {metrics['system_health']['memory_usage']:.1f}%"
            })
        
        if metrics["system_health"]["disk_usage"] > 95:
            metrics["alerts"].append({
                "level": "critical",
                "message": f"High disk usage: {metrics['system_health']['disk_usage']:.1f}%"
            })
        
        if metrics["system_health"]["error_rate"] > 0.1:
            metrics["alerts"].append({
                "level": "warning",
                "message": f"High error rate: {metrics['system_health']['error_rate']:.1%}"
            })
        
        if metrics["progress"]["current_throughput"] < 10:
            metrics["alerts"].append({
                "level": "warning",
                "message": f"Low throughput: {metrics['progress']['current_throughput']:.1f} stocks/min"
            })
        
        return metrics
        
    except Exception as e:
        logging.error(f"Failed to collect system metrics: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


def display_live_dashboard(metrics: Dict[str, Any], threshold: float):
    """Display live dashboard with real-time metrics."""
    clear_screen()
    
    print("🖥️  CONTINUOUS DATA COLLECTION - SYSTEM HEALTH DASHBOARD")
    print("=" * 80)
    print(f"Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    if "error" in metrics:
        print(f"❌ Error collecting metrics: {metrics['error']}")
        return
    
    # System Health Overview
    health = metrics.get("system_health", {})
    overall_status = "🟢 HEALTHY" if health.get("is_healthy", False) else "🔴 UNHEALTHY"
    
    print(f"🏥 SYSTEM HEALTH: {overall_status}")
    print("-" * 40)
    
    cpu_indicator = create_status_indicator(health.get("cpu_usage", 0), threshold, reverse=True)
    memory_indicator = create_status_indicator(health.get("memory_usage", 0), threshold, reverse=True)
    disk_indicator = create_status_indicator(health.get("disk_usage", 0), threshold, reverse=True)
    
    print(f"  {cpu_indicator} CPU Usage:      {health.get('cpu_usage', 0):.1f}%")
    print(f"  {memory_indicator} Memory Usage:   {health.get('memory_usage', 0):.1f}%")
    print(f"  {disk_indicator} Disk Usage:     {health.get('disk_usage', 0):.1f}%")
    print(f"  👥 Active Workers:  {health.get('active_workers', 0)}")
    print(f"  📊 Collection Rate: {health.get('collection_rate', 0):.1f} stocks/min")
    print(f"  ⚠️  Error Rate:      {health.get('error_rate', 0):.1%}")
    
    # API Connectivity
    api_status = health.get("api_connectivity", {})
    if api_status:
        print(f"\n🌐 API CONNECTIVITY:")
        for api, status in api_status.items():
            indicator = "🟢" if status else "🔴"
            print(f"  {indicator} {api.title()}: {'Connected' if status else 'Disconnected'}")
    
    # Progress Overview
    progress = metrics.get("progress", {})
    print(f"\n📈 COLLECTION PROGRESS:")
    print("-" * 40)
    
    total = progress.get("total_target", 0)
    completed = progress.get("completed", 0)
    failed = progress.get("failed", 0)
    pending = progress.get("pending", 0)
    
    completion_pct = (completed / total * 100) if total > 0 else 0
    success_rate = progress.get("success_rate", 0)
    
    # Progress bar
    bar_width = 30
    filled = int(bar_width * completion_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    print(f"  Progress:     [{bar}] {completion_pct:.1f}%")
    print(f"  Completed:    {completed:,} / {total:,}")
    print(f"  Failed:       {failed:,}")
    print(f"  Pending:      {pending:,}")
    print(f"  Success Rate: {success_rate:.1%}")
    print(f"  Throughput:   {progress.get('current_throughput', 0):.1f} stocks/min")
    
    eta = progress.get("eta")
    if eta:
        eta_dt = datetime.fromisoformat(eta.replace('Z', '+00:00'))
        eta_delta = eta_dt.replace(tzinfo=None) - datetime.utcnow()
        if eta_delta.total_seconds() > 0:
            print(f"  ETA:          {format_uptime(eta_delta.total_seconds())}")
        else:
            print(f"  ETA:          Complete")
    
    # Performance Details
    perf = metrics.get("performance", {})
    if "error" not in perf:
        print(f"\n⚡ PERFORMANCE DETAILS:")
        print("-" * 40)
        
        cpu = perf.get("cpu", {})
        memory = perf.get("memory", {})
        disk = perf.get("disk", {})
        
        print(f"  CPU Cores:    {cpu.get('count', 0)}")
        if cpu.get("load_avg"):
            load_avg = cpu["load_avg"]
            print(f"  Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
        
        print(f"  Memory Total: {format_bytes(memory.get('total', 0))}")
        print(f"  Memory Free:  {format_bytes(memory.get('available', 0))}")
        print(f"  Disk Total:   {format_bytes(disk.get('total', 0))}")
        print(f"  Disk Free:    {format_bytes(disk.get('free', 0))}")
    
    # Active Alerts
    alerts = metrics.get("alerts", [])
    if alerts:
        print(f"\n🚨 ACTIVE ALERTS ({len(alerts)}):")
        print("-" * 40)
        
        for alert in alerts:
            level_icon = "🔴" if alert["level"] == "critical" else "🟡"
            print(f"  {level_icon} {alert['message']}")
    else:
        print(f"\n✅ NO ACTIVE ALERTS")
    
    # Issues
    issues = health.get("issues", [])
    if issues:
        print(f"\n⚠️  SYSTEM ISSUES ({len(issues)}):")
        print("-" * 40)
        for issue in issues:
            print(f"  • {issue}")
    
    print(f"\n{'=' * 80}")
    print(f"Press Ctrl+C to exit")


def display_metrics_only(metrics: Dict[str, Any]):
    """Display performance metrics only."""
    if "error" in metrics:
        print(f"❌ Error: {metrics['error']}")
        return
    
    print("📊 PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Timestamp: {metrics['timestamp']}")
    
    # Key metrics
    health = metrics.get("system_health", {})
    progress = metrics.get("progress", {})
    
    print(f"\nCollection Rate:  {health.get('collection_rate', 0):.1f} stocks/min")
    print(f"Success Rate:     {progress.get('success_rate', 0):.1%}")
    print(f"Error Rate:       {health.get('error_rate', 0):.1%}")
    print(f"CPU Usage:        {health.get('cpu_usage', 0):.1f}%")
    print(f"Memory Usage:     {health.get('memory_usage', 0):.1f}%")
    print(f"Disk Usage:       {health.get('disk_usage', 0):.1f}%")
    print(f"Active Workers:   {health.get('active_workers', 0)}")
    
    # Progress
    total = progress.get("total_target", 0)
    completed = progress.get("completed", 0)
    completion_pct = (completed / total * 100) if total > 0 else 0
    
    print(f"\nProgress:         {completion_pct:.1f}% ({completed:,}/{total:,})")
    print(f"Pending:          {progress.get('pending', 0):,}")
    print(f"Failed:           {progress.get('failed', 0):,}")


def display_alerts_only(metrics: Dict[str, Any]):
    """Display active alerts only."""
    if "error" in metrics:
        print(f"❌ Error: {metrics['error']}")
        return
    
    alerts = metrics.get("alerts", [])
    issues = metrics.get("system_health", {}).get("issues", [])
    
    print("🚨 SYSTEM ALERTS")
    print("=" * 50)
    print(f"Timestamp: {metrics['timestamp']}")
    
    if not alerts and not issues:
        print("\n✅ No active alerts or issues")
        return
    
    if alerts:
        print(f"\n🔔 ALERTS ({len(alerts)}):")
        for alert in alerts:
            level_icon = "🔴" if alert["level"] == "critical" else "🟡"
            print(f"  {level_icon} [{alert['level'].upper()}] {alert['message']}")
    
    if issues:
        print(f"\n⚠️  ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"  • {issue}")


async def generate_health_report(metrics: Dict[str, Any], output_path: str, format_type: str):
    """Generate comprehensive health report."""
    if "error" in metrics:
        print(f"❌ Cannot generate report: {metrics['error']}")
        return
    
    # Enhance metrics with additional analysis
    report = {
        "report_metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "report_type": "system_health",
            "format": format_type
        },
        "executive_summary": {},
        "detailed_metrics": metrics,
        "recommendations": []
    }
    
    # Executive summary
    health = metrics.get("system_health", {})
    progress = metrics.get("progress", {})
    alerts = metrics.get("alerts", [])
    
    report["executive_summary"] = {
        "overall_health": "healthy" if health.get("is_healthy", False) else "unhealthy",
        "completion_percentage": (progress.get("completed", 0) / progress.get("total_target", 1)) * 100,
        "critical_alerts": len([a for a in alerts if a["level"] == "critical"]),
        "warning_alerts": len([a for a in alerts if a["level"] == "warning"]),
        "collection_rate": health.get("collection_rate", 0),
        "success_rate": progress.get("success_rate", 0)
    }
    
    # Recommendations
    if health.get("cpu_usage", 0) > 80:
        report["recommendations"].append("Consider reducing worker count or optimizing CPU usage")
    
    if health.get("memory_usage", 0) > 80:
        report["recommendations"].append("Monitor memory usage and consider increasing available RAM")
    
    if health.get("error_rate", 0) > 0.05:
        report["recommendations"].append("Investigate high error rate and improve error handling")
    
    if progress.get("current_throughput", 0) < 20:
        report["recommendations"].append("Optimize collection throughput or check API connectivity")
    
    # Save report
    try:
        if format_type == "json":
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format_type == "html":
            html_content = generate_html_report(report)
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        elif format_type == "txt":
            txt_content = generate_text_report(report)
            with open(output_path, 'w') as f:
                f.write(txt_content)
        
        print(f"✅ Health report generated: {output_path}")
        
    except Exception as e:
        print(f"❌ Failed to save report: {e}")


def generate_html_report(report: Dict[str, Any]) -> str:
    """Generate HTML health report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>System Health Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ margin: 10px 0; }}
            .alert {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .critical {{ background: #ffebee; border-left: 4px solid #f44336; }}
            .warning {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
            .healthy {{ color: #4caf50; }}
            .unhealthy {{ color: #f44336; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>System Health Report</h1>
            <p>Generated: {report['report_metadata']['generated_at']}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric">Overall Health: <span class="{report['executive_summary']['overall_health']}">{report['executive_summary']['overall_health'].title()}</span></div>
            <div class="metric">Completion: {report['executive_summary']['completion_percentage']:.1f}%</div>
            <div class="metric">Collection Rate: {report['executive_summary']['collection_rate']:.1f} stocks/min</div>
            <div class="metric">Success Rate: {report['executive_summary']['success_rate']:.1%}</div>
        </div>
        
        <div class="section">
            <h2>Active Alerts</h2>
    """
    
    alerts = report['detailed_metrics'].get('alerts', [])
    if alerts:
        for alert in alerts:
            html += f'<div class="alert {alert["level"]}">{alert["message"]}</div>'
    else:
        html += '<p>No active alerts</p>'
    
    html += """
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
    """
    
    recommendations = report.get('recommendations', [])
    if recommendations:
        html += '<ul>'
        for rec in recommendations:
            html += f'<li>{rec}</li>'
        html += '</ul>'
    else:
        html += '<p>No recommendations at this time</p>'
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html


def generate_text_report(report: Dict[str, Any]) -> str:
    """Generate text health report."""
    lines = [
        "SYSTEM HEALTH REPORT",
        "=" * 50,
        f"Generated: {report['report_metadata']['generated_at']}",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 20,
        f"Overall Health: {report['executive_summary']['overall_health'].title()}",
        f"Completion: {report['executive_summary']['completion_percentage']:.1f}%",
        f"Collection Rate: {report['executive_summary']['collection_rate']:.1f} stocks/min",
        f"Success Rate: {report['executive_summary']['success_rate']:.1%}",
        f"Critical Alerts: {report['executive_summary']['critical_alerts']}",
        f"Warning Alerts: {report['executive_summary']['warning_alerts']}",
        "",
        "ACTIVE ALERTS",
        "-" * 20
    ]
    
    alerts = report['detailed_metrics'].get('alerts', [])
    if alerts:
        for alert in alerts:
            lines.append(f"[{alert['level'].upper()}] {alert['message']}")
    else:
        lines.append("No active alerts")
    
    lines.extend([
        "",
        "RECOMMENDATIONS",
        "-" * 20
    ])
    
    recommendations = report.get('recommendations', [])
    if recommendations:
        for rec in recommendations:
            lines.append(f"• {rec}")
    else:
        lines.append("No recommendations at this time")
    
    return "\n".join(lines)


async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(level="WARNING")  # Reduce log noise for dashboard
        
        # Check if config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
        
        if args.mode == "live":
            # Live dashboard mode
            print("Starting live dashboard...")
            try:
                while True:
                    metrics = await collect_system_metrics(str(config_path))
                    display_live_dashboard(metrics, args.threshold)
                    await asyncio.sleep(args.refresh)
            except KeyboardInterrupt:
                print("\n👋 Dashboard stopped")
        
        elif args.mode == "report":
            # Report generation mode
            print("Generating health report...")
            metrics = await collect_system_metrics(str(config_path))
            
            if not args.output:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                args.output = f"health_report_{timestamp}.{args.format}"
            
            await generate_health_report(metrics, args.output, args.format)
        
        elif args.mode == "metrics":
            # Metrics only mode
            if args.refresh > 0:
                try:
                    while True:
                        metrics = await collect_system_metrics(str(config_path))
                        clear_screen()
                        display_metrics_only(metrics)
                        print(f"\nRefreshing in {args.refresh} seconds... (Ctrl+C to exit)")
                        await asyncio.sleep(args.refresh)
                except KeyboardInterrupt:
                    print("\n👋 Metrics monitoring stopped")
            else:
                metrics = await collect_system_metrics(str(config_path))
                display_metrics_only(metrics)
        
        elif args.mode == "alerts":
            # Alerts only mode
            metrics = await collect_system_metrics(str(config_path))
            display_alerts_only(metrics)
        
    except Exception as e:
        print(f"\n❌ Dashboard failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())