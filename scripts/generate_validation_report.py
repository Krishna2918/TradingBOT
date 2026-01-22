#!/usr/bin/env python3
"""
Validation Report Generator

Generates HTML validation reports for CI/CD pipeline results.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

def generate_validation_report(results_file: str = 'ci-validation-results.json') -> str:
    """
    Generate HTML validation report from CI results.
    
    Args:
        results_file: Path to the CI validation results JSON file
        
    Returns:
        HTML report content
    """
    
    # Load results
    if not os.path.exists(results_file):
        return generate_error_report("Results file not found")
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        return generate_error_report(f"Failed to load results: {e}")
    
    # Generate HTML report
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Trading System Validation Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header .timestamp {{
                margin-top: 10px;
                opacity: 0.9;
                font-size: 1.1em;
            }}
            .status-banner {{
                padding: 20px;
                text-align: center;
                font-size: 1.5em;
                font-weight: bold;
            }}
            .status-passed {{
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            .status-failed {{
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }}
            .section {{
                margin: 30px;
                padding: 20px;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }}
            .section h2 {{
                margin-top: 0;
                color: #667eea;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #667eea;
            }}
            .metric-label {{
                font-weight: bold;
                color: #666;
                font-size: 0.9em;
            }}
            .metric-value {{
                font-size: 1.5em;
                color: #333;
                margin-top: 5px;
            }}
            .test-results {{
                margin: 20px 0;
            }}
            .test-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
                background: #f8f9fa;
            }}
            .test-passed {{
                background: #d4edda;
                color: #155724;
            }}
            .test-failed {{
                background: #f8d7da;
                color: #721c24;
            }}
            .test-status {{
                font-weight: bold;
            }}
            .error-details {{
                background: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 4px;
                margin: 10px 0;
                font-family: monospace;
                white-space: pre-wrap;
            }}
            .performance-chart {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 6px;
                margin: 20px 0;
            }}
            .regression-alert {{
                background: #fff3cd;
                color: #856404;
                padding: 15px;
                border-radius: 4px;
                border: 1px solid #ffeaa7;
                margin: 10px 0;
            }}
            .footer {{
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                color: #666;
                border-top: 1px solid #e0e0e0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Trading System Validation Report</h1>
                <div class="timestamp">
                    Generated: {results.get('timestamp', 'Unknown')}
                </div>
            </div>
    """
    
    # Status banner
    summary = results.get('summary', {})
    status = summary.get('status', 'UNKNOWN')
    status_class = 'status-passed' if status == 'PASSED' else 'status-failed'
    status_emoji = '✅' if status == 'PASSED' else '❌'
    
    html += f"""
            <div class="status-banner {status_class}">
                {status_emoji} Validation Status: {status}
            </div>
    """
    
    # Summary metrics
    html += """
            <div class="section">
                <h2>Summary</h2>
                <div class="metric-grid">
    """
    
    total_time = summary.get('total_time_seconds', 0)
    ai_limit = summary.get('ai_limit', 'N/A')
    
    html += f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Execution Time</div>
                        <div class="metric-value">{total_time}s</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">AI Limit</div>
                        <div class="metric-value">{ai_limit}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Validation Status</div>
                        <div class="metric-value">{status}</div>
                    </div>
    """
    
    html += """
                </div>
            </div>
    """
    
    # System Health
    if 'system_health' in results:
        html += generate_system_health_section(results['system_health'])
    
    # Test Results
    if 'tests' in results:
        html += generate_test_results_section(results['tests'])
    
    # Performance
    if 'performance' in results:
        html += generate_performance_section(results['performance'])
    
    # Regressions
    if 'regressions' in results:
        html += generate_regressions_section(results['regressions'])
    
    # Footer
    html += f"""
            <div class="footer">
                <p>Report generated by Trading System CI/CD Pipeline</p>
                <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def generate_system_health_section(health: Dict[str, Any]) -> str:
    """Generate system health section."""
    html = """
            <div class="section">
                <h2>System Health</h2>
                <div class="metric-grid">
    """
    
    if 'error' in health:
        html += f"""
                    <div class="error-details">
                        System Health Check Error: {health['error']}
                    </div>
        """
    else:
        # Python version
        python_version = health.get('python_version', 'Unknown')
        html += f"""
                    <div class="metric-card">
                        <div class="metric-label">Python Version</div>
                        <div class="metric-value">{python_version}</div>
                    </div>
        """
        
        # Memory
        memory_available = health.get('memory_available_gb', 0)
        memory_percent = health.get('memory_percent_used', 0)
        html += f"""
                    <div class="metric-card">
                        <div class="metric-label">Available Memory</div>
                        <div class="metric-value">{memory_available} GB</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Memory Usage</div>
                        <div class="metric-value">{memory_percent}%</div>
                    </div>
        """
        
        # Disk
        disk_free = health.get('disk_free_gb', 0)
        disk_percent = health.get('disk_percent_used', 0)
        html += f"""
                    <div class="metric-card">
                        <div class="metric-label">Free Disk Space</div>
                        <div class="metric-value">{disk_free} GB</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Disk Usage</div>
                        <div class="metric-value">{disk_percent}%</div>
                    </div>
        """
        
        # CPU
        cpu_percent = health.get('cpu_percent', 0)
        html += f"""
                    <div class="metric-card">
                        <div class="metric-label">CPU Usage</div>
                        <div class="metric-value">{cpu_percent}%</div>
                    </div>
        """
    
    html += """
                </div>
            </div>
    """
    
    return html

def generate_test_results_section(tests: Dict[str, Any]) -> str:
    """Generate test results section."""
    html = """
            <div class="section">
                <h2>Test Results</h2>
    """
    
    for test_category, test_results in tests.items():
        html += f"""
                <h3>{test_category.replace('_', ' ').title()}</h3>
                <div class="test-results">
        """
        
        if isinstance(test_results, dict) and 'error' in test_results:
            html += f"""
                    <div class="error-details">
                        {test_results['error']}
                    </div>
            """
        else:
            for test_name, test_result in test_results.items():
                if isinstance(test_result, bool):
                    status_class = 'test-passed' if test_result else 'test-failed'
                    status_text = 'PASSED' if test_result else 'FAILED'
                    html += f"""
                        <div class="test-item {status_class}">
                            <span>{test_name.replace('_', ' ').title()}</span>
                            <span class="test-status">{status_text}</span>
                        </div>
                    """
                elif isinstance(test_result, dict):
                    # Handle nested test results
                    for nested_name, nested_result in test_result.items():
                        if isinstance(nested_result, bool):
                            status_class = 'test-passed' if nested_result else 'test-failed'
                            status_text = 'PASSED' if nested_result else 'FAILED'
                            html += f"""
                                <div class="test-item {status_class}">
                                    <span>{nested_name.replace('_', ' ').title()}</span>
                                    <span class="test-status">{status_text}</span>
                                </div>
                            """
        
        html += """
                </div>
        """
    
    html += """
            </div>
    """
    
    return html

def generate_performance_section(performance: Dict[str, Any]) -> str:
    """Generate performance section."""
    html = """
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="performance-chart">
    """
    
    if 'error' in performance:
        html += f"""
                    <div class="error-details">
                        Performance Benchmark Error: {performance['error']}
                    </div>
        """
    else:
        startup_time = performance.get('startup_time_seconds', 0)
        db_time = performance.get('database_operations_seconds', 0)
        ai_time = performance.get('ai_operations_seconds', 0)
        memory_usage = performance.get('memory_usage_mb', 0)
        
        html += f"""
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-label">Startup Time</div>
                            <div class="metric-value">{startup_time}s</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Database Operations</div>
                            <div class="metric-value">{db_time}s</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">AI Operations</div>
                            <div class="metric-value">{ai_time}s</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Memory Usage</div>
                            <div class="metric-value">{memory_usage} MB</div>
                        </div>
                    </div>
        """
    
    html += """
                </div>
            </div>
    """
    
    return html

def generate_regressions_section(regressions: Dict[str, Any]) -> str:
    """Generate regressions section."""
    html = """
            <div class="section">
                <h2>Regression Detection</h2>
    """
    
    if 'error' in regressions:
        html += f"""
                <div class="error-details">
                    Regression Detection Error: {regressions['error']}
                </div>
        """
    else:
        missing_files = regressions.get('missing_files', [])
        import_errors = regressions.get('import_errors', [])
        config_issues = regressions.get('config_issues', [])
        
        if missing_files or import_errors or config_issues:
            html += """
                <div class="regression-alert">
                    ⚠️ Potential regressions detected!
                </div>
            """
            
            if missing_files:
                html += """
                    <h3>Missing Files</h3>
                    <ul>
                """
                for file_path in missing_files:
                    html += f"<li>{file_path}</li>"
                html += "</ul>"
            
            if import_errors:
                html += """
                    <h3>Import Errors</h3>
                    <ul>
                """
                for error in import_errors:
                    html += f"<li>{error}</li>"
                html += "</ul>"
            
            if config_issues:
                html += """
                    <h3>Configuration Issues</h3>
                    <ul>
                """
                for issue in config_issues:
                    html += f"<li>{issue}</li>"
                html += "</ul>"
        else:
            html += """
                <div class="test-item test-passed">
                    <span>No regressions detected</span>
                    <span class="test-status">PASSED</span>
                </div>
            """
    
    html += """
            </div>
    """
    
    return html

def generate_error_report(error_message: str) -> str:
    """Generate error report when validation fails."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Trading System Validation Report - Error</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: #dc3545;
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .error-content {{
                padding: 30px;
                text-align: center;
            }}
            .error-details {{
                background: #f8d7da;
                color: #721c24;
                padding: 20px;
                border-radius: 4px;
                margin: 20px 0;
                font-family: monospace;
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>❌ Validation Report Error</h1>
            </div>
            <div class="error-content">
                <h2>Failed to generate validation report</h2>
                <div class="error-details">{error_message}</div>
                <p>Please check the CI validation results and try again.</p>
            </div>
        </div>
    </body>
    </html>
    """

def main():
    """Generate and save validation report."""
    print("Generating validation report...")
    
    try:
        html_content = generate_validation_report()
        
        # Save to file
        report_file = 'validation-report.html'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Validation report generated: {report_file}")
        
    except Exception as e:
        print(f"Failed to generate validation report: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
