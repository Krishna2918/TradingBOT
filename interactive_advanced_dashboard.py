#!/usr/bin/env python3
"""
Advanced AI Trading Dashboard with Comprehensive Logging
"""
import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import threading
import time
import asyncio
import logging
import os
import sys
import json
import psutil
import io

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import components
try:
    from src.dashboard.clean_state_manager import state_manager
    from src.dashboard.maximum_power_ai_engine import max_power_ai_engine
    from src.dashboard.advanced_ai_logger import advanced_ai_logger
    from src.dashboard.services import get_demo_price, is_market_open, get_random_tsx_stock
    from src.config.mode_manager import get_mode_manager
    from src.monitoring.system_monitor import SystemMonitor
    
    # Initialize components
    mode_manager = get_mode_manager()
    system_monitor = SystemMonitor()
    
    # Initialize MAXIMUM POWER AI Engine
    if max_power_ai_engine.initialize():
        logger.info("🚀 MAXIMUM POWER AI Engine ready!")
    else:
        logger.error("❌ MAXIMUM POWER AI Engine initialization failed")
    
    MAXIMUM_POWER_AVAILABLE = True
    logger.info("🚀 Connected to ADVANCED AI trading system!")
except Exception as e:
    logger.error(f"❌ Component initialization failed: {e}")
    MAXIMUM_POWER_AVAILABLE = False

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Advanced AI Trading Dashboard"

# Custom CSS for dark theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            :root {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --bg-tertiary: #3d3d3d;
                --text-primary: #ffffff;
                --text-secondary: #b0b0b0;
                --accent-primary: #00d4aa;
                --accent-secondary: #ff6b6b;
                --border-color: #404040;
            }
            
            body {
                background-color: var(--bg-primary);
                color: var(--text-primary);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .card {
                background-color: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
            }
            
            .card-header {
                background-color: var(--bg-tertiary);
                border-bottom: 1px solid var(--border-color);
                color: var(--text-primary);
            }
            
            .modal-content {
                background-color: var(--bg-secondary);
                color: var(--text-primary);
            }
            
            .modal-header {
                background-color: var(--bg-tertiary);
                border-bottom: 1px solid var(--border-color);
            }
            
            .modal-title {
                color: var(--text-primary);
            }
            
            .modal-body {
                color: var(--text-primary);
            }
            
            .log-entry {
                background-color: var(--bg-tertiary);
                border-left: 3px solid var(--accent-primary);
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
                color: var(--text-primary);
            }
            
            .log-timestamp {
                color: var(--text-secondary);
                font-size: 0.8em;
            }
            
            .ai-status-active {
                color: var(--accent-primary);
                font-weight: bold;
            }
            
            .ai-status-idle {
                color: var(--text-secondary);
            }
            
            .ai-status-error {
                color: var(--accent-secondary);
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def create_ai_status_cards():
    """Create AI component status cards"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🧠 AI Components Status"),
                dbc.CardBody([
                    html.Div(id="ai-components-status", children="Loading...")
                ])
            ])
        ], width=12)
    ])

def create_advanced_logs_section():
    """Create advanced logs section"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Advanced AI Logs & Analytics"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            html.Div(id="ai-activity-logs", children="Loading activity logs...")
                        ], label="🔍 Activity Logs", tab_id="activity"),
                        dbc.Tab([
                            html.Div(id="trading-decisions", children="Loading trading decisions...")
                        ], label="💰 Trading Decisions", tab_id="decisions"),
                        dbc.Tab([
                            html.Div(id="performance-metrics", children="Loading performance metrics...")
                        ], label="📈 Performance", tab_id="performance"),
                        dbc.Tab([
                            html.Div(id="system-status", children="Loading system status...")
                        ], label="⚙️ System Status", tab_id="system")
                    ], id="advanced-logs-tabs", active_tab="activity")
                ])
            ])
        ], width=12)
    ])

def create_main_layout():
    """Create main dashboard layout"""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🚀 Advanced AI Trading Dashboard", className="text-center mb-4"),
                html.P("Real-time AI trading with comprehensive logging and analytics", 
                      className="text-center text-muted mb-4")
            ])
        ]),
        
        # Control Panel
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎮 Control Panel"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Starting Capital ($)", className="form-label"),
                                dbc.Input(
                                    id="capital-input",
                                    type="number",
                                    value=10000,
                                    min=1,
                                    step=1,
                                    placeholder="Enter capital for AI trading"
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Trading Mode", className="form-label"),
                                dbc.Select(
                                    id="mode-select",
                                    options=[
                                        {"label": "DEMO", "value": "DEMO"},
                                        {"label": "LIVE", "value": "LIVE"}
                                    ],
                                    value="DEMO"
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Actions", className="form-label"),
                                dbc.ButtonGroup([
                                    dbc.Button("🚀 Start Advanced AI", id="start-advanced-btn", 
                                              color="success", className="me-2"),
                                    dbc.Button("⏹️ Stop AI", id="stop-ai-btn", 
                                              color="danger")
                                ], vertical=False)
                            ], width=4)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # AI Status Cards
        create_ai_status_cards(),
        
        # Advanced Logs Section
        create_advanced_logs_section(),
        
        # Hidden divs for state
        dcc.Store(id="ai-trading-active", data=False),
        dcc.Store(id="current-mode", data="DEMO"),
        
        # Auto-refresh interval
        dcc.Interval(
            id='interval-component',
            interval=2*1000,  # 2 seconds
            n_intervals=0
        )
    ], fluid=True)

# Set layout
app.layout = create_main_layout()

# Callbacks
@app.callback(
    [Output('ai-components-status', 'children'),
     Output('ai-activity-logs', 'children'),
     Output('trading-decisions', 'children'),
     Output('performance-metrics', 'children'),
     Output('system-status', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_advanced_dashboard(n):
    """Update all dashboard components"""
    try:
        # Get AI component status
        component_status = advanced_ai_logger.get_component_status()
        ai_status_cards = []
        
        for name, status in component_status.items():
            status_color = "success" if status['status'] == 'active' else "secondary"
            status_icon = "🟢" if status['status'] == 'active' else "⚪"
            
            ai_status_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(f"{status_icon} {name}", className="card-title"),
                            html.P(f"Status: {status['status'].title()}", className="card-text"),
                            html.Small(f"Task: {status['current_task']}", className="text-muted")
                        ])
                    ], color=status_color, outline=True)
                ], width=2, className="mb-2")
            )
        
        # Get recent activity logs
        recent_activity = advanced_ai_logger.get_recent_activity(20)
        activity_logs = []
        
        for log in recent_activity[-10:]:  # Show last 10
            activity_logs.append(
                html.Div([
                    html.Span(f"[{log['timestamp'][:19]}] ", className="log-timestamp"),
                    html.Strong(f"{log['component']}: "),
                    html.Span(f"{log['activity_type']} - {log['details'].get('task', 'Processing')}")
                ], className="log-entry")
            )
        
        # Get recent trading decisions
        recent_decisions = advanced_ai_logger.get_recent_decisions(10)
        decision_logs = []
        
        for decision in recent_decisions[-5:]:  # Show last 5
            action_color = "success" if decision['action'] == 'BUY' else "danger" if decision['action'] == 'SELL' else "warning"
            decision_logs.append(
                dbc.Card([
                    dbc.CardBody([
                        html.H6(f"{decision['action']} {decision['symbol']}", className="card-title"),
                        html.P(f"Confidence: {decision['confidence']:.2%}", className="card-text"),
                        html.P(f"Reasoning: {decision['reasoning']}", className="card-text"),
                        html.Small(f"Models: {', '.join(decision['ai_models_used'])}", className="text-muted")
                    ])
                ], color=action_color, outline=True, className="mb-2")
            )
        
        # Get performance summary
        performance_summary = advanced_ai_logger.get_performance_summary()
        performance_metrics = [
            html.Div([
                html.H6("📊 Performance Summary"),
                html.P(f"Total Decisions: {performance_summary.get('total_decisions', 0)}"),
                html.P(f"Average Confidence: {performance_summary.get('avg_decision_confidence', 0):.2%}"),
                html.P(f"Success Rate: {performance_summary.get('success_rate', 0):.2%}"),
                html.P(f"Active Components: {performance_summary.get('active_components', 0)}")
            ])
        ]
        
        # System status
        system_status = [
            html.Div([
                html.H6("⚙️ System Status"),
                html.P(f"CPU Usage: {psutil.cpu_percent():.1f}%"),
                html.P(f"Memory Usage: {psutil.virtual_memory().percent:.1f}%"),
                html.P(f"AI Engine: {'Active' if MAXIMUM_POWER_AVAILABLE else 'Inactive'}"),
                html.P(f"Mode: {mode_manager.get_current_mode()}")
            ])
        ]
        
        return (
            dbc.Row(ai_status_cards),
            activity_logs,
            decision_logs,
            performance_metrics,
            system_status
        )
        
    except Exception as e:
        logger.error(f"Dashboard update error: {e}")
        return "Error loading data", "Error loading logs", "Error loading decisions", "Error loading metrics", "Error loading status"

@app.callback(
    [Output('ai-trading-active', 'data')],
    [Input('start-advanced-btn', 'n_clicks'),
     Input('stop-ai-btn', 'n_clicks')],
    [State('capital-input', 'value'),
     State('mode-select', 'value')]
)
def start_stop_advanced_trading(start_clicks, stop_clicks, capital, mode):
    """Start or stop advanced AI trading"""
    if not ctx.triggered:
        return [False]
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-advanced-btn' and start_clicks:
        try:
            # Clear logs for fresh session
            advanced_ai_logger.clear_logs()
            
            # Start new session with capital and mode
            session_id = state_manager.start_new_session(capital or 10000, mode or 'DEMO')
            
            # Start AI trading in background thread
            def run_ai_trading():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(max_power_ai_engine.execute_maximum_power_cycle())
                except Exception as e:
                    logger.error(f"AI trading error: {e}")
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_ai_trading, daemon=True)
            thread.start()
            
            logger.info("🚀 Advanced AI Trading started!")
            return [True]
            
        except Exception as e:
            logger.error(f"Failed to start AI trading: {e}")
            return [False]
    
    elif button_id == 'stop-ai-btn' and stop_clicks:
        try:
            # Stop AI trading by ending the session
            if state_manager.current_session:
                state_manager.current_session.is_active = False
                state_manager.save()
            logger.info("⏹️ AI Trading stopped!")
            return [False]
        except Exception as e:
            logger.error(f"Failed to stop AI trading: {e}")
            return [False]
    
    return [False]

if __name__ == '__main__':
    # Fix Unicode encoding for Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print("=" * 80)
    print("ADVANCED AI Trading Dashboard Starting...")
    print("=" * 80)
    print("Features:")
    print("   - Advanced AI Engine with comprehensive logging")
    print("   - Real-time component status tracking")
    print("   - Detailed trading decision analytics")
    print("   - Performance metrics and system monitoring")
    print("   - Organized activity logs and error tracking")
    print("Dashboard URL: http://localhost:8058")
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    
    app.run_server(debug=True, host='127.0.0.1', port=8058)
