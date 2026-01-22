"""
Dashboard Configuration Management
Centralized configuration for the Trading Dashboard system.
"""

DASHBOARD_CONFIG = {
    "update_intervals": {
        "portfolio": 5000,    # 5 seconds
        "agents": 2000,       # 2 seconds  
        "system": 10000,      # 10 seconds
        "charts": 30000       # 30 seconds
    },
    "websocket": {
        "reconnect_interval": 3000,
        "max_reconnect_attempts": 5
    },
    "api": {
        "timeout": 10000,
        "retry_attempts": 3
    },
    "ui": {
        "default_mode": "demo",
        "chart_periods": ["1D", "1W", "1M", "3M", "1Y"],
        "max_chart_points": 100
    }
}

# API Base URLs
API_BASE_URL = "http://localhost:8000"
DASHBOARD_BASE_URL = "http://localhost:8001"
WEBSOCKET_URL = "ws://localhost:8001/ws/dashboard"

# Agent Configuration
AGENT_CONFIG = {
    "critical_agents": ["risk_agent", "monitoring_agent", "execution_agent"],
    "important_agents": ["portfolio_agent", "market_analysis_agent"],
    "optional_agents": ["learning_agent"]
}

# Chart Configuration
CHART_CONFIG = {
    "portfolio": {
        "colors": ["#2E8B57", "#FF6B6B", "#4ECDC4", "#45B7D1"],
        "background": "rgba(46, 139, 87, 0.1)"
    },
    "market": {
        "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
        "background": "rgba(255, 107, 107, 0.1)"
    },
    "learning": {
        "colors": ["#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"],
        "background": "rgba(69, 183, 209, 0.1)"
    }
}
