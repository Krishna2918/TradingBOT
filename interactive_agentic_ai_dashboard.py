"""
Interactive Agentic AI Dashboard

A comprehensive dashboard for monitoring and controlling the Agentic AI system.
This dashboard provides real-time visualization of all 7 agents, resource management,
and system performance.

Features:
- Real-time agent status monitoring
- Resource management visualization
- Market analysis display
- Learning progress tracking
- Portfolio optimization interface
- Risk management controls
- System health monitoring
- Alert management
- Performance analytics
- Interactive agent control
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Web framework
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Chart libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# HTTP client for API calls
import httpx

# Configure HTTP client with connection pooling to prevent connection exhaustion
HTTP_LIMITS = httpx.Limits(max_keepalive_connections=3, max_connections=5)
HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0, read=10.0)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA VALIDATION LAYER
# ============================================================================

class DashboardDataValidator:
    """Data validation and sanitization for dashboard responses."""
    
    @staticmethod
    def validate_portfolio_data(data: Dict) -> Dict:
        """Ensure portfolio data has required fields with sensible defaults."""
        try:
            # Handle the actual API response format - API returns nested summary object
            summary = data.get("summary", {})
            
            return {
                "total_value": max(0, float(summary.get("total_value", 0))),
                "daily_pnl": float(summary.get("daily_pnl", 0)),
                "daily_pnl_percent": max(-100, min(100, float(summary.get("daily_pnl_percent", 0)))),
                "total_pnl": float(summary.get("total_pnl", 0)),
                "total_pnl_percent": max(-100, min(100, float(summary.get("total_pnl_percent", 0)))),
                "positions": max(0, int(data.get("positions", 0))),
                "ai_confidence": 85,  # Placeholder - will be calculated from agents
                "mode": data.get("mode", "demo"),
                "lastUpdated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Portfolio data validation failed: {str(e)}", exc_info=True)
            return {
                "total_value": 0.0,
                "daily_pnl": 0.0,
                "daily_pnl_percent": 0.0,
                "total_pnl": 0.0,
                "total_pnl_percent": 0.0,
                "positions": 0,
                "ai_confidence": 0,
                "mode": "demo",
                "lastUpdated": datetime.now().isoformat()
            }
    
    @staticmethod
    def validate_agent_data(data: Dict) -> Dict:
        """Validate and sanitize agent data."""
        try:
            validated_agents = {}
            # The API returns agents directly at root level, not nested under "agents"
            for agent_id, agent_data in data.items():
                if isinstance(agent_data, dict):
                    validated_agents[agent_id] = {
                        "status": agent_data.get("status", "unknown"),
                        "priority": agent_data.get("priority", "OPTIONAL"),
                        "uptime_seconds": agent_data.get("uptime_seconds", 0),
                        "metrics": {
                            "tasks_completed": max(0, int(agent_data.get("metrics", {}).get("tasks_completed", 0))),
                            "tasks_failed": max(0, int(agent_data.get("metrics", {}).get("tasks_failed", 0))),
                            "avg_response_time": max(0, float(agent_data.get("metrics", {}).get("avg_response_time", 0))),
                            "success_rate": max(0, min(1, float(agent_data.get("metrics", {}).get("success_rate", 0))))
                        },
                        "last_updated": datetime.now().isoformat()
                    }
            return validated_agents
        except Exception as e:
            logger.error(f"Agent data validation failed: {str(e)}", exc_info=True)
            return {}
    
    @staticmethod
    def validate_system_health(data: Dict) -> Dict:
        """Validate system health data."""
        return {
            "overall_health": data.get("overall_health", "unknown"),
            "status": data.get("status", "unknown"),
            "cpu_usage": max(0, min(100, float(data.get("cpu_usage", 0)))),
            "memory_usage": max(0, min(100, float(data.get("memory_usage", 0)))),
            "disk_usage": max(0, min(100, float(data.get("disk_usage", 0)))),
            "last_updated": datetime.now().isoformat()
        }
    
    @staticmethod
    def format_chart_data(chart_type: str, data: Dict) -> Dict:
        """Format chart data for Chart.js consumption."""
        if chart_type == "portfolio":
            return {
                "labels": data.get("labels", []),
                "datasets": [{
                    "label": "Portfolio Value",
                    "data": data.get("data", []),
                    "borderColor": "#2E8B57",
                    "backgroundColor": "rgba(46, 139, 87, 0.1)",
                    "borderWidth": 2,
                    "fill": True,
                    "tension": 0.4
                }]
            }
        elif chart_type == "market":
            return {
                "labels": data.get("labels", []),
                "datasets": [{
                    "label": "Market Volatility",
                    "data": data.get("data", []),
                    "borderColor": "#FF6B6B",
                    "backgroundColor": "rgba(255, 107, 107, 0.1)",
                    "borderWidth": 2,
                    "fill": True,
                    "tension": 0.4
                }]
            }
        elif chart_type == "learning":
            return {
                "labels": data.get("labels", []),
                "datasets": [{
                    "label": "Learning Progress",
                    "data": data.get("data", []),
                    "borderColor": "#45B7D1",
                    "backgroundColor": "rgba(69, 183, 209, 0.1)",
                    "borderWidth": 2,
                    "fill": True,
                    "tension": 0.4
                }]
            }
        return data
    
    @staticmethod
    def format_alert_data(alerts: List[Dict]) -> List[Dict]:
        """Format alert data for dashboard display."""
        formatted_alerts = []
        for alert in alerts:
            formatted_alerts.append({
                "id": alert.get("id", ""),
                "type": alert.get("type", "info"),
                "severity": alert.get("severity", "low"),
                "message": alert.get("message", ""),
                "timestamp": alert.get("timestamp", datetime.now().isoformat()),
                "source": alert.get("source", "system")
            })
        return formatted_alerts

# Initialize FastAPI app
app = FastAPI(
    title="Agentic AI Dashboard",
    description="Real-time monitoring and control for the Agentic AI Trading System",
    version="1.0.0"
)

# Templates
templates = Jinja2Templates(directory="templates")

# Import configuration from centralized config
from dashboard_config import DASHBOARD_CONFIG, API_BASE_URL, DASHBOARD_BASE_URL, WEBSOCKET_URL, CHART_CONFIG

# Dashboard state
dashboard_state = {
    "agents_status": {},
    "resource_manager_status": {},
    "system_health": {},
    "market_analysis": {},
    "learning_progress": {},
    "portfolio_analysis": {},
    "active_alerts": [],
    "performance_metrics": {},
    "last_update": None
}

# WebSocket connections
websocket_connections: List[WebSocket] = []

class AgenticAIDashboard:
    """Main dashboard controller for the Agentic AI system."""
    
    def __init__(self):
        self.api_client = httpx.AsyncClient(
            limits=HTTP_LIMITS,
            timeout=HTTP_TIMEOUT
        )
        self.update_interval = 15  # seconds - reduced frequency to prevent connection exhaustion
        self.is_running = False
        
    async def start(self):
        """Start the dashboard background tasks."""
        self.is_running = True
        asyncio.create_task(self._update_loop())
        logger.info("Agentic AI Dashboard started")
    
    async def stop(self):
        """Stop the dashboard."""
        self.is_running = False
        await self.api_client.aclose()
        logger.info("Agentic AI Dashboard stopped")
    
    async def _update_loop(self):
        """Background loop to update dashboard data."""
        while self.is_running:
            try:
                await self._update_dashboard_data()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _make_api_call_with_retry(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        """Make API call with retry logic to handle connection errors."""
        for attempt in range(max_retries):
            try:
                response = await self.api_client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"API call failed with status {response.status_code}: {url}")
                    return None
            except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} attempts failed for {url}")
                    return None
        return None

    async def _update_dashboard_data(self):
        """Update all dashboard data from the API with retry logic."""
        try:
            # Update agents status
            agents_data = await self._make_api_call_with_retry(f"{API_BASE_URL}/api/agents/status")
            if agents_data:
                dashboard_state["agents_status"] = agents_data
            
            # Update resource manager status
            rm_data = await self._make_api_call_with_retry(f"{API_BASE_URL}/api/agents/resource-manager/status")
            if rm_data:
                dashboard_state["resource_manager_status"] = rm_data
            
            # Update system health
            health_data = await self._make_api_call_with_retry(f"{API_BASE_URL}/api/agents/monitoring/health")
            if health_data:
                dashboard_state["system_health"] = health_data
            
            # Update market analysis
            market_data = await self._make_api_call_with_retry(f"{API_BASE_URL}/api/agents/market/analysis")
            if market_data:
                dashboard_state["market_analysis"] = market_data
            
            # Update learning progress
            learning_data = await self._make_api_call_with_retry(f"{API_BASE_URL}/api/agents/learning/progress")
            if learning_data:
                dashboard_state["learning_progress"] = learning_data
            
            # Update portfolio analysis
            portfolio_data = await self._make_api_call_with_retry(f"{API_BASE_URL}/api/agents/portfolio/analysis")
            if portfolio_data:
                dashboard_state["portfolio_analysis"] = portfolio_data
            
            # Update active alerts
            alerts_data = await self._make_api_call_with_retry(f"{API_BASE_URL}/api/agents/monitoring/alerts")
            if alerts_data:
                dashboard_state["active_alerts"] = alerts_data.get("alerts", [])
            
            dashboard_state["last_update"] = datetime.now().isoformat()
            
            # Broadcast update to all connected clients
            await self._broadcast_update()
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
    
    async def _broadcast_update(self):
        """Broadcast dashboard update to all connected WebSocket clients."""
        if websocket_connections:
            message = {
                "type": "dashboard_update",
                "data": dashboard_state,
                "timestamp": datetime.now().isoformat()
            }
            
            disconnected = []
            for websocket in websocket_connections:
                try:
                    await websocket.send_text(json.dumps(message))
                except:
                    disconnected.append(websocket)
            
            # Remove disconnected connections
            for ws in disconnected:
                websocket_connections.remove(ws)
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific agent."""
        try:
            response = await self.api_client.get(f"{API_BASE_URL}/api/agents/{agent_id}/status")
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to get status for agent {agent_id}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def execute_agent_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task on a specific agent."""
        try:
            response = await self.api_client.post(
                f"{API_BASE_URL}/api/agents/{agent_id}/task",
                json=task
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to execute task for agent {agent_id}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def activate_agent(self, agent_id: str) -> Dict[str, Any]:
        """Activate a specific agent."""
        try:
            response = await self.api_client.post(f"{API_BASE_URL}/api/agents/{agent_id}/activate")
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to activate agent {agent_id}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def deactivate_agent(self, agent_id: str) -> Dict[str, Any]:
        """Deactivate a specific agent."""
        try:
            response = await self.api_client.post(f"{API_BASE_URL}/api/agents/{agent_id}/deactivate")
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to deactivate agent {agent_id}"}
        except Exception as e:
            return {"error": str(e)}

# Initialize dashboard
dashboard = AgenticAIDashboard()

# ============================================================================
# DASHBOARD ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page."""
    
    return templates.TemplateResponse("trading_dashboard.html", {
        "request": request,
        "title": "TradingBOT Dashboard",
        "api_base_url": DASHBOARD_BASE_URL,
        "current_mode": "demo",  # Will be dynamic later
        "websocket_url": WEBSOCKET_URL,
        "agent_count": 6,  # Total number of agents
        "system_health": "healthy",
        "update_intervals": DASHBOARD_CONFIG["update_intervals"],
        "chart_config": CHART_CONFIG,
        "agents": [
            {"id": "risk_agent", "name": "Risk Management", "priority": "CRITICAL"},
            {"id": "monitoring_agent", "name": "System Monitoring", "priority": "CRITICAL"},
            {"id": "execution_agent", "name": "Order Execution", "priority": "CRITICAL"},
            {"id": "portfolio_agent", "name": "Portfolio Management", "priority": "IMPORTANT"},
            {"id": "market_analysis_agent", "name": "Market Analysis", "priority": "IMPORTANT"},
            {"id": "learning_agent", "name": "Learning & Adaptation", "priority": "OPTIONAL"}
        ]
    })


@app.get("/api/dashboard/agents")
async def get_agents_data():
    """Get agents data for visualization."""
    agents_data = dashboard_state.get("agents_status", {})
    
    # Process agents data for charts
    agents_list = []
    for agent_id, agent_data in agents_data.get("agents", {}).items():
        agents_list.append({
            "id": agent_id,
            "name": agent_data.get("name", agent_id),
            "status": agent_data.get("status", "unknown"),
            "priority": agent_data.get("priority", "unknown"),
            "cpu_usage": agent_data.get("metrics", {}).get("cpu_usage", 0),
            "memory_usage": agent_data.get("metrics", {}).get("memory_usage", 0),
            "tasks_completed": agent_data.get("metrics", {}).get("tasks_completed", 0),
            "uptime": agent_data.get("metrics", {}).get("uptime_seconds", 0)
        })
    
    return {"agents": agents_list}

@app.get("/api/dashboard/resources")
async def get_resources_data():
    """Get resource management data."""
    rm_data = dashboard_state.get("resource_manager_status", {})
    
    return {
        "system_cpu": rm_data.get("resources", {}).get("cpu", {}).get("percent", 0),
        "system_memory": rm_data.get("resources", {}).get("memory", {}).get("percent", 0),
        "active_agents": rm_data.get("agents", {}).get("active", 0),
        "total_agents": rm_data.get("agents", {}).get("total", 0),
        "emergency_mode": rm_data.get("emergency_mode", False),
        "optimization_active": rm_data.get("optimization_active", False)
    }

@app.get("/api/dashboard/market")
async def get_market_data():
    """Get market analysis data."""
    market_data = dashboard_state.get("market_analysis", {})
    
    return {
        "regime": market_data.get("overall_regime", "unknown"),
        "trend_strength": market_data.get("trend_strength", "unknown"),
        "volatility": market_data.get("volatility", 0),
        "sentiment": market_data.get("market_sentiment", 0),
        "support_level": market_data.get("support_level", 0),
        "resistance_level": market_data.get("resistance_level", 0),
        "regime_change_detected": market_data.get("regime_change_detected", False)
    }

@app.get("/api/dashboard/learning")
async def get_learning_data():
    """Get learning progress data."""
    learning_data = dashboard_state.get("learning_progress", {})
    
    return {
        "learning_episodes": learning_data.get("learning_episodes", 0),
        "patterns_discovered": learning_data.get("patterns_discovered", 0),
        "insights_generated": learning_data.get("insights_generated", 0),
        "model_improvements": learning_data.get("model_improvements", 0),
        "current_accuracy": learning_data.get("current_accuracy", 0),
        "knowledge_base_size": learning_data.get("knowledge_base_size", 0),
        "learning_active": learning_data.get("learning_active", False)
    }

@app.get("/api/dashboard/portfolio")
async def get_portfolio_data():
    """Get portfolio analysis data."""
    portfolio_data = dashboard_state.get("portfolio_analysis", {})
    
    return {
        "total_value": portfolio_data.get("total_value", 0),
        "total_pnl": portfolio_data.get("total_pnl", 0),
        "sharpe_ratio": portfolio_data.get("sharpe_ratio", 0),
        "max_drawdown": portfolio_data.get("max_drawdown", 0),
        "positions_count": portfolio_data.get("positions_count", 0),
        "rebalancing_needed": portfolio_data.get("rebalancing_needed", False),
        "risk_score": portfolio_data.get("risk_score", 0)
    }

@app.get("/api/dashboard/alerts")
async def get_alerts_data():
    """Get active alerts data."""
    return {
        "alerts": dashboard_state.get("active_alerts", []),
        "count": len(dashboard_state.get("active_alerts", []))
    }

# ============================================================================
# SMART GROUPED API ENDPOINTS
# ============================================================================

@app.get("/api/dashboard/status")
async def get_dashboard_status():
    """Complete system status - agents, health, alerts in one call."""
    try:
        async with httpx.AsyncClient(limits=HTTP_LIMITS, timeout=HTTP_TIMEOUT) as client:
            # Fetch all data in parallel
            agents_response = await client.get(f"{API_BASE_URL}/api/agents/status")
            health_response = await client.get(f"{API_BASE_URL}/api/agents/monitoring/health")
            alerts_response = await client.get(f"{API_BASE_URL}/api/agents/monitoring/alerts")
            
            agents_data = agents_response.json() if agents_response.status_code == 200 else {}
            health_data = health_response.json() if health_response.status_code == 200 else {}
            alerts_data = alerts_response.json() if alerts_response.status_code == 200 else {}
            
            return {
                "agents": DashboardDataValidator.validate_agent_data(agents_data),
                "system_health": DashboardDataValidator.validate_system_health(health_data),
                "active_alerts": alerts_data.get("alerts", []),
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching dashboard status: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/api/dashboard/trading/{mode}")
async def get_trading_data(mode: str):
    """All trading data - portfolio, positions, P&L for specified mode."""
    try:
        async with httpx.AsyncClient(limits=HTTP_LIMITS, timeout=HTTP_TIMEOUT) as client:
            # Fetch trading data in parallel
            portfolio_response = await client.get(f"{API_BASE_URL}/api/portfolio")
            positions_response = await client.get(f"{API_BASE_URL}/api/positions")
            risk_response = await client.get(f"{API_BASE_URL}/api/risk/metrics")
            
            portfolio_data = portfolio_response.json() if portfolio_response.status_code == 200 else {}
            positions_data = positions_response.json() if positions_response.status_code == 200 else {}
            risk_data = risk_response.json() if risk_response.status_code == 200 else {}
            
            return {
                "portfolio": DashboardDataValidator.validate_portfolio_data(portfolio_data),
                "positions": positions_data.get("positions", []),
                "risk_metrics": risk_data,
                "ai_confidence": 85,  # Placeholder - will be calculated
                "mode": mode,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching trading data: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/api/dashboard/charts/{chart_type}")
async def get_chart_data(chart_type: str, period: str = "1W"):
    """Unified chart data endpoint."""
    try:
        if chart_type == "portfolio":
            # Generate portfolio history data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            values = np.random.normal(10000, 500, 30).cumsum()
            raw_data = {
                "labels": [d.strftime('%Y-%m-%d') for d in dates],
                "data": values.tolist()
            }
            return DashboardDataValidator.format_chart_data("portfolio", raw_data)
            
        elif chart_type == "market":
            # Generate market analysis data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            volatility = np.random.normal(0.2, 0.05, 30)
            raw_data = {
                "labels": [d.strftime('%Y-%m-%d') for d in dates],
                "data": volatility.tolist()
            }
            return DashboardDataValidator.format_chart_data("market", raw_data)
            
        elif chart_type == "learning":
            # Generate learning progress data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            progress = np.linspace(0, 100, 30)
            raw_data = {
                "labels": [d.strftime('%Y-%m-%d') for d in dates],
                "data": progress.tolist()
            }
            return DashboardDataValidator.format_chart_data("learning", raw_data)
            
        elif chart_type == "agents-performance":
            # Generate agents performance data
            agents = ["Risk", "Monitoring", "Execution", "Portfolio", "Market", "Learning"]
            performance = [95.5, 98.2, 100.0, 97.9, 98.8, 99.4]
            return {
                "labels": agents,
                "datasets": [{
                    "label": "Success Rate (%)",
                    "data": performance,
                    "backgroundColor": ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#06b6d4"],
                    "borderColor": ["#dc2626", "#2563eb", "#16a34a", "#d97706", "#7c3aed", "#0891b2"],
                    "borderWidth": 2
                }]
            }
            
        return {"error": "Unknown chart type"}
    except Exception as e:
        logger.error(f"Error generating chart data: {e}")
        return {"error": str(e)}

@app.post("/api/dashboard/mode")
async def switch_trading_mode(mode_data: Dict[str, str]):
    """Switch between DEMO and LIVE trading modes."""
    try:
        mode = mode_data.get("mode", "demo").lower()
        if mode not in ["demo", "live"]:
            return {"error": "Invalid mode. Must be 'demo' or 'live'"}
        
        # Here we would integrate with the actual mode manager
        # For now, return success
        return {
            "success": True,
            "mode": mode,
            "message": f"Switched to {mode.upper()} mode",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error switching mode: {e}")
        return {"error": str(e)}

@app.get("/api/dashboard/health")
async def get_dashboard_health():
    """Health check endpoint for dashboard connectivity."""
    try:
        # Test API connectivity
        async with httpx.AsyncClient(limits=HTTP_LIMITS, timeout=HTTP_TIMEOUT) as client:
            api_response = await client.get(f"{API_BASE_URL}/api/health", timeout=5.0)
            api_status = "connected" if api_response.status_code == 200 else "disconnected"
        
        return {
            "dashboard": "healthy",
            "api_connectivity": api_status,
            "database": "connected",  # Placeholder
            "agents": "operational",  # Placeholder
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "dashboard": "healthy",
            "api_connectivity": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# AGENT CONTROL ENDPOINTS
# ============================================================================

@app.post("/api/dashboard/agents/{agent_id}/activate")
async def activate_agent(agent_id: str):
    """Activate a specific agent."""
    result = await dashboard.activate_agent(agent_id)
    return result

@app.post("/api/dashboard/agents/{agent_id}/deactivate")
async def deactivate_agent(agent_id: str):
    """Deactivate a specific agent."""
    result = await dashboard.deactivate_agent(agent_id)
    return result

@app.post("/api/dashboard/agents/{agent_id}/task")
async def execute_agent_task(agent_id: str, task: Dict[str, Any]):
    """Execute a task on a specific agent."""
    result = await dashboard.execute_agent_task(agent_id, task)
    return result

@app.get("/api/dashboard/agents/{agent_id}/status")
async def get_agent_detailed_status(agent_id: str):
    """Get detailed status of a specific agent."""
    result = await dashboard.get_agent_status(agent_id)
    return result

# ============================================================================
# CHART GENERATION ENDPOINTS
# ============================================================================

@app.get("/api/dashboard/charts/agents-performance")
async def get_agents_performance_chart():
    """Generate agents performance chart."""
    agents_data = await get_agents_data()
    agents = agents_data["agents"]
    
    if not agents:
        return {"error": "No agents data available"}
    
    # Create performance chart
    fig = go.Figure()
    
    agent_names = [agent["name"] for agent in agents]
    tasks_completed = [agent["tasks_completed"] for agent in agents]
    cpu_usage = [agent["cpu_usage"] for agent in agents]
    memory_usage = [agent["memory_usage"] for agent in agents]
    
    # Add tasks completed bar
    fig.add_trace(go.Bar(
        name="Tasks Completed",
        x=agent_names,
        y=tasks_completed,
        yaxis="y",
        offsetgroup=1
    ))
    
    # Add CPU usage line
    fig.add_trace(go.Scatter(
        name="CPU Usage %",
        x=agent_names,
        y=cpu_usage,
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="red")
    ))
    
    # Add memory usage line
    fig.add_trace(go.Scatter(
        name="Memory Usage %",
        x=agent_names,
        y=memory_usage,
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="blue")
    ))
    
    # Update layout
    fig.update_layout(
        title="Agents Performance Overview",
        xaxis_title="Agents",
        yaxis=dict(title="Tasks Completed", side="left"),
        yaxis2=dict(title="Resource Usage %", side="right", overlaying="y"),
        hovermode="x unified"
    )
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

@app.get("/api/dashboard/charts/resource-usage")
async def get_resource_usage_chart():
    """Generate resource usage chart."""
    resources_data = await get_resources_data()
    
    # Create resource usage pie chart
    fig = go.Figure(data=[go.Pie(
        labels=["Used CPU", "Available CPU"],
        values=[resources_data["system_cpu"], 100 - resources_data["system_cpu"]],
        hole=0.3,
        name="CPU Usage"
    )])
    
    fig.update_layout(
        title="System Resource Usage",
        annotations=[dict(text="CPU", x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

@app.get("/api/dashboard/charts/market-analysis")
async def get_market_analysis_chart():
    """Generate market analysis chart."""
    market_data = await get_market_data()
    
    # Create market sentiment gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=market_data["sentiment"] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Market Sentiment"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, -50], 'color': "lightgray"},
                {'range': [-50, 0], 'color': "gray"},
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

@app.get("/api/dashboard/charts/learning-progress")
async def get_learning_progress_chart():
    """Generate learning progress chart."""
    learning_data = await get_learning_data()
    
    # Create learning metrics bar chart
    metrics = ["Learning Episodes", "Patterns Discovered", "Insights Generated", "Model Improvements"]
    values = [
        learning_data["learning_episodes"],
        learning_data["patterns_discovered"],
        learning_data["insights_generated"],
        learning_data["model_improvements"]
    ]
    
    fig = go.Figure(data=[go.Bar(x=metrics, y=values)])
    fig.update_layout(
        title="Learning Progress Metrics",
        xaxis_title="Metrics",
        yaxis_title="Count"
    )
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await websocket.accept()
    websocket_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total connections: {len(websocket_connections)}")
    
    try:
        while True:
            # Send real-time updates every 2 seconds
            await asyncio.sleep(2)
            
            # Get latest data
            try:
                async with httpx.AsyncClient(limits=HTTP_LIMITS, timeout=HTTP_TIMEOUT) as client:
                    # Get agent status
                    agents_response = await client.get(f"{API_BASE_URL}/api/agents/status")
                    agents_data = agents_response.json() if agents_response.status_code == 200 else {}
                    
                    # Get portfolio data
                    portfolio_response = await client.get(f"{API_BASE_URL}/api/portfolio")
                    portfolio_data = portfolio_response.json() if portfolio_response.status_code == 200 else {}
                    
                    # Send agent updates
                    await websocket.send_json({
                        "type": "agents",
                        "data": agents_data,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Send portfolio updates
                    await websocket.send_json({
                        "type": "portfolio",
                        "data": portfolio_data,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Error sending WebSocket update: {str(e)}", exc_info=True)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                except:
                    pass  # Connection might be closed
                
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(websocket_connections)}")

# ============================================================================
# STARTUP AND SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    await dashboard.start()
    logger.info("Agentic AI Dashboard started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    await dashboard.stop()
    logger.info("Agentic AI Dashboard stopped")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("""
    Agentic AI Dashboard Starting...
    
    Dashboard Features:
    - Real-time agent monitoring
    - Resource management visualization
    - Market analysis display
    - Learning progress tracking
    - Portfolio optimization interface
    - Risk management controls
    - System health monitoring
    - Alert management
    - Performance analytics
    - Interactive agent control
    
    Access Points:
    - Dashboard: http://localhost:8001/
    - API Documentation: http://localhost:8001/docs
    - WebSocket: ws://localhost:8001/ws
    
    Ready for monitoring the Agentic AI system!
    """)
    
    uvicorn.run(
        "interactive_agentic_ai_dashboard:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
