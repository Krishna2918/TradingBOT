"""
Test Script for Final Trading API

This script tests all major endpoints of the Final Trading API
to ensure everything is working correctly.
"""

import requests
import json
import time
import asyncio
import websockets
from datetime import datetime

# API Configuration
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

class APITester:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test(self, test_name, success, message="", response_data=None):
        """Log test results."""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "response": response_data
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        
        if response_data and not success:
            print(f"   Response: {json.dumps(response_data, indent=2)}")
    
    def test_system_status(self):
        """Test system status endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/api/status")
            if response.status_code == 200:
                data = response.json()
                self.log_test("System Status", True, f"Status: {data.get('status', 'Unknown')}", data)
                return True
            else:
                self.log_test("System Status", False, f"HTTP {response.status_code}", response.text)
                return False
        except Exception as e:
            self.log_test("System Status", False, str(e))
            return False
    
    def test_system_health(self):
        """Test system health endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                self.log_test("System Health", True, f"Health: {data.get('health', {}).get('status', 'Unknown')}", data)
                return True
            else:
                self.log_test("System Health", False, f"HTTP {response.status_code}", response.text)
                return False
        except Exception as e:
            self.log_test("System Health", False, str(e))
            return False
    
    def test_session_management(self):
        """Test session management endpoints."""
        try:
            # Start session
            response = self.session.post(
                f"{self.base_url}/api/session/start",
                json={"capital": 10000, "mode": "DEMO"}
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.log_test("Start Session", True, f"Session ID: {data.get('session_id', 'Unknown')}", data)
                    
                    # Get session status
                    response = self.session.get(f"{self.base_url}/api/session/status")
                    if response.status_code == 200:
                        status_data = response.json()
                        self.log_test("Get Session Status", True, "Session status retrieved", status_data)
                        
                        # Stop session
                        response = self.session.post(f"{self.base_url}/api/session/stop")
                        if response.status_code == 200:
                            stop_data = response.json()
                            self.log_test("Stop Session", True, "Session stopped", stop_data)
                            return True
                        else:
                            self.log_test("Stop Session", False, f"HTTP {response.status_code}", response.text)
                    else:
                        self.log_test("Get Session Status", False, f"HTTP {response.status_code}", response.text)
                else:
                    self.log_test("Start Session", False, data.get("error", "Unknown error"), data)
            else:
                self.log_test("Start Session", False, f"HTTP {response.status_code}", response.text)
            return False
        except Exception as e:
            self.log_test("Session Management", False, str(e))
            return False
    
    def test_ai_system(self):
        """Test AI system endpoints."""
        try:
            # Start session first
            self.session.post(
                f"{self.base_url}/api/session/start",
                json={"capital": 10000, "mode": "DEMO"}
            )
            
            # Start AI trading
            response = self.session.post(f"{self.base_url}/api/ai/start")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.log_test("Start AI Trading", True, "AI trading started", data)
                    
                    # Run AI analysis
                    response = self.session.post(
                        f"{self.base_url}/api/ai/analyze",
                        json={"symbol": "AAPL"}
                    )
                    if response.status_code == 200:
                        analysis_data = response.json()
                        if analysis_data.get("success"):
                            decision = analysis_data.get("decision", {})
                            self.log_test("AI Analysis", True, f"Decision: {decision.get('action', 'Unknown')}", analysis_data)
                        else:
                            self.log_test("AI Analysis", False, analysis_data.get("error", "Unknown error"), analysis_data)
                    
                    # Stop AI trading
                    response = self.session.post(f"{self.base_url}/api/ai/stop")
                    if response.status_code == 200:
                        stop_data = response.json()
                        self.log_test("Stop AI Trading", True, "AI trading stopped", stop_data)
                        return True
                    else:
                        self.log_test("Stop AI Trading", False, f"HTTP {response.status_code}", response.text)
                else:
                    self.log_test("Start AI Trading", False, data.get("error", "Unknown error"), data)
            else:
                self.log_test("Start AI Trading", False, f"HTTP {response.status_code}", response.text)
            return False
        except Exception as e:
            self.log_test("AI System", False, str(e))
            return False
    
    def test_trading_operations(self):
        """Test trading operations endpoints."""
        try:
            # Start session first
            self.session.post(
                f"{self.base_url}/api/session/start",
                json={"capital": 10000, "mode": "DEMO"}
            )
            
            # Place order
            response = self.session.post(
                f"{self.base_url}/api/orders/place",
                json={
                    "symbol": "AAPL",
                    "quantity": 100,
                    "order_type": "LIMIT",
                    "side": "BUY",
                    "price": 150.0
                }
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.log_test("Place Order", True, f"Order placed: {data.get('order', {}).get('order_id', 'Unknown')}", data)
                else:
                    self.log_test("Place Order", False, data.get("error", "Unknown error"), data)
            
            # Get orders
            response = self.session.get(f"{self.base_url}/api/orders")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    orders_count = data.get("count", 0)
                    self.log_test("Get Orders", True, f"Retrieved {orders_count} orders", data)
                else:
                    self.log_test("Get Orders", False, data.get("error", "Unknown error"), data)
            
            return True
        except Exception as e:
            self.log_test("Trading Operations", False, str(e))
            return False
    
    def test_portfolio_management(self):
        """Test portfolio management endpoints."""
        try:
            # Get portfolio
            response = self.session.get(f"{self.base_url}/api/portfolio")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    portfolio = data.get("portfolio", [])
                    summary = data.get("summary", {})
                    self.log_test("Get Portfolio", True, f"Portfolio: {len(portfolio)} positions, P&L: {summary.get('total_pnl', 0)}", data)
                else:
                    self.log_test("Get Portfolio", False, data.get("error", "Unknown error"), data)
            
            # Get positions
            response = self.session.get(f"{self.base_url}/api/positions")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    positions_count = data.get("count", 0)
                    self.log_test("Get Positions", True, f"Retrieved {positions_count} positions", data)
                else:
                    self.log_test("Get Positions", False, data.get("error", "Unknown error"), data)
            
            return True
        except Exception as e:
            self.log_test("Portfolio Management", False, str(e))
            return False
    
    def test_risk_management(self):
        """Test risk management endpoints."""
        try:
            # Get risk metrics
            response = self.session.get(f"{self.base_url}/api/risk/metrics")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    risk_metrics = data.get("risk_metrics", {})
                    self.log_test("Get Risk Metrics", True, f"Portfolio risk: {risk_metrics.get('portfolio_risk', 'Unknown')}", data)
                else:
                    self.log_test("Get Risk Metrics", False, data.get("error", "Unknown error"), data)
            
            # Check risk limits
            response = self.session.post(
                f"{self.base_url}/api/risk/check",
                params={"symbol": "AAPL", "quantity": 100, "price": 150.0}
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    risk_check = data.get("risk_check", {})
                    self.log_test("Check Risk Limits", True, f"Allowed: {risk_check.get('allowed', 'Unknown')}", data)
                else:
                    self.log_test("Check Risk Limits", False, data.get("error", "Unknown error"), data)
            
            return True
        except Exception as e:
            self.log_test("Risk Management", False, str(e))
            return False
    
    def test_market_data(self):
        """Test market data endpoints."""
        try:
            # Get market data
            response = self.session.get(f"{self.base_url}/api/market/data/AAPL?period=1d")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    market_data = data.get("data", [])
                    self.log_test("Get Market Data", True, f"Retrieved {len(market_data)} data points", data)
                else:
                    self.log_test("Get Market Data", False, data.get("error", "Unknown error"), data)
            
            # Get current price
            response = self.session.get(f"{self.base_url}/api/market/price/AAPL")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    price = data.get("price", 0)
                    self.log_test("Get Current Price", True, f"Price: ${price}", data)
                else:
                    self.log_test("Get Current Price", False, data.get("error", "Unknown error"), data)
            
            return True
        except Exception as e:
            self.log_test("Market Data", False, str(e))
            return False
    
    def test_analytics(self):
        """Test analytics endpoints."""
        try:
            # Get performance analytics
            response = self.session.get(f"{self.base_url}/api/analytics/performance")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    analytics = data.get("analytics", {})
                    self.log_test("Get Performance Analytics", True, f"Total trades: {analytics.get('total_trades', 0)}, Win rate: {analytics.get('win_rate', 0)}%", data)
                else:
                    self.log_test("Get Performance Analytics", False, data.get("error", "Unknown error"), data)
            
            # Get AI logs
            response = self.session.get(f"{self.base_url}/api/logs/ai?limit=10")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logs = data.get("logs", {})
                    self.log_test("Get AI Logs", True, f"Retrieved AI logs", data)
                else:
                    self.log_test("Get AI Logs", False, data.get("error", "Unknown error"), data)
            
            return True
        except Exception as e:
            self.log_test("Analytics", False, str(e))
            return False
    
    async def test_websocket(self):
        """Test WebSocket connection."""
        try:
            async with websockets.connect(WS_URL) as websocket:
                # Send a test message
                await websocket.send(json.dumps({"type": "test", "data": "Hello WebSocket"}))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                
                self.log_test("WebSocket Connection", True, f"Received: {data.get('type', 'Unknown')}", data)
                return True
        except Exception as e:
            self.log_test("WebSocket Connection", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all API tests."""
        print("🚀 Starting Final Trading API Tests...")
        print("=" * 60)
        
        # Test system endpoints
        self.test_system_status()
        self.test_system_health()
        
        # Test session management
        self.test_session_management()
        
        # Test AI system
        self.test_ai_system()
        
        # Test trading operations
        self.test_trading_operations()
        
        # Test portfolio management
        self.test_portfolio_management()
        
        # Test risk management
        self.test_risk_management()
        
        # Test market data
        self.test_market_data()
        
        # Test analytics
        self.test_analytics()
        
        # Test WebSocket
        print("\n🌐 Testing WebSocket connection...")
        asyncio.run(self.test_websocket())
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['message']}")
        
        print("\n🎯 API Testing Complete!")
        return passed_tests == total_tests

def main():
    """Main test function."""
    print("Final Trading API Test Suite")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/api/status", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running or not accessible")
            print(f"   Expected: http://localhost:8000")
            print(f"   Status Code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Cannot connect to API")
        print(f"   Error: {e}")
        print(f"   Make sure the API is running on {BASE_URL}")
        return False
    
    # Run tests
    tester = APITester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return success

if __name__ == "__main__":
    main()
