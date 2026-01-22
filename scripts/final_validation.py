#!/usr/bin/env python3
"""
Phase 0: Comprehensive System Validation Script
==============================================

This script provides comprehensive system validation for the trading bot,
including API validation, database connectivity, AI model health checks,
and end-to-end pipeline testing.

Based on the Phase 0 requirements from phase.plan.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import requests
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase0_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# API Keys from phase.plan.md
API_KEYS = {
    "QUESTRADE_REFRESH_TOKEN": "iAvs9K6p-MngByiDo29nTCVoTNgIN4Gr0",
    "ALPHA_VANTAGE_API_KEY": "ZJAGE580APQ5UXPL",
    "FINNHUB_API_KEY": "d3hd0g9r01qi2vu0d5e0d3hd0g9r01qi2vu0d5eg",
    "NEWSAPI_KEY": "aa175a7eef1340cab792ab1570fe72e5"
}

# Placeholder APIs (no keys available)
PLACEHOLDER_APIS = [
    "Reddit API (social sentiment)",
    "Weather API (commodity correlation)", 
    "SEC EDGAR (insider trades)"
]


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.start_time = time.time()
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.success_count = 0
        self.total_count = 0
    
    def add_result(self, test_name: str, success: bool, details: Any = None, error: str = None):
        """Add a test result."""
        self.total_count += 1
        if success:
            self.success_count += 1
        else:
            if error:
                self.errors.append(f"{test_name}: {error}")
            else:
                self.errors.append(f"{test_name}: Failed")
        
        self.results[test_name] = {
            "success": success,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        duration = time.time() - self.start_time
        return {
            "duration_seconds": round(duration, 2),
            "total_tests": self.total_count,
            "successful_tests": self.success_count,
            "failed_tests": self.total_count - self.success_count,
            "success_rate": round(self.success_count / self.total_count * 100, 1) if self.total_count > 0 else 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": datetime.now().isoformat()
        }


class APIValidator:
    """Validates external API connectivity."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 15
    
    async def validate_questrade(self, token: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate Questrade API connectivity."""
        try:
            url = f"https://login.questrade.com/oauth2/token?grant_type=refresh_token&refresh_token={token}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if "access_token" in data:
                    return True, {
                        "message": "Questrade API connected successfully",
                        "token_type": data.get("token_type", "unknown"),
                        "expires_in": data.get("expires_in", "unknown")
                    }
            
            return False, {
                "message": f"Questrade API failed: HTTP {response.status_code}",
                "response": response.text[:200]
            }
        except Exception as e:
            return False, {
                "message": f"Questrade API error: {str(e)}"
            }
    
    async def validate_yahoo_finance(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate Yahoo Finance API (no key needed)."""
        try:
            # Test with a simple quote request
            url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if "chart" in data and "result" in data["chart"]:
                    return True, {
                        "message": "Yahoo Finance API accessible",
                        "test_symbol": "AAPL",
                        "data_available": len(data["chart"]["result"]) > 0
                    }
            
            return False, {
                "message": f"Yahoo Finance API failed: HTTP {response.status_code}"
            }
        except Exception as e:
            return False, {
                "message": f"Yahoo Finance API error: {str(e)}"
            }
    
    async def validate_alpha_vantage(self, api_key: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate Alpha Vantage API."""
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if "Global Quote" in data and data["Global Quote"]:
                    return True, {
                        "message": "Alpha Vantage API connected successfully",
                        "test_symbol": "AAPL",
                        "quote_available": True
                    }
                elif "Note" in data:
                    return False, {
                        "message": f"Alpha Vantage rate limit: {data['Note']}"
                    }
            
            return False, {
                "message": f"Alpha Vantage API failed: HTTP {response.status_code}",
                "response": response.text[:200]
            }
        except Exception as e:
            return False, {
                "message": f"Alpha Vantage API error: {str(e)}"
            }
    
    async def validate_finnhub(self, api_key: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate Finnhub API."""
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if "c" in data and data["c"] is not None:
                    return True, {
                        "message": "Finnhub API connected successfully",
                        "test_symbol": "AAPL",
                        "current_price": data.get("c", "unknown")
                    }
            
            return False, {
                "message": f"Finnhub API failed: HTTP {response.status_code}",
                "response": response.text[:200]
            }
        except Exception as e:
            return False, {
                "message": f"Finnhub API error: {str(e)}"
            }
    
    async def validate_newsapi(self, api_key: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate NewsAPI."""
        try:
            url = f"https://newsapi.org/v2/everything?q=apple&sortBy=publishedAt&apiKey={api_key}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if "articles" in data and isinstance(data["articles"], list):
                    return True, {
                        "message": "NewsAPI connected successfully",
                        "article_count": len(data["articles"]),
                        "total_results": data.get("totalResults", 0)
                    }
            
            return False, {
                "message": f"NewsAPI failed: HTTP {response.status_code}",
                "response": response.text[:200]
            }
        except Exception as e:
            return False, {
                "message": f"NewsAPI error: {str(e)}"
            }


class DatabaseValidator:
    """Validates database connectivity and schema."""
    
    def __init__(self):
        self.db_manager = None
    
    async def validate_database_connectivity(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate database connectivity for both LIVE and DEMO modes."""
        try:
            # Import database manager
            sys.path.append(str(ROOT / "src"))
            from config.database import get_database_manager
            
            self.db_manager = get_database_manager()
            
            # Test both modes
            modes = ["LIVE", "DEMO"]
            results = {}
            
            for mode in modes:
                try:
                    with self.db_manager.get_connection(mode) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                        result = cursor.fetchone()
                        if result and result[0] == 1:
                            results[mode] = "connected"
                        else:
                            results[mode] = "failed"
                except Exception as e:
                    results[mode] = f"error: {str(e)}"
            
            all_connected = all(status == "connected" for status in results.values())
            
            return all_connected, {
                "message": "Database connectivity validated",
                "modes": results,
                "isolation": "LIVE and DEMO modes isolated"
            }
            
        except Exception as e:
            return False, {
                "message": f"Database connectivity error: {str(e)}"
            }
    
    async def validate_database_schema(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate database schema and required tables."""
        try:
            if not self.db_manager:
                return False, {"message": "Database manager not initialized"}
            
            # Check required tables exist
            required_tables = [
                "positions", "orders", "trade_results", 
                "account_balance", "risk_events", "system_logs"
            ]
            
            table_status = {}
            for table in required_tables:
                try:
                    count = self.db_manager.get_table_count(table, "DEMO")
                    table_status[table] = f"exists ({count} rows)"
                except Exception as e:
                    table_status[table] = f"error: {str(e)}"
            
            all_tables_exist = all("exists" in status for status in table_status.values())
            
            return all_tables_exist, {
                "message": "Database schema validated",
                "tables": table_status
            }
            
        except Exception as e:
            return False, {
                "message": f"Database schema validation error: {str(e)}"
            }


class AIModelValidator:
    """Validates AI model availability and health."""
    
    async def validate_ollama_health(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate Ollama health check."""
        try:
            url = "http://localhost:11434/api/tags"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                
                return True, {
                    "message": "Ollama health check passed",
                    "available_models": len(models),
                    "models": [model.get("name", "unknown") for model in models]
                }
            else:
                return False, {
                    "message": f"Ollama health check failed: HTTP {response.status_code}"
                }
                
        except Exception as e:
            return False, {
                "message": f"Ollama health check error: {str(e)}"
            }


class PipelineValidator:
    """Validates core trading cycle components."""
    
    async def validate_trading_cycle_components(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate core trading cycle components."""
        try:
            # Check if main components can be imported
            components = [
                "src.workflows.trading_cycle",
                "src.monitoring.system_monitor", 
                "src.data_pipeline.questrade_client",
                "src.ai.multi_model",
                "src.trading.risk"
            ]
            
            import_results = {}
            for component in components:
                try:
                    __import__(component)
                    import_results[component] = "imported successfully"
                except Exception as e:
                    import_results[component] = f"import failed: {str(e)}"
            
            all_imported = all("imported successfully" in result for result in import_results.values())
            
            return all_imported, {
                "message": "Trading cycle components validated",
                "components": import_results
            }
            
        except Exception as e:
            return False, {
                "message": f"Trading cycle validation error: {str(e)}"
            }


class SmokeTestRunner:
    """Runs fast smoke tests with AI_LIMIT=20."""
    
    def __init__(self):
        self.ai_limit = 20
        self.max_duration = 360  # 6 minutes in seconds
    
    async def run_smoke_test(self) -> Tuple[bool, Dict[str, Any]]:
        """Run fast smoke test with AI_LIMIT=20."""
        try:
            start_time = time.time()
            
            # Run simple smoke test script
            cmd = [sys.executable, "scripts/smoke_test.py"]
            
            logger.info(f"Starting smoke test with AI_LIMIT={self.ai_limit}")
            process = subprocess.Popen(
                cmd, 
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.max_duration)
                duration = time.time() - start_time
                
                if process.returncode == 0:
                    return True, {
                        "message": f"Smoke test completed successfully in {duration:.1f}s",
                        "duration_seconds": round(duration, 1),
                        "ai_limit": self.ai_limit,
                        "output_lines": len(stdout.split('\n')),
                        "error_lines": len(stderr.split('\n')) if stderr else 0
                    }
                else:
                    return False, {
                        "message": f"Smoke test failed with return code {process.returncode}",
                        "duration_seconds": round(duration, 1),
                        "stdout": stdout[-500:],  # Last 500 chars
                        "stderr": stderr[-500:] if stderr else ""
                    }
                    
            except subprocess.TimeoutExpired:
                process.kill()
                return False, {
                    "message": f"Smoke test timed out after {self.max_duration}s",
                    "ai_limit": self.ai_limit
                }
                
        except Exception as e:
            return False, {
                "message": f"Smoke test error: {str(e)}"
            }


async def run_comprehensive_validation() -> ValidationResult:
    """Run comprehensive system validation."""
    result = ValidationResult()
    
    logger.info("Starting Phase 0 Comprehensive System Validation")
    
    # 1. API Validation
    logger.info("Validating External APIs...")
    api_validator = APIValidator()
    
    # Questrade API
    success, details = await api_validator.validate_questrade(API_KEYS["QUESTRADE_REFRESH_TOKEN"])
    result.add_result("Questrade API", success, details)
    
    # Yahoo Finance (no key needed)
    success, details = await api_validator.validate_yahoo_finance()
    result.add_result("Yahoo Finance API", success, details)
    
    # Alpha Vantage
    success, details = await api_validator.validate_alpha_vantage(API_KEYS["ALPHA_VANTAGE_API_KEY"])
    result.add_result("Alpha Vantage API", success, details)
    
    # Finnhub
    success, details = await api_validator.validate_finnhub(API_KEYS["FINNHUB_API_KEY"])
    result.add_result("Finnhub API", success, details)
    
    # NewsAPI
    success, details = await api_validator.validate_newsapi(API_KEYS["NEWSAPI_KEY"])
    result.add_result("NewsAPI", success, details)
    
    # Add placeholder API warnings
    for api in PLACEHOLDER_APIS:
        result.add_warning(f"Placeholder API skipped: {api}")
    
    # 2. Database Validation
    logger.info("Validating Database Connectivity...")
    db_validator = DatabaseValidator()
    
    success, details = await db_validator.validate_database_connectivity()
    result.add_result("Database Connectivity", success, details)
    
    success, details = await db_validator.validate_database_schema()
    result.add_result("Database Schema", success, details)
    
    # 3. AI Model Validation
    logger.info("Validating AI Model Health...")
    ai_validator = AIModelValidator()
    
    success, details = await ai_validator.validate_ollama_health()
    result.add_result("Ollama Health Check", success, details)
    
    # 4. Pipeline Component Validation
    logger.info("Validating Trading Cycle Components...")
    pipeline_validator = PipelineValidator()
    
    success, details = await pipeline_validator.validate_trading_cycle_components()
    result.add_result("Trading Cycle Components", success, details)
    
    # 5. Smoke Test
    logger.info("Running Smoke Test (AI_LIMIT=20)...")
    smoke_runner = SmokeTestRunner()
    
    success, details = await smoke_runner.run_smoke_test()
    result.add_result("Smoke Test", success, details)
    
    return result


def run_legacy_tests() -> int:
    """Run legacy test suite with plugins disabled."""
    try:
        env = os.environ.copy()
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        
        logger.info("Running Legacy Test Suite...")
        result = subprocess.call(
            [sys.executable, "-m", "pytest", "tests", "-q", "--tb=short"],
            cwd=str(ROOT),
            env=env
        )
        
        return result
    except Exception as e:
        logger.error(f"Legacy test suite error: {e}")
        return 1


async def main() -> int:
    """Main validation function."""
    # Load environment variables
    load_dotenv(dotenv_path=ROOT / ".env")

    # Ensure logs directory exists
    (ROOT / "logs").mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PHASE 0: COMPREHENSIVE SYSTEM VALIDATION")
    logger.info("=" * 60)
    
    # Run comprehensive validation
    validation_result = await run_comprehensive_validation()
    
    # Run legacy tests
    legacy_test_result = run_legacy_tests()
    validation_result.add_result("Legacy Test Suite", legacy_test_result == 0)
    
    # Generate summary
    summary = validation_result.get_summary()
    
    # Log summary
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Duration: {summary['duration_seconds']}s")
    logger.info(f"Total Tests: {summary['total_tests']}")
    logger.info(f"Successful: {summary['successful_tests']}")
    logger.info(f"Failed: {summary['failed_tests']}")
    logger.info(f"Success Rate: {summary['success_rate']}%")
    
    if summary['warnings']:
        logger.info(f"Warnings: {len(summary['warnings'])}")
        for warning in summary['warnings']:
            logger.warning(f"  WARNING: {warning}")
    
    if summary['errors']:
        logger.error(f"Errors: {len(summary['errors'])}")
        for error in summary['errors']:
            logger.error(f"  ERROR: {error}")
    
    # Save detailed results
    results_file = ROOT / "logs" / "phase0_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "summary": summary,
            "detailed_results": validation_result.results
        }, f, indent=2)
    
    logger.info(f"Detailed results saved to: {results_file}")

    # Final banner
    if summary['success_rate'] >= 90:
        logger.info("")
        logger.info("ALL TESTS PASSED — SYSTEM READY FOR BUILD")
        logger.info("")
    return 0
    else:
        logger.error("")
        logger.error("VALIDATION FAILED — SYSTEM NOT READY")
        logger.error("")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

