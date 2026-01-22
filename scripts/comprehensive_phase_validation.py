#!/usr/bin/env python3
"""
Comprehensive Phase 0-11 Validation Script
Tests all implemented phases and validates system readiness for Phase 12-13
"""

import asyncio
import sys
import os
import time
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveValidator:
    """Comprehensive validation of all phases 0-11"""
    
    def __init__(self):
        self.results = {
            'phase_0': {'status': 'pending', 'details': {}},
            'phase_1': {'status': 'pending', 'details': {}},
            'phase_2': {'status': 'pending', 'details': {}},
            'phase_3': {'status': 'pending', 'details': {}},
            'phase_4': {'status': 'pending', 'details': {}},
            'phase_5': {'status': 'pending', 'details': {}},
            'phase_6': {'status': 'pending', 'details': {}},
            'phase_7': {'status': 'pending', 'details': {}},
            'phase_8': {'status': 'pending', 'details': {}},
            'phase_9': {'status': 'pending', 'details': {}},
            'phase_10': {'status': 'pending', 'details': {}},
            'phase_11': {'status': 'pending', 'details': {}},
            'integration': {'status': 'pending', 'details': {}},
            'performance': {'status': 'pending', 'details': {}},
            'readiness': {'status': 'pending', 'details': {}}
        }
        self.start_time = time.time()
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests"""
        logger.info("🚀 Starting Comprehensive Phase 0-11 Validation")
        
        try:
            # Phase 0: System Baseline
            await self.validate_phase_0()
            
            # Phase 1: Observability
            await self.validate_phase_1()
            
            # Phase 2: API Budgets & Caching
            await self.validate_phase_2()
            
            # Phase 3: Data Quality
            await self.validate_phase_3()
            
            # Phase 4: Confidence Calibration
            await self.validate_phase_4()
            
            # Phase 5: Adaptive Weights
            await self.validate_phase_5()
            
            # Phase 6: Drawdown-Aware Kelly & ATR
            await self.validate_phase_6()
            
            # Phase 7: Regime Awareness
            await self.validate_phase_7()
            
            # Phase 8: Dashboard Enhancements
            await self.validate_phase_8()
            
            # Phase 9: GPU & Ollama Lifecycle
            await self.validate_phase_9()
            
            # Phase 10: CI & Automation
            await self.validate_phase_10()
            
            # Phase 11: Feature Flags & Safety
            await self.validate_phase_11()
            
            # Integration Tests
            await self.validate_integration()
            
            # Performance Tests
            await self.validate_performance()
            
            # Readiness Assessment
            await self.assess_readiness()
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            logger.error(traceback.format_exc())
            self.results['overall'] = {'status': 'failed', 'error': str(e)}
        
        # Generate final report
        return self.generate_report()
    
    async def validate_phase_0(self):
        """Validate Phase 0: System Baseline"""
        logger.info("🔍 Validating Phase 0: System Baseline")
        
        try:
            # Test database connectivity
            from config.database import get_connection
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
            required_tables = ['api_validation_log', 'phase_execution_tracking', 'api_usage_metrics']
            missing_tables = [t for t in required_tables if t not in tables]
            
            if missing_tables:
                self.results['phase_0']['status'] = 'failed'
                self.results['phase_0']['details']['missing_tables'] = missing_tables
                logger.error(f"❌ Phase 0: Missing tables: {missing_tables}")
            else:
                self.results['phase_0']['status'] = 'passed'
                self.results['phase_0']['details']['tables'] = len(tables)
                logger.info(f"✅ Phase 0: Database OK ({len(tables)} tables)")
            
            # Test API validation script exists
            validation_script = Path("scripts/final_validation.py")
            if validation_script.exists():
                self.results['phase_0']['details']['validation_script'] = True
                logger.info("✅ Phase 0: Validation script exists")
            else:
                self.results['phase_0']['status'] = 'failed'
                self.results['phase_0']['details']['validation_script'] = False
                logger.error("❌ Phase 0: Validation script missing")
            
        except Exception as e:
            self.results['phase_0']['status'] = 'failed'
            self.results['phase_0']['details']['error'] = str(e)
            logger.error(f"❌ Phase 0 failed: {e}")
    
    async def validate_phase_1(self):
        """Validate Phase 1: Observability Foundation"""
        logger.info("🔍 Validating Phase 1: Observability Foundation")
        
        try:
            # Test system monitor
            from monitoring.system_monitor import SystemMonitor
            monitor = SystemMonitor()
            
            # Test phase timing
            monitor.start_phase_timer("test_phase")
            await asyncio.sleep(0.1)
            duration = monitor.end_phase_timer("test_phase")
            
            if duration > 0:
                self.results['phase_1']['status'] = 'passed'
                self.results['phase_1']['details']['phase_timing'] = True
                logger.info("✅ Phase 1: Phase timing works")
            else:
                self.results['phase_1']['status'] = 'failed'
                self.results['phase_1']['details']['phase_timing'] = False
                logger.error("❌ Phase 1: Phase timing failed")
            
            # Test JSON logging
            log_file = Path("logs/system.log")
            if log_file.exists():
                self.results['phase_1']['details']['json_logging'] = True
                logger.info("✅ Phase 1: JSON logging active")
            else:
                self.results['phase_1']['status'] = 'failed'
                self.results['phase_1']['details']['json_logging'] = False
                logger.error("❌ Phase 1: JSON logging not working")
            
            # Test performance analytics
            from monitoring.performance_analytics import PerformanceAnalytics
            analytics = PerformanceAnalytics()
            self.results['phase_1']['details']['performance_analytics'] = True
            logger.info("✅ Phase 1: Performance analytics available")
            
        except Exception as e:
            self.results['phase_1']['status'] = 'failed'
            self.results['phase_1']['details']['error'] = str(e)
            logger.error(f"❌ Phase 1 failed: {e}")
    
    async def validate_phase_2(self):
        """Validate Phase 2: API Budgets & Caching"""
        logger.info("🔍 Validating Phase 2: API Budgets & Caching")
        
        try:
            # Test API budget manager
            from data_pipeline.api_budget_manager import API_Budget_Manager
            budget_manager = API_Budget_Manager()
            
            # Test budget tracking
            budget_manager.track_api_call("test_api", 1)
            remaining = budget_manager.get_remaining_budget("test_api")
            
            if remaining is not None:
                self.results['phase_2']['status'] = 'passed'
                self.results['phase_2']['details']['budget_tracking'] = True
                logger.info("✅ Phase 2: API budget tracking works")
            else:
                self.results['phase_2']['status'] = 'failed'
                self.results['phase_2']['details']['budget_tracking'] = False
                logger.error("❌ Phase 2: API budget tracking failed")
            
            # Test caching
            cache_key = "test_cache_key"
            test_data = {"test": "data"}
            budget_manager.cache.set(cache_key, test_data, ttl=60)
            cached_data = budget_manager.cache.get(cache_key)
            
            if cached_data == test_data:
                self.results['phase_2']['details']['caching'] = True
                logger.info("✅ Phase 2: Caching works")
            else:
                self.results['phase_2']['status'] = 'failed'
                self.results['phase_2']['details']['caching'] = False
                logger.error("❌ Phase 2: Caching failed")
            
        except Exception as e:
            self.results['phase_2']['status'] = 'failed'
            self.results['phase_2']['details']['error'] = str(e)
            logger.error(f"❌ Phase 2 failed: {e}")
    
    async def validate_phase_3(self):
        """Validate Phase 3: Data Quality"""
        logger.info("🔍 Validating Phase 3: Data Quality")
        
        try:
            # Test data quality validation
            from validation.data_quality import DataQualityValidator
            validator = DataQualityValidator()
            
            # Test with sample data
            sample_data = {
                'symbol': 'AAPL',
                'atr': 2.5,
                'volume': 1000000,
                'bb_upper': 150.0,
                'bb_middle': 145.0,
                'bb_lower': 140.0,
                'adx': 25.0,
                'rsi': 50.0
            }
            
            quality_score = validator.validate_indicators(sample_data)
            
            if quality_score > 0:
                self.results['phase_3']['status'] = 'passed'
                self.results['phase_3']['details']['quality_validation'] = True
                self.results['phase_3']['details']['quality_score'] = quality_score
                logger.info(f"✅ Phase 3: Data quality validation works (score: {quality_score})")
            else:
                self.results['phase_3']['status'] = 'failed'
                self.results['phase_3']['details']['quality_validation'] = False
                logger.error("❌ Phase 3: Data quality validation failed")
            
            # Test database schema
            from config.database import get_connection
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data_provenance'")
                if cursor.fetchone():
                    self.results['phase_3']['details']['provenance_table'] = True
                    logger.info("✅ Phase 3: Data provenance table exists")
                else:
                    self.results['phase_3']['status'] = 'failed'
                    self.results['phase_3']['details']['provenance_table'] = False
                    logger.error("❌ Phase 3: Data provenance table missing")
            
        except Exception as e:
            self.results['phase_3']['status'] = 'failed'
            self.results['phase_3']['details']['error'] = str(e)
            logger.error(f"❌ Phase 3 failed: {e}")
    
    async def validate_phase_4(self):
        """Validate Phase 4: Confidence Calibration"""
        logger.info("🔍 Validating Phase 4: Confidence Calibration")
        
        try:
            # Test confidence calibration
            from adaptive.confidence_calibration import ConfidenceCalibrator
            calibrator = ConfidenceCalibrator()
            
            # Test calibration
            raw_confidence = 0.8
            calibrated = calibrator.calibrate_confidence("test_model", raw_confidence)
            
            if 0 <= calibrated <= 1:
                self.results['phase_4']['status'] = 'passed'
                self.results['phase_4']['details']['calibration'] = True
                self.results['phase_4']['details']['raw_confidence'] = raw_confidence
                self.results['phase_4']['details']['calibrated_confidence'] = calibrated
                logger.info(f"✅ Phase 4: Confidence calibration works ({raw_confidence} -> {calibrated})")
            else:
                self.results['phase_4']['status'] = 'failed'
                self.results['phase_4']['details']['calibration'] = False
                logger.error("❌ Phase 4: Confidence calibration failed")
            
            # Test database schema
            from config.database import get_connection
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='confidence_calibration'")
                if cursor.fetchone():
                    self.results['phase_4']['details']['calibration_table'] = True
                    logger.info("✅ Phase 4: Confidence calibration table exists")
                else:
                    self.results['phase_4']['status'] = 'failed'
                    self.results['phase_4']['details']['calibration_table'] = False
                    logger.error("❌ Phase 4: Confidence calibration table missing")
            
        except Exception as e:
            self.results['phase_4']['status'] = 'failed'
            self.results['phase_4']['details']['error'] = str(e)
            logger.error(f"❌ Phase 4 failed: {e}")
    
    async def validate_phase_5(self):
        """Validate Phase 5: Adaptive Weights"""
        logger.info("🔍 Validating Phase 5: Adaptive Weights")
        
        try:
            # Test adaptive weights
            from ai.adaptive_weights import AdaptiveWeightManager
            weight_manager = AdaptiveWeightManager()
            
            # Test weight calculation
            model_performance = {
                'model1': {'brier_score': 0.2, 'accuracy': 0.8},
                'model2': {'brier_score': 0.3, 'accuracy': 0.7}
            }
            
            weights = weight_manager.calculate_weights(model_performance)
            
            if weights and sum(weights.values()) > 0.9:  # Should sum to ~1
                self.results['phase_5']['status'] = 'passed'
                self.results['phase_5']['details']['weight_calculation'] = True
                self.results['phase_5']['details']['weights'] = weights
                logger.info(f"✅ Phase 5: Adaptive weights work: {weights}")
            else:
                self.results['phase_5']['status'] = 'failed'
                self.results['phase_5']['details']['weight_calculation'] = False
                logger.error("❌ Phase 5: Adaptive weights failed")
            
            # Test database schema
            from config.database import get_connection
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_performance'")
                if cursor.fetchone():
                    self.results['phase_5']['details']['performance_table'] = True
                    logger.info("✅ Phase 5: Model performance table exists")
                else:
                    self.results['phase_5']['status'] = 'failed'
                    self.results['phase_5']['details']['performance_table'] = False
                    logger.error("❌ Phase 5: Model performance table missing")
            
        except Exception as e:
            self.results['phase_5']['status'] = 'failed'
            self.results['phase_5']['details']['error'] = str(e)
            logger.error(f"❌ Phase 5 failed: {e}")
    
    async def validate_phase_6(self):
        """Validate Phase 6: Drawdown-Aware Kelly & ATR"""
        logger.info("🔍 Validating Phase 6: Drawdown-Aware Kelly & ATR")
        
        try:
            # Test drawdown calculation
            from trading.risk import RiskManager
            risk_manager = RiskManager()
            
            # Test Kelly calculation with drawdown
            kelly_fraction = risk_manager.calculate_kelly_fraction(0.6, 0.4, 0.2)
            if 0 <= kelly_fraction <= 1:
                self.results['phase_6']['status'] = 'passed'
                self.results['phase_6']['details']['kelly_calculation'] = True
                self.results['phase_6']['details']['kelly_fraction'] = kelly_fraction
                logger.info(f"✅ Phase 6: Kelly calculation works: {kelly_fraction}")
            else:
                self.results['phase_6']['status'] = 'failed'
                self.results['phase_6']['details']['kelly_calculation'] = False
                logger.error("❌ Phase 6: Kelly calculation failed")
            
            # Test ATR brackets
            from trading.atr_brackets import ATRBracketManager
            bracket_manager = ATRBracketManager()
            
            # Test bracket calculation
            brackets = bracket_manager.calculate_atr_brackets(
                symbol="AAPL",
                current_price=150.0,
                atr_value=2.5,
                volatility_multiplier=2.0,
                risk_reward_ratio=2.0
            )
            
            if brackets and 'stop_loss' in brackets and 'take_profit' in brackets:
                self.results['phase_6']['details']['atr_brackets'] = True
                self.results['phase_6']['details']['brackets'] = brackets
                logger.info(f"✅ Phase 6: ATR brackets work: {brackets}")
            else:
                self.results['phase_6']['status'] = 'failed'
                self.results['phase_6']['details']['atr_brackets'] = False
                logger.error("❌ Phase 6: ATR brackets failed")
            
        except Exception as e:
            self.results['phase_6']['status'] = 'failed'
            self.results['phase_6']['details']['error'] = str(e)
            logger.error(f"❌ Phase 6 failed: {e}")
    
    async def validate_phase_7(self):
        """Validate Phase 7: Regime Awareness"""
        logger.info("🔍 Validating Phase 7: Regime Awareness")
        
        try:
            # Test regime detection
            from ai.regime_detection import RegimeDetector
            regime_detector = RegimeDetector()
            
            # Test regime classification
            market_data = {
                'atr': 2.5,
                'volatility': 0.15,
                'trend_strength': 0.3
            }
            
            regime = regime_detector.detect_regime(market_data)
            
            if regime:
                self.results['phase_7']['status'] = 'passed'
                self.results['phase_7']['details']['regime_detection'] = True
                self.results['phase_7']['details']['detected_regime'] = regime
                logger.info(f"✅ Phase 7: Regime detection works: {regime}")
            else:
                self.results['phase_7']['status'] = 'failed'
                self.results['phase_7']['details']['regime_detection'] = False
                logger.error("❌ Phase 7: Regime detection failed")
            
            # Test regime policies
            from config.regime_policy_manager import RegimePolicyManager
            policy_manager = RegimePolicyManager()
            
            policies = policy_manager.get_policies()
            if policies:
                self.results['phase_7']['details']['regime_policies'] = True
                logger.info("✅ Phase 7: Regime policies loaded")
            else:
                self.results['phase_7']['status'] = 'failed'
                self.results['phase_7']['details']['regime_policies'] = False
                logger.error("❌ Phase 7: Regime policies failed")
            
        except Exception as e:
            self.results['phase_7']['status'] = 'failed'
            self.results['phase_7']['details']['error'] = str(e)
            logger.error(f"❌ Phase 7 failed: {e}")
    
    async def validate_phase_8(self):
        """Validate Phase 8: Dashboard Enhancements"""
        logger.info("🔍 Validating Phase 8: Dashboard Enhancements")
        
        try:
            # Test dashboard connector
            from dashboard.connector import DashboardConnector
            connector = DashboardConnector()
            
            # Test data providers
            api_budget_data = connector.get_api_budget_data()
            if api_budget_data is not None:
                self.results['phase_8']['status'] = 'passed'
                self.results['phase_8']['details']['api_budget_data'] = True
                logger.info("✅ Phase 8: API budget data provider works")
            else:
                self.results['phase_8']['status'] = 'failed'
                self.results['phase_8']['details']['api_budget_data'] = False
                logger.error("❌ Phase 8: API budget data provider failed")
            
            # Test enhanced layout
            from dashboard.enhanced_layout import EnhancedDashboardLayout
            layout = EnhancedDashboardLayout()
            
            if layout:
                self.results['phase_8']['details']['enhanced_layout'] = True
                logger.info("✅ Phase 8: Enhanced layout available")
            else:
                self.results['phase_8']['status'] = 'failed'
                self.results['phase_8']['details']['enhanced_layout'] = False
                logger.error("❌ Phase 8: Enhanced layout failed")
            
        except Exception as e:
            self.results['phase_8']['status'] = 'failed'
            self.results['phase_8']['details']['error'] = str(e)
            logger.error(f"❌ Phase 8 failed: {e}")
    
    async def validate_phase_9(self):
        """Validate Phase 9: GPU & Ollama Lifecycle"""
        logger.info("🔍 Validating Phase 9: GPU & Ollama Lifecycle")
        
        try:
            # Test Ollama lifecycle
            from ai.ollama_lifecycle import OllamaLifecycleManager
            lifecycle_manager = OllamaLifecycleManager()
            
            # Test health check
            health_status = await lifecycle_manager.check_ollama_health()
            
            if health_status:
                self.results['phase_9']['status'] = 'passed'
                self.results['phase_9']['details']['ollama_health'] = True
                logger.info("✅ Phase 9: Ollama health check works")
            else:
                self.results['phase_9']['status'] = 'failed'
                self.results['phase_9']['details']['ollama_health'] = False
                logger.error("❌ Phase 9: Ollama health check failed")
            
            # Test model management
            models_ready = await lifecycle_manager.ensure_models_ready(['llama2'])
            
            if models_ready:
                self.results['phase_9']['details']['model_management'] = True
                logger.info("✅ Phase 9: Model management works")
            else:
                self.results['phase_9']['status'] = 'failed'
                self.results['phase_9']['details']['model_management'] = False
                logger.error("❌ Phase 9: Model management failed")
            
        except Exception as e:
            self.results['phase_9']['status'] = 'failed'
            self.results['phase_9']['details']['error'] = str(e)
            logger.error(f"❌ Phase 9 failed: {e}")
    
    async def validate_phase_10(self):
        """Validate Phase 10: CI & Automation"""
        logger.info("🔍 Validating Phase 10: CI & Automation")
        
        try:
            # Test CI validation script
            ci_script = Path("scripts/ci_validation.py")
            if ci_script.exists():
                self.results['phase_10']['status'] = 'passed'
                self.results['phase_10']['details']['ci_script'] = True
                logger.info("✅ Phase 10: CI validation script exists")
            else:
                self.results['phase_10']['status'] = 'failed'
                self.results['phase_10']['details']['ci_script'] = False
                logger.error("❌ Phase 10: CI validation script missing")
            
            # Test GitHub workflow
            workflow_file = Path(".github/workflows/nightly-validation.yml")
            if workflow_file.exists():
                self.results['phase_10']['details']['github_workflow'] = True
                logger.info("✅ Phase 10: GitHub workflow exists")
            else:
                self.results['phase_10']['status'] = 'failed'
                self.results['phase_10']['details']['github_workflow'] = False
                logger.error("❌ Phase 10: GitHub workflow missing")
            
            # Test pre-commit config
            precommit_config = Path(".pre-commit-config.yaml")
            if precommit_config.exists():
                self.results['phase_10']['details']['precommit_config'] = True
                logger.info("✅ Phase 10: Pre-commit config exists")
            else:
                self.results['phase_10']['status'] = 'failed'
                self.results['phase_10']['details']['precommit_config'] = False
                logger.error("❌ Phase 10: Pre-commit config missing")
            
        except Exception as e:
            self.results['phase_10']['status'] = 'failed'
            self.results['phase_10']['details']['error'] = str(e)
            logger.error(f"❌ Phase 10 failed: {e}")
    
    async def validate_phase_11(self):
        """Validate Phase 11: Feature Flags & Safety"""
        logger.info("🔍 Validating Phase 11: Feature Flags & Safety")
        
        try:
            # Test feature flags
            from config.feature_flags import FeatureFlagManager
            flag_manager = FeatureFlagManager()
            
            # Test flag operations
            flag_manager.set_flag('test_flag', True)
            flag_value = flag_manager.get_flag('test_flag')
            
            if flag_value:
                self.results['phase_11']['status'] = 'passed'
                self.results['phase_11']['details']['feature_flags'] = True
                logger.info("✅ Phase 11: Feature flags work")
            else:
                self.results['phase_11']['status'] = 'failed'
                self.results['phase_11']['details']['feature_flags'] = False
                logger.error("❌ Phase 11: Feature flags failed")
            
            # Test safety controls
            from dashboard.safety_controls import SafetyControlsDashboard
            safety_controls = SafetyControlsDashboard()
            
            if safety_controls:
                self.results['phase_11']['details']['safety_controls'] = True
                logger.info("✅ Phase 11: Safety controls available")
            else:
                self.results['phase_11']['status'] = 'failed'
                self.results['phase_11']['details']['safety_controls'] = False
                logger.error("❌ Phase 11: Safety controls failed")
            
            # Test rollout plan
            rollout_plan = Path("docs/ROLLOUT_PLAN.md")
            if rollout_plan.exists():
                self.results['phase_11']['details']['rollout_plan'] = True
                logger.info("✅ Phase 11: Rollout plan exists")
            else:
                self.results['phase_11']['status'] = 'failed'
                self.results['phase_11']['details']['rollout_plan'] = False
                logger.error("❌ Phase 11: Rollout plan missing")
            
        except Exception as e:
            self.results['phase_11']['status'] = 'failed'
            self.results['phase_11']['details']['error'] = str(e)
            logger.error(f"❌ Phase 11 failed: {e}")
    
    async def validate_integration(self):
        """Validate integration between phases"""
        logger.info("🔍 Validating Phase Integration")
        
        try:
            # Test trading cycle integration
            from workflows.trading_cycle import TradingCycle
            trading_cycle = TradingCycle()
            
            # Test end-to-end pipeline
            result = await trading_cycle.run_cycle(ai_limit=5)
            
            if result and 'picks' in result:
                self.results['integration']['status'] = 'passed'
                self.results['integration']['details']['trading_cycle'] = True
                self.results['integration']['details']['picks_generated'] = len(result['picks'])
                logger.info(f"✅ Integration: Trading cycle works ({len(result['picks'])} picks)")
            else:
                self.results['integration']['status'] = 'failed'
                self.results['integration']['details']['trading_cycle'] = False
                logger.error("❌ Integration: Trading cycle failed")
            
        except Exception as e:
            self.results['integration']['status'] = 'failed'
            self.results['integration']['details']['error'] = str(e)
            logger.error(f"❌ Integration failed: {e}")
    
    async def validate_performance(self):
        """Validate performance benchmarks"""
        logger.info("🔍 Validating Performance")
        
        try:
            # Test API performance
            from data_pipeline.api_budget_manager import API_Budget_Manager
            budget_manager = API_Budget_Manager()
            
            start_time = time.time()
            for i in range(10):
                budget_manager.track_api_call("perf_test", 1)
            end_time = time.time()
            
            duration = end_time - start_time
            if duration < 1.0:  # Should be fast
                self.results['performance']['status'] = 'passed'
                self.results['performance']['details']['api_performance'] = True
                self.results['performance']['details']['api_duration'] = duration
                logger.info(f"✅ Performance: API tracking fast ({duration:.3f}s)")
            else:
                self.results['performance']['status'] = 'failed'
                self.results['performance']['details']['api_performance'] = False
                logger.error(f"❌ Performance: API tracking slow ({duration:.3f}s)")
            
        except Exception as e:
            self.results['performance']['status'] = 'failed'
            self.results['performance']['details']['error'] = str(e)
            logger.error(f"❌ Performance validation failed: {e}")
    
    async def assess_readiness(self):
        """Assess readiness for Phase 12-13"""
        logger.info("🔍 Assessing Readiness for Phase 12-13")
        
        try:
            # Count passed phases
            passed_phases = sum(1 for phase in self.results.values() 
                              if isinstance(phase, dict) and phase.get('status') == 'passed')
            total_phases = 12  # Phases 0-11
            
            readiness_score = passed_phases / total_phases
            
            if readiness_score >= 0.9:  # 90% of phases must pass
                self.results['readiness']['status'] = 'ready'
                self.results['readiness']['details']['readiness_score'] = readiness_score
                self.results['readiness']['details']['passed_phases'] = passed_phases
                self.results['readiness']['details']['total_phases'] = total_phases
                logger.info(f"✅ Ready for Phase 12-13: {passed_phases}/{total_phases} phases passed ({readiness_score:.1%})")
            else:
                self.results['readiness']['status'] = 'not_ready'
                self.results['readiness']['details']['readiness_score'] = readiness_score
                self.results['readiness']['details']['passed_phases'] = passed_phases
                self.results['readiness']['details']['total_phases'] = total_phases
                logger.error(f"❌ Not ready for Phase 12-13: {passed_phases}/{total_phases} phases passed ({readiness_score:.1%})")
            
        except Exception as e:
            self.results['readiness']['status'] = 'failed'
            self.results['readiness']['details']['error'] = str(e)
            logger.error(f"❌ Readiness assessment failed: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_time = time.time() - self.start_time
        
        # Count results
        passed = sum(1 for phase in self.results.values() 
                    if isinstance(phase, dict) and phase.get('status') == 'passed')
        failed = sum(1 for phase in self.results.values() 
                    if isinstance(phase, dict) and phase.get('status') == 'failed')
        pending = sum(1 for phase in self.results.values() 
                     if isinstance(phase, dict) and phase.get('status') == 'pending')
        
        report = {
            'summary': {
                'total_time': total_time,
                'passed': passed,
                'failed': failed,
                'pending': pending,
                'readiness': self.results['readiness']['status']
            },
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_file = Path("logs/comprehensive_validation_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 COMPREHENSIVE VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total Time: {total_time:.2f}s")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⏳ Pending: {pending}")
        logger.info(f"🚀 Readiness: {self.results['readiness']['status']}")
        logger.info("=" * 60)
        
        if self.results['readiness']['status'] == 'ready':
            logger.info("🎉 SYSTEM READY FOR PHASE 12-13! 🎉")
        else:
            logger.info("⚠️  SYSTEM NOT READY - FIX ISSUES FIRST")
        
        return report

async def main():
    """Main validation function"""
    validator = ComprehensiveValidator()
    report = await validator.run_all_validations()
    
    # Exit with appropriate code
    if report['summary']['readiness'] == 'ready':
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

