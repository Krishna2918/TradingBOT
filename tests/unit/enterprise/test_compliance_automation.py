"""
Comprehensive Unit Tests for Complex Compliance Automation

This module contains comprehensive unit tests for all compliance automation
components implemented in Phase 14B: Complex Compliance Automation.

Author: AI Trading System
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from enterprise.compliance.sec_compliance import SECCompliance, SECRule, ComplianceViolation, RuleSeverity, RuleStatus
from enterprise.compliance.risk_enforcement import RiskEnforcer, RiskLimit, RiskViolation, RiskLimitType, RiskSeverity, EnforcementAction
from enterprise.compliance.audit_trail import AuditTrail, AuditEvent, AuditLogger, EventType, EventSeverity, EventStatus
from enterprise.compliance.regulatory_reporting import RegulatoryReporter, ReportGenerator, ReportDefinition, ReportInstance, ReportType, ReportStatus, ReportFormat
from enterprise.compliance.trade_surveillance import TradeSurveillance, SurveillanceAlert, AnomalyDetector, AlertType, AlertSeverity, AlertStatus

class TestSECCompliance:
    """Test cases for SECCompliance class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def sec_compliance(self, temp_db):
        """Create SECCompliance instance for testing."""
        return SECCompliance(db_path=temp_db)
    
    @pytest.fixture
    def sample_trade_data(self):
        """Create sample trade data for testing."""
        return {
            'symbol': 'AAPL',
            'quantity': 1000,
            'price': 150.0,
            'side': 'BUY',
            'order_type': 'MARKET',
            'position_size': 0.06,  # 6% of portfolio
            'trade_amount': 150000,
            'order_value': 150000
        }
    
    def test_sec_compliance_initialization(self, sec_compliance):
        """Test SECCompliance initialization."""
        assert len(sec_compliance.rules) > 0
        assert sec_compliance.compliance_thresholds is not None
        assert RuleSeverity.CRITICAL in sec_compliance.compliance_thresholds
    
    def test_add_rule(self, sec_compliance):
        """Test adding a new SEC rule."""
        new_rule = SECRule(
            rule_id="TEST_RULE_001",
            name="Test Rule",
            description="Test rule for unit testing",
            category="TEST",
            severity=RuleSeverity.MEDIUM,
            status=RuleStatus.ACTIVE,
            effective_date=datetime.now(),
            validation_logic={'test_param': 0.1},
            enforcement_actions=['WARNING'],
            reporting_requirements=['TEST_REPORT']
        )
        
        sec_compliance.add_rule(new_rule)
        assert "TEST_RULE_001" in sec_compliance.rules
        assert sec_compliance.rules["TEST_RULE_001"].name == "Test Rule"
    
    def test_validate_trade_compliant(self, sec_compliance, sample_trade_data):
        """Test trade validation for compliant trade."""
        # Modify trade data to be compliant
        compliant_trade = sample_trade_data.copy()
        compliant_trade['position_size'] = 0.03  # 3% - within limits
        
        is_compliant, violations = sec_compliance.validate_trade(compliant_trade)
        
        assert is_compliant
        assert len(violations) == 0
    
    def test_validate_trade_position_limit_violation(self, sec_compliance, sample_trade_data):
        """Test trade validation for position limit violation."""
        # Modify trade data to violate position limits
        violating_trade = sample_trade_data.copy()
        violating_trade['position_size'] = 0.08  # 8% - exceeds 5% limit
        
        is_compliant, violations = sec_compliance.validate_trade(violating_trade)
        
        assert not is_compliant
        assert len(violations) > 0
        assert any(v.rule_id == "SEC_RULE_001" for v in violations)
    
    def test_validate_trade_wash_sale_violation(self, sec_compliance, sample_trade_data):
        """Test trade validation for wash sale violation."""
        # Mock recent sales data
        with patch.object(sec_compliance, '_get_recent_sales') as mock_sales:
            mock_sales.return_value = [{
                'quantity': 1000,
                'loss': 0.02,  # 2% loss
                'trade_id': 'SALE_001'
            }]
            
            violating_trade = sample_trade_data.copy()
            violating_trade['side'] = 'BUY'
            violating_trade['quantity'] = 1000  # Same quantity as recent sale
            
            is_compliant, violations = sec_compliance.validate_trade(violating_trade)
            
            assert not is_compliant
            assert len(violations) > 0
            assert any(v.rule_id == "SEC_RULE_002" for v in violations)
    
    def test_validate_trade_market_manipulation_violation(self, sec_compliance, sample_trade_data):
        """Test trade validation for market manipulation violation."""
        # Modify trade data to violate manipulation limits
        violating_trade = sample_trade_data.copy()
        violating_trade['order_value'] = 2000000  # $2M - exceeds $1M limit
        
        is_compliant, violations = sec_compliance.validate_trade(violating_trade)
        
        assert not is_compliant
        assert len(violations) > 0
        assert any(v.rule_id == "SEC_RULE_003" for v in violations)
    
    def test_get_compliance_metrics(self, sec_compliance):
        """Test compliance metrics calculation."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        metrics = sec_compliance.get_compliance_metrics(start_date, end_date)
        
        assert 'total_violations' in metrics
        assert 'resolved_violations' in metrics
        assert 'compliance_score' in metrics
        assert 'severity_breakdown' in metrics
        assert 'date_range' in metrics
    
    def test_generate_compliance_report(self, sec_compliance):
        """Test compliance report generation."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        report = sec_compliance.generate_compliance_report(start_date, end_date)
        
        assert 'report_period' in report
        assert 'summary_metrics' in report
        assert 'detailed_violations' in report
        assert 'recommendations' in report
        assert 'generated_at' in report
    
    def test_resolve_violation(self, sec_compliance):
        """Test violation resolution."""
        # Create a test violation
        violation = ComplianceViolation(
            violation_id="TEST_VIOL_001",
            rule_id="SEC_RULE_001",
            rule_name="Test Rule",
            severity=RuleSeverity.MEDIUM,
            violation_type="TEST_VIOLATION",
            description="Test violation",
            affected_entities=["AAPL"],
            violation_data={'test': 'data'},
            detected_at=datetime.now()
        )
        
        sec_compliance.violations.append(violation)
        
        # Resolve the violation
        success = sec_compliance.resolve_violation("TEST_VIOL_001", "Test resolution", "Test action")
        
        assert success
        assert violation.resolved_at is not None
        assert violation.resolution_notes == "Test resolution"
        assert violation.enforcement_action == "Test action"


class TestRiskEnforcement:
    """Test cases for RiskEnforcement class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def risk_enforcer(self, temp_db):
        """Create RiskEnforcer instance for testing."""
        return RiskEnforcer(db_path=temp_db)
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Create sample portfolio data for testing."""
        return {
            'total_value': 1000000,  # $1M portfolio
            'cash': 100000,  # $100K cash
            'positions': [
                {
                    'symbol': 'AAPL',
                    'market_value': 50000,  # 5% position
                    'weight': 0.05,
                    'volatility': 0.25,
                    'beta': 1.2,
                    'sector': 'Technology'
                },
                {
                    'symbol': 'GOOGL',
                    'market_value': 30000,  # 3% position
                    'weight': 0.03,
                    'volatility': 0.30,
                    'beta': 1.1,
                    'sector': 'Technology'
                }
            ]
        }
    
    def test_risk_enforcer_initialization(self, risk_enforcer):
        """Test RiskEnforcer initialization."""
        assert len(risk_enforcer.risk_limits) > 0
        assert risk_enforcer.risk_params is not None
        assert 'confidence_level' in risk_enforcer.risk_params
    
    def test_add_risk_limit(self, risk_enforcer):
        """Test adding a new risk limit."""
        new_limit = RiskLimit(
            limit_id="TEST_LIMIT_001",
            name="Test Risk Limit",
            limit_type=RiskLimitType.POSITION_SIZE,
            value=0.10,  # 10%
            threshold=0.08,  # 8% warning
            severity=RiskSeverity.HIGH,
            enforcement_action=EnforcementAction.REJECT_ORDER,
            description="Test risk limit"
        )
        
        risk_enforcer.add_risk_limit(new_limit)
        assert "TEST_LIMIT_001" in risk_enforcer.risk_limits
        assert risk_enforcer.risk_limits["TEST_LIMIT_001"].name == "Test Risk Limit"
    
    def test_update_portfolio_data(self, risk_enforcer, sample_portfolio_data):
        """Test portfolio data update."""
        risk_enforcer.update_portfolio_data(sample_portfolio_data)
        assert risk_enforcer.portfolio_data == sample_portfolio_data
    
    def test_validate_risk_limits_compliant(self, risk_enforcer, sample_portfolio_data):
        """Test risk limit validation for compliant portfolio."""
        risk_enforcer.update_portfolio_data(sample_portfolio_data)
        
        is_compliant, violations = risk_enforcer.validate_risk_limits()
        
        assert is_compliant
        assert len(violations) == 0
    
    def test_validate_risk_limits_position_violation(self, risk_enforcer, sample_portfolio_data):
        """Test risk limit validation for position size violation."""
        # Modify portfolio to have large position
        violating_portfolio = sample_portfolio_data.copy()
        violating_portfolio['positions'][0]['market_value'] = 100000  # 10% position
        violating_portfolio['positions'][0]['weight'] = 0.10
        
        risk_enforcer.update_portfolio_data(violating_portfolio)
        
        is_compliant, violations = risk_enforcer.validate_risk_limits()
        
        assert not is_compliant
        assert len(violations) > 0
        assert any(v.limit_type == RiskLimitType.POSITION_SIZE for v in violations)
    
    def test_validate_risk_limits_daily_loss_violation(self, risk_enforcer, sample_portfolio_data):
        """Test risk limit validation for daily loss violation."""
        # Add daily loss to portfolio data
        violating_portfolio = sample_portfolio_data.copy()
        violating_portfolio['daily_pnl'] = -25000  # 2.5% loss
        
        risk_enforcer.update_portfolio_data(violating_portfolio)
        
        is_compliant, violations = risk_enforcer.validate_risk_limits()
        
        assert not is_compliant
        assert len(violations) > 0
        assert any(v.limit_type == RiskLimitType.DAILY_LOSS for v in violations)
    
    def test_calculate_risk_metrics(self, risk_enforcer, sample_portfolio_data):
        """Test risk metrics calculation."""
        risk_enforcer.update_portfolio_data(sample_portfolio_data)
        
        metrics = risk_enforcer._calculate_risk_metrics()
        
        assert 'portfolio_value' in metrics
        assert 'position_count' in metrics
        assert 'max_position_size' in metrics
        assert 'concentration_risk' in metrics
        assert 'portfolio_volatility' in metrics
        assert 'portfolio_beta' in metrics
    
    def test_get_risk_metrics_summary(self, risk_enforcer, sample_portfolio_data):
        """Test risk metrics summary."""
        risk_enforcer.update_portfolio_data(sample_portfolio_data)
        
        summary = risk_enforcer.get_risk_metrics_summary()
        
        assert 'portfolio_value' in summary
        assert 'position_count' in summary
        assert 'max_position_size' in summary
        assert 'portfolio_volatility' in summary
        assert 'calculated_at' in summary
    
    def test_execute_enforcement_action(self, risk_enforcer):
        """Test enforcement action execution."""
        # Create a test violation
        violation = RiskViolation(
            violation_id="TEST_VIOL_001",
            limit_id="RISK_LIMIT_001",
            limit_name="Test Limit",
            limit_type=RiskLimitType.POSITION_SIZE,
            current_value=0.08,
            limit_value=0.05,
            excess_amount=0.03,
            severity=RiskSeverity.HIGH,
            enforcement_action=EnforcementAction.WARNING,
            description="Test violation",
            affected_positions=["AAPL"],
            violation_data={'test': 'data'},
            detected_at=datetime.now()
        )
        
        # Execute enforcement action
        success = risk_enforcer.execute_enforcement_action(violation)
        
        assert success
        assert violation.action_taken == EnforcementAction.WARNING.value
    
    def test_resolve_violation(self, risk_enforcer):
        """Test violation resolution."""
        # Create a test violation
        violation = RiskViolation(
            violation_id="TEST_VIOL_001",
            limit_id="RISK_LIMIT_001",
            limit_name="Test Limit",
            limit_type=RiskLimitType.POSITION_SIZE,
            current_value=0.08,
            limit_value=0.05,
            excess_amount=0.03,
            severity=RiskSeverity.HIGH,
            enforcement_action=EnforcementAction.WARNING,
            description="Test violation",
            affected_positions=["AAPL"],
            violation_data={'test': 'data'},
            detected_at=datetime.now()
        )
        
        risk_enforcer.violations.append(violation)
        
        # Resolve the violation
        success = risk_enforcer.resolve_violation("TEST_VIOL_001", "Test resolution")
        
        assert success
        assert violation.resolved_at is not None
        assert violation.resolution_notes == "Test resolution"


class TestAuditTrail:
    """Test cases for AuditTrail class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def audit_trail(self, temp_db):
        """Create AuditTrail instance for testing."""
        return AuditTrail(db_path=temp_db)
    
    @pytest.fixture
    def sample_trade_data(self):
        """Create sample trade data for testing."""
        return {
            'trade_id': 'TRADE_001',
            'symbol': 'AAPL',
            'quantity': 1000,
            'price': 150.0,
            'side': 'BUY',
            'order_type': 'MARKET',
            'execution_time': datetime.now().isoformat(),
            'commission': 1.0,
            'slippage': 0.01
        }
    
    def test_audit_trail_initialization(self, audit_trail):
        """Test AuditTrail initialization."""
        assert audit_trail.logger is not None
        assert isinstance(audit_trail.logger, AuditLogger)
    
    def test_log_trade(self, audit_trail, sample_trade_data):
        """Test trade logging."""
        event_id = audit_trail.log_trade(sample_trade_data, user_id="USER_001")
        
        assert event_id is not None
        assert len(event_id) > 0
    
    def test_log_order(self, audit_trail):
        """Test order logging."""
        order_data = {
            'order_id': 'ORDER_001',
            'symbol': 'AAPL',
            'quantity': 1000,
            'side': 'BUY',
            'order_type': 'LIMIT',
            'limit_price': 150.0,
            'time_in_force': 'DAY'
        }
        
        event_id = audit_trail.log_order(order_data, user_id="USER_001")
        
        assert event_id is not None
        assert len(event_id) > 0
    
    def test_log_risk_violation(self, audit_trail):
        """Test risk violation logging."""
        violation_data = {
            'violation_id': 'RISK_VIOL_001',
            'violation_type': 'POSITION_SIZE_EXCEEDED',
            'limit_type': 'POSITION_SIZE',
            'current_value': 0.08,
            'limit_value': 0.05,
            'excess_amount': 0.03,
            'severity': 'HIGH',
            'enforcement_action': 'WARNING'
        }
        
        event_id = audit_trail.log_risk_violation(violation_data)
        
        assert event_id is not None
        assert len(event_id) > 0
    
    def test_log_compliance_violation(self, audit_trail):
        """Test compliance violation logging."""
        violation_data = {
            'violation_id': 'COMP_VIOL_001',
            'violation_type': 'WASH_SALE_DETECTED',
            'rule_id': 'SEC_RULE_002',
            'rule_name': 'Wash Sale Prevention',
            'severity': 'CRITICAL',
            'affected_entities': ['AAPL'],
            'enforcement_action': 'REJECT_ORDER'
        }
        
        event_id = audit_trail.log_compliance_violation(violation_data)
        
        assert event_id is not None
        assert len(event_id) > 0
    
    def test_log_config_change(self, audit_trail):
        """Test configuration change logging."""
        config_data = {
            'config_id': 'CONFIG_001',
            'config_type': 'RISK_LIMITS',
            'config_section': 'POSITION_LIMITS',
            'old_values': {'max_position': 0.05},
            'new_values': {'max_position': 0.08},
            'change_reason': 'Risk tolerance adjustment'
        }
        
        event_id = audit_trail.log_config_change(config_data, user_id="USER_001")
        
        assert event_id is not None
        assert len(event_id) > 0
    
    def test_log_user_action(self, audit_trail):
        """Test user action logging."""
        action_data = {
            'action': 'LOGIN',
            'target_system': 'TRADING_PLATFORM',
            'entity_type': 'USER',
            'entity_id': 'USER_001',
            'action_type': 'AUTHENTICATION',
            'resource': '/login',
            'permissions': ['READ', 'WRITE'],
            'success': True
        }
        
        event_id = audit_trail.log_user_action(action_data, user_id="USER_001")
        
        assert event_id is not None
        assert len(event_id) > 0
    
    def test_get_trade_history(self, audit_trail, sample_trade_data):
        """Test trade history retrieval."""
        # Log some trades
        audit_trail.log_trade(sample_trade_data, user_id="USER_001")
        
        # Get trade history
        history = audit_trail.get_trade_history(symbol="AAPL")
        
        assert len(history) > 0
        assert all(event.event_type == EventType.TRADE_EXECUTION for event in history)
    
    def test_get_violation_history(self, audit_trail):
        """Test violation history retrieval."""
        # Log some violations
        risk_violation = {
            'violation_id': 'RISK_VIOL_001',
            'violation_type': 'POSITION_SIZE_EXCEEDED',
            'severity': 'HIGH'
        }
        audit_trail.log_risk_violation(risk_violation)
        
        # Get violation history
        history = audit_trail.get_violation_history()
        
        assert len(history) > 0
        assert any(event.event_type == EventType.RISK_VIOLATION for event in history)
    
    def test_generate_compliance_report(self, audit_trail):
        """Test compliance report generation."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        report = audit_trail.generate_compliance_report(start_date, end_date)
        
        assert 'report_period' in report
        assert 'summary' in report
        assert 'event_type_breakdown' in report
        assert 'severity_breakdown' in report
        assert 'generated_at' in report


class TestRegulatoryReporting:
    """Test cases for RegulatoryReporting class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def regulatory_reporter(self, temp_db):
        """Create RegulatoryReporter instance for testing."""
        return RegulatoryReporter(db_path=temp_db)
    
    @pytest.fixture
    def sample_holdings_data(self):
        """Create sample holdings data for testing."""
        return [
            {
                'symbol': 'AAPL',
                'shares': 1000,
                'market_value': 150000,
                'percentage': 0.15
            },
            {
                'symbol': 'GOOGL',
                'shares': 500,
                'market_value': 100000,
                'percentage': 0.10
            }
        ]
    
    def test_regulatory_reporter_initialization(self, regulatory_reporter):
        """Test RegulatoryReporter initialization."""
        assert regulatory_reporter.generator is not None
        assert isinstance(regulatory_reporter.generator, ReportGenerator)
    
    def test_generate_form_13f(self, regulatory_reporter, sample_holdings_data):
        """Test Form 13F report generation."""
        period_start = datetime(2023, 1, 1)
        period_end = datetime(2023, 3, 31)
        
        instance_id = regulatory_reporter.generate_form_13f(
            period_start, period_end, sample_holdings_data
        )
        
        assert instance_id is not None
        assert len(instance_id) > 0
    
    def test_generate_trace_report(self, regulatory_reporter):
        """Test TRACE report generation."""
        trade_date = datetime.now()
        trade_data = [
            {
                'symbol': 'BOND_001',
                'volume': 1000000,
                'price': 100.5,
                'side': 'BUY'
            }
        ]
        
        instance_id = regulatory_reporter.generate_trace_report(trade_date, trade_data)
        
        assert instance_id is not None
        assert len(instance_id) > 0
    
    def test_generate_compliance_report(self, regulatory_reporter):
        """Test compliance report generation."""
        period_start = datetime.now() - timedelta(days=30)
        period_end = datetime.now()
        compliance_data = {
            'violation_count': 5,
            'compliance_score': 95.0,
            'risk_violations': [],
            'sec_violations': [],
            'recommendations': ['Improve monitoring']
        }
        
        instance_id = regulatory_reporter.generate_compliance_report(
            period_start, period_end, compliance_data
        )
        
        assert instance_id is not None
        assert len(instance_id) > 0
    
    def test_generate_risk_report(self, regulatory_reporter):
        """Test risk report generation."""
        report_date = datetime.now()
        risk_data = {
            'portfolio_value': 1000000,
            'var_95': 25000,
            'var_99': 35000,
            'max_drawdown': -0.05,
            'sharpe_ratio': 1.5,
            'beta': 1.2,
            'volatility': 0.20,
            'leverage': 1.1,
            'violations': []
        }
        
        instance_id = regulatory_reporter.generate_risk_report(report_date, risk_data)
        
        assert instance_id is not None
        assert len(instance_id) > 0
    
    def test_submit_report(self, regulatory_reporter):
        """Test report submission."""
        # First create and approve a report
        period_start = datetime.now() - timedelta(days=30)
        period_end = datetime.now()
        compliance_data = {'violation_count': 0, 'compliance_score': 100.0}
        
        instance_id = regulatory_reporter.generate_compliance_report(
            period_start, period_end, compliance_data
        )
        
        # Approve the report
        approval_success = regulatory_reporter.approve_report(instance_id, "Test approval")
        
        if approval_success:
            # Submit the report
            submission_id = regulatory_reporter.submit_report(instance_id, "Test submission")
            
            assert submission_id is not None
            assert len(submission_id) > 0
    
    def test_get_due_reports(self, regulatory_reporter):
        """Test getting due reports."""
        due_reports = regulatory_reporter.get_due_reports(days_ahead=7)
        
        assert isinstance(due_reports, list)
    
    def test_get_compliance_summary(self, regulatory_reporter):
        """Test compliance summary generation."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        summary = regulatory_reporter.get_compliance_summary(start_date, end_date)
        
        assert 'summary' in summary
        assert 'by_report_type' in summary
        assert 'period' in summary
        assert 'generated_at' in summary


class TestTradeSurveillance:
    """Test cases for TradeSurveillance class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def trade_surveillance(self, temp_db):
        """Create TradeSurveillance instance for testing."""
        return TradeSurveillance(db_path=temp_db)
    
    @pytest.fixture
    def sample_trade_data(self):
        """Create sample trade data for testing."""
        return {
            'trade_id': 'TRADE_001',
            'symbol': 'AAPL',
            'timestamp': datetime.now().isoformat(),
            'price': 150.0,
            'quantity': 1000,
            'side': 'BUY',
            'order_type': 'MARKET',
            'execution_venue': 'NASDAQ',
            'market_impact': 0.001
        }
    
    @pytest.fixture
    def sample_order_data(self):
        """Create sample order data for testing."""
        return {
            'order_id': 'ORDER_001',
            'symbol': 'AAPL',
            'timestamp': datetime.now().isoformat(),
            'price': 150.0,
            'quantity': 1000,
            'side': 'BUY',
            'order_type': 'LIMIT',
            'time_in_force': 'DAY',
            'status': 'NEW',
            'venue': 'NASDAQ'
        }
    
    def test_trade_surveillance_initialization(self, trade_surveillance):
        """Test TradeSurveillance initialization."""
        assert len(trade_surveillance.anomaly_detectors) > 0
        assert trade_surveillance.surveillance_params is not None
        assert 'wash_sale_window' in trade_surveillance.surveillance_params
    
    def test_add_anomaly_detector(self, trade_surveillance):
        """Test adding a new anomaly detector."""
        new_detector = AnomalyDetector(
            detector_id="TEST_DETECTOR_001",
            name="Test Detector",
            alert_type=AlertType.ANOMALOUS_PATTERN,
            parameters={'test_param': 0.1},
            thresholds={'confidence_threshold': 0.8}
        )
        
        trade_surveillance.add_anomaly_detector(new_detector)
        assert "TEST_DETECTOR_001" in trade_surveillance.anomaly_detectors
        assert trade_surveillance.anomaly_detectors["TEST_DETECTOR_001"].name == "Test Detector"
    
    def test_process_trade(self, trade_surveillance, sample_trade_data):
        """Test trade processing for surveillance."""
        alerts = trade_surveillance.process_trade(sample_trade_data)
        
        assert isinstance(alerts, list)
        # Alerts may or may not be generated depending on the data
    
    def test_process_order(self, trade_surveillance, sample_order_data):
        """Test order processing for surveillance."""
        alerts = trade_surveillance.process_order(sample_order_data)
        
        assert isinstance(alerts, list)
        # Alerts may or may not be generated depending on the data
    
    def test_detect_wash_sales(self, trade_surveillance):
        """Test wash sale detection."""
        # Mock recent sales data
        with patch.object(trade_surveillance, '_get_recent_sales') as mock_sales:
            mock_sales.return_value = [{
                'quantity': 1000,
                'loss': 0.02,
                'trade_id': 'SALE_001'
            }]
            
            detector = trade_surveillance.anomaly_detectors["DETECTOR_001"]
            trade_data = {
                'symbol': 'AAPL',
                'side': 'BUY',
                'quantity': 1000,
                'trade_id': 'TRADE_001'
            }
            
            alerts = trade_surveillance._detect_wash_sales(detector, trade_data)
            
            assert isinstance(alerts, list)
    
    def test_detect_spoofing(self, trade_surveillance):
        """Test spoofing detection."""
        # Mock recent orders data
        with patch.object(trade_surveillance, '_get_recent_orders') as mock_orders:
            mock_orders.return_value = [
                {'order_id': 'ORDER_001', 'status': 'CANCELLED', 'price': 150.0},
                {'order_id': 'ORDER_002', 'status': 'CANCELLED', 'price': 150.1},
                {'order_id': 'ORDER_003', 'status': 'CANCELLED', 'price': 150.2}
            ]
            
            detector = trade_surveillance.anomaly_detectors["DETECTOR_002"]
            order_data = {
                'symbol': 'AAPL',
                'order_type': 'LIMIT',
                'order_id': 'ORDER_004'
            }
            
            alerts = trade_surveillance._detect_spoofing(detector, order_data)
            
            assert isinstance(alerts, list)
    
    def test_detect_unusual_volume(self, trade_surveillance):
        """Test unusual volume detection."""
        # Mock historical volumes
        with patch.object(trade_surveillance, '_get_historical_volumes') as mock_volumes:
            mock_volumes.return_value = [1000, 1200, 1100, 1300, 1000, 1200, 1100, 1300, 1000, 1200]
            
            detector = trade_surveillance.anomaly_detectors["DETECTOR_004"]
            trade_data = {
                'symbol': 'AAPL',
                'quantity': 5000  # High volume
            }
            
            alerts = trade_surveillance._detect_unusual_volume(detector, trade_data)
            
            assert isinstance(alerts, list)
    
    def test_get_alerts(self, trade_surveillance):
        """Test alert retrieval."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        alerts = trade_surveillance.get_alerts(start_date=start_date, end_date=end_date)
        
        assert isinstance(alerts, list)
    
    def test_update_alert_status(self, trade_surveillance):
        """Test alert status update."""
        # Create a test alert
        alert = SurveillanceAlert(
            alert_id="TEST_ALERT_001",
            alert_type=AlertType.UNUSUAL_VOLUME,
            severity=AlertSeverity.MEDIUM,
            status=AlertStatus.NEW,
            timestamp=datetime.now(),
            symbol="AAPL",
            description="Test alert",
            confidence_score=0.8,
            suspicious_activities=[],
            affected_orders=[],
            affected_trades=[],
            market_impact={},
            regulatory_relevance=False
        )
        
        trade_surveillance.alerts.append(alert)
        
        # Update alert status
        success = trade_surveillance.update_alert_status(
            "TEST_ALERT_001", AlertStatus.INVESTIGATING, "Under investigation"
        )
        
        assert success
    
    def test_get_surveillance_summary(self, trade_surveillance):
        """Test surveillance summary generation."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        summary = trade_surveillance.get_surveillance_summary(start_date, end_date)
        
        assert 'summary' in summary
        assert 'alert_type_breakdown' in summary
        assert 'severity_breakdown' in summary
        assert 'status_breakdown' in summary
        assert 'period' in summary
        assert 'generated_at' in summary


class TestComplianceIntegration:
    """Integration tests for all compliance automation components."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    def test_compliance_workflow(self, temp_db):
        """Test complete compliance workflow."""
        # Initialize all compliance components
        sec_compliance = SECCompliance(db_path=temp_db)
        risk_enforcer = RiskEnforcer(db_path=temp_db)
        audit_trail = AuditTrail(db_path=temp_db)
        regulatory_reporter = RegulatoryReporter(db_path=temp_db)
        trade_surveillance = TradeSurveillance(db_path=temp_db)
        
        # Test trade data
        trade_data = {
            'symbol': 'AAPL',
            'quantity': 1000,
            'price': 150.0,
            'side': 'BUY',
            'order_type': 'MARKET',
            'position_size': 0.06,  # 6% - violates position limit
            'trade_amount': 150000,
            'order_value': 150000,
            'trade_id': 'TRADE_001',
            'order_id': 'ORDER_001'
        }
        
        # 1. SEC Compliance Check
        is_sec_compliant, sec_violations = sec_compliance.validate_trade(trade_data)
        
        # 2. Risk Enforcement Check
        portfolio_data = {
            'total_value': 1000000,
            'cash': 100000,
            'positions': [{'symbol': 'AAPL', 'market_value': 60000, 'weight': 0.06}]
        }
        risk_enforcer.update_portfolio_data(portfolio_data)
        is_risk_compliant, risk_violations = risk_enforcer.validate_risk_limits()
        
        # 3. Trade Surveillance
        surveillance_alerts = trade_surveillance.process_trade(trade_data)
        
        # 4. Audit Logging
        audit_trail.log_trade(trade_data, user_id="USER_001")
        
        if sec_violations:
            for violation in sec_violations:
                audit_trail.log_compliance_violation({
                    'violation_id': violation.violation_id,
                    'violation_type': violation.violation_type,
                    'rule_id': violation.rule_id,
                    'severity': violation.severity.value
                })
        
        if risk_violations:
            for violation in risk_violations:
                audit_trail.log_risk_violation({
                    'violation_id': violation.violation_id,
                    'violation_type': violation.limit_type.value,
                    'severity': violation.severity.value
                })
        
        # 5. Generate Compliance Report
        period_start = datetime.now() - timedelta(days=30)
        period_end = datetime.now()
        
        compliance_data = {
            'violation_count': len(sec_violations) + len(risk_violations),
            'compliance_score': 95.0 if is_sec_compliant and is_risk_compliant else 85.0,
            'sec_violations': [{'type': v.violation_type, 'severity': v.severity.value} for v in sec_violations],
            'risk_violations': [{'type': v.limit_type.value, 'severity': v.severity.value} for v in risk_violations],
            'recommendations': ['Review position limits', 'Improve risk monitoring']
        }
        
        report_instance_id = regulatory_reporter.generate_compliance_report(
            period_start, period_end, compliance_data
        )
        
        # Verify all components are working
        assert isinstance(is_sec_compliant, bool)
        assert isinstance(is_risk_compliant, bool)
        assert isinstance(surveillance_alerts, list)
        assert report_instance_id is not None
        
        # Verify audit trail
        trade_history = audit_trail.get_trade_history(symbol="AAPL")
        assert len(trade_history) > 0
        
        violation_history = audit_trail.get_violation_history()
        assert len(violation_history) >= len(sec_violations) + len(risk_violations)
    
    def test_compliance_metrics_integration(self, temp_db):
        """Test compliance metrics integration across components."""
        # Initialize components
        sec_compliance = SECCompliance(db_path=temp_db)
        risk_enforcer = RiskEnforcer(db_path=temp_db)
        audit_trail = AuditTrail(db_path=temp_db)
        regulatory_reporter = RegulatoryReporter(db_path=temp_db)
        trade_surveillance = TradeSurveillance(db_path=temp_db)
        
        # Generate some test data
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Get metrics from each component
        sec_metrics = sec_compliance.get_compliance_metrics(start_date, end_date)
        risk_summary = risk_enforcer.get_violations_summary(start_date, end_date)
        audit_report = audit_trail.generate_compliance_report(start_date, end_date)
        surveillance_summary = trade_surveillance.get_surveillance_summary(start_date, end_date)
        reporting_summary = regulatory_reporter.get_compliance_summary(start_date, end_date)
        
        # Verify all metrics are generated
        assert 'total_violations' in sec_metrics
        assert 'total_violations' in risk_summary
        assert 'summary' in audit_report
        assert 'summary' in surveillance_summary
        assert 'summary' in reporting_summary
        
        # Verify metrics structure
        assert 'compliance_score' in sec_metrics
        assert 'severity_breakdown' in sec_metrics
        assert 'severity_breakdown' in risk_summary
        assert 'event_type_breakdown' in audit_report
        assert 'alert_type_breakdown' in surveillance_summary
        assert 'by_report_type' in reporting_summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
