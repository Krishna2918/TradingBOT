"""
Comprehensive Unit Tests for Professional Penetration Testing Framework

This module contains comprehensive unit tests for the penetration testing,
vulnerability scanning, security auditing, threat modeling, security monitoring,
and cryptographic validation systems.

Author: AI Trading System
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import sqlite3
import tempfile
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from enterprise.security.penetration_testing import (
    PenetrationTester, SecurityAssessment, VulnerabilityScan, ThreatModel,
    AssessmentType, SeverityLevel, AssessmentStatus
)
from enterprise.security.vulnerability_scanner import (
    VulnerabilityScanner, Vulnerability, ScanType, ScanStatus
)
from enterprise.security.security_auditor import (
    SecurityAuditor, SecurityAudit, AuditFinding, ComplianceCheck,
    AuditType, ComplianceFramework, AuditStatus, FindingSeverity
)
from enterprise.security.threat_modeling import (
    ThreatModeler, Threat, SecurityControl, RiskAssessment,
    ThreatCategory, AttackVector, SecurityControl, RiskLevel
)
from enterprise.security.security_monitoring import (
    SecurityMonitor, SecurityEvent, ThreatIntelligence, IncidentResponse,
    SecurityEventType, ThreatLevel, IncidentStatus, ThreatIntelligenceType
)
from enterprise.security.crypto_validation import (
    CryptoValidator, EncryptionCheck, KeyManagement, CertificateValidation,
    EncryptionAlgorithm, KeyType, CertificateStatus, ValidationResult
)

class TestPenetrationTesting:
    """Test cases for penetration testing framework."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def penetration_tester(self, temp_db):
        """Create penetration tester instance."""
        return PenetrationTester(db_path=temp_db)
    
    def test_penetration_tester_initialization(self, penetration_tester):
        """Test penetration tester initialization."""
        assert penetration_tester is not None
        assert penetration_tester.db_path is not None
        assert len(penetration_tester.assessments) == 0
        assert len(penetration_tester.vulnerabilities) == 0
        assert len(penetration_tester.scan_results) == 0
    
    def test_create_assessment(self, penetration_tester):
        """Test creating a security assessment."""
        assessment_id = penetration_tester.create_assessment(
            name="Test Assessment",
            assessment_type=AssessmentType.WEB_APPLICATION,
            target="example.com",
            scope=["web application", "API"],
            objectives=["identify vulnerabilities", "test security controls"]
        )
        
        assert assessment_id is not None
        assert assessment_id.startswith("ASSESS_")
        assert assessment_id in penetration_tester.assessments
        
        assessment = penetration_tester.assessments[assessment_id]
        assert assessment.name == "Test Assessment"
        assert assessment.assessment_type == AssessmentType.WEB_APPLICATION
        assert assessment.target == "example.com"
        assert assessment.status == AssessmentStatus.PLANNED
    
    def test_start_assessment(self, penetration_tester):
        """Test starting a security assessment."""
        assessment_id = penetration_tester.create_assessment(
            name="Test Assessment",
            assessment_type=AssessmentType.WEB_APPLICATION,
            target="example.com",
            scope=["web application"],
            objectives=["test security"]
        )
        
        result = penetration_tester.start_assessment(assessment_id)
        assert result is True
        
        assessment = penetration_tester.assessments[assessment_id]
        assert assessment.status == AssessmentStatus.IN_PROGRESS
    
    @patch('socket.socket')
    def test_run_vulnerability_scan(self, mock_socket, penetration_tester):
        """Test running a vulnerability scan."""
        # Mock socket connection
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 0  # Port is open
        mock_socket.return_value = mock_sock
        
        scan_id = penetration_tester.run_vulnerability_scan("example.com", "comprehensive")
        
        assert scan_id is not None
        assert scan_id.startswith("SCAN_")
        assert len(penetration_tester.scan_results) == 1
        
        scan = penetration_tester.scan_results[0]
        assert scan.target == "example.com"
        assert scan.scan_type == "comprehensive"
        assert scan.status == AssessmentStatus.COMPLETED
    
    def test_create_threat_model(self, penetration_tester):
        """Test creating a threat model."""
        architecture = {
            "components": ["web server", "database", "API"],
            "data_flows": ["user -> web server", "web server -> database"]
        }
        
        model_id = penetration_tester.create_threat_model(
            application_name="Test App",
            version="1.0.0",
            architecture=architecture
        )
        
        assert model_id is not None
        assert model_id.startswith("THREAT_MODEL_")
    
    def test_analyze_threats(self, penetration_tester):
        """Test threat analysis."""
        architecture = {
            "components": ["web server", "database"],
            "data_flows": ["user -> web server"]
        }
        
        model_id = penetration_tester.create_threat_model(
            application_name="Test App",
            version="1.0.0",
            architecture=architecture
        )
        
        threats = penetration_tester.analyze_threats(model_id)
        
        assert len(threats) == 6  # STRIDE threats
        assert all(threat['category'] in ['Spoofing', 'Tampering', 'Repudiation', 
                                        'Information Disclosure', 'Denial of Service', 
                                        'Elevation of Privilege'] for threat in threats)
    
    def test_generate_security_report(self, penetration_tester):
        """Test generating security report."""
        assessment_id = penetration_tester.create_assessment(
            name="Test Assessment",
            assessment_type=AssessmentType.WEB_APPLICATION,
            target="example.com",
            scope=["web application"],
            objectives=["test security"]
        )
        
        # Add some findings
        assessment = penetration_tester.assessments[assessment_id]
        assessment.findings = [
            Mock(severity=SeverityLevel.HIGH, __dict__={'severity': SeverityLevel.HIGH}),
            Mock(severity=SeverityLevel.MEDIUM, __dict__={'severity': SeverityLevel.MEDIUM})
        ]
        
        report = penetration_tester.generate_security_report(assessment_id)
        
        assert report is not None
        assert 'assessment_id' in report
        assert 'risk_score' in report
        assert 'findings' in report
        assert 'recommendations' in report

class TestVulnerabilityScanner:
    """Test cases for vulnerability scanner."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def vulnerability_scanner(self, temp_db):
        """Create vulnerability scanner instance."""
        return VulnerabilityScanner(db_path=temp_db)
    
    def test_vulnerability_scanner_initialization(self, vulnerability_scanner):
        """Test vulnerability scanner initialization."""
        assert vulnerability_scanner is not None
        assert vulnerability_scanner.db_path is not None
        assert len(vulnerability_scanner.scan_results) == 0
        assert len(vulnerability_scanner.vulnerability_signatures) > 0
    
    @patch('socket.socket')
    def test_run_network_scan(self, mock_socket, vulnerability_scanner):
        """Test running network vulnerability scan."""
        # Mock socket connection
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 0  # Port is open
        mock_socket.return_value = mock_sock
        
        scan_id = vulnerability_scanner.run_scan("example.com", ScanType.NETWORK_SCAN)
        
        assert scan_id is not None
        assert scan_id.startswith("SCAN_")
        assert len(vulnerability_scanner.scan_results) == 1
        
        scan = vulnerability_scanner.scan_results[0]
        assert scan.target == "example.com"
        assert scan.scan_type == ScanType.NETWORK_SCAN
        assert scan.status == ScanStatus.COMPLETED
    
    @patch('requests.get')
    def test_run_web_application_scan(self, mock_get, vulnerability_scanner):
        """Test running web application vulnerability scan."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        scan_id = vulnerability_scanner.run_scan("example.com", ScanType.WEB_APPLICATION_SCAN)
        
        assert scan_id is not None
        assert len(vulnerability_scanner.scan_results) == 1
        
        scan = vulnerability_scanner.scan_results[0]
        assert scan.target == "example.com"
        assert scan.scan_type == ScanType.WEB_APPLICATION_SCAN
    
    def test_get_scan_results(self, vulnerability_scanner):
        """Test getting scan results."""
        # Add some test scan results
        scan_result = Mock()
        scan_result.scan_id = "test_scan_1"
        scan_result.target = "example.com"
        scan_result.scan_type = ScanType.NETWORK_SCAN
        vulnerability_scanner.scan_results = [scan_result]
        
        results = vulnerability_scanner.get_scan_results(target="example.com")
        assert len(results) == 1
        assert results[0].target == "example.com"
    
    def test_get_vulnerabilities(self, vulnerability_scanner):
        """Test getting vulnerabilities."""
        vulnerabilities = vulnerability_scanner.get_vulnerabilities(severity=SeverityLevel.HIGH)
        assert isinstance(vulnerabilities, list)
    
    def test_mark_vulnerability_verified(self, vulnerability_scanner):
        """Test marking vulnerability as verified."""
        # This would require a vulnerability to exist in the database
        result = vulnerability_scanner.mark_vulnerability_verified("test_vuln_id", True)
        assert isinstance(result, bool)
    
    def test_generate_vulnerability_report(self, vulnerability_scanner):
        """Test generating vulnerability report."""
        report = vulnerability_scanner.generate_vulnerability_report()
        
        assert report is not None
        assert 'summary' in report
        assert 'severity_breakdown' in report
        assert 'category_breakdown' in report

class TestSecurityAuditor:
    """Test cases for security auditor."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def security_auditor(self, temp_db):
        """Create security auditor instance."""
        return SecurityAuditor(db_path=temp_db)
    
    def test_security_auditor_initialization(self, security_auditor):
        """Test security auditor initialization."""
        assert security_auditor is not None
        assert security_auditor.db_path is not None
        assert len(security_auditor.audits) == 0
        assert len(security_auditor.compliance_checks) > 0
    
    def test_add_compliance_check(self, security_auditor):
        """Test adding compliance check."""
        check = ComplianceCheck(
            check_id="TEST_CHECK_001",
            name="Test Check",
            description="Test compliance check",
            framework=ComplianceFramework.ISO_27001,
            requirement="A.9.1.1",
            check_type="Access Control",
            automated=True,
            check_script="test_check()",
            expected_result="Test passed",
            severity=FindingSeverity.MEDIUM
        )
        
        security_auditor.add_compliance_check(check)
        assert check.check_id in security_auditor.compliance_checks
    
    def test_create_audit(self, security_auditor):
        """Test creating security audit."""
        audit_id = security_auditor.create_audit(
            name="Test Audit",
            audit_type=AuditType.COMPLIANCE_AUDIT,
            framework=ComplianceFramework.ISO_27001,
            scope=["system1", "system2"],
            objectives=["compliance check", "security assessment"],
            auditor="Test Auditor"
        )
        
        assert audit_id is not None
        assert audit_id.startswith("AUDIT_")
        assert audit_id in security_auditor.audits
        
        audit = security_auditor.audits[audit_id]
        assert audit.name == "Test Audit"
        assert audit.audit_type == AuditType.COMPLIANCE_AUDIT
        assert audit.framework == ComplianceFramework.ISO_27001
    
    def test_start_audit(self, security_auditor):
        """Test starting security audit."""
        audit_id = security_auditor.create_audit(
            name="Test Audit",
            audit_type=AuditType.COMPLIANCE_AUDIT,
            framework=ComplianceFramework.ISO_27001,
            scope=["system1"],
            objectives=["test"]
        )
        
        result = security_auditor.start_audit(audit_id)
        assert result is True
        
        audit = security_auditor.audits[audit_id]
        assert audit.status == AuditStatus.IN_PROGRESS
    
    def test_run_compliance_audit(self, security_auditor):
        """Test running compliance audit."""
        audit_id = security_auditor.create_audit(
            name="Test Audit",
            audit_type=AuditType.COMPLIANCE_AUDIT,
            framework=ComplianceFramework.ISO_27001,
            scope=["system1"],
            objectives=["test"]
        )
        
        findings = security_auditor.run_compliance_audit(audit_id)
        
        assert isinstance(findings, list)
        assert len(findings) > 0  # Should have some default findings
        
        audit = security_auditor.audits[audit_id]
        assert len(audit.findings) > 0
        assert audit.compliance_score >= 0.0
    
    def test_complete_audit(self, security_auditor):
        """Test completing security audit."""
        audit_id = security_auditor.create_audit(
            name="Test Audit",
            audit_type=AuditType.COMPLIANCE_AUDIT,
            framework=ComplianceFramework.ISO_27001,
            scope=["system1"],
            objectives=["test"]
        )
        
        result = security_auditor.complete_audit(audit_id)
        assert result is True
        
        audit = security_auditor.audits[audit_id]
        assert audit.status == AuditStatus.COMPLETED
        assert audit.end_date is not None
    
    def test_generate_audit_report(self, security_auditor):
        """Test generating audit report."""
        audit_id = security_auditor.create_audit(
            name="Test Audit",
            audit_type=AuditType.COMPLIANCE_AUDIT,
            framework=ComplianceFramework.ISO_27001,
            scope=["system1"],
            objectives=["test"]
        )
        
        # Add some findings
        audit = security_auditor.audits[audit_id]
        audit.findings = [
            Mock(severity=FindingSeverity.HIGH, __dict__={'severity': FindingSeverity.HIGH}),
            Mock(severity=FindingSeverity.MEDIUM, __dict__={'severity': FindingSeverity.MEDIUM})
        ]
        
        report = security_auditor.generate_audit_report(audit_id)
        
        assert report is not None
        assert 'audit_id' in report
        assert 'compliance_score' in report
        assert 'findings' in report
        assert 'recommendations' in report

class TestThreatModeling:
    """Test cases for threat modeling system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def threat_modeler(self, temp_db):
        """Create threat modeler instance."""
        return ThreatModeler(db_path=temp_db)
    
    def test_threat_modeler_initialization(self, threat_modeler):
        """Test threat modeler initialization."""
        assert threat_modeler is not None
        assert threat_modeler.db_path is not None
        assert len(threat_modeler.threat_models) == 0
        assert len(threat_modeler.threat_library) > 0
        assert len(threat_modeler.security_controls) > 0
    
    def test_create_threat_model(self, threat_modeler):
        """Test creating threat model."""
        architecture = {
            "components": ["web server", "database", "API"],
            "data_flows": ["user -> web server", "web server -> database"]
        }
        
        model_id = threat_modeler.create_threat_model(
            application_name="Test App",
            version="1.0.0",
            description="Test application",
            architecture=architecture
        )
        
        assert model_id is not None
        assert model_id.startswith("TM_")
        assert model_id in threat_modeler.threat_models
        
        model = threat_modeler.threat_models[model_id]
        assert model.application_name == "Test App"
        assert model.version == "1.0.0"
        assert model.architecture == architecture
    
    def test_add_data_flow(self, threat_modeler):
        """Test adding data flow to threat model."""
        architecture = {"components": ["web server"]}
        model_id = threat_modeler.create_threat_model(
            application_name="Test App",
            version="1.0.0",
            description="Test app",
            architecture=architecture
        )
        
        data_flow = {
            "flow_id": "flow_001",
            "source": "user",
            "destination": "web server",
            "data_type": "authentication",
            "protocol": "https"
        }
        
        result = threat_modeler.add_data_flow(model_id, data_flow)
        assert result is True
        
        model = threat_modeler.threat_models[model_id]
        assert len(model.data_flows) == 1
        assert model.data_flows[0] == data_flow
    
    def test_add_trust_boundary(self, threat_modeler):
        """Test adding trust boundary to threat model."""
        architecture = {"components": ["web server"]}
        model_id = threat_modeler.create_threat_model(
            application_name="Test App",
            version="1.0.0",
            description="Test app",
            architecture=architecture
        )
        
        trust_boundary = {
            "boundary_id": "boundary_001",
            "name": "Internet Boundary",
            "components": ["web server"],
            "trust_level": "untrusted"
        }
        
        result = threat_modeler.add_trust_boundary(model_id, trust_boundary)
        assert result is True
        
        model = threat_modeler.threat_models[model_id]
        assert len(model.trust_boundaries) == 1
        assert model.trust_boundaries[0] == trust_boundary
    
    def test_analyze_threats(self, threat_modeler):
        """Test analyzing threats."""
        architecture = {"components": ["web server", "database"]}
        model_id = threat_modeler.create_threat_model(
            application_name="Test App",
            version="1.0.0",
            description="Test app",
            architecture=architecture
        )
        
        # Add data flow
        data_flow = {
            "flow_id": "flow_001",
            "source": "user",
            "destination": "web server",
            "data_type": "authentication",
            "protocol": "https"
        }
        threat_modeler.add_data_flow(model_id, data_flow)
        
        threats = threat_modeler.analyze_threats(model_id)
        
        assert isinstance(threats, list)
        assert len(threats) > 0  # Should identify some threats
        
        model = threat_modeler.threat_models[model_id]
        assert len(model.threats) > 0
        assert len(model.risk_assessments) > 0
        assert len(model.security_controls) > 0
    
    def test_generate_threat_model_report(self, threat_modeler):
        """Test generating threat model report."""
        architecture = {"components": ["web server"]}
        model_id = threat_modeler.create_threat_model(
            application_name="Test App",
            version="1.0.0",
            description="Test app",
            architecture=architecture
        )
        
        # Add some threats
        model = threat_modeler.threat_models[model_id]
        model.threats = [
            Mock(risk_score=0.8, __dict__={'risk_score': 0.8}),
            Mock(risk_score=0.6, __dict__={'risk_score': 0.6})
        ]
        
        report = threat_modeler.generate_threat_model_report(model_id)
        
        assert report is not None
        assert 'model_id' in report
        assert 'summary' in report
        assert 'threats' in report
        assert 'security_controls' in report
        assert 'risk_assessments' in report

class TestSecurityMonitoring:
    """Test cases for security monitoring system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def security_monitor(self, temp_db):
        """Create security monitor instance."""
        return SecurityMonitor(db_path=temp_db)
    
    def test_security_monitor_initialization(self, security_monitor):
        """Test security monitor initialization."""
        assert security_monitor is not None
        assert security_monitor.db_path is not None
        assert len(security_monitor.security_events) == 0
        assert len(security_monitor.threat_intelligence) > 0
        assert len(security_monitor.incidents) == 0
        assert len(security_monitor.detection_rules) > 0
    
    def test_process_security_event(self, security_monitor):
        """Test processing security event."""
        event_data = {
            'event_type': 'AUTHENTICATION_FAILURE',
            'threat_level': 'MEDIUM',
            'timestamp': datetime.now().isoformat(),
            'source_ip': '192.168.1.100',
            'source_user': 'testuser',
            'target_system': 'web server',
            'description': 'Failed login attempt',
            'indicators': ['failed_login'],
            'context': {'attempts': 3}
        }
        
        event = security_monitor.process_security_event(event_data)
        
        assert event is not None
        assert event.event_type == SecurityEventType.AUTHENTICATION_FAILURE
        assert event.threat_level == ThreatLevel.MEDIUM
        assert event.source_ip == '192.168.1.100'
        assert len(security_monitor.security_events) == 1
    
    def test_add_threat_intelligence(self, security_monitor):
        """Test adding threat intelligence."""
        ti_data = {
            'indicator_type': 'IP_ADDRESS',
            'indicator_value': '192.168.1.200',
            'threat_level': 'HIGH',
            'confidence': 0.8,
            'source': 'Internal Honeypot',
            'description': 'Known malicious IP',
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'tags': ['malware', 'botnet'],
            'metadata': {'country': 'Unknown'}
        }
        
        ti_id = security_monitor.add_threat_intelligence(ti_data)
        
        assert ti_id is not None
        assert ti_id.startswith("TI_")
        assert ti_id in security_monitor.threat_intelligence
        
        ti = security_monitor.threat_intelligence[ti_id]
        assert ti.indicator_value == '192.168.1.200'
        assert ti.threat_level == ThreatLevel.HIGH
    
    def test_update_incident_status(self, security_monitor):
        """Test updating incident status."""
        # Create an incident first
        incident = IncidentResponse(
            incident_id="test_incident_001",
            title="Test Incident",
            description="Test security incident",
            threat_level=ThreatLevel.HIGH,
            status=IncidentStatus.NEW,
            assigned_to=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            resolved_at=None,
            affected_systems=["system1"],
            indicators=["indicator1"],
            response_actions=[],
            lessons_learned=[]
        )
        security_monitor.incidents.append(incident)
        
        result = security_monitor.update_incident_status(
            "test_incident_001",
            IncidentStatus.IN_PROGRESS,
            assigned_to="analyst1",
            response_actions=["investigation_started"]
        )
        
        assert result is True
        assert incident.status == IncidentStatus.IN_PROGRESS
        assert incident.assigned_to == "analyst1"
        assert len(incident.response_actions) == 1
    
    def test_get_security_events(self, security_monitor):
        """Test getting security events with filters."""
        # Add some test events
        event1 = SecurityEvent(
            event_id="event_001",
            event_type=SecurityEventType.AUTHENTICATION_FAILURE,
            threat_level=ThreatLevel.HIGH,
            timestamp=datetime.now(),
            source_ip="192.168.1.100",
            source_user="user1",
            target_system="system1",
            description="Failed login",
            raw_data={},
            indicators=[],
            context={}
        )
        event2 = SecurityEvent(
            event_id="event_002",
            event_type=SecurityEventType.NETWORK_ANOMALY,
            threat_level=ThreatLevel.MEDIUM,
            timestamp=datetime.now(),
            source_ip="192.168.1.200",
            source_user=None,
            target_system="system2",
            description="Network anomaly",
            raw_data={},
            indicators=[],
            context={}
        )
        security_monitor.security_events = [event1, event2]
        
        # Test filtering by event type
        auth_events = security_monitor.get_security_events(
            event_type=SecurityEventType.AUTHENTICATION_FAILURE
        )
        assert len(auth_events) == 1
        assert auth_events[0].event_type == SecurityEventType.AUTHENTICATION_FAILURE
        
        # Test filtering by threat level
        high_threat_events = security_monitor.get_security_events(
            threat_level=ThreatLevel.HIGH
        )
        assert len(high_threat_events) == 1
        assert high_threat_events[0].threat_level == ThreatLevel.HIGH
    
    def test_generate_security_report(self, security_monitor):
        """Test generating security report."""
        # Add some test data
        event = SecurityEvent(
            event_id="event_001",
            event_type=SecurityEventType.AUTHENTICATION_FAILURE,
            threat_level=ThreatLevel.HIGH,
            timestamp=datetime.now(),
            source_ip="192.168.1.100",
            source_user="user1",
            target_system="system1",
            description="Failed login",
            raw_data={},
            indicators=[],
            context={}
        )
        security_monitor.security_events = [event]
        
        incident = IncidentResponse(
            incident_id="incident_001",
            title="Test Incident",
            description="Test incident",
            threat_level=ThreatLevel.HIGH,
            status=IncidentStatus.NEW,
            assigned_to=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            resolved_at=None,
            affected_systems=["system1"],
            indicators=["indicator1"],
            response_actions=[],
            lessons_learned=[]
        )
        security_monitor.incidents = [incident]
        
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        report = security_monitor.generate_security_report(start_date, end_date)
        
        assert report is not None
        assert 'summary' in report
        assert 'event_statistics' in report
        assert 'incident_statistics' in report
        assert 'top_events' in report
        assert 'active_incidents' in report

class TestCryptoValidation:
    """Test cases for cryptographic validation system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def crypto_validator(self, temp_db):
        """Create crypto validator instance."""
        return CryptoValidator(db_path=temp_db)
    
    def test_crypto_validator_initialization(self, crypto_validator):
        """Test crypto validator initialization."""
        assert crypto_validator is not None
        assert crypto_validator.db_path is not None
        assert len(crypto_validator.encryption_checks) > 0
        assert len(crypto_validator.key_management) == 0
        assert len(crypto_validator.certificate_validations) == 0
    
    def test_validate_encryption_algorithm(self, crypto_validator):
        """Test validating encryption algorithm."""
        # Test strong algorithm
        check = crypto_validator.validate_encryption_algorithm(
            EncryptionAlgorithm.AES_256, 256
        )
        
        assert check is not None
        assert check.algorithm == EncryptionAlgorithm.AES_256
        assert check.key_size == 256
        assert check.strength == "Strong"
        assert check.status == ValidationResult.PASS
        
        # Test weak algorithm
        check = crypto_validator.validate_encryption_algorithm(
            EncryptionAlgorithm.MD5, 128
        )
        
        assert check is not None
        assert check.algorithm == EncryptionAlgorithm.MD5
        assert check.strength == "Deprecated"
        assert check.status == ValidationResult.FAIL
    
    def test_validate_key_management(self, crypto_validator):
        """Test validating key management."""
        key_data = {
            'key_type': 'SYMMETRIC',
            'algorithm': 'AES_256',
            'key_size': 256,
            'creation_date': datetime.now().isoformat(),
            'expiration_date': (datetime.now() + timedelta(days=365)).isoformat(),
            'usage': ['encryption', 'decryption'],
            'status': 'Active',
            'rotation_policy': 'Annual',
            'storage_location': 'HSM',
            'access_controls': ['admin_only']
        }
        
        key_mgmt = crypto_validator.validate_key_management(key_data)
        
        assert key_mgmt is not None
        assert key_mgmt.key_type == KeyType.SYMMETRIC
        assert key_mgmt.algorithm == EncryptionAlgorithm.AES_256
        assert key_mgmt.key_size == 256
        assert key_mgmt.status == 'Active'
    
    @patch('ssl.create_default_context')
    @patch('socket.create_connection')
    def test_validate_certificate(self, mock_connection, mock_context, crypto_validator):
        """Test validating certificate."""
        # Mock SSL context and connection
        mock_ctx = Mock()
        mock_ctx.wrap_socket.return_value.__enter__.return_value.getpeercert.return_value = {
            'subject': [('commonName', 'example.com')],
            'issuer': [('commonName', 'CA')],
            'serialNumber': '123456789',
            'notBefore': 'Jan 01 00:00:00 2023 GMT',
            'notAfter': 'Jan 01 00:00:00 2024 GMT'
        }
        mock_context.return_value = mock_ctx
        
        mock_sock = Mock()
        mock_connection.return_value.__enter__.return_value = mock_sock
        
        cert_validation = crypto_validator.validate_certificate("example.com", 443)
        
        assert cert_validation is not None
        assert cert_validation.subject == "example.com"
        assert cert_validation.issuer == "CA"
        assert cert_validation.status in [CertificateStatus.VALID, CertificateStatus.EXPIRED]
    
    def test_generate_crypto_compliance_report(self, crypto_validator):
        """Test generating crypto compliance report."""
        # Add some test data
        check = EncryptionCheck(
            check_id="test_check_001",
            name="Test Check",
            algorithm=EncryptionAlgorithm.AES_256,
            key_size=256,
            strength="Strong",
            status=ValidationResult.PASS,
            description="Test encryption check",
            recommendations=["Continue using AES-256"]
        )
        crypto_validator.encryption_checks = [check]
        
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        report = crypto_validator.generate_crypto_compliance_report(start_date, end_date)
        
        assert report is not None
        assert 'summary' in report
        assert 'encryption_check_statistics' in report
        assert 'certificate_statistics' in report
        assert 'failed_checks' in report
        assert 'recommendations' in report

class TestIntegration:
    """Integration tests for the penetration testing framework."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    def test_end_to_end_penetration_testing_workflow(self, temp_db):
        """Test complete penetration testing workflow."""
        # Initialize all components
        penetration_tester = PenetrationTester(db_path=temp_db)
        vulnerability_scanner = VulnerabilityScanner(db_path=temp_db)
        security_auditor = SecurityAuditor(db_path=temp_db)
        threat_modeler = ThreatModeler(db_path=temp_db)
        security_monitor = SecurityMonitor(db_path=temp_db)
        crypto_validator = CryptoValidator(db_path=temp_db)
        
        # 1. Create security assessment
        assessment_id = penetration_tester.create_assessment(
            name="Comprehensive Security Assessment",
            assessment_type=AssessmentType.WEB_APPLICATION,
            target="example.com",
            scope=["web application", "API", "database"],
            objectives=["identify vulnerabilities", "assess security controls", "compliance check"]
        )
        
        # 2. Start assessment
        penetration_tester.start_assessment(assessment_id)
        
        # 3. Run vulnerability scan
        scan_id = vulnerability_scanner.run_scan("example.com", ScanType.WEB_APPLICATION_SCAN)
        
        # 4. Create compliance audit
        audit_id = security_auditor.create_audit(
            name="ISO 27001 Compliance Audit",
            audit_type=AuditType.COMPLIANCE_AUDIT,
            framework=ComplianceFramework.ISO_27001,
            scope=["example.com"],
            objectives=["compliance validation"]
        )
        
        # 5. Run compliance audit
        findings = security_auditor.run_compliance_audit(audit_id)
        
        # 6. Create threat model
        architecture = {
            "components": ["web server", "database", "API"],
            "data_flows": ["user -> web server", "web server -> database", "API -> database"]
        }
        model_id = threat_modeler.create_threat_model(
            application_name="Example Application",
            version="1.0.0",
            description="Web application with API and database",
            architecture=architecture
        )
        
        # 7. Analyze threats
        threats = threat_modeler.analyze_threats(model_id)
        
        # 8. Process security events
        event_data = {
            'event_type': 'AUTHENTICATION_FAILURE',
            'threat_level': 'HIGH',
            'timestamp': datetime.now().isoformat(),
            'source_ip': '192.168.1.100',
            'source_user': 'testuser',
            'target_system': 'web server',
            'description': 'Multiple failed login attempts',
            'indicators': ['brute_force'],
            'context': {'attempts': 10}
        }
        security_monitor.process_security_event(event_data)
        
        # 9. Validate encryption
        encryption_check = crypto_validator.validate_encryption_algorithm(
            EncryptionAlgorithm.AES_256, 256
        )
        
        # 10. Generate reports
        security_report = penetration_tester.generate_security_report(assessment_id)
        audit_report = security_auditor.generate_audit_report(audit_id)
        threat_report = threat_modeler.generate_threat_model_report(model_id)
        monitoring_report = security_monitor.generate_security_report(
            datetime.now() - timedelta(days=1), datetime.now()
        )
        crypto_report = crypto_validator.generate_crypto_compliance_report(
            datetime.now() - timedelta(days=1), datetime.now()
        )
        
        # Verify all components worked together
        assert assessment_id is not None
        assert scan_id is not None
        assert audit_id is not None
        assert model_id is not None
        assert len(findings) > 0
        assert len(threats) > 0
        assert len(security_monitor.security_events) > 0
        assert encryption_check is not None
        assert security_report is not None
        assert audit_report is not None
        assert threat_report is not None
        assert monitoring_report is not None
        assert crypto_report is not None
    
    def test_security_monitoring_integration(self, temp_db):
        """Test security monitoring integration with other components."""
        security_monitor = SecurityMonitor(db_path=temp_db)
        vulnerability_scanner = VulnerabilityScanner(db_path=temp_db)
        
        # Add threat intelligence
        ti_data = {
            'indicator_type': 'IP_ADDRESS',
            'indicator_value': '192.168.1.100',
            'threat_level': 'HIGH',
            'confidence': 0.9,
            'source': 'External Feed',
            'description': 'Known malicious IP',
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'tags': ['malware', 'botnet'],
            'metadata': {'country': 'Unknown'}
        }
        ti_id = security_monitor.add_threat_intelligence(ti_data)
        
        # Process security event from known malicious IP
        event_data = {
            'event_type': 'INTRUSION_ATTEMPT',
            'threat_level': 'MEDIUM',
            'timestamp': datetime.now().isoformat(),
            'source_ip': '192.168.1.100',  # Known malicious IP
            'source_user': None,
            'target_system': 'web server',
            'description': 'Suspicious connection attempt',
            'indicators': ['port_scan'],
            'context': {'ports': [22, 80, 443]}
        }
        event = security_monitor.process_security_event(event_data)
        
        # Verify threat intelligence was applied
        assert event is not None
        assert event.threat_level == ThreatLevel.HIGH  # Should be elevated due to TI
        assert 'threat_intelligence' in event.context
        
        # Verify incident was created
        assert len(security_monitor.incidents) > 0
        incident = security_monitor.incidents[0]
        assert incident.threat_level == ThreatLevel.HIGH

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
