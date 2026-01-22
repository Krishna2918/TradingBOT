"""
Security Auditor

This module implements a comprehensive security auditing system with
automated compliance checks, security policy validation, and audit
reporting capabilities.

Author: AI Trading System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import sqlite3
from pathlib import Path
import subprocess
import os
import hashlib
import re
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AuditType(Enum):
    """Types of security audits."""
    COMPLIANCE_AUDIT = "COMPLIANCE_AUDIT"
    SECURITY_POLICY_AUDIT = "SECURITY_POLICY_AUDIT"
    ACCESS_CONTROL_AUDIT = "ACCESS_CONTROL_AUDIT"
    DATA_PROTECTION_AUDIT = "DATA_PROTECTION_AUDIT"
    NETWORK_SECURITY_AUDIT = "NETWORK_SECURITY_AUDIT"
    APPLICATION_SECURITY_AUDIT = "APPLICATION_SECURITY_AUDIT"
    INFRASTRUCTURE_AUDIT = "INFRASTRUCTURE_AUDIT"
    INCIDENT_RESPONSE_AUDIT = "INCIDENT_RESPONSE_AUDIT"
    BUSINESS_CONTINUITY_AUDIT = "BUSINESS_CONTINUITY_AUDIT"
    THIRD_PARTY_AUDIT = "THIRD_PARTY_AUDIT"

class ComplianceFramework(Enum):
    """Compliance frameworks."""
    ISO_27001 = "ISO_27001"
    SOC_2 = "SOC_2"
    PCI_DSS = "PCI_DSS"
    HIPAA = "HIPAA"
    GDPR = "GDPR"
    NIST_CSF = "NIST_CSF"
    CIS_CONTROLS = "CIS_CONTROLS"
    OWASP = "OWASP"
    CUSTOM = "CUSTOM"

class AuditStatus(Enum):
    """Audit status."""
    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class FindingSeverity(Enum):
    """Audit finding severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"

@dataclass
class AuditFinding:
    """Security audit finding."""
    finding_id: str
    title: str
    description: str
    severity: FindingSeverity
    category: str
    compliance_requirement: str
    affected_components: List[str]
    evidence: List[str]
    recommendation: str
    remediation_priority: str
    estimated_effort: str
    business_impact: str
    discovered_at: datetime
    verified: bool = False
    remediated: bool = False
    false_positive: bool = False
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ComplianceCheck:
    """Compliance check definition."""
    check_id: str
    name: str
    description: str
    framework: ComplianceFramework
    requirement: str
    check_type: str
    automated: bool
    check_script: Optional[str]
    expected_result: str
    severity: FindingSeverity
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SecurityAudit:
    """Security audit definition."""
    audit_id: str
    name: str
    audit_type: AuditType
    framework: ComplianceFramework
    scope: List[str]
    objectives: List[str]
    status: AuditStatus
    start_date: datetime
    end_date: Optional[datetime]
    findings: List[AuditFinding]
    compliance_score: float
    recommendations: List[str]
    report_path: Optional[str] = None
    auditor: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

class SecurityAuditor:
    """
    Comprehensive security auditing system.
    
    Features:
    - Automated compliance auditing
    - Security policy validation
    - Access control auditing
    - Data protection auditing
    - Network security auditing
    - Application security auditing
    - Infrastructure auditing
    - Audit reporting and recommendations
    """
    
    def __init__(self, db_path: str = "data/security_auditor.db"):
        """
        Initialize security auditor.
        
        Args:
            db_path: Path to security auditor database
        """
        self.db_path = db_path
        self.audits: Dict[str, SecurityAudit] = {}
        self.compliance_checks: Dict[str, ComplianceCheck] = {}
        self.audit_findings: List[AuditFinding] = []
        
        # Audit methodologies
        self.audit_methodologies = {
            AuditType.COMPLIANCE_AUDIT: "ISO 19011",
            AuditType.SECURITY_POLICY_AUDIT: "NIST SP 800-53",
            AuditType.ACCESS_CONTROL_AUDIT: "ISO 27001",
            AuditType.DATA_PROTECTION_AUDIT: "GDPR Article 32",
            AuditType.NETWORK_SECURITY_AUDIT: "NIST SP 800-41",
            AuditType.APPLICATION_SECURITY_AUDIT: "OWASP ASVS",
            AuditType.INFRASTRUCTURE_AUDIT: "CIS Controls",
            AuditType.INCIDENT_RESPONSE_AUDIT: "NIST SP 800-61",
            AuditType.BUSINESS_CONTINUITY_AUDIT: "ISO 22301",
            AuditType.THIRD_PARTY_AUDIT: "ISO 27001"
        }
        
        # Initialize database
        self._init_database()
        
        # Load default compliance checks
        self._load_default_compliance_checks()
        
        logger.info("Security Auditor initialized")
    
    def _init_database(self) -> None:
        """Initialize security auditor database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create audits table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_audits (
                audit_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                audit_type TEXT NOT NULL,
                framework TEXT NOT NULL,
                scope TEXT,
                objectives TEXT,
                status TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                findings TEXT,
                compliance_score REAL,
                recommendations TEXT,
                report_path TEXT,
                auditor TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create audit findings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_findings (
                finding_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                severity TEXT NOT NULL,
                category TEXT,
                compliance_requirement TEXT,
                affected_components TEXT,
                evidence TEXT,
                recommendation TEXT,
                remediation_priority TEXT,
                estimated_effort TEXT,
                business_impact TEXT,
                discovered_at TEXT,
                verified INTEGER,
                remediated INTEGER,
                false_positive INTEGER,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create compliance checks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_checks (
                check_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                framework TEXT NOT NULL,
                requirement TEXT,
                check_type TEXT,
                automated INTEGER,
                check_script TEXT,
                expected_result TEXT,
                severity TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_compliance_checks(self) -> None:
        """Load default compliance checks."""
        default_checks = [
            ComplianceCheck(
                check_id="CHECK_001",
                name="Password Policy Compliance",
                description="Verify password policy implementation",
                framework=ComplianceFramework.ISO_27001,
                requirement="A.9.2.3",
                check_type="Policy",
                automated=True,
                check_script="check_password_policy()",
                expected_result="Strong password policy implemented",
                severity=FindingSeverity.HIGH
            ),
            ComplianceCheck(
                check_id="CHECK_002",
                name="Access Control Review",
                description="Review user access controls",
                framework=ComplianceFramework.ISO_27001,
                requirement="A.9.1.1",
                check_type="Access Control",
                automated=True,
                check_script="check_access_controls()",
                expected_result="Proper access controls implemented",
                severity=FindingSeverity.HIGH
            ),
            ComplianceCheck(
                check_id="CHECK_003",
                name="Data Encryption",
                description="Verify data encryption implementation",
                framework=ComplianceFramework.ISO_27001,
                requirement="A.10.1.1",
                check_type="Cryptography",
                automated=True,
                check_script="check_data_encryption()",
                expected_result="Data encrypted at rest and in transit",
                severity=FindingSeverity.CRITICAL
            ),
            ComplianceCheck(
                check_id="CHECK_004",
                name="Network Security",
                description="Review network security controls",
                framework=ComplianceFramework.ISO_27001,
                requirement="A.13.1.1",
                check_type="Network",
                automated=True,
                check_script="check_network_security()",
                expected_result="Network security controls implemented",
                severity=FindingSeverity.HIGH
            ),
            ComplianceCheck(
                check_id="CHECK_005",
                name="Incident Response",
                description="Verify incident response procedures",
                framework=ComplianceFramework.ISO_27001,
                requirement="A.16.1.1",
                check_type="Incident Response",
                automated=False,
                check_script=None,
                expected_result="Incident response procedures documented and tested",
                severity=FindingSeverity.MEDIUM
            ),
            ComplianceCheck(
                check_id="CHECK_006",
                name="Backup and Recovery",
                description="Verify backup and recovery procedures",
                framework=ComplianceFramework.ISO_27001,
                requirement="A.12.3.1",
                check_type="Backup",
                automated=True,
                check_script="check_backup_procedures()",
                expected_result="Backup and recovery procedures implemented",
                severity=FindingSeverity.HIGH
            ),
            ComplianceCheck(
                check_id="CHECK_007",
                name="Security Monitoring",
                description="Verify security monitoring implementation",
                framework=ComplianceFramework.ISO_27001,
                requirement="A.12.4.1",
                check_type="Monitoring",
                automated=True,
                check_script="check_security_monitoring()",
                expected_result="Security monitoring implemented",
                severity=FindingSeverity.HIGH
            ),
            ComplianceCheck(
                check_id="CHECK_008",
                name="Vulnerability Management",
                description="Verify vulnerability management process",
                framework=ComplianceFramework.ISO_27001,
                requirement="A.12.6.1",
                check_type="Vulnerability Management",
                automated=True,
                check_script="check_vulnerability_management()",
                expected_result="Vulnerability management process implemented",
                severity=FindingSeverity.HIGH
            ),
            ComplianceCheck(
                check_id="CHECK_009",
                name="Security Awareness",
                description="Verify security awareness training",
                framework=ComplianceFramework.ISO_27001,
                requirement="A.7.2.2",
                check_type="Training",
                automated=False,
                check_script=None,
                expected_result="Security awareness training provided",
                severity=FindingSeverity.MEDIUM
            ),
            ComplianceCheck(
                check_id="CHECK_010",
                name="Third Party Security",
                description="Verify third party security controls",
                framework=ComplianceFramework.ISO_27001,
                requirement="A.15.1.1",
                check_type="Third Party",
                automated=False,
                check_script=None,
                expected_result="Third party security controls implemented",
                severity=FindingSeverity.MEDIUM
            )
        ]
        
        for check in default_checks:
            self.add_compliance_check(check)
    
    def add_compliance_check(self, check: ComplianceCheck) -> None:
        """
        Add a new compliance check.
        
        Args:
            check: Compliance check definition
        """
        self.compliance_checks[check.check_id] = check
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO compliance_checks 
            (check_id, name, description, framework, requirement, check_type,
             automated, check_script, expected_result, severity, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            check.check_id, check.name, check.description, check.framework.value,
            check.requirement, check.check_type, check.automated, check.check_script,
            check.expected_result, check.severity.value, check.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added compliance check: {check.check_id} - {check.name}")
    
    def create_audit(self, name: str, audit_type: AuditType, framework: ComplianceFramework,
                    scope: List[str], objectives: List[str], auditor: str = None) -> str:
        """
        Create a new security audit.
        
        Args:
            name: Audit name
            audit_type: Type of audit
            framework: Compliance framework
            scope: Audit scope
            objectives: Audit objectives
            auditor: Auditor name
            
        Returns:
            Audit ID
        """
        audit_id = f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        audit = SecurityAudit(
            audit_id=audit_id,
            name=name,
            audit_type=audit_type,
            framework=framework,
            scope=scope,
            objectives=objectives,
            status=AuditStatus.PLANNED,
            start_date=datetime.now(),
            end_date=None,
            findings=[],
            compliance_score=0.0,
            recommendations=[],
            auditor=auditor
        )
        
        self.audits[audit_id] = audit
        self._store_audit(audit)
        
        logger.info(f"Created security audit: {audit_id} - {name}")
        return audit_id
    
    def start_audit(self, audit_id: str) -> bool:
        """
        Start a security audit.
        
        Args:
            audit_id: Audit ID
            
        Returns:
            True if started successfully
        """
        if audit_id not in self.audits:
            return False
        
        audit = self.audits[audit_id]
        audit.status = AuditStatus.IN_PROGRESS
        audit.start_date = datetime.now()
        
        self._update_audit(audit)
        
        logger.info(f"Started security audit: {audit_id}")
        return True
    
    def run_compliance_audit(self, audit_id: str) -> List[AuditFinding]:
        """
        Run compliance audit checks.
        
        Args:
            audit_id: Audit ID
            
        Returns:
            List of audit findings
        """
        if audit_id not in self.audits:
            return []
        
        audit = self.audits[audit_id]
        findings = []
        
        # Run compliance checks based on framework
        framework_checks = [check for check in self.compliance_checks.values() 
                           if check.framework == audit.framework]
        
        for check in framework_checks:
            try:
                if check.automated and check.check_script:
                    # Run automated check
                    finding = self._run_automated_check(check, audit.scope)
                    if finding:
                        findings.append(finding)
                else:
                    # Manual check - create placeholder finding
                    finding = AuditFinding(
                        finding_id=f"FINDING_{check.check_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        title=f"Manual Check Required: {check.name}",
                        description=f"Manual verification required for {check.description}",
                        severity=check.severity,
                        category=check.check_type,
                        compliance_requirement=check.requirement,
                        affected_components=audit.scope,
                        evidence=[],
                        recommendation=check.expected_result,
                        remediation_priority="Medium",
                        estimated_effort="2-4 hours",
                        business_impact="Compliance requirement",
                        discovered_at=datetime.now()
                    )
                    findings.append(finding)
            
            except Exception as e:
                logger.error(f"Error running compliance check {check.check_id}: {e}")
        
        # Update audit with findings
        audit.findings = findings
        audit.compliance_score = self._calculate_compliance_score(findings)
        audit.recommendations = self._generate_recommendations(findings)
        
        self._update_audit(audit)
        
        # Store individual findings
        for finding in findings:
            self._store_audit_finding(finding)
        
        logger.info(f"Completed compliance audit: {audit_id} - Found {len(findings)} findings")
        return findings
    
    def _run_automated_check(self, check: ComplianceCheck, scope: List[str]) -> Optional[AuditFinding]:
        """Run an automated compliance check."""
        try:
            if check.check_script == "check_password_policy()":
                return self._check_password_policy(scope)
            elif check.check_script == "check_access_controls()":
                return self._check_access_controls(scope)
            elif check.check_script == "check_data_encryption()":
                return self._check_data_encryption(scope)
            elif check.check_script == "check_network_security()":
                return self._check_network_security(scope)
            elif check.check_script == "check_backup_procedures()":
                return self._check_backup_procedures(scope)
            elif check.check_script == "check_security_monitoring()":
                return self._check_security_monitoring(scope)
            elif check.check_script == "check_vulnerability_management()":
                return self._check_vulnerability_management(scope)
        
        except Exception as e:
            logger.error(f"Error running automated check {check.check_id}: {e}")
        
        return None
    
    def _check_password_policy(self, scope: List[str]) -> Optional[AuditFinding]:
        """Check password policy compliance."""
        # This would typically check system password policies
        # For now, we'll simulate a check
        
        # Simulate finding a weak password policy
        finding = AuditFinding(
            finding_id=f"PASSWORD_POLICY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Weak Password Policy",
            description="Password policy does not meet minimum security requirements",
            severity=FindingSeverity.HIGH,
            category="Authentication",
            compliance_requirement="A.9.2.3",
            affected_components=scope,
            evidence=["Password minimum length: 6 characters", "No complexity requirements"],
            recommendation="Implement strong password policy with minimum 12 characters, complexity requirements, and regular rotation",
            remediation_priority="High",
            estimated_effort="1-2 days",
            business_impact="Increased risk of account compromise",
            discovered_at=datetime.now()
        )
        
        return finding
    
    def _check_access_controls(self, scope: List[str]) -> Optional[AuditFinding]:
        """Check access control compliance."""
        # This would typically check user access controls
        # For now, we'll simulate a check
        
        # Simulate finding excessive privileges
        finding = AuditFinding(
            finding_id=f"ACCESS_CONTROL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Excessive User Privileges",
            description="Users have excessive privileges beyond their job requirements",
            severity=FindingSeverity.MEDIUM,
            category="Access Control",
            compliance_requirement="A.9.1.1",
            affected_components=scope,
            evidence=["5 users with administrative privileges", "No privilege review process"],
            recommendation="Implement principle of least privilege and regular access reviews",
            remediation_priority="Medium",
            estimated_effort="1 week",
            business_impact="Increased risk of privilege escalation",
            discovered_at=datetime.now()
        )
        
        return finding
    
    def _check_data_encryption(self, scope: List[str]) -> Optional[AuditFinding]:
        """Check data encryption compliance."""
        # This would typically check data encryption implementation
        # For now, we'll simulate a check
        
        # Simulate finding unencrypted data
        finding = AuditFinding(
            finding_id=f"DATA_ENCRYPTION_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Unencrypted Sensitive Data",
            description="Sensitive data is stored without encryption",
            severity=FindingSeverity.CRITICAL,
            category="Data Protection",
            compliance_requirement="A.10.1.1",
            affected_components=scope,
            evidence=["Database contains unencrypted PII", "Backup files not encrypted"],
            recommendation="Implement encryption for data at rest and in transit",
            remediation_priority="Critical",
            estimated_effort="2-3 weeks",
            business_impact="High risk of data breach and regulatory penalties",
            discovered_at=datetime.now()
        )
        
        return finding
    
    def _check_network_security(self, scope: List[str]) -> Optional[AuditFinding]:
        """Check network security compliance."""
        # This would typically check network security controls
        # For now, we'll simulate a check
        
        # Simulate finding weak network controls
        finding = AuditFinding(
            finding_id=f"NETWORK_SECURITY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Weak Network Security Controls",
            description="Network security controls are insufficient",
            severity=FindingSeverity.HIGH,
            category="Network Security",
            compliance_requirement="A.13.1.1",
            affected_components=scope,
            evidence=["No network segmentation", "Weak firewall rules", "No intrusion detection"],
            recommendation="Implement network segmentation, strengthen firewall rules, and deploy IDS/IPS",
            remediation_priority="High",
            estimated_effort="2-4 weeks",
            business_impact="Increased risk of network attacks",
            discovered_at=datetime.now()
        )
        
        return finding
    
    def _check_backup_procedures(self, scope: List[str]) -> Optional[AuditFinding]:
        """Check backup and recovery compliance."""
        # This would typically check backup procedures
        # For now, we'll simulate a check
        
        # Simulate finding inadequate backup procedures
        finding = AuditFinding(
            finding_id=f"BACKUP_PROCEDURES_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Inadequate Backup Procedures",
            description="Backup and recovery procedures are insufficient",
            severity=FindingSeverity.MEDIUM,
            category="Backup and Recovery",
            compliance_requirement="A.12.3.1",
            affected_components=scope,
            evidence=["No automated backups", "Backup testing not performed", "No offsite storage"],
            recommendation="Implement automated backups, regular testing, and offsite storage",
            remediation_priority="Medium",
            estimated_effort="1-2 weeks",
            business_impact="Risk of data loss and business disruption",
            discovered_at=datetime.now()
        )
        
        return finding
    
    def _check_security_monitoring(self, scope: List[str]) -> Optional[AuditFinding]:
        """Check security monitoring compliance."""
        # This would typically check security monitoring implementation
        # For now, we'll simulate a check
        
        # Simulate finding inadequate monitoring
        finding = AuditFinding(
            finding_id=f"SECURITY_MONITORING_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Inadequate Security Monitoring",
            description="Security monitoring and logging are insufficient",
            severity=FindingSeverity.HIGH,
            category="Security Monitoring",
            compliance_requirement="A.12.4.1",
            affected_components=scope,
            evidence=["No SIEM implementation", "Limited log retention", "No real-time monitoring"],
            recommendation="Implement SIEM, extend log retention, and deploy real-time monitoring",
            remediation_priority="High",
            estimated_effort="3-4 weeks",
            business_impact="Delayed detection of security incidents",
            discovered_at=datetime.now()
        )
        
        return finding
    
    def _check_vulnerability_management(self, scope: List[str]) -> Optional[AuditFinding]:
        """Check vulnerability management compliance."""
        # This would typically check vulnerability management process
        # For now, we'll simulate a check
        
        # Simulate finding inadequate vulnerability management
        finding = AuditFinding(
            finding_id=f"VULNERABILITY_MANAGEMENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Inadequate Vulnerability Management",
            description="Vulnerability management process is insufficient",
            severity=FindingSeverity.HIGH,
            category="Vulnerability Management",
            compliance_requirement="A.12.6.1",
            affected_components=scope,
            evidence=["No regular vulnerability scanning", "No patch management process", "No risk assessment"],
            recommendation="Implement regular vulnerability scanning, patch management, and risk assessment",
            remediation_priority="High",
            estimated_effort="2-3 weeks",
            business_impact="Increased risk of exploitation",
            discovered_at=datetime.now()
        )
        
        return finding
    
    def _calculate_compliance_score(self, findings: List[AuditFinding]) -> float:
        """Calculate compliance score based on findings."""
        if not findings:
            return 100.0
        
        severity_weights = {
            FindingSeverity.CRITICAL: 20.0,
            FindingSeverity.HIGH: 15.0,
            FindingSeverity.MEDIUM: 10.0,
            FindingSeverity.LOW: 5.0,
            FindingSeverity.INFORMATIONAL: 1.0
        }
        
        total_penalty = sum(severity_weights.get(f.severity, 0) for f in findings)
        max_possible_penalty = len(findings) * 20.0
        
        compliance_score = max(0.0, 100.0 - (total_penalty / max_possible_penalty * 100))
        return compliance_score
    
    def _generate_recommendations(self, findings: List[AuditFinding]) -> List[str]:
        """Generate audit recommendations based on findings."""
        recommendations = []
        
        # Group findings by category
        categories = {}
        for finding in findings:
            if finding.category not in categories:
                categories[finding.category] = []
            categories[finding.category].append(finding)
        
        # Generate category-specific recommendations
        for category, category_findings in categories.items():
            if category == "Authentication":
                recommendations.append("Strengthen authentication mechanisms and implement multi-factor authentication")
            elif category == "Access Control":
                recommendations.append("Implement principle of least privilege and regular access reviews")
            elif category == "Data Protection":
                recommendations.append("Implement comprehensive data encryption and access controls")
            elif category == "Network Security":
                recommendations.append("Strengthen network security controls and implement network segmentation")
            elif category == "Backup and Recovery":
                recommendations.append("Implement automated backup procedures and regular testing")
            elif category == "Security Monitoring":
                recommendations.append("Implement comprehensive security monitoring and logging")
            elif category == "Vulnerability Management":
                recommendations.append("Implement regular vulnerability scanning and patch management")
        
        # Add general recommendations
        if len(findings) > 0:
            recommendations.extend([
                "Establish regular security assessments and audits",
                "Implement security awareness training program",
                "Develop incident response procedures",
                "Establish security metrics and monitoring",
                "Regularly review and update security policies"
            ])
        
        return recommendations
    
    def complete_audit(self, audit_id: str) -> bool:
        """
        Complete a security audit.
        
        Args:
            audit_id: Audit ID
            
        Returns:
            True if completed successfully
        """
        if audit_id not in self.audits:
            return False
        
        audit = self.audits[audit_id]
        audit.status = AuditStatus.COMPLETED
        audit.end_date = datetime.now()
        
        self._update_audit(audit)
        
        logger.info(f"Completed security audit: {audit_id}")
        return True
    
    def generate_audit_report(self, audit_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive audit report.
        
        Args:
            audit_id: Audit ID
            
        Returns:
            Audit report dictionary
        """
        if audit_id not in self.audits:
            return {}
        
        audit = self.audits[audit_id]
        
        # Create report
        report = {
            'audit_id': audit.audit_id,
            'name': audit.name,
            'audit_type': audit.audit_type.value,
            'framework': audit.framework.value,
            'scope': audit.scope,
            'objectives': audit.objectives,
            'status': audit.status.value,
            'start_date': audit.start_date.isoformat(),
            'end_date': audit.end_date.isoformat() if audit.end_date else None,
            'auditor': audit.auditor,
            'compliance_score': audit.compliance_score,
            'compliance_level': self._get_compliance_level(audit.compliance_score),
            'summary': {
                'total_findings': len(audit.findings),
                'critical_findings': len([f for f in audit.findings if f.severity == FindingSeverity.CRITICAL]),
                'high_findings': len([f for f in audit.findings if f.severity == FindingSeverity.HIGH]),
                'medium_findings': len([f for f in audit.findings if f.severity == FindingSeverity.MEDIUM]),
                'low_findings': len([f for f in audit.findings if f.severity == FindingSeverity.LOW])
            },
            'findings': [
                {
                    'finding_id': f.finding_id,
                    'title': f.title,
                    'description': f.description,
                    'severity': f.severity.value,
                    'category': f.category,
                    'compliance_requirement': f.compliance_requirement,
                    'affected_components': f.affected_components,
                    'evidence': f.evidence,
                    'recommendation': f.recommendation,
                    'remediation_priority': f.remediation_priority,
                    'estimated_effort': f.estimated_effort,
                    'business_impact': f.business_impact
                }
                for f in audit.findings
            ],
            'recommendations': audit.recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        report_path = f"reports/security_audit_{audit_id}.json"
        Path("reports").mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        audit.report_path = report_path
        self._update_audit(audit)
        
        return report
    
    def _get_compliance_level(self, score: float) -> str:
        """Get compliance level based on score."""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 80:
            return "GOOD"
        elif score >= 70:
            return "SATISFACTORY"
        elif score >= 60:
            return "NEEDS IMPROVEMENT"
        else:
            return "POOR"
    
    def _store_audit(self, audit: SecurityAudit) -> None:
        """Store security audit in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO security_audits 
            (audit_id, name, audit_type, framework, scope, objectives, status,
             start_date, end_date, findings, compliance_score, recommendations, report_path, auditor, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            audit.audit_id, audit.name, audit.audit_type.value, audit.framework.value,
            json.dumps(audit.scope), json.dumps(audit.objectives), audit.status.value,
            audit.start_date.isoformat(), audit.end_date.isoformat() if audit.end_date else None,
            json.dumps([f.__dict__ for f in audit.findings]), audit.compliance_score,
            json.dumps(audit.recommendations), audit.report_path, audit.auditor,
            audit.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_audit(self, audit: SecurityAudit) -> None:
        """Update security audit in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE security_audits
            SET status = ?, start_date = ?, end_date = ?, findings = ?, 
                compliance_score = ?, recommendations = ?, report_path = ?
            WHERE audit_id = ?
        """, (
            audit.status.value, audit.start_date.isoformat(),
            audit.end_date.isoformat() if audit.end_date else None,
            json.dumps([f.__dict__ for f in audit.findings]), audit.compliance_score,
            json.dumps(audit.recommendations), audit.report_path, audit.audit_id
        ))
        
        conn.commit()
        conn.close()
    
    def _store_audit_finding(self, finding: AuditFinding) -> None:
        """Store audit finding in database."""
        self.audit_findings.append(finding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO audit_findings 
            (finding_id, title, description, severity, category, compliance_requirement,
             affected_components, evidence, recommendation, remediation_priority,
             estimated_effort, business_impact, discovered_at, verified, remediated, false_positive, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            finding.finding_id, finding.title, finding.description, finding.severity.value,
            finding.category, finding.compliance_requirement, json.dumps(finding.affected_components),
            json.dumps(finding.evidence), finding.recommendation, finding.remediation_priority,
            finding.estimated_effort, finding.business_impact, finding.discovered_at.isoformat(),
            finding.verified, finding.remediated, finding.false_positive, finding.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_audit_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Get audit summary for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Audit summary dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get audit counts by type
        cursor.execute("""
            SELECT audit_type, COUNT(*) as count
            FROM security_audits
            WHERE start_date BETWEEN ? AND ?
            GROUP BY audit_type
        """, (start_date.isoformat(), end_date.isoformat()))
        
        audit_type_counts = dict(cursor.fetchall())
        
        # Get finding counts by severity
        cursor.execute("""
            SELECT severity, COUNT(*) as count
            FROM audit_findings
            WHERE discovered_at BETWEEN ? AND ?
            GROUP BY severity
        """, (start_date.isoformat(), end_date.isoformat()))
        
        severity_counts = dict(cursor.fetchall())
        
        # Get average compliance score
        cursor.execute("""
            SELECT AVG(compliance_score) as avg_score
            FROM security_audits
            WHERE start_date BETWEEN ? AND ? AND status = 'COMPLETED'
        """, (start_date.isoformat(), end_date.isoformat()))
        
        avg_score = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        return {
            'summary': {
                'total_audits': sum(audit_type_counts.values()),
                'average_compliance_score': avg_score,
                'total_findings': sum(severity_counts.values())
            },
            'audit_type_breakdown': audit_type_counts,
            'severity_breakdown': severity_counts,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat()
        }
