"""
Professional Penetration Testing Framework

This module implements a comprehensive penetration testing framework with
automated security assessments, vulnerability scanning, and threat modeling
capabilities for enterprise-grade security validation.

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
import socket
import ssl
import requests
import hashlib
import base64
import secrets
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AssessmentType(Enum):
    """Types of security assessments."""
    NETWORK_PENETRATION = "NETWORK_PENETRATION"
    WEB_APPLICATION = "WEB_APPLICATION"
    MOBILE_APPLICATION = "MOBILE_APPLICATION"
    API_SECURITY = "API_SECURITY"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    SOCIAL_ENGINEERING = "SOCIAL_ENGINEERING"
    PHYSICAL_SECURITY = "PHYSICAL_SECURITY"
    WIRELESS_SECURITY = "WIRELESS_SECURITY"
    DATABASE_SECURITY = "DATABASE_SECURITY"
    CLOUD_SECURITY = "CLOUD_SECURITY"

class SeverityLevel(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"

class AssessmentStatus(Enum):
    """Assessment status."""
    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class Vulnerability:
    """Vulnerability definition."""
    vuln_id: str
    name: str
    description: str
    severity: SeverityLevel
    category: str
    cve_id: Optional[str]
    cvss_score: Optional[float]
    affected_components: List[str]
    attack_vector: str
    impact: str
    remediation: str
    references: List[str]
    discovered_at: datetime
    verified: bool = False
    exploited: bool = False
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SecurityAssessment:
    """Security assessment definition."""
    assessment_id: str
    name: str
    assessment_type: AssessmentType
    target: str
    scope: List[str]
    objectives: List[str]
    methodology: str
    status: AssessmentStatus
    start_date: datetime
    end_date: Optional[datetime]
    findings: List[Vulnerability]
    risk_score: float
    recommendations: List[str]
    report_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class VulnerabilityScan:
    """Vulnerability scan definition."""
    scan_id: str
    target: str
    scan_type: str
    start_time: datetime
    end_time: Optional[datetime]
    status: AssessmentStatus
    vulnerabilities_found: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    info_count: int
    scan_results: List[Vulnerability]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ThreatModel:
    """Threat model definition."""
    model_id: str
    application_name: str
    version: str
    architecture: Dict[str, Any]
    data_flows: List[Dict[str, Any]]
    trust_boundaries: List[Dict[str, Any]]
    threats: List[Dict[str, Any]]
    mitigations: List[Dict[str, Any]]
    risk_ratings: Dict[str, str]
    created_at: datetime = field(default_factory=datetime.now)

class PenetrationTester:
    """
    Professional penetration testing framework.
    
    Features:
    - Automated security assessments
    - Vulnerability scanning and analysis
    - Threat modeling and risk assessment
    - Security testing methodologies
    - Report generation and recommendations
    - Compliance validation
    """
    
    def __init__(self, db_path: str = "data/penetration_testing.db"):
        """
        Initialize penetration testing framework.
        
        Args:
            db_path: Path to penetration testing database
        """
        self.db_path = db_path
        self.assessments: Dict[str, SecurityAssessment] = {}
        self.vulnerabilities: List[Vulnerability] = []
        self.scan_results: List[VulnerabilityScan] = []
        
        # Testing methodologies
        self.methodologies = {
            AssessmentType.NETWORK_PENETRATION: "OSSTMM",
            AssessmentType.WEB_APPLICATION: "OWASP",
            AssessmentType.API_SECURITY: "OWASP API Security",
            AssessmentType.INFRASTRUCTURE: "NIST SP 800-115",
            AssessmentType.CLOUD_SECURITY: "CSA CCM"
        }
        
        # Vulnerability databases
        self.vuln_databases = {
            'cve': 'https://cve.mitre.org/cve/',
            'nvd': 'https://nvd.nist.gov/vuln/detail/',
            'exploit_db': 'https://www.exploit-db.com/',
            'owasp': 'https://owasp.org/'
        }
        
        # Initialize database
        self._init_database()
        
        # Load default vulnerability signatures
        self._load_default_vulnerabilities()
        
        logger.info("Professional Penetration Testing framework initialized")
    
    def _init_database(self) -> None:
        """Initialize penetration testing database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create assessments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_assessments (
                assessment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                assessment_type TEXT NOT NULL,
                target TEXT NOT NULL,
                scope TEXT,
                objectives TEXT,
                methodology TEXT,
                status TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                findings TEXT,
                risk_score REAL,
                recommendations TEXT,
                report_path TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create vulnerabilities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vulnerabilities (
                vuln_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                severity TEXT NOT NULL,
                category TEXT,
                cve_id TEXT,
                cvss_score REAL,
                affected_components TEXT,
                attack_vector TEXT,
                impact TEXT,
                remediation TEXT,
                references TEXT,
                discovered_at TEXT,
                verified INTEGER,
                exploited INTEGER,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create vulnerability scans table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vulnerability_scans (
                scan_id TEXT PRIMARY KEY,
                target TEXT NOT NULL,
                scan_type TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT NOT NULL,
                vulnerabilities_found INTEGER,
                critical_count INTEGER,
                high_count INTEGER,
                medium_count INTEGER,
                low_count INTEGER,
                info_count INTEGER,
                scan_results TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create threat models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threat_models (
                model_id TEXT PRIMARY KEY,
                application_name TEXT NOT NULL,
                version TEXT,
                architecture TEXT,
                data_flows TEXT,
                trust_boundaries TEXT,
                threats TEXT,
                mitigations TEXT,
                risk_ratings TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_vulnerabilities(self) -> None:
        """Load default vulnerability signatures."""
        default_vulnerabilities = [
            Vulnerability(
                vuln_id="VULN_001",
                name="SQL Injection",
                description="SQL injection vulnerability allowing unauthorized database access",
                severity=SeverityLevel.HIGH,
                category="Injection",
                cve_id="CWE-89",
                cvss_score=8.8,
                affected_components=["Database", "Web Application"],
                attack_vector="Network",
                impact="Data breach, unauthorized access, data manipulation",
                remediation="Use parameterized queries, input validation, WAF",
                references=["https://owasp.org/www-community/attacks/SQL_Injection"],
                discovered_at=datetime.now()
            ),
            Vulnerability(
                vuln_id="VULN_002",
                name="Cross-Site Scripting (XSS)",
                description="XSS vulnerability allowing script injection",
                severity=SeverityLevel.MEDIUM,
                category="Cross-Site Scripting",
                cve_id="CWE-79",
                cvss_score=6.1,
                affected_components=["Web Application", "Browser"],
                attack_vector="Network",
                impact="Session hijacking, data theft, defacement",
                remediation="Input validation, output encoding, CSP headers",
                references=["https://owasp.org/www-community/attacks/xss/"],
                discovered_at=datetime.now()
            ),
            Vulnerability(
                vuln_id="VULN_003",
                name="Insecure Direct Object Reference",
                description="Direct access to internal objects without authorization",
                severity=SeverityLevel.MEDIUM,
                category="Access Control",
                cve_id="CWE-639",
                cvss_score=5.3,
                affected_components=["Web Application", "API"],
                attack_vector="Network",
                impact="Unauthorized data access, privilege escalation",
                remediation="Access control checks, indirect object references",
                references=["https://owasp.org/www-community/attacks/Insecure_Direct_Object_References"],
                discovered_at=datetime.now()
            ),
            Vulnerability(
                vuln_id="VULN_004",
                name="Weak Authentication",
                description="Weak authentication mechanisms",
                severity=SeverityLevel.HIGH,
                category="Authentication",
                cve_id="CWE-287",
                cvss_score=7.5,
                affected_components=["Authentication System"],
                attack_vector="Network",
                impact="Account takeover, unauthorized access",
                remediation="Strong passwords, MFA, account lockout policies",
                references=["https://owasp.org/www-community/controls/Authentication"],
                discovered_at=datetime.now()
            ),
            Vulnerability(
                vuln_id="VULN_005",
                name="Insecure Communication",
                description="Data transmitted over unencrypted channels",
                severity=SeverityLevel.HIGH,
                category="Cryptography",
                cve_id="CWE-319",
                cvss_score=7.4,
                affected_components=["Network", "Application"],
                attack_vector="Network",
                impact="Data interception, man-in-the-middle attacks",
                remediation="Use TLS/SSL, certificate validation, HSTS",
                references=["https://owasp.org/www-community/controls/Transport_Layer_Protection"],
                discovered_at=datetime.now()
            ),
            Vulnerability(
                vuln_id="VULN_006",
                name="Security Misconfiguration",
                description="Insecure default configurations",
                severity=SeverityLevel.MEDIUM,
                category="Configuration",
                cve_id="CWE-16",
                cvss_score=5.3,
                affected_components=["Server", "Application", "Database"],
                attack_vector="Network",
                impact="Information disclosure, unauthorized access",
                remediation="Secure configuration, regular updates, hardening",
                references=["https://owasp.org/www-community/controls/Security_Configuration"],
                discovered_at=datetime.now()
            ),
            Vulnerability(
                vuln_id="VULN_007",
                name="Sensitive Data Exposure",
                description="Exposure of sensitive information",
                severity=SeverityLevel.HIGH,
                category="Data Protection",
                cve_id="CWE-200",
                cvss_score=7.5,
                affected_components=["Database", "Application", "Storage"],
                attack_vector="Network",
                impact="Data breach, privacy violation, compliance issues",
                remediation="Data encryption, access controls, data classification",
                references=["https://owasp.org/www-community/controls/Protect_Data"],
                discovered_at=datetime.now()
            ),
            Vulnerability(
                vuln_id="VULN_008",
                name="Broken Access Control",
                description="Inadequate access control mechanisms",
                severity=SeverityLevel.HIGH,
                category="Access Control",
                cve_id="CWE-284",
                cvss_score=8.1,
                affected_components=["Application", "API", "Database"],
                attack_vector="Network",
                impact="Unauthorized access, privilege escalation",
                remediation="Role-based access control, principle of least privilege",
                references=["https://owasp.org/www-community/controls/Access_Control"],
                discovered_at=datetime.now()
            )
        ]
        
        for vuln in default_vulnerabilities:
            self._store_vulnerability(vuln)
    
    def create_assessment(self, name: str, assessment_type: AssessmentType, 
                         target: str, scope: List[str], objectives: List[str]) -> str:
        """
        Create a new security assessment.
        
        Args:
            name: Assessment name
            assessment_type: Type of assessment
            target: Target system/application
            scope: Assessment scope
            objectives: Assessment objectives
            
        Returns:
            Assessment ID
        """
        assessment_id = f"ASSESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        methodology = self.methodologies.get(assessment_type, "Custom")
        
        assessment = SecurityAssessment(
            assessment_id=assessment_id,
            name=name,
            assessment_type=assessment_type,
            target=target,
            scope=scope,
            objectives=objectives,
            methodology=methodology,
            status=AssessmentStatus.PLANNED,
            start_date=datetime.now(),
            end_date=None,
            findings=[],
            risk_score=0.0,
            recommendations=[]
        )
        
        self.assessments[assessment_id] = assessment
        self._store_assessment(assessment)
        
        logger.info(f"Created security assessment: {assessment_id} - {name}")
        return assessment_id
    
    def start_assessment(self, assessment_id: str) -> bool:
        """
        Start a security assessment.
        
        Args:
            assessment_id: Assessment ID
            
        Returns:
            True if started successfully
        """
        if assessment_id not in self.assessments:
            return False
        
        assessment = self.assessments[assessment_id]
        assessment.status = AssessmentStatus.IN_PROGRESS
        assessment.start_date = datetime.now()
        
        self._update_assessment(assessment)
        
        logger.info(f"Started security assessment: {assessment_id}")
        return True
    
    def run_vulnerability_scan(self, target: str, scan_type: str = "comprehensive") -> str:
        """
        Run a vulnerability scan on a target.
        
        Args:
            target: Target to scan
            scan_type: Type of scan to run
            
        Returns:
            Scan ID
        """
        scan_id = f"SCAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        scan = VulnerabilityScan(
            scan_id=scan_id,
            target=target,
            scan_type=scan_type,
            start_time=datetime.now(),
            end_time=None,
            status=AssessmentStatus.IN_PROGRESS,
            vulnerabilities_found=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            info_count=0,
            scan_results=[]
        )
        
        # Run the actual scan
        vulnerabilities = self._perform_vulnerability_scan(target, scan_type)
        
        # Update scan results
        scan.scan_results = vulnerabilities
        scan.vulnerabilities_found = len(vulnerabilities)
        scan.critical_count = len([v for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL])
        scan.high_count = len([v for v in vulnerabilities if v.severity == SeverityLevel.HIGH])
        scan.medium_count = len([v for v in vulnerabilities if v.severity == SeverityLevel.MEDIUM])
        scan.low_count = len([v for v in vulnerabilities if v.severity == SeverityLevel.LOW])
        scan.info_count = len([v for v in vulnerabilities if v.severity == SeverityLevel.INFORMATIONAL])
        scan.end_time = datetime.now()
        scan.status = AssessmentStatus.COMPLETED
        
        self.scan_results.append(scan)
        self._store_vulnerability_scan(scan)
        
        logger.info(f"Completed vulnerability scan: {scan_id} - Found {scan.vulnerabilities_found} vulnerabilities")
        return scan_id
    
    def _perform_vulnerability_scan(self, target: str, scan_type: str) -> List[Vulnerability]:
        """Perform actual vulnerability scanning."""
        vulnerabilities = []
        
        try:
            # Network port scanning
            if scan_type in ["comprehensive", "network"]:
                vulnerabilities.extend(self._scan_network_ports(target))
            
            # Web application scanning
            if scan_type in ["comprehensive", "web"]:
                vulnerabilities.extend(self._scan_web_application(target))
            
            # SSL/TLS scanning
            if scan_type in ["comprehensive", "ssl"]:
                vulnerabilities.extend(self._scan_ssl_tls(target))
            
            # API security scanning
            if scan_type in ["comprehensive", "api"]:
                vulnerabilities.extend(self._scan_api_security(target))
            
        except Exception as e:
            logger.error(f"Error during vulnerability scan: {e}")
        
        return vulnerabilities
    
    def _scan_network_ports(self, target: str) -> List[Vulnerability]:
        """Scan for open network ports."""
        vulnerabilities = []
        
        try:
            # Common ports to scan
            common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3389, 5432, 3306, 1433]
            
            for port in common_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((target, port))
                    sock.close()
                    
                    if result == 0:
                        # Port is open - check for potential vulnerabilities
                        vuln = self._check_port_vulnerabilities(target, port)
                        if vuln:
                            vulnerabilities.append(vuln)
                
                except Exception:
                    continue
        
        except Exception as e:
            logger.error(f"Error scanning network ports: {e}")
        
        return vulnerabilities
    
    def _check_port_vulnerabilities(self, target: str, port: int) -> Optional[Vulnerability]:
        """Check for vulnerabilities on a specific port."""
        # Port-specific vulnerability checks
        port_vulns = {
            21: "FTP service may allow anonymous access",
            22: "SSH service may have weak authentication",
            23: "Telnet service transmits data in plaintext",
            25: "SMTP service may allow open relay",
            80: "HTTP service may be vulnerable to various attacks",
            443: "HTTPS service may have SSL/TLS vulnerabilities",
            3389: "RDP service may allow brute force attacks",
            5432: "PostgreSQL service may have default credentials",
            3306: "MySQL service may have default credentials",
            1433: "SQL Server service may have default credentials"
        }
        
        if port in port_vulns:
            return Vulnerability(
                vuln_id=f"PORT_{port}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"Open Port {port}",
                description=port_vulns[port],
                severity=SeverityLevel.MEDIUM,
                category="Network",
                cve_id=None,
                cvss_score=5.0,
                affected_components=[f"{target}:{port}"],
                attack_vector="Network",
                impact="Potential unauthorized access or information disclosure",
                remediation="Review service configuration and access controls",
                references=[],
                discovered_at=datetime.now()
            )
        
        return None
    
    def _scan_web_application(self, target: str) -> List[Vulnerability]:
        """Scan web application for vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Check for common web vulnerabilities
            url = f"http://{target}" if not target.startswith(('http://', 'https://')) else target
            
            # Test for common paths
            common_paths = [
                "/admin", "/login", "/api", "/.env", "/config", "/backup",
                "/test", "/dev", "/staging", "/.git", "/.svn", "/robots.txt"
            ]
            
            for path in common_paths:
                try:
                    response = requests.get(f"{url}{path}", timeout=5, allow_redirects=False)
                    
                    if response.status_code == 200:
                        vuln = Vulnerability(
                            vuln_id=f"WEB_{path.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            name=f"Exposed Path: {path}",
                            description=f"Path {path} is accessible and may expose sensitive information",
                            severity=SeverityLevel.MEDIUM,
                            category="Information Disclosure",
                            cve_id=None,
                            cvss_score=5.3,
                            affected_components=[f"{url}{path}"],
                            attack_vector="Network",
                            impact="Information disclosure, potential unauthorized access",
                            remediation="Restrict access to sensitive paths, implement proper authentication",
                            references=[],
                            discovered_at=datetime.now()
                        )
                        vulnerabilities.append(vuln)
                
                except Exception:
                    continue
        
        except Exception as e:
            logger.error(f"Error scanning web application: {e}")
        
        return vulnerabilities
    
    def _scan_ssl_tls(self, target: str) -> List[Vulnerability]:
        """Scan SSL/TLS configuration for vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Extract hostname and port from target
            if ':' in target:
                hostname, port = target.split(':')
                port = int(port)
            else:
                hostname = target
                port = 443
            
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            # Connect and get certificate info
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    version = ssock.version()
            
            # Check for weak SSL/TLS versions
            if version in ['SSLv2', 'SSLv3', 'TLSv1', 'TLSv1.1']:
                vuln = Vulnerability(
                    vuln_id=f"SSL_VERSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    name="Weak SSL/TLS Version",
                    description=f"Server supports weak SSL/TLS version: {version}",
                    severity=SeverityLevel.HIGH,
                    category="Cryptography",
                    cve_id="CVE-2014-3566",
                    cvss_score=7.4,
                    affected_components=[f"{hostname}:{port}"],
                    attack_vector="Network",
                    impact="Man-in-the-middle attacks, data interception",
                    remediation="Disable weak SSL/TLS versions, use TLS 1.2 or higher",
                    references=["https://tools.ietf.org/html/rfc7568"],
                    discovered_at=datetime.now()
                )
                vulnerabilities.append(vuln)
            
            # Check certificate validity
            if cert:
                not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                if not_after < datetime.now() + timedelta(days=30):
                    vuln = Vulnerability(
                        vuln_id=f"SSL_CERT_EXPIRY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        name="SSL Certificate Expiring Soon",
                        description=f"SSL certificate expires on {cert['notAfter']}",
                        severity=SeverityLevel.MEDIUM,
                        category="Cryptography",
                        cve_id=None,
                        cvss_score=4.3,
                        affected_components=[f"{hostname}:{port}"],
                        attack_vector="Network",
                        impact="Service disruption, potential security issues",
                        remediation="Renew SSL certificate before expiration",
                        references=[],
                        discovered_at=datetime.now()
                    )
                    vulnerabilities.append(vuln)
        
        except Exception as e:
            logger.error(f"Error scanning SSL/TLS: {e}")
        
        return vulnerabilities
    
    def _scan_api_security(self, target: str) -> List[Vulnerability]:
        """Scan API for security vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Common API endpoints to test
            api_endpoints = [
                "/api/v1/users", "/api/users", "/api/v1/auth", "/api/auth",
                "/api/v1/data", "/api/data", "/api/v1/admin", "/api/admin"
            ]
            
            base_url = f"http://{target}" if not target.startswith(('http://', 'https://')) else target
            
            for endpoint in api_endpoints:
                try:
                    # Test for authentication bypass
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    
                    if response.status_code == 200:
                        vuln = Vulnerability(
                            vuln_id=f"API_AUTH_BYPASS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            name="API Authentication Bypass",
                            description=f"API endpoint {endpoint} is accessible without authentication",
                            severity=SeverityLevel.HIGH,
                            category="Authentication",
                            cve_id="CWE-287",
                            cvss_score=7.5,
                            affected_components=[f"{base_url}{endpoint}"],
                            attack_vector="Network",
                            impact="Unauthorized access to API endpoints",
                            remediation="Implement proper authentication and authorization",
                            references=["https://owasp.org/www-community/controls/Authentication"],
                            discovered_at=datetime.now()
                        )
                        vulnerabilities.append(vuln)
                
                except Exception:
                    continue
        
        except Exception as e:
            logger.error(f"Error scanning API security: {e}")
        
        return vulnerabilities
    
    def create_threat_model(self, application_name: str, version: str, 
                           architecture: Dict[str, Any]) -> str:
        """
        Create a threat model for an application.
        
        Args:
            application_name: Name of the application
            version: Application version
            architecture: Application architecture
            
        Returns:
            Threat model ID
        """
        model_id = f"THREAT_MODEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        threat_model = ThreatModel(
            model_id=model_id,
            application_name=application_name,
            version=version,
            architecture=architecture,
            data_flows=[],
            trust_boundaries=[],
            threats=[],
            mitigations=[],
            risk_ratings={}
        )
        
        self._store_threat_model(threat_model)
        
        logger.info(f"Created threat model: {model_id} - {application_name}")
        return model_id
    
    def analyze_threats(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Analyze threats for a threat model.
        
        Args:
            model_id: Threat model ID
            
        Returns:
            List of identified threats
        """
        # Load threat model
        threat_model = self._load_threat_model(model_id)
        if not threat_model:
            return []
        
        # STRIDE threat analysis
        stride_threats = [
            {
                'threat_id': 'STRIDE_001',
                'category': 'Spoofing',
                'description': 'Impersonation of users or systems',
                'impact': 'Unauthorized access, identity theft',
                'likelihood': 'Medium',
                'severity': 'High'
            },
            {
                'threat_id': 'STRIDE_002',
                'category': 'Tampering',
                'description': 'Modification of data or code',
                'impact': 'Data integrity compromise, system compromise',
                'likelihood': 'Medium',
                'severity': 'High'
            },
            {
                'threat_id': 'STRIDE_003',
                'category': 'Repudiation',
                'description': 'Denial of actions or transactions',
                'impact': 'Audit trail compromise, legal issues',
                'likelihood': 'Low',
                'severity': 'Medium'
            },
            {
                'threat_id': 'STRIDE_004',
                'category': 'Information Disclosure',
                'description': 'Exposure of sensitive information',
                'impact': 'Privacy breach, competitive advantage loss',
                'likelihood': 'High',
                'severity': 'High'
            },
            {
                'threat_id': 'STRIDE_005',
                'category': 'Denial of Service',
                'description': 'Service unavailability',
                'impact': 'Business disruption, revenue loss',
                'likelihood': 'Medium',
                'severity': 'Medium'
            },
            {
                'threat_id': 'STRIDE_006',
                'category': 'Elevation of Privilege',
                'description': 'Unauthorized privilege escalation',
                'impact': 'System compromise, data breach',
                'likelihood': 'Low',
                'severity': 'Critical'
            }
        ]
        
        threat_model.threats = stride_threats
        self._update_threat_model(threat_model)
        
        return stride_threats
    
    def generate_security_report(self, assessment_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive security assessment report.
        
        Args:
            assessment_id: Assessment ID
            
        Returns:
            Security report dictionary
        """
        if assessment_id not in self.assessments:
            return {}
        
        assessment = self.assessments[assessment_id]
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(assessment.findings)
        assessment.risk_score = risk_score
        
        # Generate recommendations
        recommendations = self._generate_recommendations(assessment.findings)
        assessment.recommendations = recommendations
        
        # Create report
        report = {
            'assessment_id': assessment.assessment_id,
            'name': assessment.name,
            'assessment_type': assessment.assessment_type.value,
            'target': assessment.target,
            'scope': assessment.scope,
            'objectives': assessment.objectives,
            'methodology': assessment.methodology,
            'status': assessment.status.value,
            'start_date': assessment.start_date.isoformat(),
            'end_date': assessment.end_date.isoformat() if assessment.end_date else None,
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'summary': {
                'total_findings': len(assessment.findings),
                'critical_findings': len([f for f in assessment.findings if f.severity == SeverityLevel.CRITICAL]),
                'high_findings': len([f for f in assessment.findings if f.severity == SeverityLevel.HIGH]),
                'medium_findings': len([f for f in assessment.findings if f.severity == SeverityLevel.MEDIUM]),
                'low_findings': len([f for f in assessment.findings if f.severity == SeverityLevel.LOW])
            },
            'findings': [
                {
                    'vuln_id': f.vuln_id,
                    'name': f.name,
                    'description': f.description,
                    'severity': f.severity.value,
                    'category': f.category,
                    'cve_id': f.cve_id,
                    'cvss_score': f.cvss_score,
                    'affected_components': f.affected_components,
                    'impact': f.impact,
                    'remediation': f.remediation
                }
                for f in assessment.findings
            ],
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        report_path = f"reports/security_assessment_{assessment_id}.json"
        Path("reports").mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        assessment.report_path = report_path
        self._update_assessment(assessment)
        
        return report
    
    def _calculate_risk_score(self, findings: List[Vulnerability]) -> float:
        """Calculate overall risk score based on findings."""
        if not findings:
            return 0.0
        
        severity_weights = {
            SeverityLevel.CRITICAL: 10.0,
            SeverityLevel.HIGH: 7.5,
            SeverityLevel.MEDIUM: 5.0,
            SeverityLevel.LOW: 2.5,
            SeverityLevel.INFORMATIONAL: 1.0
        }
        
        total_weighted_score = sum(severity_weights.get(f.severity, 0) for f in findings)
        max_possible_score = len(findings) * 10.0
        
        return (total_weighted_score / max_possible_score * 100) if max_possible_score > 0 else 0.0
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level based on risk score."""
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_recommendations(self, findings: List[Vulnerability]) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        # Group findings by category
        categories = {}
        for finding in findings:
            if finding.category not in categories:
                categories[finding.category] = []
            categories[finding.category].append(finding)
        
        # Generate category-specific recommendations
        for category, category_findings in categories.items():
            if category == "Injection":
                recommendations.append("Implement input validation and parameterized queries to prevent injection attacks")
            elif category == "Cross-Site Scripting":
                recommendations.append("Implement output encoding and Content Security Policy headers")
            elif category == "Authentication":
                recommendations.append("Implement strong authentication mechanisms and multi-factor authentication")
            elif category == "Cryptography":
                recommendations.append("Use strong encryption algorithms and secure key management")
            elif category == "Access Control":
                recommendations.append("Implement proper access controls and principle of least privilege")
            elif category == "Configuration":
                recommendations.append("Review and harden system configurations")
            elif category == "Data Protection":
                recommendations.append("Implement data encryption and access controls")
            elif category == "Network":
                recommendations.append("Review network security and firewall configurations")
        
        # Add general recommendations
        if len(findings) > 0:
            recommendations.extend([
                "Implement regular security assessments and penetration testing",
                "Establish incident response procedures",
                "Provide security awareness training to staff",
                "Implement security monitoring and logging",
                "Regularly update and patch systems"
            ])
        
        return recommendations
    
    def _store_assessment(self, assessment: SecurityAssessment) -> None:
        """Store security assessment in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO security_assessments 
            (assessment_id, name, assessment_type, target, scope, objectives, methodology,
             status, start_date, end_date, findings, risk_score, recommendations, report_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assessment.assessment_id, assessment.name, assessment.assessment_type.value,
            assessment.target, json.dumps(assessment.scope), json.dumps(assessment.objectives),
            assessment.methodology, assessment.status.value, assessment.start_date.isoformat(),
            assessment.end_date.isoformat() if assessment.end_date else None,
            json.dumps([f.__dict__ for f in assessment.findings]), assessment.risk_score,
            json.dumps(assessment.recommendations), assessment.report_path,
            assessment.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_assessment(self, assessment: SecurityAssessment) -> None:
        """Update security assessment in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE security_assessments
            SET status = ?, start_date = ?, end_date = ?, findings = ?, risk_score = ?, 
                recommendations = ?, report_path = ?
            WHERE assessment_id = ?
        """, (
            assessment.status.value, assessment.start_date.isoformat(),
            assessment.end_date.isoformat() if assessment.end_date else None,
            json.dumps([f.__dict__ for f in assessment.findings]), assessment.risk_score,
            json.dumps(assessment.recommendations), assessment.report_path,
            assessment.assessment_id
        ))
        
        conn.commit()
        conn.close()
    
    def _store_vulnerability(self, vulnerability: Vulnerability) -> None:
        """Store vulnerability in database."""
        self.vulnerabilities.append(vulnerability)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO vulnerabilities 
            (vuln_id, name, description, severity, category, cve_id, cvss_score,
             affected_components, attack_vector, impact, remediation, references,
             discovered_at, verified, exploited, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vulnerability.vuln_id, vulnerability.name, vulnerability.description,
            vulnerability.severity.value, vulnerability.category, vulnerability.cve_id,
            vulnerability.cvss_score, json.dumps(vulnerability.affected_components),
            vulnerability.attack_vector, vulnerability.impact, vulnerability.remediation,
            json.dumps(vulnerability.references), vulnerability.discovered_at.isoformat(),
            vulnerability.verified, vulnerability.exploited, vulnerability.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_vulnerability_scan(self, scan: VulnerabilityScan) -> None:
        """Store vulnerability scan in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO vulnerability_scans 
            (scan_id, target, scan_type, start_time, end_time, status, vulnerabilities_found,
             critical_count, high_count, medium_count, low_count, info_count, scan_results, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan.scan_id, scan.target, scan.scan_type, scan.start_time.isoformat(),
            scan.end_time.isoformat() if scan.end_time else None, scan.status.value,
            scan.vulnerabilities_found, scan.critical_count, scan.high_count,
            scan.medium_count, scan.low_count, scan.info_count,
            json.dumps([v.__dict__ for v in scan.scan_results]), scan.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_threat_model(self, threat_model: ThreatModel) -> None:
        """Store threat model in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO threat_models 
            (model_id, application_name, version, architecture, data_flows, trust_boundaries,
             threats, mitigations, risk_ratings, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            threat_model.model_id, threat_model.application_name, threat_model.version,
            json.dumps(threat_model.architecture), json.dumps(threat_model.data_flows),
            json.dumps(threat_model.trust_boundaries), json.dumps(threat_model.threats),
            json.dumps(threat_model.mitigations), json.dumps(threat_model.risk_ratings),
            threat_model.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_threat_model(self, threat_model: ThreatModel) -> None:
        """Update threat model in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE threat_models
            SET threats = ?, mitigations = ?, risk_ratings = ?
            WHERE model_id = ?
        """, (
            json.dumps(threat_model.threats), json.dumps(threat_model.mitigations),
            json.dumps(threat_model.risk_ratings), threat_model.model_id
        ))
        
        conn.commit()
        conn.close()
    
    def _load_threat_model(self, model_id: str) -> Optional[ThreatModel]:
        """Load threat model from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM threat_models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return ThreatModel(
                model_id=row[0],
                application_name=row[1],
                version=row[2],
                architecture=json.loads(row[3]),
                data_flows=json.loads(row[4]),
                trust_boundaries=json.loads(row[5]),
                threats=json.loads(row[6]),
                mitigations=json.loads(row[7]),
                risk_ratings=json.loads(row[8]),
                created_at=datetime.fromisoformat(row[9])
            )
        
        return None
    
    def get_assessment_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Get assessment summary for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Assessment summary dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get assessment counts by type
        cursor.execute("""
            SELECT assessment_type, COUNT(*) as count
            FROM security_assessments
            WHERE start_date BETWEEN ? AND ?
            GROUP BY assessment_type
        """, (start_date.isoformat(), end_date.isoformat()))
        
        assessment_type_counts = dict(cursor.fetchall())
        
        # Get vulnerability counts by severity
        cursor.execute("""
            SELECT severity, COUNT(*) as count
            FROM vulnerabilities
            WHERE discovered_at BETWEEN ? AND ?
            GROUP BY severity
        """, (start_date.isoformat(), end_date.isoformat()))
        
        severity_counts = dict(cursor.fetchall())
        
        # Get scan statistics
        cursor.execute("""
            SELECT COUNT(*) as total_scans, AVG(vulnerabilities_found) as avg_vulns
            FROM vulnerability_scans
            WHERE start_time BETWEEN ? AND ?
        """, (start_date.isoformat(), end_date.isoformat()))
        
        total_scans, avg_vulns = cursor.fetchone()
        
        conn.close()
        
        return {
            'summary': {
                'total_assessments': sum(assessment_type_counts.values()),
                'total_scans': total_scans or 0,
                'average_vulnerabilities': avg_vulns or 0
            },
            'assessment_type_breakdown': assessment_type_counts,
            'severity_breakdown': severity_counts,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat()
        }
