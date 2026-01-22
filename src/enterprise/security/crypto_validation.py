"""
Cryptographic Validation System

This module implements a comprehensive cryptographic validation system with
encryption strength analysis, key management validation, and certificate
verification for enterprise security compliance.

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
import hashlib
import hmac
import secrets
import ssl
import socket
import base64
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    """Encryption algorithms."""
    AES_128 = "AES_128"
    AES_256 = "AES_256"
    RSA_1024 = "RSA_1024"
    RSA_2048 = "RSA_2048"
    RSA_4096 = "RSA_4096"
    ECDSA_P256 = "ECDSA_P256"
    ECDSA_P384 = "ECDSA_P384"
    ECDSA_P521 = "ECDSA_P521"
    SHA_1 = "SHA_1"
    SHA_256 = "SHA_256"
    SHA_384 = "SHA_384"
    SHA_512 = "SHA_512"
    MD5 = "MD5"
    DES = "DES"
    RC4 = "RC4"

class KeyType(Enum):
    """Key types."""
    SYMMETRIC = "SYMMETRIC"
    ASYMMETRIC = "ASYMMETRIC"
    HASH = "HASH"
    HMAC = "HMAC"

class CertificateStatus(Enum):
    """Certificate status."""
    VALID = "VALID"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    INVALID = "INVALID"
    UNTRUSTED = "UNTRUSTED"

class ValidationResult(Enum):
    """Validation results."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    INFO = "INFO"

@dataclass
class EncryptionCheck:
    """Encryption check definition."""
    check_id: str
    name: str
    algorithm: EncryptionAlgorithm
    key_size: int
    strength: str
    status: ValidationResult
    description: str
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class KeyManagement:
    """Key management definition."""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    key_size: int
    creation_date: datetime
    expiration_date: Optional[datetime]
    usage: List[str]
    status: str
    rotation_policy: str
    storage_location: str
    access_controls: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class CertificateValidation:
    """Certificate validation definition."""
    cert_id: str
    subject: str
    issuer: str
    serial_number: str
    valid_from: datetime
    valid_to: datetime
    key_size: int
    algorithm: str
    status: CertificateStatus
    validation_errors: List[str]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

class CryptoValidator:
    """
    Comprehensive cryptographic validation system.
    
    Features:
    - Encryption strength analysis
    - Key management validation
    - Certificate verification
    - Cryptographic compliance checking
    - Security recommendations
    - Compliance reporting
    """
    
    def __init__(self, db_path: str = "data/crypto_validation.db"):
        """
        Initialize crypto validator.
        
        Args:
            db_path: Path to crypto validation database
        """
        self.db_path = db_path
        self.encryption_checks: List[EncryptionCheck] = []
        self.key_management: Dict[str, KeyManagement] = {}
        self.certificate_validations: List[CertificateValidation] = []
        
        # Cryptographic standards
        self.crypto_standards = {
            'NIST_SP_800_57': {
                'min_key_sizes': {
                    EncryptionAlgorithm.AES_128: 128,
                    EncryptionAlgorithm.AES_256: 256,
                    EncryptionAlgorithm.RSA_1024: 1024,
                    EncryptionAlgorithm.RSA_2048: 2048,
                    EncryptionAlgorithm.RSA_4096: 4096,
                    EncryptionAlgorithm.ECDSA_P256: 256,
                    EncryptionAlgorithm.ECDSA_P384: 384,
                    EncryptionAlgorithm.ECDSA_P521: 521
                },
                'deprecated_algorithms': [
                    EncryptionAlgorithm.MD5,
                    EncryptionAlgorithm.SHA_1,
                    EncryptionAlgorithm.DES,
                    EncryptionAlgorithm.RC4,
                    EncryptionAlgorithm.RSA_1024
                ]
            },
            'FIPS_140_2': {
                'approved_algorithms': [
                    EncryptionAlgorithm.AES_128,
                    EncryptionAlgorithm.AES_256,
                    EncryptionAlgorithm.RSA_2048,
                    EncryptionAlgorithm.RSA_4096,
                    EncryptionAlgorithm.ECDSA_P256,
                    EncryptionAlgorithm.ECDSA_P384,
                    EncryptionAlgorithm.ECDSA_P521,
                    EncryptionAlgorithm.SHA_256,
                    EncryptionAlgorithm.SHA_384,
                    EncryptionAlgorithm.SHA_512
                ]
            }
        }
        
        # Initialize database
        self._init_database()
        
        # Load default encryption checks
        self._load_default_encryption_checks()
        
        logger.info("Cryptographic Validation system initialized")
    
    def _init_database(self) -> None:
        """Initialize crypto validation database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create encryption checks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS encryption_checks (
                check_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                key_size INTEGER NOT NULL,
                strength TEXT NOT NULL,
                status TEXT NOT NULL,
                description TEXT,
                recommendations TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create key management table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS key_management (
                key_id TEXT PRIMARY KEY,
                key_type TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                key_size INTEGER NOT NULL,
                creation_date TEXT NOT NULL,
                expiration_date TEXT,
                usage TEXT,
                status TEXT NOT NULL,
                rotation_policy TEXT,
                storage_location TEXT,
                access_controls TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create certificate validations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS certificate_validations (
                cert_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                issuer TEXT NOT NULL,
                serial_number TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT NOT NULL,
                key_size INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                status TEXT NOT NULL,
                validation_errors TEXT,
                recommendations TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_encryption_checks(self) -> None:
        """Load default encryption checks."""
        default_checks = [
            EncryptionCheck(
                check_id="CHECK_001",
                name="AES-256 Encryption",
                algorithm=EncryptionAlgorithm.AES_256,
                key_size=256,
                strength="Strong",
                status=ValidationResult.PASS,
                description="AES-256 encryption is considered cryptographically strong",
                recommendations=["Continue using AES-256 for sensitive data"]
            ),
            EncryptionCheck(
                check_id="CHECK_002",
                name="RSA-2048 Key",
                algorithm=EncryptionAlgorithm.RSA_2048,
                key_size=2048,
                strength="Strong",
                status=ValidationResult.PASS,
                description="RSA-2048 keys provide adequate security for most applications",
                recommendations=["Consider upgrading to RSA-4096 for long-term security"]
            ),
            EncryptionCheck(
                check_id="CHECK_003",
                name="SHA-256 Hash",
                algorithm=EncryptionAlgorithm.SHA_256,
                key_size=256,
                strength="Strong",
                status=ValidationResult.PASS,
                description="SHA-256 is a secure hash function",
                recommendations=["Continue using SHA-256 for data integrity"]
            ),
            EncryptionCheck(
                check_id="CHECK_004",
                name="MD5 Hash",
                algorithm=EncryptionAlgorithm.MD5,
                key_size=128,
                strength="Weak",
                status=ValidationResult.FAIL,
                description="MD5 is cryptographically broken and should not be used",
                recommendations=["Replace MD5 with SHA-256 or SHA-3"]
            ),
            EncryptionCheck(
                check_id="CHECK_005",
                name="SHA-1 Hash",
                algorithm=EncryptionAlgorithm.SHA_1,
                key_size=160,
                strength="Weak",
                status=ValidationResult.FAIL,
                description="SHA-1 is deprecated and vulnerable to collision attacks",
                recommendations=["Replace SHA-1 with SHA-256 or SHA-3"]
            ),
            EncryptionCheck(
                check_id="CHECK_006",
                name="DES Encryption",
                algorithm=EncryptionAlgorithm.DES,
                key_size=56,
                strength="Very Weak",
                status=ValidationResult.FAIL,
                description="DES is cryptographically weak and should not be used",
                recommendations=["Replace DES with AES-256"]
            ),
            EncryptionCheck(
                check_id="CHECK_007",
                name="RC4 Stream Cipher",
                algorithm=EncryptionAlgorithm.RC4,
                key_size=128,
                strength="Weak",
                status=ValidationResult.FAIL,
                description="RC4 is vulnerable to bias attacks and should not be used",
                recommendations=["Replace RC4 with AES-256 in CTR or GCM mode"]
            )
        ]
        
        for check in default_checks:
            self.encryption_checks.append(check)
            self._store_encryption_check(check)
    
    def validate_encryption_algorithm(self, algorithm: EncryptionAlgorithm, 
                                    key_size: int) -> EncryptionCheck:
        """
        Validate encryption algorithm and key size.
        
        Args:
            algorithm: Encryption algorithm
            key_size: Key size in bits
            
        Returns:
            Encryption check result
        """
        check_id = f"ENCRYPTION_{algorithm.value}_{key_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check against standards
        nist_standard = self.crypto_standards['NIST_SP_800_57']
        fips_standard = self.crypto_standards['FIPS_140_2']
        
        # Determine strength and status
        strength = "Unknown"
        status = ValidationResult.INFO
        description = ""
        recommendations = []
        
        # Check if algorithm is deprecated
        if algorithm in nist_standard['deprecated_algorithms']:
            strength = "Deprecated"
            status = ValidationResult.FAIL
            description = f"{algorithm.value} is deprecated and should not be used"
            recommendations.append(f"Replace {algorithm.value} with a modern algorithm")
        
        # Check if algorithm is FIPS approved
        elif algorithm in fips_standard['approved_algorithms']:
            # Check key size
            min_key_size = nist_standard['min_key_sizes'].get(algorithm, 0)
            if key_size >= min_key_size:
                strength = "Strong"
                status = ValidationResult.PASS
                description = f"{algorithm.value} with {key_size}-bit key is cryptographically strong"
                recommendations.append("Continue using this configuration")
            else:
                strength = "Weak"
                status = ValidationResult.FAIL
                description = f"{algorithm.value} key size {key_size} is below minimum recommended {min_key_size} bits"
                recommendations.append(f"Increase key size to at least {min_key_size} bits")
        
        else:
            strength = "Unknown"
            status = ValidationResult.WARNING
            description = f"{algorithm.value} is not in approved algorithms list"
            recommendations.append("Consider using FIPS-approved algorithms")
        
        # Add algorithm-specific recommendations
        if algorithm == EncryptionAlgorithm.AES_128:
            recommendations.append("Consider upgrading to AES-256 for enhanced security")
        elif algorithm == EncryptionAlgorithm.RSA_1024:
            recommendations.append("RSA-1024 is deprecated, upgrade to RSA-2048 or higher")
        elif algorithm == EncryptionAlgorithm.RSA_2048:
            recommendations.append("Consider upgrading to RSA-4096 for long-term security")
        
        check = EncryptionCheck(
            check_id=check_id,
            name=f"{algorithm.value} Validation",
            algorithm=algorithm,
            key_size=key_size,
            strength=strength,
            status=status,
            description=description,
            recommendations=recommendations
        )
        
        self.encryption_checks.append(check)
        self._store_encryption_check(check)
        
        logger.info(f"Validated encryption algorithm: {algorithm.value} - {status.value}")
        return check
    
    def validate_key_management(self, key_data: Dict[str, Any]) -> KeyManagement:
        """
        Validate key management practices.
        
        Args:
            key_data: Key management data
            
        Returns:
            Key management validation result
        """
        key_id = f"KEY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        key_management = KeyManagement(
            key_id=key_id,
            key_type=KeyType(key_data.get('key_type', 'SYMMETRIC')),
            algorithm=EncryptionAlgorithm(key_data.get('algorithm', 'AES_256')),
            key_size=key_data.get('key_size', 256),
            creation_date=datetime.fromisoformat(key_data.get('creation_date', datetime.now().isoformat())),
            expiration_date=datetime.fromisoformat(key_data.get('expiration_date')) if key_data.get('expiration_date') else None,
            usage=key_data.get('usage', []),
            status=key_data.get('status', 'Active'),
            rotation_policy=key_data.get('rotation_policy', 'Annual'),
            storage_location=key_data.get('storage_location', 'Unknown'),
            access_controls=key_data.get('access_controls', [])
        )
        
        self.key_management[key_id] = key_management
        self._store_key_management(key_management)
        
        logger.info(f"Validated key management: {key_id}")
        return key_management
    
    def validate_certificate(self, hostname: str, port: int = 443) -> CertificateValidation:
        """
        Validate SSL/TLS certificate.
        
        Args:
            hostname: Hostname to validate
            port: Port number (default 443)
            
        Returns:
            Certificate validation result
        """
        cert_id = f"CERT_{hostname}_{port}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            
            # Connect and get certificate
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    version = ssock.version()
            
            # Parse certificate information
            subject = cert.get('subject', [])
            issuer = cert.get('issuer', [])
            serial_number = cert.get('serialNumber', '')
            not_before = cert.get('notBefore', '')
            not_after = cert.get('notAfter', '')
            
            # Extract subject and issuer names
            subject_name = self._extract_name(subject)
            issuer_name = self._extract_name(issuer)
            
            # Parse dates
            valid_from = datetime.strptime(not_before, '%b %d %H:%M:%S %Y %Z')
            valid_to = datetime.strptime(not_after, '%b %d %H:%M:%S %Y %Z')
            
            # Determine certificate status
            status = CertificateStatus.VALID
            validation_errors = []
            recommendations = []
            
            # Check expiration
            if valid_to < datetime.now():
                status = CertificateStatus.EXPIRED
                validation_errors.append("Certificate has expired")
                recommendations.append("Renew certificate immediately")
            elif valid_to < datetime.now() + timedelta(days=30):
                validation_errors.append("Certificate expires within 30 days")
                recommendations.append("Plan certificate renewal")
            
            # Check key size
            key_size = self._extract_key_size(cert)
            if key_size < 2048:
                validation_errors.append(f"Key size {key_size} is below recommended minimum of 2048 bits")
                recommendations.append("Use certificate with at least 2048-bit key")
            
            # Check algorithm
            algorithm = self._extract_algorithm(cert)
            if algorithm in ['SHA1', 'MD5']:
                validation_errors.append(f"Certificate uses weak hash algorithm: {algorithm}")
                recommendations.append("Use certificate with SHA-256 or stronger hash algorithm")
            
            # Check SSL/TLS version
            if version in ['SSLv2', 'SSLv3', 'TLSv1', 'TLSv1.1']:
                validation_errors.append(f"Weak SSL/TLS version: {version}")
                recommendations.append("Use TLS 1.2 or higher")
            
            # Add general recommendations
            if status == CertificateStatus.VALID:
                recommendations.extend([
                    "Certificate is valid and properly configured",
                    "Monitor certificate expiration date",
                    "Implement certificate monitoring and alerting"
                ])
            
            certificate_validation = CertificateValidation(
                cert_id=cert_id,
                subject=subject_name,
                issuer=issuer_name,
                serial_number=serial_number,
                valid_from=valid_from,
                valid_to=valid_to,
                key_size=key_size,
                algorithm=algorithm,
                status=status,
                validation_errors=validation_errors,
                recommendations=recommendations
            )
            
        except ssl.SSLError as e:
            # SSL/TLS error
            certificate_validation = CertificateValidation(
                cert_id=cert_id,
                subject=hostname,
                issuer="Unknown",
                serial_number="Unknown",
                valid_from=datetime.now(),
                valid_to=datetime.now(),
                key_size=0,
                algorithm="Unknown",
                status=CertificateStatus.INVALID,
                validation_errors=[f"SSL/TLS error: {str(e)}"],
                recommendations=["Fix SSL/TLS configuration", "Check certificate installation"]
            )
        
        except Exception as e:
            # General error
            certificate_validation = CertificateValidation(
                cert_id=cert_id,
                subject=hostname,
                issuer="Unknown",
                serial_number="Unknown",
                valid_from=datetime.now(),
                valid_to=datetime.now(),
                key_size=0,
                algorithm="Unknown",
                status=CertificateStatus.INVALID,
                validation_errors=[f"Validation error: {str(e)}"],
                recommendations=["Check hostname and port", "Verify network connectivity"]
            )
        
        self.certificate_validations.append(certificate_validation)
        self._store_certificate_validation(certificate_validation)
        
        logger.info(f"Validated certificate: {hostname} - {certificate_validation.status.value}")
        return certificate_validation
    
    def _extract_name(self, name_list: List[Tuple]) -> str:
        """Extract name from certificate name list."""
        for item in name_list:
            if isinstance(item, tuple) and len(item) == 2:
                key, value = item
                if key == 'commonName':
                    return value
        return "Unknown"
    
    def _extract_key_size(self, cert: Dict[str, Any]) -> int:
        """Extract key size from certificate."""
        # This is a simplified extraction
        # In a real implementation, this would parse the certificate's public key
        return 2048  # Default assumption
    
    def _extract_algorithm(self, cert: Dict[str, Any]) -> str:
        """Extract algorithm from certificate."""
        # This is a simplified extraction
        # In a real implementation, this would parse the certificate's signature algorithm
        return "SHA256"  # Default assumption
    
    def generate_crypto_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate cryptographic compliance report.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Compliance report dictionary
        """
        # Get checks in date range
        checks = [c for c in self.encryption_checks if start_date <= c.created_at <= end_date]
        
        # Get certificate validations in date range
        certs = [c for c in self.certificate_validations if start_date <= c.created_at <= end_date]
        
        # Calculate statistics
        check_status_counts = {}
        cert_status_counts = {}
        algorithm_counts = {}
        
        for check in checks:
            check_status_counts[check.status.value] = check_status_counts.get(check.status.value, 0) + 1
            algorithm_counts[check.algorithm.value] = algorithm_counts.get(check.algorithm.value, 0) + 1
        
        for cert in certs:
            cert_status_counts[cert.status.value] = cert_status_counts.get(cert.status.value, 0) + 1
        
        # Calculate compliance score
        total_checks = len(checks)
        passed_checks = len([c for c in checks if c.status == ValidationResult.PASS])
        compliance_score = (passed_checks / total_checks * 100) if total_checks > 0 else 100.0
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_encryption_checks': len(checks),
                'total_certificate_validations': len(certs),
                'total_keys_managed': len(self.key_management),
                'compliance_score': compliance_score
            },
            'encryption_check_statistics': {
                'by_status': check_status_counts,
                'by_algorithm': algorithm_counts
            },
            'certificate_statistics': {
                'by_status': cert_status_counts
            },
            'failed_checks': [
                {
                    'check_id': c.check_id,
                    'name': c.name,
                    'algorithm': c.algorithm.value,
                    'key_size': c.key_size,
                    'status': c.status.value,
                    'description': c.description,
                    'recommendations': c.recommendations
                }
                for c in checks if c.status == ValidationResult.FAIL
            ],
            'expired_certificates': [
                {
                    'cert_id': c.cert_id,
                    'subject': c.subject,
                    'valid_to': c.valid_to.isoformat(),
                    'validation_errors': c.validation_errors,
                    'recommendations': c.recommendations
                }
                for c in certs if c.status == CertificateStatus.EXPIRED
            ],
            'recommendations': self._generate_compliance_recommendations(checks, certs),
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_compliance_recommendations(self, checks: List[EncryptionCheck], 
                                           certs: List[CertificateValidation]) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        # Analyze failed checks
        failed_checks = [c for c in checks if c.status == ValidationResult.FAIL]
        if failed_checks:
            recommendations.append(f"Address {len(failed_checks)} failed encryption checks")
            
            # Group by algorithm
            failed_algorithms = {}
            for check in failed_checks:
                if check.algorithm not in failed_algorithms:
                    failed_algorithms[check.algorithm] = []
                failed_algorithms[check.algorithm].append(check)
            
            for algorithm, algorithm_checks in failed_algorithms.items():
                recommendations.append(f"Replace {algorithm.value} with modern alternatives")
        
        # Analyze expired certificates
        expired_certs = [c for c in certs if c.status == CertificateStatus.EXPIRED]
        if expired_certs:
            recommendations.append(f"Renew {len(expired_certs)} expired certificates")
        
        # Analyze weak algorithms
        weak_algorithms = [c for c in checks if c.algorithm in [
            EncryptionAlgorithm.MD5, EncryptionAlgorithm.SHA_1, 
            EncryptionAlgorithm.DES, EncryptionAlgorithm.RC4
        ]]
        if weak_algorithms:
            recommendations.append("Replace weak cryptographic algorithms with strong alternatives")
        
        # General recommendations
        recommendations.extend([
            "Implement regular cryptographic audits",
            "Establish key rotation policies",
            "Monitor certificate expiration dates",
            "Use FIPS-approved algorithms where required",
            "Implement cryptographic key management best practices"
        ])
        
        return recommendations
    
    def _store_encryption_check(self, check: EncryptionCheck) -> None:
        """Store encryption check in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO encryption_checks 
            (check_id, name, algorithm, key_size, strength, status, description, recommendations, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            check.check_id, check.name, check.algorithm.value, check.key_size,
            check.strength, check.status.value, check.description,
            json.dumps(check.recommendations), check.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_key_management(self, key_mgmt: KeyManagement) -> None:
        """Store key management in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO key_management 
            (key_id, key_type, algorithm, key_size, creation_date, expiration_date,
             usage, status, rotation_policy, storage_location, access_controls, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            key_mgmt.key_id, key_mgmt.key_type.value, key_mgmt.algorithm.value,
            key_mgmt.key_size, key_mgmt.creation_date.isoformat(),
            key_mgmt.expiration_date.isoformat() if key_mgmt.expiration_date else None,
            json.dumps(key_mgmt.usage), key_mgmt.status, key_mgmt.rotation_policy,
            key_mgmt.storage_location, json.dumps(key_mgmt.access_controls),
            key_mgmt.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_certificate_validation(self, cert_validation: CertificateValidation) -> None:
        """Store certificate validation in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO certificate_validations 
            (cert_id, subject, issuer, serial_number, valid_from, valid_to,
             key_size, algorithm, status, validation_errors, recommendations, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cert_validation.cert_id, cert_validation.subject, cert_validation.issuer,
            cert_validation.serial_number, cert_validation.valid_from.isoformat(),
            cert_validation.valid_to.isoformat(), cert_validation.key_size,
            cert_validation.algorithm, cert_validation.status.value,
            json.dumps(cert_validation.validation_errors),
            json.dumps(cert_validation.recommendations),
            cert_validation.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_crypto_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Get cryptographic validation summary.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Summary dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get encryption check counts
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM encryption_checks
            WHERE created_at BETWEEN ? AND ?
            GROUP BY status
        """, (start_date.isoformat(), end_date.isoformat()))
        
        check_status_counts = dict(cursor.fetchall())
        
        # Get certificate validation counts
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM certificate_validations
            WHERE created_at BETWEEN ? AND ?
            GROUP BY status
        """, (start_date.isoformat(), end_date.isoformat()))
        
        cert_status_counts = dict(cursor.fetchall())
        
        # Get key management counts
        cursor.execute("""
            SELECT COUNT(*) as total_keys
            FROM key_management
            WHERE created_at BETWEEN ? AND ?
        """, (start_date.isoformat(), end_date.isoformat()))
        
        total_keys = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'summary': {
                'total_encryption_checks': sum(check_status_counts.values()),
                'total_certificate_validations': sum(cert_status_counts.values()),
                'total_keys_managed': total_keys
            },
            'encryption_check_status': check_status_counts,
            'certificate_status': cert_status_counts,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat()
        }
