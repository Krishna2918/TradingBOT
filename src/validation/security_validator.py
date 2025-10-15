"""
Security Validator Module

This module implements comprehensive security validation to prevent data leaks,
detect PII exposure, and ensure no credentials are exposed in code or logs.
"""

import re
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class SecurityIssue:
    """Represents a security issue found during validation."""
    issue_type: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    description: str
    location: str
    recommendation: str
    timestamp: datetime

@dataclass
class SecurityReport:
    """Comprehensive security validation report."""
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    issues: List[SecurityIssue]
    validation_timestamp: datetime
    overall_status: str  # "SECURE", "WARNING", "CRITICAL"

class SecurityValidator:
    """Comprehensive security validation system."""
    
    def __init__(self):
        # Common API key patterns
        self.api_key_patterns = [
            r'api[_-]?key["\s]*[:=]["\s]*([a-zA-Z0-9_-]{20,})',
            r'access[_-]?token["\s]*[:=]["\s]*([a-zA-Z0-9_-]{20,})',
            r'secret[_-]?key["\s]*[:=]["\s]*([a-zA-Z0-9_-]{20,})',
            r'bearer["\s]*[:=]["\s]*([a-zA-Z0-9_-]{20,})',
            r'token["\s]*[:=]["\s]*([a-zA-Z0-9_-]{20,})',
            r'password["\s]*[:=]["\s]*([^"\s]{8,})',
            r'passwd["\s]*[:=]["\s]*([^"\s]{8,})',
            r'pwd["\s]*[:=]["\s]*([^"\s]{8,})'
        ]
        
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        }
        
        # File extensions to check
        self.sensitive_extensions = ['.env', '.key', '.pem', '.p12', '.pfx', '.crt', '.cer']
        
        # Common credential keywords
        self.credential_keywords = [
            'password', 'passwd', 'pwd', 'secret', 'private', 'key', 'token',
            'api_key', 'access_token', 'bearer', 'auth', 'credential'
        ]
        
        logger.info("Security Validator initialized")
    
    def detect_api_key_leaks(self, text: str, context: str = "unknown") -> List[SecurityIssue]:
        """Detect potential API key leaks in text."""
        issues = []
        
        for pattern in self.api_key_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Check if it's not a placeholder or example
                matched_text = match.group(1) if match.groups() else match.group(0)
                
                if self._is_placeholder(matched_text):
                    continue
                
                # Hash the potential key for logging (don't log actual keys)
                key_hash = hashlib.sha256(matched_text.encode()).hexdigest()[:8]
                
                issue = SecurityIssue(
                    issue_type="API_KEY_LEAK",
                    severity="CRITICAL",
                    description=f"Potential API key detected (hash: {key_hash})",
                    location=context,
                    recommendation="Remove or mask the API key. Use environment variables or secure storage.",
                    timestamp=datetime.now()
                )
                issues.append(issue)
        
        return issues
    
    def detect_pii_leaks(self, data: Any, context: str = "unknown") -> List[SecurityIssue]:
        """Detect PII leaks in data."""
        issues = []
        
        # Convert data to string for pattern matching
        text = str(data)
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Check if it's not a placeholder or example
                matched_text = match.group(0)
                
                if self._is_placeholder(matched_text):
                    continue
                
                # Hash the PII for logging
                pii_hash = hashlib.sha256(matched_text.encode()).hexdigest()[:8]
                
                issue = SecurityIssue(
                    issue_type="PII_LEAK",
                    severity="HIGH",
                    description=f"Potential {pii_type.upper()} detected (hash: {pii_hash})",
                    location=context,
                    recommendation=f"Remove or mask the {pii_type}. Use data anonymization techniques.",
                    timestamp=datetime.now()
                )
                issues.append(issue)
        
        return issues
    
    def validate_no_credentials_in_code(self, file_path: str) -> List[SecurityIssue]:
        """Validate that no credentials are hardcoded in source files."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check for hardcoded credentials
            for keyword in self.credential_keywords:
                # Look for assignment patterns
                pattern = rf'{keyword}\s*[=:]\s*["\']([^"\']+)["\']'
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    value = match.group(1)
                    
                    # Skip if it's a placeholder or environment variable reference
                    if (self._is_placeholder(value) or 
                        value.startswith('${') or 
                        value.startswith('os.getenv') or
                        value.startswith('os.environ')):
                        continue
                    
                    # Check if it looks like a real credential
                    if len(value) > 8 and not value.isdigit():
                        issue = SecurityIssue(
                            issue_type="HARDCODED_CREDENTIAL",
                            severity="CRITICAL",
                            description=f"Potential hardcoded {keyword} found",
                            location=file_path,
                            recommendation="Use environment variables or secure configuration management.",
                            timestamp=datetime.now()
                        )
                        issues.append(issue)
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            issue = SecurityIssue(
                issue_type="VALIDATION_ERROR",
                severity="MEDIUM",
                description=f"Could not validate file: {str(e)}",
                location=file_path,
                recommendation="Check file permissions and encoding.",
                timestamp=datetime.now()
            )
            issues.append(issue)
        
        return issues
    
    def sanitize_logs(self, log_text: str) -> str:
        """Sanitize log text to remove sensitive information."""
        sanitized = log_text
        
        # Remove API keys
        for pattern in self.api_key_patterns:
            sanitized = re.sub(pattern, r'\1***MASKED***', sanitized, flags=re.IGNORECASE)
        
        # Remove PII
        for pii_type, pattern in self.pii_patterns.items():
            sanitized = re.sub(pattern, f'***{pii_type.upper()}_MASKED***', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def validate_file_security(self, file_path: str) -> List[SecurityIssue]:
        """Comprehensive file security validation."""
        issues = []
        
        # Check file extension
        if any(file_path.endswith(ext) for ext in self.sensitive_extensions):
            issue = SecurityIssue(
                issue_type="SENSITIVE_FILE",
                severity="HIGH",
                description=f"Sensitive file type detected: {file_path}",
                location=file_path,
                recommendation="Ensure sensitive files are properly secured and not committed to version control.",
                timestamp=datetime.now()
            )
            issues.append(issue)
        
        # Check for credentials in code
        if file_path.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c')):
            issues.extend(self.validate_no_credentials_in_code(file_path))
        
        return issues
    
    def validate_configuration_security(self, config: Dict[str, Any]) -> List[SecurityIssue]:
        """Validate configuration for security issues."""
        issues = []
        
        # Convert config to string for pattern matching
        config_text = json.dumps(config, indent=2)
        
        # Check for API key leaks
        issues.extend(self.detect_api_key_leaks(config_text, "configuration"))
        
        # Check for PII leaks
        issues.extend(self.detect_pii_leaks(config, "configuration"))
        
        return issues
    
    def run_comprehensive_security_scan(self, paths: List[str]) -> SecurityReport:
        """Run comprehensive security scan on multiple paths."""
        all_issues = []
        
        for path in paths:
            try:
                # Validate file security
                issues = self.validate_file_security(path)
                all_issues.extend(issues)
                
                # If it's a text file, check content
                if path.endswith(('.py', '.js', '.ts', '.json', '.yaml', '.yml', '.txt', '.log')):
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Check for API key leaks
                        all_issues.extend(self.detect_api_key_leaks(content, path))
                        
                        # Check for PII leaks
                        all_issues.extend(self.detect_pii_leaks(content, path))
                        
                    except Exception as e:
                        logger.warning(f"Could not read file {path}: {e}")
                        
            except Exception as e:
                logger.error(f"Error scanning path {path}: {e}")
        
        # Categorize issues by severity
        critical_issues = [i for i in all_issues if i.severity == "CRITICAL"]
        high_issues = [i for i in all_issues if i.severity == "HIGH"]
        medium_issues = [i for i in all_issues if i.severity == "MEDIUM"]
        low_issues = [i for i in all_issues if i.severity == "LOW"]
        
        # Determine overall status
        if critical_issues:
            overall_status = "CRITICAL"
        elif high_issues:
            overall_status = "WARNING"
        else:
            overall_status = "SECURE"
        
        return SecurityReport(
            total_issues=len(all_issues),
            critical_issues=len(critical_issues),
            high_issues=len(high_issues),
            medium_issues=len(medium_issues),
            low_issues=len(low_issues),
            issues=all_issues,
            validation_timestamp=datetime.now(),
            overall_status=overall_status
        )
    
    def _is_placeholder(self, text: str) -> bool:
        """Check if text is a placeholder or example."""
        placeholders = [
            'your_api_key', 'your_secret', 'example', 'placeholder', 'test',
            'dummy', 'sample', 'xxx', 'yyy', 'zzz', '123456', 'password123',
            'admin', 'user', 'demo', 'fake', 'mock', 'temp', 'temporary'
        ]
        
        text_lower = text.lower()
        return any(placeholder in text_lower for placeholder in placeholders)
    
    def generate_security_report(self, report: SecurityReport) -> str:
        """Generate a human-readable security report."""
        report_lines = [
            "=" * 60,
            "SECURITY VALIDATION REPORT",
            "=" * 60,
            f"Timestamp: {report.validation_timestamp}",
            f"Overall Status: {report.overall_status}",
            "",
            "SUMMARY:",
            f"  Total Issues: {report.total_issues}",
            f"  Critical: {report.critical_issues}",
            f"  High: {report.high_issues}",
            f"  Medium: {report.medium_issues}",
            f"  Low: {report.low_issues}",
            ""
        ]
        
        if report.issues:
            report_lines.append("DETAILED ISSUES:")
            report_lines.append("-" * 40)
            
            for issue in report.issues:
                report_lines.extend([
                    f"Type: {issue.issue_type}",
                    f"Severity: {issue.severity}",
                    f"Description: {issue.description}",
                    f"Location: {issue.location}",
                    f"Recommendation: {issue.recommendation}",
                    f"Timestamp: {issue.timestamp}",
                    ""
                ])
        else:
            report_lines.append("✅ No security issues found!")
        
        return "\n".join(report_lines)

# Global security validator instance
_security_validator: Optional[SecurityValidator] = None

def get_security_validator() -> SecurityValidator:
    """Get the global security validator instance."""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator()
    return _security_validator

def detect_api_key_leaks(text: str, context: str = "unknown") -> List[SecurityIssue]:
    """Detect API key leaks in text."""
    return get_security_validator().detect_api_key_leaks(text, context)

def detect_pii_leaks(data: Any, context: str = "unknown") -> List[SecurityIssue]:
    """Detect PII leaks in data."""
    return get_security_validator().detect_pii_leaks(data, context)

def sanitize_logs(log_text: str) -> str:
    """Sanitize log text to remove sensitive information."""
    return get_security_validator().sanitize_logs(log_text)

def validate_file_security(file_path: str) -> List[SecurityIssue]:
    """Validate file security."""
    return get_security_validator().validate_file_security(file_path)

def run_comprehensive_security_scan(paths: List[str]) -> SecurityReport:
    """Run comprehensive security scan."""
    return get_security_validator().run_comprehensive_security_scan(paths)
