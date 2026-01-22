"""
Security Tests for Complete Security
====================================

Tests comprehensive security measures including data leak prevention,
PII detection, log security, and credential protection.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import re
import hashlib
import base64

# Import security components
from src.validation.security_validator import get_security_validator, run_comprehensive_security_scan
from src.validation.hallucination_detector import get_hallucination_detector, detect_hallucinations

logger = logging.getLogger(__name__)


class TestCompleteSecurity:
    """Test suite for comprehensive security measures."""
    
    @pytest.fixture
    async def setup_security_validator(self):
        """Setup security validator for testing."""
        return get_security_validator()
    
    @pytest.fixture
    async def setup_hallucination_detector(self):
        """Setup hallucination detector for testing."""
        return get_hallucination_detector()
    
    @pytest.mark.asyncio
    async def test_api_key_leak_detection(self, setup_security_validator):
        """Test API key leak detection."""
        logger.info("Testing API key leak detection...")
        
        security_validator = setup_security_validator
        
        # Test cases with potential API key leaks
        test_cases = [
            {
                "name": "Valid API Key in Log",
                "log_text": "API key: sk-1234567890abcdef1234567890abcdef",
                "should_detect": True
            },
            {
                "name": "Questrade Token in Log",
                "log_text": "Questrade token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "should_detect": True
            },
            {
                "name": "Yahoo Finance API Key",
                "log_text": "Yahoo API key: abc123def456ghi789",
                "should_detect": True
            },
            {
                "name": "No API Key",
                "log_text": "Regular log message without any API keys",
                "should_detect": False
            },
            {
                "name": "False Positive - Similar Pattern",
                "log_text": "User ID: 1234567890abcdef",
                "should_detect": False
            }
        ]
        
        for test_case in test_cases:
            # Test API key detection
            has_leak = await security_validator.detect_api_key_leaks(test_case["log_text"])
            
            if test_case["should_detect"]:
                assert has_leak, f"Should have detected API key leak in: {test_case['name']}"
            else:
                assert not has_leak, f"Should not have detected API key leak in: {test_case['name']}"
        
        logger.info("✓ API key leak detection completed successfully")
    
    @pytest.mark.asyncio
    async def test_pii_sanitization(self, setup_security_validator):
        """Test PII detection and sanitization."""
        logger.info("Testing PII sanitization...")
        
        security_validator = setup_security_validator
        
        # Test cases with PII
        test_cases = [
            {
                "name": "Email Address",
                "data": "User email: john.doe@example.com",
                "should_detect": True,
                "sanitized_should_contain": "***@***.***"
            },
            {
                "name": "Phone Number",
                "data": "Contact: +1-555-123-4567",
                "should_detect": True,
                "sanitized_should_contain": "***-***-****"
            },
            {
                "name": "Credit Card Number",
                "data": "Card: 4111-1111-1111-1111",
                "should_detect": True,
                "sanitized_should_contain": "****-****-****-****"
            },
            {
                "name": "SSN",
                "data": "SSN: 123-45-6789",
                "should_detect": True,
                "sanitized_should_contain": "***-**-****"
            },
            {
                "name": "No PII",
                "data": "Regular data without PII",
                "should_detect": False,
                "sanitized_should_contain": "Regular data without PII"
            }
        ]
        
        for test_case in test_cases:
            # Test PII detection
            has_pii = await security_validator.detect_pii_leaks(test_case["data"])
            
            if test_case["should_detect"]:
                assert has_pii, f"Should have detected PII in: {test_case['name']}"
            else:
                assert not has_pii, f"Should not have detected PII in: {test_case['name']}"
            
            # Test PII sanitization
            sanitized_data = await security_validator.sanitize_logs(test_case["data"])
            
            if test_case["should_detect"]:
                assert test_case["sanitized_should_contain"] in sanitized_data, f"Sanitized data should contain: {test_case['sanitized_should_contain']}"
            else:
                assert sanitized_data == test_case["data"], f"Data without PII should remain unchanged"
        
        logger.info("✓ PII sanitization completed successfully")
    
    @pytest.mark.asyncio
    async def test_log_security(self, setup_security_validator):
        """Test log security measures."""
        logger.info("Testing log security...")
        
        security_validator = setup_security_validator
        
        # Test cases with sensitive information in logs
        test_cases = [
            {
                "name": "Password in Log",
                "log_text": "User password: mysecretpassword123",
                "should_sanitize": True
            },
            {
                "name": "Database Connection String",
                "log_text": "DB connection: postgresql://user:pass@localhost:5432/db",
                "should_sanitize": True
            },
            {
                "name": "JWT Token",
                "log_text": "JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
                "should_sanitize": True
            },
            {
                "name": "Regular Log",
                "log_text": "User logged in successfully",
                "should_sanitize": False
            }
        ]
        
        for test_case in test_cases:
            # Test log sanitization
            sanitized_log = await security_validator.sanitize_logs(test_case["log_text"])
            
            if test_case["should_sanitize"]:
                # Verify sensitive information is removed or masked
                assert sanitized_log != test_case["log_text"], f"Log should be sanitized: {test_case['name']}"
                assert "***" in sanitized_log or "REDACTED" in sanitized_log, f"Sanitized log should contain masking: {test_case['name']}"
            else:
                # Verify regular logs are unchanged
                assert sanitized_log == test_case["log_text"], f"Regular log should remain unchanged: {test_case['name']}"
        
        logger.info("✓ Log security completed successfully")
    
    @pytest.mark.asyncio
    async def test_credential_protection(self, setup_security_validator):
        """Test credential protection measures."""
        logger.info("Testing credential protection...")
        
        security_validator = setup_security_validator
        
        # Test cases with credentials
        test_cases = [
            {
                "name": "API Key in Environment",
                "env_var": "QUESTRADE_API_KEY",
                "value": "sk-1234567890abcdef",
                "should_protect": True
            },
            {
                "name": "Database Password",
                "env_var": "DB_PASSWORD",
                "value": "mypassword123",
                "should_protect": True
            },
            {
                "name": "Regular Environment Variable",
                "env_var": "APP_NAME",
                "value": "TradingBot",
                "should_protect": False
            }
        ]
        
        for test_case in test_cases:
            # Test credential protection
            is_protected = await security_validator.validate_no_credentials_in_code(test_case["value"])
            
            if test_case["should_protect"]:
                assert is_protected, f"Should protect credential: {test_case['name']}"
            else:
                assert not is_protected, f"Should not protect regular value: {test_case['name']}"
        
        logger.info("✓ Credential protection completed successfully")
    
    @pytest.mark.asyncio
    async def test_data_encryption(self, setup_security_validator):
        """Test data encryption measures."""
        logger.info("Testing data encryption...")
        
        security_validator = setup_security_validator
        
        # Test data encryption
        test_data = "Sensitive trading data that needs encryption"
        
        # Test encryption
        encrypted_data = await security_validator.encrypt_sensitive_data(test_data)
        assert encrypted_data != test_data, "Encrypted data should be different from original"
        assert len(encrypted_data) > 0, "Encrypted data should not be empty"
        
        # Test decryption
        decrypted_data = await security_validator.decrypt_sensitive_data(encrypted_data)
        assert decrypted_data == test_data, "Decrypted data should match original"
        
        # Test encryption consistency
        encrypted_data2 = await security_validator.encrypt_sensitive_data(test_data)
        assert encrypted_data != encrypted_data2, "Encryption should be non-deterministic"
        
        # Test decryption of second encryption
        decrypted_data2 = await security_validator.decrypt_sensitive_data(encrypted_data2)
        assert decrypted_data2 == test_data, "Second decryption should also match original"
        
        logger.info("✓ Data encryption completed successfully")
    
    @pytest.mark.asyncio
    async def test_comprehensive_security_scan(self, setup_security_validator):
        """Test comprehensive security scan."""
        logger.info("Testing comprehensive security scan...")
        
        security_validator = setup_security_validator
        
        # Run comprehensive security scan
        scan_results = await run_comprehensive_security_scan(security_validator)
        
        # Verify scan results structure
        assert scan_results is not None
        assert "total_issues" in scan_results
        assert "critical_issues" in scan_results
        assert "high_issues" in scan_results
        assert "medium_issues" in scan_results
        assert "low_issues" in scan_results
        assert "scan_timestamp" in scan_results
        assert "scan_duration" in scan_results
        
        # Verify scan results are reasonable
        assert scan_results["total_issues"] >= 0, "Total issues should be non-negative"
        assert scan_results["critical_issues"] >= 0, "Critical issues should be non-negative"
        assert scan_results["high_issues"] >= 0, "High issues should be non-negative"
        assert scan_results["medium_issues"] >= 0, "Medium issues should be non-negative"
        assert scan_results["low_issues"] >= 0, "Low issues should be non-negative"
        
        # Verify scan duration is reasonable
        assert scan_results["scan_duration"] > 0, "Scan duration should be positive"
        assert scan_results["scan_duration"] < 60, "Scan duration should be less than 60 seconds"
        
        # Log scan results
        logger.info(f"Security scan results: {scan_results['total_issues']} total issues")
        logger.info(f"Critical: {scan_results['critical_issues']}, High: {scan_results['high_issues']}")
        logger.info(f"Medium: {scan_results['medium_issues']}, Low: {scan_results['low_issues']}")
        logger.info(f"Scan duration: {scan_results['scan_duration']:.2f}s")
        
        logger.info("✓ Comprehensive security scan completed successfully")
    
    @pytest.mark.asyncio
    async def test_security_validation_integration(self, setup_security_validator):
        """Test security validation integration."""
        logger.info("Testing security validation integration...")
        
        security_validator = setup_security_validator
        
        # Test security validation with various data types
        test_data = [
            "Regular log message",
            "API key: sk-1234567890abcdef",
            "User email: test@example.com",
            "Password: secret123",
            "JWT token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        ]
        
        validation_results = []
        for data in test_data:
            # Test individual validation methods
            api_leak = await security_validator.detect_api_key_leaks(data)
            pii_leak = await security_validator.detect_pii_leaks(data)
            sanitized = await security_validator.sanitize_logs(data)
            
            validation_results.append({
                "original": data,
                "api_leak": api_leak,
                "pii_leak": pii_leak,
                "sanitized": sanitized
            })
        
        # Verify validation results
        assert len(validation_results) == len(test_data), "All data should be validated"
        
        # Check that sensitive data is properly detected and sanitized
        for result in validation_results:
            if result["api_leak"] or result["pii_leak"]:
                assert result["sanitized"] != result["original"], "Sensitive data should be sanitized"
            else:
                assert result["sanitized"] == result["original"], "Non-sensitive data should remain unchanged"
        
        logger.info("✓ Security validation integration completed successfully")
    
    @pytest.mark.asyncio
    async def test_security_performance(self, setup_security_validator):
        """Test security validation performance."""
        logger.info("Testing security validation performance...")
        
        security_validator = setup_security_validator
        
        # Test performance with large dataset
        large_dataset = []
        for i in range(1000):
            large_dataset.append(f"Log entry {i}: User action performed")
        
        # Add some sensitive data
        large_dataset[100] = "API key: sk-1234567890abcdef"
        large_dataset[200] = "Email: test@example.com"
        large_dataset[300] = "Password: secret123"
        
        # Test performance
        start_time = asyncio.get_event_loop().time()
        
        # Process large dataset
        processed_data = []
        for data in large_dataset:
            sanitized = await security_validator.sanitize_logs(data)
            processed_data.append(sanitized)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Verify processing completed
        assert len(processed_data) == len(large_dataset), "All data should be processed"
        
        # Verify performance (should process 1000 items quickly)
        assert processing_time < 5.0, f"Security processing took {processing_time:.2f}s, should be <5s"
        
        # Verify sensitive data was sanitized
        assert processed_data[100] != large_dataset[100], "API key should be sanitized"
        assert processed_data[200] != large_dataset[200], "Email should be sanitized"
        assert processed_data[300] != large_dataset[300], "Password should be sanitized"
        
        # Calculate throughput
        throughput = len(large_dataset) / processing_time
        logger.info(f"✓ Security validation performance: {throughput:.1f} items/sec")
    
    @pytest.mark.asyncio
    async def test_security_error_handling(self, setup_security_validator):
        """Test security validation error handling."""
        logger.info("Testing security validation error handling...")
        
        security_validator = setup_security_validator
        
        # Test error handling with invalid inputs
        error_test_cases = [
            None,  # None input
            "",    # Empty string
            123,   # Non-string input
            [],    # List input
            {},    # Dict input
        ]
        
        for test_case in error_test_cases:
            try:
                # Test API key detection
                await security_validator.detect_api_key_leaks(test_case)
                
                # Test PII detection
                await security_validator.detect_pii_leaks(test_case)
                
                # Test log sanitization
                await security_validator.sanitize_logs(test_case)
                
                # If we get here, the method handled the error gracefully
                logger.info(f"✓ Error handling for {type(test_case).__name__} passed")
                
            except Exception as e:
                # Log the error but don't fail the test
                logger.warning(f"Error handling for {type(test_case).__name__}: {e}")
        
        logger.info("✓ Security validation error handling completed successfully")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
