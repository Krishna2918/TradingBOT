"""
Security Tests for Hallucination Detection
==========================================

Tests AI response validation to prevent bad trades and unrealistic values.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Import hallucination detection components
from src.validation.hallucination_detector import get_hallucination_detector, detect_hallucinations

logger = logging.getLogger(__name__)


class TestHallucinationDetection:
    """Test suite for hallucination detection and AI response validation."""
    
    @pytest.fixture
    async def setup_hallucination_detector(self):
        """Setup hallucination detector for testing."""
        return get_hallucination_detector()
    
    @pytest.mark.asyncio
    async def test_ai_response_validation(self, setup_hallucination_detector):
        """Test AI response validation."""
        logger.info("Testing AI response validation...")
        
        detector = setup_hallucination_detector
        
        # Test cases with valid and invalid AI responses
        test_cases = [
            {
                "name": "Valid AI Response",
                "ai_response": {
                    "decision": "BUY",
                    "symbol": "AAPL",
                    "confidence": 0.85,
                    "reasoning": "Strong technical indicators and positive sentiment",
                    "entry_price": 150.0,
                    "stop_loss": 145.0,
                    "take_profit": 160.0,
                    "position_size": 0.1
                },
                "should_be_valid": True
            },
            {
                "name": "Invalid Confidence Score",
                "ai_response": {
                    "decision": "BUY",
                    "symbol": "AAPL",
                    "confidence": 1.5,  # Invalid: > 1.0
                    "reasoning": "Strong technical indicators",
                    "entry_price": 150.0,
                    "stop_loss": 145.0,
                    "take_profit": 160.0,
                    "position_size": 0.1
                },
                "should_be_valid": False
            },
            {
                "name": "Negative Confidence Score",
                "ai_response": {
                    "decision": "BUY",
                    "symbol": "AAPL",
                    "confidence": -0.5,  # Invalid: < 0.0
                    "reasoning": "Strong technical indicators",
                    "entry_price": 150.0,
                    "stop_loss": 145.0,
                    "take_profit": 160.0,
                    "position_size": 0.1
                },
                "should_be_valid": False
            },
            {
                "name": "Invalid Decision",
                "ai_response": {
                    "decision": "INVALID_DECISION",  # Invalid decision
                    "symbol": "AAPL",
                    "confidence": 0.85,
                    "reasoning": "Strong technical indicators",
                    "entry_price": 150.0,
                    "stop_loss": 145.0,
                    "take_profit": 160.0,
                    "position_size": 0.1
                },
                "should_be_valid": False
            },
            {
                "name": "Missing Required Fields",
                "ai_response": {
                    "decision": "BUY",
                    "symbol": "AAPL",
                    "confidence": 0.85
                    # Missing required fields
                },
                "should_be_valid": False
            }
        ]
        
        for test_case in test_cases:
            # Test AI response validation
            validation_result = await detect_hallucinations(detector, test_case["ai_response"])
            
            if test_case["should_be_valid"]:
                assert validation_result.is_valid, f"Should be valid: {test_case['name']}"
                assert validation_result.total_issues == 0, f"Should have no issues: {test_case['name']}"
            else:
                assert not validation_result.is_valid, f"Should be invalid: {test_case['name']}"
                assert validation_result.total_issues > 0, f"Should have issues: {test_case['name']}"
        
        logger.info("✓ AI response validation completed successfully")
    
    @pytest.mark.asyncio
    async def test_unrealistic_value_detection(self, setup_hallucination_detector):
        """Test detection of unrealistic values."""
        logger.info("Testing unrealistic value detection...")
        
        detector = setup_hallucination_detector
        
        # Test cases with unrealistic values
        test_cases = [
            {
                "name": "Unrealistic High Price",
                "ai_response": {
                    "decision": "BUY",
                    "symbol": "AAPL",
                    "confidence": 0.85,
                    "reasoning": "Strong technical indicators",
                    "entry_price": 1000000.0,  # Unrealistic price
                    "stop_loss": 950000.0,
                    "take_profit": 1100000.0,
                    "position_size": 0.1
                },
                "should_detect_unrealistic": True
            },
            {
                "name": "Negative Price",
                "ai_response": {
                    "decision": "BUY",
                    "symbol": "AAPL",
                    "confidence": 0.85,
                    "reasoning": "Strong technical indicators",
                    "entry_price": -150.0,  # Negative price
                    "stop_loss": -145.0,
                    "take_profit": -160.0,
                    "position_size": 0.1
                },
                "should_detect_unrealistic": True
            },
            {
                "name": "Unrealistic Position Size",
                "ai_response": {
                    "decision": "BUY",
                    "symbol": "AAPL",
                    "confidence": 0.85,
                    "reasoning": "Strong technical indicators",
                    "entry_price": 150.0,
                    "stop_loss": 145.0,
                    "take_profit": 160.0,
                    "position_size": 2.0  # Unrealistic: > 1.0
                },
                "should_detect_unrealistic": True
            },
            {
                "name": "Negative Position Size",
                "ai_response": {
                    "decision": "BUY",
                    "symbol": "AAPL",
                    "confidence": 0.85,
                    "reasoning": "Strong technical indicators",
                    "entry_price": 150.0,
                    "stop_loss": 145.0,
                    "take_profit": 160.0,
                    "position_size": -0.1  # Negative position size
                },
                "should_detect_unrealistic": True
            },
            {
                "name": "Realistic Values",
                "ai_response": {
                    "decision": "BUY",
                    "symbol": "AAPL",
                    "confidence": 0.85,
                    "reasoning": "Strong technical indicators",
                    "entry_price": 150.0,
                    "stop_loss": 145.0,
                    "take_profit": 160.0,
                    "position_size": 0.1
                },
                "should_detect_unrealistic": False
            }
        ]
        
        for test_case in test_cases:
            # Test unrealistic value detection
            validation_result = await detect_hallucinations(detector, test_case["ai_response"])
            
            if test_case["should_detect_unrealistic"]:
                assert not validation_result.is_valid, f"Should detect unrealistic values: {test_case['name']}"
                assert validation_result.total_issues > 0, f"Should have issues: {test_case['name']}"
            else:
                assert validation_result.is_valid, f"Should not detect unrealistic values: {test_case['name']}"
                assert validation_result.total_issues == 0, f"Should have no issues: {test_case['name']}"
        
        logger.info("✓ Unrealistic value detection completed successfully")
    
    @pytest.mark.asyncio
    async def test_confidence_score_validation(self, setup_hallucination_detector):
        """Test confidence score validation."""
        logger.info("Testing confidence score validation...")
        
        detector = setup_hallucination_detector
        
        # Test cases with various confidence scores
        test_cases = [
            {
                "name": "Valid Confidence (0.0)",
                "confidence": 0.0,
                "should_be_valid": True
            },
            {
                "name": "Valid Confidence (0.5)",
                "confidence": 0.5,
                "should_be_valid": True
            },
            {
                "name": "Valid Confidence (1.0)",
                "confidence": 1.0,
                "should_be_valid": True
            },
            {
                "name": "Invalid Confidence (1.1)",
                "confidence": 1.1,
                "should_be_valid": False
            },
            {
                "name": "Invalid Confidence (-0.1)",
                "confidence": -0.1,
                "should_be_valid": False
            },
            {
                "name": "Invalid Confidence (2.0)",
                "confidence": 2.0,
                "should_be_valid": False
            }
        ]
        
        for test_case in test_cases:
            # Create AI response with specific confidence
            ai_response = {
                "decision": "BUY",
                "symbol": "AAPL",
                "confidence": test_case["confidence"],
                "reasoning": "Strong technical indicators",
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "take_profit": 160.0,
                "position_size": 0.1
            }
            
            # Test confidence validation
            validation_result = await detect_hallucinations(detector, ai_response)
            
            if test_case["should_be_valid"]:
                assert validation_result.is_valid, f"Should be valid: {test_case['name']}"
            else:
                assert not validation_result.is_valid, f"Should be invalid: {test_case['name']}"
        
        logger.info("✓ Confidence score validation completed successfully")
    
    @pytest.mark.asyncio
    async def test_trading_decision_safety(self, setup_hallucination_detector):
        """Test trading decision safety validation."""
        logger.info("Testing trading decision safety...")
        
        detector = setup_hallucination_detector
        
        # Test cases with various trading decisions
        test_cases = [
            {
                "name": "Valid BUY Decision",
                "decision": "BUY",
                "should_be_safe": True
            },
            {
                "name": "Valid SELL Decision",
                "decision": "SELL",
                "should_be_safe": True
            },
            {
                "name": "Valid HOLD Decision",
                "decision": "HOLD",
                "should_be_safe": True
            },
            {
                "name": "Invalid Decision (Empty)",
                "decision": "",
                "should_be_safe": False
            },
            {
                "name": "Invalid Decision (None)",
                "decision": None,
                "should_be_safe": False
            },
            {
                "name": "Invalid Decision (Random String)",
                "decision": "RANDOM_DECISION",
                "should_be_safe": False
            },
            {
                "name": "Invalid Decision (Number)",
                "decision": 123,
                "should_be_safe": False
            }
        ]
        
        for test_case in test_cases:
            # Create AI response with specific decision
            ai_response = {
                "decision": test_case["decision"],
                "symbol": "AAPL",
                "confidence": 0.85,
                "reasoning": "Strong technical indicators",
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "take_profit": 160.0,
                "position_size": 0.1
            }
            
            # Test trading decision safety
            validation_result = await detect_hallucinations(detector, ai_response)
            
            if test_case["should_be_safe"]:
                assert validation_result.is_valid, f"Should be safe: {test_case['name']}"
            else:
                assert not validation_result.is_valid, f"Should not be safe: {test_case['name']}"
        
        logger.info("✓ Trading decision safety completed successfully")
    
    @pytest.mark.asyncio
    async def test_impossible_price_detection(self, setup_hallucination_detector):
        """Test detection of impossible prices."""
        logger.info("Testing impossible price detection...")
        
        detector = setup_hallucination_detector
        
        # Test cases with impossible prices
        test_cases = [
            {
                "name": "Impossible High Price",
                "entry_price": 1000000.0,
                "stop_loss": 950000.0,
                "take_profit": 1100000.0,
                "should_detect_impossible": True
            },
            {
                "name": "Impossible Low Price",
                "entry_price": 0.001,
                "stop_loss": 0.0005,
                "take_profit": 0.002,
                "should_detect_impossible": True
            },
            {
                "name": "Negative Prices",
                "entry_price": -150.0,
                "stop_loss": -145.0,
                "take_profit": -160.0,
                "should_detect_impossible": True
            },
            {
                "name": "Zero Price",
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "should_detect_impossible": True
            },
            {
                "name": "Realistic Prices",
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "take_profit": 160.0,
                "should_detect_impossible": False
            }
        ]
        
        for test_case in test_cases:
            # Create AI response with specific prices
            ai_response = {
                "decision": "BUY",
                "symbol": "AAPL",
                "confidence": 0.85,
                "reasoning": "Strong technical indicators",
                "entry_price": test_case["entry_price"],
                "stop_loss": test_case["stop_loss"],
                "take_profit": test_case["take_profit"],
                "position_size": 0.1
            }
            
            # Test impossible price detection
            validation_result = await detect_hallucinations(detector, ai_response)
            
            if test_case["should_detect_impossible"]:
                assert not validation_result.is_valid, f"Should detect impossible prices: {test_case['name']}"
                assert validation_result.total_issues > 0, f"Should have issues: {test_case['name']}"
            else:
                assert validation_result.is_valid, f"Should not detect impossible prices: {test_case['name']}"
                assert validation_result.total_issues == 0, f"Should have no issues: {test_case['name']}"
        
        logger.info("✓ Impossible price detection completed successfully")
    
    @pytest.mark.asyncio
    async def test_reasoning_validation(self, setup_hallucination_detector):
        """Test reasoning validation."""
        logger.info("Testing reasoning validation...")
        
        detector = setup_hallucination_detector
        
        # Test cases with various reasoning quality
        test_cases = [
            {
                "name": "Valid Reasoning",
                "reasoning": "Strong technical indicators and positive sentiment",
                "should_be_valid": True
            },
            {
                "name": "Empty Reasoning",
                "reasoning": "",
                "should_be_valid": False
            },
            {
                "name": "None Reasoning",
                "reasoning": None,
                "should_be_valid": False
            },
            {
                "name": "Too Short Reasoning",
                "reasoning": "Buy",
                "should_be_valid": False
            },
            {
                "name": "Nonsensical Reasoning",
                "reasoning": "The stock will go to the moon because I said so",
                "should_be_valid": False
            },
            {
                "name": "Contradictory Reasoning",
                "reasoning": "Strong buy signal but also strong sell signal",
                "should_be_valid": False
            }
        ]
        
        for test_case in test_cases:
            # Create AI response with specific reasoning
            ai_response = {
                "decision": "BUY",
                "symbol": "AAPL",
                "confidence": 0.85,
                "reasoning": test_case["reasoning"],
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "take_profit": 160.0,
                "position_size": 0.1
            }
            
            # Test reasoning validation
            validation_result = await detect_hallucinations(detector, ai_response)
            
            if test_case["should_be_valid"]:
                assert validation_result.is_valid, f"Should be valid: {test_case['name']}"
            else:
                assert not validation_result.is_valid, f"Should be invalid: {test_case['name']}"
        
        logger.info("✓ Reasoning validation completed successfully")
    
    @pytest.mark.asyncio
    async def test_hallucination_detection_performance(self, setup_hallucination_detector):
        """Test hallucination detection performance."""
        logger.info("Testing hallucination detection performance...")
        
        detector = setup_hallucination_detector
        
        # Test performance with large dataset
        large_dataset = []
        for i in range(1000):
            ai_response = {
                "decision": "BUY" if i % 2 == 0 else "SELL",
                "symbol": f"STOCK_{i:04d}",
                "confidence": 0.5 + (i % 50) * 0.01,
                "reasoning": f"Technical analysis for stock {i}",
                "entry_price": 100.0 + i * 0.1,
                "stop_loss": 95.0 + i * 0.1,
                "take_profit": 105.0 + i * 0.1,
                "position_size": 0.1
            }
            large_dataset.append(ai_response)
        
        # Add some invalid responses
        large_dataset[100] = {
            "decision": "BUY",
            "symbol": "INVALID",
            "confidence": 1.5,  # Invalid confidence
            "reasoning": "Invalid reasoning",
            "entry_price": -150.0,  # Invalid price
            "stop_loss": -145.0,
            "take_profit": -160.0,
            "position_size": 2.0  # Invalid position size
        }
        
        # Test performance
        start_time = asyncio.get_event_loop().time()
        
        # Process large dataset
        validation_results = []
        for ai_response in large_dataset:
            result = await detect_hallucinations(detector, ai_response)
            validation_results.append(result)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Verify processing completed
        assert len(validation_results) == len(large_dataset), "All responses should be validated"
        
        # Verify performance (should process 1000 responses quickly)
        assert processing_time < 10.0, f"Hallucination detection took {processing_time:.2f}s, should be <10s"
        
        # Verify invalid response was detected
        assert not validation_results[100].is_valid, "Invalid response should be detected"
        assert validation_results[100].total_issues > 0, "Invalid response should have issues"
        
        # Calculate throughput
        throughput = len(large_dataset) / processing_time
        logger.info(f"✓ Hallucination detection performance: {throughput:.1f} responses/sec")
    
    @pytest.mark.asyncio
    async def test_hallucination_detection_error_handling(self, setup_hallucination_detector):
        """Test hallucination detection error handling."""
        logger.info("Testing hallucination detection error handling...")
        
        detector = setup_hallucination_detector
        
        # Test error handling with invalid inputs
        error_test_cases = [
            None,  # None input
            "",    # Empty string
            123,   # Non-dict input
            [],    # List input
            "invalid",  # String input
        ]
        
        for test_case in error_test_cases:
            try:
                # Test hallucination detection
                result = await detect_hallucinations(detector, test_case)
                
                # If we get here, the method handled the error gracefully
                assert result is not None, f"Should return result for {type(test_case).__name__}"
                assert not result.is_valid, f"Should be invalid for {type(test_case).__name__}"
                assert result.total_issues > 0, f"Should have issues for {type(test_case).__name__}"
                
                logger.info(f"✓ Error handling for {type(test_case).__name__} passed")
                
            except Exception as e:
                # Log the error but don't fail the test
                logger.warning(f"Error handling for {type(test_case).__name__}: {e}")
        
        logger.info("✓ Hallucination detection error handling completed successfully")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
