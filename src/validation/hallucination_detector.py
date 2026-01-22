"""
Hallucination Detector Module

This module implements comprehensive hallucination detection to validate AI responses
and prevent unrealistic or impossible trading decisions.
"""

import re
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class HallucinationIssue:
    """Represents a hallucination detected in AI response."""
    issue_type: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    description: str
    detected_value: Any
    expected_range: Tuple[float, float]
    confidence: float
    recommendation: str
    timestamp: datetime

@dataclass
class HallucinationReport:
    """Comprehensive hallucination detection report."""
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    issues: List[HallucinationIssue]
    validation_timestamp: datetime
    overall_status: str  # "VALID", "WARNING", "INVALID"
    ai_response_valid: bool

class HallucinationDetector:
    """Comprehensive hallucination detection system for AI trading responses."""
    
    def __init__(self):
        # Realistic trading ranges
        self.trading_ranges = {
            'confidence': (0.0, 1.0),
            'price': (0.01, 1000000.0),  # $0.01 to $1M per share
            'volume': (1, 1000000000),  # 1 to 1B shares
            'market_cap': (1000, 10000000000000),  # $1K to $10T
            'percentage': (0.0, 100.0),
            'rsi': (0.0, 100.0),
            'macd': (-1000.0, 1000.0),
            'bollinger_position': (0.0, 1.0),
            'sentiment': (-1.0, 1.0),
            'volatility': (0.0, 10.0),  # 0% to 1000% annualized
            'sharpe_ratio': (-10.0, 10.0),
            'beta': (-5.0, 5.0),
            'pe_ratio': (0.0, 1000.0),
            'dividend_yield': (0.0, 50.0),  # 0% to 50%
            'position_size': (0.0, 1.0),  # 0% to 100% of portfolio
            'stop_loss': (0.0, 0.5),  # 0% to 50% loss
            'take_profit': (0.0, 10.0),  # 0% to 1000% gain
        }
        
        # Impossible value patterns
        self.impossible_patterns = [
            r'infinity', r'infinite', r'undefined', r'null', r'nan',
            r'999999999', r'1e\+?\d+', r'0\.0+1', r'1\.0+0'
        ]
        
        # Contradiction keywords
        self.contradiction_pairs = [
            ('buy', 'sell'), ('bullish', 'bearish'), ('long', 'short'),
            ('increase', 'decrease'), ('rise', 'fall'), ('gain', 'loss'),
            ('positive', 'negative'), ('strong', 'weak'), ('high', 'low')
        ]
        
        # Unrealistic trading scenarios
        self.unrealistic_scenarios = [
            'guaranteed profit', 'risk-free', 'sure thing', 'cannot lose',
            '100% success', 'perfect trade', 'no downside', 'guaranteed return'
        ]
        
        logger.info("Hallucination Detector initialized")
    
    def detect_unrealistic_values(self, data: Dict[str, Any]) -> List[HallucinationIssue]:
        """Detect unrealistic values in trading data."""
        issues = []
        
        for key, value in data.items():
            if not isinstance(value, (int, float)):
                continue
            
            # Check if key has a defined range
            if key.lower() in self.trading_ranges:
                expected_min, expected_max = self.trading_ranges[key.lower()]
                
                if value < expected_min or value > expected_max:
                    severity = "CRITICAL" if key.lower() in ['confidence', 'price', 'position_size'] else "HIGH"
                    
                    issue = HallucinationIssue(
                        issue_type="UNREALISTIC_VALUE",
                        severity=severity,
                        description=f"Value {value} for {key} is outside realistic range",
                        detected_value=value,
                        expected_range=(expected_min, expected_max),
                        confidence=1.0,
                        recommendation=f"Ensure {key} is within range [{expected_min}, {expected_max}]",
                        timestamp=datetime.now()
                    )
                    issues.append(issue)
        
        return issues
    
    def detect_impossible_values(self, data: Dict[str, Any]) -> List[HallucinationIssue]:
        """Detect impossible values (NaN, infinity, etc.)."""
        issues = []
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                # Check for NaN
                if np.isnan(value):
                    issue = HallucinationIssue(
                        issue_type="IMPOSSIBLE_VALUE",
                        severity="CRITICAL",
                        description=f"NaN value detected for {key}",
                        detected_value=value,
                        expected_range=(0, 0),
                        confidence=1.0,
                        recommendation="Replace NaN with a valid numerical value",
                        timestamp=datetime.now()
                    )
                    issues.append(issue)
                
                # Check for infinity
                elif np.isinf(value):
                    issue = HallucinationIssue(
                        issue_type="IMPOSSIBLE_VALUE",
                        severity="CRITICAL",
                        description=f"Infinite value detected for {key}",
                        detected_value=value,
                        expected_range=(0, 0),
                        confidence=1.0,
                        recommendation="Replace infinite value with a realistic finite value",
                        timestamp=datetime.now()
                    )
                    issues.append(issue)
        
        return issues
    
    def detect_contradictions(self, text: str) -> List[HallucinationIssue]:
        """Detect contradictory statements in AI reasoning."""
        issues = []
        text_lower = text.lower()
        
        for positive, negative in self.contradiction_pairs:
            positive_found = positive in text_lower
            negative_found = negative in text_lower
            
            if positive_found and negative_found:
                issue = HallucinationIssue(
                    issue_type="CONTRADICTION",
                    severity="HIGH",
                    description=f"Contradictory statements detected: '{positive}' and '{negative}'",
                    detected_value=f"{positive} + {negative}",
                    expected_range=(0, 0),
                    confidence=0.8,
                    recommendation="Clarify the reasoning to resolve the contradiction",
                    timestamp=datetime.now()
                )
                issues.append(issue)
        
        return issues
    
    def detect_unrealistic_scenarios(self, text: str) -> List[HallucinationIssue]:
        """Detect unrealistic trading scenarios."""
        issues = []
        text_lower = text.lower()
        
        for scenario in self.unrealistic_scenarios:
            if scenario in text_lower:
                issue = HallucinationIssue(
                    issue_type="UNREALISTIC_SCENARIO",
                    severity="HIGH",
                    description=f"Unrealistic trading scenario detected: '{scenario}'",
                    detected_value=scenario,
                    expected_range=(0, 0),
                    confidence=0.9,
                    recommendation="Remove unrealistic claims about guaranteed outcomes",
                    timestamp=datetime.now()
                )
                issues.append(issue)
        
        return issues
    
    def detect_impossible_patterns(self, text: str) -> List[HallucinationIssue]:
        """Detect impossible numerical patterns."""
        issues = []
        
        for pattern in self.impossible_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issue = HallucinationIssue(
                    issue_type="IMPOSSIBLE_PATTERN",
                    severity="MEDIUM",
                    description=f"Impossible pattern detected: '{match.group(0)}'",
                    detected_value=match.group(0),
                    expected_range=(0, 0),
                    confidence=0.7,
                    recommendation="Replace with realistic numerical values",
                    timestamp=datetime.now()
                )
                issues.append(issue)
        
        return issues
    
    def validate_confidence_consistency(self, confidence: float, reasoning: str) -> List[HallucinationIssue]:
        """Validate that confidence level is consistent with reasoning quality."""
        issues = []
        
        # Check reasoning quality indicators
        reasoning_lower = reasoning.lower()
        
        # High confidence with weak reasoning
        weak_indicators = ['maybe', 'possibly', 'might', 'could', 'perhaps', 'unclear', 'uncertain']
        strong_indicators = ['definitely', 'certainly', 'clearly', 'obviously', 'evident', 'proven']
        
        weak_count = sum(1 for indicator in weak_indicators if indicator in reasoning_lower)
        strong_count = sum(1 for indicator in strong_indicators if indicator in reasoning_lower)
        
        if confidence > 0.8 and weak_count > strong_count:
            issue = HallucinationIssue(
                issue_type="CONFIDENCE_INCONSISTENCY",
                severity="MEDIUM",
                description="High confidence with weak reasoning",
                detected_value=confidence,
                expected_range=(0.0, 0.7),
                confidence=0.6,
                recommendation="Adjust confidence to match reasoning quality",
                timestamp=datetime.now()
            )
            issues.append(issue)
        
        elif confidence < 0.3 and strong_count > weak_count:
            issue = HallucinationIssue(
                issue_type="CONFIDENCE_INCONSISTENCY",
                severity="MEDIUM",
                description="Low confidence with strong reasoning",
                detected_value=confidence,
                expected_range=(0.4, 1.0),
                confidence=0.6,
                recommendation="Increase confidence to match reasoning quality",
                timestamp=datetime.now()
            )
            issues.append(issue)
        
        return issues
    
    def validate_trading_decision_logic(self, decision: str, market_data: Dict[str, Any]) -> List[HallucinationIssue]:
        """Validate trading decision logic against market data."""
        issues = []
        decision_lower = decision.lower()
        
        # Check for buy/sell decision consistency
        if 'buy' in decision_lower:
            # Check if market data supports buy decision
            if 'sentiment' in market_data and market_data['sentiment'] < -0.5:
                issue = HallucinationIssue(
                    issue_type="DECISION_INCONSISTENCY",
                    severity="HIGH",
                    description="Buy decision contradicts negative sentiment",
                    detected_value="BUY",
                    expected_range=(0, 0),
                    confidence=0.7,
                    recommendation="Reconsider buy decision given negative sentiment",
                    timestamp=datetime.now()
                )
                issues.append(issue)
        
        elif 'sell' in decision_lower:
            # Check if market data supports sell decision
            if 'sentiment' in market_data and market_data['sentiment'] > 0.5:
                issue = HallucinationIssue(
                    issue_type="DECISION_INCONSISTENCY",
                    severity="HIGH",
                    description="Sell decision contradicts positive sentiment",
                    detected_value="SELL",
                    expected_range=(0, 0),
                    confidence=0.7,
                    recommendation="Reconsider sell decision given positive sentiment",
                    timestamp=datetime.now()
                )
                issues.append(issue)
        
        return issues
    
    def detect_hallucinations(self, ai_response: Dict[str, Any]) -> HallucinationReport:
        """Comprehensive hallucination detection on AI response."""
        issues = []
        
        # Extract components
        confidence = ai_response.get('confidence', 0.0)
        reasoning = ai_response.get('reasoning', '')
        recommendation = ai_response.get('recommendation', '')
        market_data = ai_response.get('market_data', {})
        
        # Combine all text for analysis
        all_text = f"{reasoning} {recommendation}"
        
        # Run all detection methods
        issues.extend(self.detect_unrealistic_values(ai_response))
        issues.extend(self.detect_impossible_values(ai_response))
        issues.extend(self.detect_contradictions(all_text))
        issues.extend(self.detect_unrealistic_scenarios(all_text))
        issues.extend(self.detect_impossible_patterns(all_text))
        issues.extend(self.validate_confidence_consistency(confidence, reasoning))
        issues.extend(self.validate_trading_decision_logic(recommendation, market_data))
        
        # Categorize issues by severity
        critical_issues = [i for i in issues if i.severity == "CRITICAL"]
        high_issues = [i for i in issues if i.severity == "HIGH"]
        medium_issues = [i for i in issues if i.severity == "MEDIUM"]
        low_issues = [i for i in issues if i.severity == "LOW"]
        
        # Determine overall status
        if critical_issues:
            overall_status = "INVALID"
            ai_response_valid = False
        elif high_issues:
            overall_status = "WARNING"
            ai_response_valid = True
        else:
            overall_status = "VALID"
            ai_response_valid = True
        
        return HallucinationReport(
            total_issues=len(issues),
            critical_issues=len(critical_issues),
            high_issues=len(high_issues),
            medium_issues=len(medium_issues),
            low_issues=len(low_issues),
            issues=issues,
            validation_timestamp=datetime.now(),
            overall_status=overall_status,
            ai_response_valid=ai_response_valid
        )
    
    def generate_hallucination_report(self, report: HallucinationReport) -> str:
        """Generate a human-readable hallucination report."""
        report_lines = [
            "=" * 60,
            "HALLUCINATION DETECTION REPORT",
            "=" * 60,
            f"Timestamp: {report.validation_timestamp}",
            f"Overall Status: {report.overall_status}",
            f"AI Response Valid: {report.ai_response_valid}",
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
                    f"Detected Value: {issue.detected_value}",
                    f"Expected Range: {issue.expected_range}",
                    f"Confidence: {issue.confidence}",
                    f"Recommendation: {issue.recommendation}",
                    f"Timestamp: {issue.timestamp}",
                    ""
                ])
        else:
            report_lines.append("✅ No hallucinations detected!")
        
        return "\n".join(report_lines)

# Global hallucination detector instance
_hallucination_detector: Optional[HallucinationDetector] = None

def get_hallucination_detector() -> HallucinationDetector:
    """Get the global hallucination detector instance."""
    global _hallucination_detector
    if _hallucination_detector is None:
        _hallucination_detector = HallucinationDetector()
    return _hallucination_detector

def detect_hallucinations(ai_response: Dict[str, Any]) -> HallucinationReport:
    """Detect hallucinations in AI response."""
    return get_hallucination_detector().detect_hallucinations(ai_response)

def validate_ai_response(ai_response: Dict[str, Any]) -> bool:
    """Quick validation of AI response."""
    report = detect_hallucinations(ai_response)
    return report.ai_response_valid
