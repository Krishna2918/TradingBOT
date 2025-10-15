"""
Data Quality Validation Module
==============================

This module implements comprehensive data quality validation for technical indicators,
ensuring data integrity before feature calculation and trading decisions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 80-89%
    FAIR = "fair"           # 70-79%
    POOR = "poor"           # 60-69%
    CRITICAL = "critical"   # <60%


@dataclass
class QualityViolation:
    """Represents a data quality violation."""
    violation_type: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    description: str
    column: str
    value: Any
    expected_range: Optional[Tuple[float, float]] = None
    actual_count: Optional[int] = None
    expected_count: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityReport:
    """Comprehensive data quality report."""
    symbol: str
    timestamp: datetime
    overall_score: float
    quality_score: float  # Alias for compatibility
    quality_level: QualityLevel
    violations: List[QualityViolation] = field(default_factory=list)
    column_scores: Dict[str, float] = field(default_factory=dict)
    missing_data_summary: Dict[str, int] = field(default_factory=dict)
    statistical_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class DataQualityValidator:
    """Validates data quality for technical indicators and market data."""
    
    def __init__(self):
        """Initialize data quality validator."""
        self.quality_thresholds = {
            "excellent": 0.90,
            "good": 0.80,
            "fair": 0.70,
            "poor": 0.60,
            "critical": 0.0
        }
        
        # Column-specific validation rules
        self.validation_rules = {
            # Price data
            "open": {"min": 0.0, "max": 10000.0, "non_negative": True},
            "high": {"min": 0.0, "max": 10000.0, "non_negative": True},
            "low": {"min": 0.0, "max": 10000.0, "non_negative": True},
            "close": {"min": 0.0, "max": 10000.0, "non_negative": True},
            "volume": {"min": 0, "max": 1000000000, "non_negative": True, "integer": True},
            
            # Technical indicators
            "atr": {"min": 0.0, "max": 100.0, "non_negative": True},
            "rsi": {"min": 0.0, "max": 100.0, "range": (0, 100)},
            "adx": {"min": 0.0, "max": 100.0, "range": (0, 100)},
            "macd": {"min": -10.0, "max": 10.0},
            "macd_signal": {"min": -10.0, "max": 10.0},
            "macd_histogram": {"min": -5.0, "max": 5.0},
            
            # Bollinger Bands
            "bb_upper": {"min": 0.0, "max": 10000.0, "non_negative": True},
            "bb_middle": {"min": 0.0, "max": 10000.0, "non_negative": True},
            "bb_lower": {"min": 0.0, "max": 10000.0, "non_negative": True},
            "bb_width": {"min": 0.0, "max": 1.0, "non_negative": True},
            "bb_position": {"min": 0.0, "max": 1.0, "range": (0, 1)},
            
            # Moving averages
            "sma_20": {"min": 0.0, "max": 10000.0, "non_negative": True},
            "sma_50": {"min": 0.0, "max": 10000.0, "non_negative": True},
            "ema_12": {"min": 0.0, "max": 10000.0, "non_negative": True},
            "ema_26": {"min": 0.0, "max": 10000.0, "non_negative": True},
            
            # Volume indicators
            "volume_sma": {"min": 0, "max": 1000000000, "non_negative": True, "integer": True},
            "volume_ratio": {"min": 0.0, "max": 10.0, "non_negative": True},
            
            # Volatility
            "volatility": {"min": 0.0, "max": 1.0, "non_negative": True},
            "vix": {"min": 0.0, "max": 100.0, "non_negative": True},
            
            # Sentiment
            "sentiment_score": {"min": -1.0, "max": 1.0, "range": (-1, 1)},
            "news_sentiment": {"min": -1.0, "max": 1.0, "range": (-1, 1)},
            
            # Fundamental
            "pe_ratio": {"min": 0.0, "max": 1000.0, "non_negative": True},
            "pb_ratio": {"min": 0.0, "max": 100.0, "non_negative": True},
            "debt_to_equity": {"min": 0.0, "max": 10.0, "non_negative": True},
        }
        
        # Missing data thresholds (percentage of missing values allowed)
        self.missing_thresholds = {
            "price_data": 0.05,  # 5% missing allowed for price data
            "volume_data": 0.10,  # 10% missing allowed for volume
            "technical_indicators": 0.15,  # 15% missing allowed for indicators
            "sentiment_data": 0.30,  # 30% missing allowed for sentiment
            "fundamental_data": 0.50,  # 50% missing allowed for fundamental
        }
        
        logger.info("Data Quality Validator initialized")
    
    def validate_dataframe(self, df: pd.DataFrame, symbol: str) -> QualityReport:
        """
        Validate a DataFrame for data quality issues.
        
        Args:
            df: DataFrame to validate
            symbol: Symbol being validated
            
        Returns:
            QualityReport with validation results
        """
        logger.info(f"Validating data quality for {symbol}")
        
        violations = []
        column_scores = {}
        missing_data_summary = {}
        statistical_summary = {}
        
        # Check for empty DataFrame
        if df.empty:
            violations.append(QualityViolation(
                violation_type="EMPTY_DATAFRAME",
                severity="CRITICAL",
                description="DataFrame is empty",
                column="ALL",
                value=0,
                expected_count=1
            ))
            return QualityReport(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_score=0.0,
                quality_score=0.0,  # Alias for compatibility
                quality_level=QualityLevel.CRITICAL,
                violations=violations
            )
        
        # Validate each column
        for column in df.columns:
            if column in self.validation_rules:
                column_violations, column_score = self._validate_column(df[column], column)
                violations.extend(column_violations)
                column_scores[column] = column_score
                
                # Calculate missing data summary
                missing_count = df[column].isna().sum()
                missing_percentage = missing_count / len(df)
                missing_data_summary[column] = missing_count
                
                # Calculate statistical summary
                if df[column].dtype in ['float64', 'int64']:
                    statistical_summary[column] = {
                        "mean": float(df[column].mean()) if not df[column].isna().all() else 0.0,
                        "std": float(df[column].std()) if not df[column].isna().all() else 0.0,
                        "min": float(df[column].min()) if not df[column].isna().all() else 0.0,
                        "max": float(df[column].max()) if not df[column].isna().all() else 0.0,
                        "missing_pct": missing_percentage
                    }
        
        # Validate cross-column relationships
        cross_violations = self._validate_cross_column_relationships(df)
        violations.extend(cross_violations)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(column_scores, violations)
        quality_level = self._determine_quality_level(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violations, missing_data_summary)
        
        report = QualityReport(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=overall_score,
            quality_score=overall_score,  # Alias for compatibility
            quality_level=quality_level,
            violations=violations,
            column_scores=column_scores,
            missing_data_summary=missing_data_summary,
            statistical_summary=statistical_summary,
            recommendations=recommendations
        )
        
        logger.info(f"Data quality validation for {symbol}: {quality_level.value} ({overall_score:.2%})")
        return report
    
    def _validate_column(self, series: pd.Series, column_name: str) -> Tuple[List[QualityViolation], float]:
        """Validate a single column against its rules."""
        violations = []
        score = 1.0
        
        if column_name not in self.validation_rules:
            return violations, score
        
        rules = self.validation_rules[column_name]
        
        # Check for non-negative values
        if rules.get("non_negative", False):
            negative_count = (series < 0).sum()
            if negative_count > 0:
                violations.append(QualityViolation(
                    violation_type="NEGATIVE_VALUES",
                    severity="HIGH",
                    description=f"Found {negative_count} negative values in {column_name}",
                    column=column_name,
                    value=negative_count,
                    expected_range=(0, float('inf'))
                ))
                score -= 0.3
        
        # Check for integer values
        if rules.get("integer", False):
            non_integer_count = (series % 1 != 0).sum()
            if non_integer_count > 0:
                violations.append(QualityViolation(
                    violation_type="NON_INTEGER_VALUES",
                    severity="MEDIUM",
                    description=f"Found {non_integer_count} non-integer values in {column_name}",
                    column=column_name,
                    value=non_integer_count
                ))
                score -= 0.2
        
        # Check value ranges
        if "min" in rules or "max" in rules:
            min_val = rules.get("min", float('-inf'))
            max_val = rules.get("max", float('inf'))
            
            out_of_range = ((series < min_val) | (series > max_val)).sum()
            if out_of_range > 0:
                violations.append(QualityViolation(
                    violation_type="OUT_OF_RANGE",
                    severity="HIGH",
                    description=f"Found {out_of_range} values out of range [{min_val}, {max_val}] in {column_name}",
                    column=column_name,
                    value=out_of_range,
                    expected_range=(min_val, max_val)
                ))
                score -= 0.4
        
        # Check specific ranges
        if "range" in rules:
            range_min, range_max = rules["range"]
            out_of_range = ((series < range_min) | (series > range_max)).sum()
            if out_of_range > 0:
                violations.append(QualityViolation(
                    violation_type="OUT_OF_RANGE",
                    severity="CRITICAL",
                    description=f"Found {out_of_range} values out of range [{range_min}, {range_max}] in {column_name}",
                    column=column_name,
                    value=out_of_range,
                    expected_range=(range_min, range_max)
                ))
                score -= 0.5
        
        # Check for missing data
        missing_count = series.isna().sum()
        missing_percentage = missing_count / len(series)
        
        # Determine missing threshold based on data type
        if column_name in ["open", "high", "low", "close"]:
            threshold = self.missing_thresholds["price_data"]
        elif column_name in ["volume", "volume_sma"]:
            threshold = self.missing_thresholds["volume_data"]
        elif column_name in ["sentiment_score", "news_sentiment"]:
            threshold = self.missing_thresholds["sentiment_data"]
        elif column_name in ["pe_ratio", "pb_ratio", "debt_to_equity"]:
            threshold = self.missing_thresholds["fundamental_data"]
        else:
            threshold = self.missing_thresholds["technical_indicators"]
        
        if missing_percentage > threshold:
            severity = "CRITICAL" if missing_percentage > 0.5 else "HIGH"
            violations.append(QualityViolation(
                violation_type="EXCESSIVE_MISSING_DATA",
                severity=severity,
                description=f"Missing {missing_percentage:.1%} of data in {column_name} (threshold: {threshold:.1%})",
                column=column_name,
                value=missing_count,
                expected_count=int(len(series) * (1 - threshold))
            ))
            score -= min(0.5, missing_percentage)
        
        # Check for constant values (no variation)
        if series.nunique() == 1 and not series.isna().all():
            violations.append(QualityViolation(
                violation_type="CONSTANT_VALUES",
                severity="MEDIUM",
                description=f"All values in {column_name} are constant",
                column=column_name,
                value=series.iloc[0]
            ))
            score -= 0.3
        
        # Check for infinite values
        infinite_count = np.isinf(series).sum()
        if infinite_count > 0:
            violations.append(QualityViolation(
                violation_type="INFINITE_VALUES",
                severity="HIGH",
                description=f"Found {infinite_count} infinite values in {column_name}",
                column=column_name,
                value=infinite_count
            ))
            score -= 0.4
        
        return violations, max(0.0, score)
    
    def _validate_cross_column_relationships(self, df: pd.DataFrame) -> List[QualityViolation]:
        """Validate relationships between columns."""
        violations = []
        
        # Validate OHLC relationships
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            # High should be >= max(open, close)
            high_violations = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
            if high_violations > 0:
                violations.append(QualityViolation(
                    violation_type="OHLC_VIOLATION",
                    severity="CRITICAL",
                    description=f"Found {high_violations} cases where high < max(open, close)",
                    column="high",
                    value=high_violations
                ))
            
            # Low should be <= min(open, close)
            low_violations = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
            if low_violations > 0:
                violations.append(QualityViolation(
                    violation_type="OHLC_VIOLATION",
                    severity="CRITICAL",
                    description=f"Found {low_violations} cases where low > min(open, close)",
                    column="low",
                    value=low_violations
                ))
        
        # Validate Bollinger Bands relationships
        if all(col in df.columns for col in ["bb_upper", "bb_middle", "bb_lower"]):
            # Upper > Middle > Lower
            upper_middle_violations = (df["bb_upper"] <= df["bb_middle"]).sum()
            if upper_middle_violations > 0:
                violations.append(QualityViolation(
                    violation_type="BOLLINGER_VIOLATION",
                    severity="HIGH",
                    description=f"Found {upper_middle_violations} cases where bb_upper <= bb_middle",
                    column="bb_upper",
                    value=upper_middle_violations
                ))
            
            middle_lower_violations = (df["bb_middle"] <= df["bb_lower"]).sum()
            if middle_lower_violations > 0:
                violations.append(QualityViolation(
                    violation_type="BOLLINGER_VIOLATION",
                    severity="HIGH",
                    description=f"Found {middle_lower_violations} cases where bb_middle <= bb_lower",
                    column="bb_middle",
                    value=middle_lower_violations
                ))
        
        # Validate volume relationships
        if "volume" in df.columns and "volume_sma" in df.columns:
            # Volume should be non-negative
            negative_volume = (df["volume"] < 0).sum()
            if negative_volume > 0:
                violations.append(QualityViolation(
                    violation_type="NEGATIVE_VOLUME",
                    severity="CRITICAL",
                    description=f"Found {negative_volume} negative volume values",
                    column="volume",
                    value=negative_volume
                ))
        
        return violations
    
    def _calculate_overall_score(self, column_scores: Dict[str, float], violations: List[QualityViolation]) -> float:
        """Calculate overall quality score."""
        if not column_scores:
            return 0.0
        
        # Base score from column scores
        base_score = np.mean(list(column_scores.values()))
        
        # Penalty for violations
        penalty = 0.0
        for violation in violations:
            if violation.severity == "CRITICAL":
                penalty += 0.2
            elif violation.severity == "HIGH":
                penalty += 0.1
            elif violation.severity == "MEDIUM":
                penalty += 0.05
            elif violation.severity == "LOW":
                penalty += 0.02
        
        final_score = max(0.0, base_score - penalty)
        return final_score
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score."""
        if score >= self.quality_thresholds["excellent"]:
            return QualityLevel.EXCELLENT
        elif score >= self.quality_thresholds["good"]:
            return QualityLevel.GOOD
        elif score >= self.quality_thresholds["fair"]:
            return QualityLevel.FAIR
        elif score >= self.quality_thresholds["poor"]:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _generate_recommendations(self, violations: List[QualityViolation], missing_data: Dict[str, int]) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        # Group violations by type
        violation_types = {}
        for violation in violations:
            if violation.violation_type not in violation_types:
                violation_types[violation.violation_type] = []
            violation_types[violation.violation_type].append(violation)
        
        # Generate recommendations
        if "EMPTY_DATAFRAME" in violation_types:
            recommendations.append("Data source is completely empty - check data pipeline connectivity")
        
        if "EXCESSIVE_MISSING_DATA" in violation_types:
            recommendations.append("High missing data detected - consider data source reliability")
        
        if "NEGATIVE_VALUES" in violation_types:
            recommendations.append("Negative values found in price/volume data - check data source")
        
        if "OUT_OF_RANGE" in violation_types:
            recommendations.append("Values outside expected ranges - validate data transformation")
        
        if "OHLC_VIOLATION" in violation_types:
            recommendations.append("OHLC price relationships violated - check data source integrity")
        
        if "BOLLINGER_VIOLATION" in violation_types:
            recommendations.append("Bollinger Band relationships violated - check indicator calculation")
        
        if "CONSTANT_VALUES" in violation_types:
            recommendations.append("Constant values detected - check for data staleness")
        
        if "INFINITE_VALUES" in violation_types:
            recommendations.append("Infinite values detected - check mathematical operations")
        
        # Add general recommendations
        if len(violations) > 10:
            recommendations.append("High number of violations - consider data source replacement")
        
        if not recommendations:
            recommendations.append("Data quality is acceptable for trading decisions")
        
        return recommendations
    
    def should_skip_sizing(self, report: QualityReport, threshold: float = 0.7) -> bool:
        """
        Determine if sizing should be skipped based on quality score.
        
        Args:
            report: Quality report
            threshold: Minimum quality score required for sizing
            
        Returns:
            True if sizing should be skipped
        """
        return report.overall_score < threshold
    
    def log_quality_event(self, report: QualityReport, mode: str = "DEMO"):
        """Log quality validation event."""
        logger.info(f"Quality validation for {report.symbol} ({mode}): "
                   f"{report.quality_level.value} ({report.overall_score:.2%}) - "
                   f"{len(report.violations)} violations")
        
        # Log critical violations
        critical_violations = [v for v in report.violations if v.severity == "CRITICAL"]
        if critical_violations:
            logger.warning(f"Critical quality violations for {report.symbol}:")
            for violation in critical_violations:
                logger.warning(f"  - {violation.violation_type}: {violation.description}")


# Global data quality validator instance
_data_quality_validator: Optional[DataQualityValidator] = None


def get_data_quality_validator() -> DataQualityValidator:
    """Get global data quality validator instance."""
    global _data_quality_validator
    if _data_quality_validator is None:
        _data_quality_validator = DataQualityValidator()
    return _data_quality_validator


# Convenience functions
def validate_dataframe(df: pd.DataFrame, symbol: str) -> QualityReport:
    """Validate a DataFrame for data quality."""
    validator = get_data_quality_validator()
    return validator.validate_dataframe(df, symbol)


def should_skip_sizing(report: QualityReport, threshold: float = 0.7) -> bool:
    """Check if sizing should be skipped based on quality."""
    validator = get_data_quality_validator()
    return validator.should_skip_sizing(report, threshold)


def log_quality_event(report: QualityReport, mode: str = "DEMO"):
    """Log quality validation event."""
    validator = get_data_quality_validator()
    validator.log_quality_event(report, mode)
