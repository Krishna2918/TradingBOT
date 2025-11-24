"""
Data Validator - Quality Assurance for Financial Data

Validates data quality, detects anomalies, and ensures data integrity.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality_score: float
    issues: List[str]
    warnings: List[str]
    metrics: Dict[str, float]

class DataValidator:
    """Comprehensive data quality validation"""
    
    def __init__(self):
        self.validation_rules = {
            "completeness_threshold": 0.95,  # 95% of data should be non-null
            "consistency_threshold": 0.90,   # 90% of OHLC should be consistent
            "max_price_change": 0.50,        # 50% max single-period change
            "min_volume": 0,                 # Minimum volume (0 = no minimum)
            "max_gap_days": 10               # Maximum gap in daily data
        }
    
    def validate_dataframe(self, data: pd.DataFrame, symbol: str, timeframe: str) -> ValidationResult:
        """Comprehensive validation of a DataFrame"""
        
        if data.empty:
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=["Empty DataFrame"],
                warnings=[],
                metrics={}
            )
        
        issues = []
        warnings = []
        metrics = {}
        
        # 1. Schema validation
        schema_issues = self._validate_schema(data)
        issues.extend(schema_issues)
        
        # 2. Completeness validation
        completeness_score, completeness_issues = self._validate_completeness(data)
        metrics["completeness"] = completeness_score
        issues.extend(completeness_issues)
        
        # 3. Consistency validation
        consistency_score, consistency_issues = self._validate_consistency(data)
        metrics["consistency"] = consistency_score
        issues.extend(consistency_issues)
        
        # 4. Range validation
        range_issues, range_warnings = self._validate_ranges(data, symbol)
        issues.extend(range_issues)
        warnings.extend(range_warnings)
        
        # 5. Temporal validation
        temporal_score, temporal_issues = self._validate_temporal(data, timeframe)
        metrics["temporal"] = temporal_score
        issues.extend(temporal_issues)
        
        # 6. Anomaly detection
        anomaly_warnings = self._detect_anomalies(data, symbol)
        warnings.extend(anomaly_warnings)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(metrics, len(issues))
        
        # Determine if data is valid (no critical issues)
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            warnings=warnings,
            metrics=metrics
        )
    
    def _validate_schema(self, data: pd.DataFrame) -> List[str]:
        """Validate DataFrame schema"""
        issues = []
        
        # Required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                issues.append(f"Column {col} is not numeric")
        
        # Check index
        if not isinstance(data.index, pd.DatetimeIndex):
            issues.append("Index is not DatetimeIndex")
        
        return issues
    
    def _validate_completeness(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Validate data completeness"""
        issues = []
        
        # Calculate completeness ratio
        total_cells = len(data) * len(data.columns)
        non_null_cells = data.count().sum()
        completeness_ratio = non_null_cells / total_cells if total_cells > 0 else 0.0
        
        if completeness_ratio < self.validation_rules["completeness_threshold"]:
            issues.append(f"Low completeness: {completeness_ratio:.3f} < {self.validation_rules['completeness_threshold']}")
        
        # Check for completely empty rows
        empty_rows = data.isnull().all(axis=1).sum()
        if empty_rows > 0:
            issues.append(f"Found {empty_rows} completely empty rows")
        
        # Check critical columns
        critical_columns = ['Open', 'High', 'Low', 'Close']
        for col in critical_columns:
            if col in data.columns:
                null_count = data[col].isnull().sum()
                null_ratio = null_count / len(data)
                if null_ratio > 0.05:  # More than 5% null values
                    issues.append(f"High null ratio in {col}: {null_ratio:.3f}")
        
        return completeness_ratio, issues
    
    def _validate_consistency(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Validate OHLC consistency"""
        issues = []
        
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            return 0.0, ["Missing OHLC columns for consistency check"]
        
        # Check High >= Low
        high_low_consistent = (data['High'] >= data['Low']).mean()
        
        # Check Close within High/Low range
        close_in_range = ((data['Close'] >= data['Low']) & (data['Close'] <= data['High'])).mean()
        
        # Check Open within High/Low range
        open_in_range = ((data['Open'] >= data['Low']) & (data['Open'] <= data['High'])).mean()
        
        # Overall consistency score
        consistency_score = (high_low_consistent + close_in_range + open_in_range) / 3
        
        if high_low_consistent < 0.99:
            issues.append(f"High < Low in {(1-high_low_consistent)*100:.1f}% of rows")
        
        if close_in_range < 0.95:
            issues.append(f"Close outside High/Low range in {(1-close_in_range)*100:.1f}% of rows")
        
        if open_in_range < 0.95:
            issues.append(f"Open outside High/Low range in {(1-open_in_range)*100:.1f}% of rows")
        
        return consistency_score, issues
    
    def _validate_ranges(self, data: pd.DataFrame, symbol: str) -> Tuple[List[str], List[str]]:
        """Validate price and volume ranges"""
        issues = []
        warnings = []
        
        # Check for non-positive prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                non_positive = (data[col] <= 0).sum()
                if non_positive > 0:
                    issues.append(f"Found {non_positive} non-positive values in {col}")
        
        # Check for negative volume
        if 'Volume' in data.columns:
            negative_volume = (data['Volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Found {negative_volume} negative volume values")
        
        # Check for extreme price changes
        if 'Close' in data.columns and len(data) > 1:
            price_changes = data['Close'].pct_change().abs()
            extreme_changes = (price_changes > self.validation_rules["max_price_change"]).sum()
            
            if extreme_changes > 0:
                max_change = price_changes.max()
                warnings.append(f"Found {extreme_changes} extreme price changes (max: {max_change:.3f})")
        
        # Check for suspiciously low prices (penny stocks should still be > $0.01)
        if 'Close' in data.columns:
            very_low_prices = (data['Close'] < 0.01).sum()
            if very_low_prices > 0:
                warnings.append(f"Found {very_low_prices} prices below $0.01")
        
        return issues, warnings
    
    def _validate_temporal(self, data: pd.DataFrame, timeframe: str) -> Tuple[float, List[str]]:
        """Validate temporal consistency"""
        issues = []
        
        if len(data) < 2:
            return 1.0, []  # Can't validate temporal consistency with < 2 rows
        
        # Check for duplicate timestamps
        duplicate_timestamps = data.index.duplicated().sum()
        if duplicate_timestamps > 0:
            issues.append(f"Found {duplicate_timestamps} duplicate timestamps")
        
        # Check for proper sorting
        if not data.index.is_monotonic_increasing:
            issues.append("Data is not sorted by timestamp")
        
        # Check for reasonable gaps (only for daily data)
        temporal_score = 1.0
        
        if timeframe == "1d":
            # Calculate gaps in business days
            business_days = pd.bdate_range(start=data.index.min(), end=data.index.max())
            expected_days = len(business_days)
            actual_days = len(data)
            
            if expected_days > 0:
                temporal_score = actual_days / expected_days
                
                # Check for large gaps
                if len(data) > 1:
                    gaps = data.index.to_series().diff().dt.days
                    large_gaps = (gaps > self.validation_rules["max_gap_days"]).sum()
                    
                    if large_gaps > 0:
                        max_gap = gaps.max()
                        issues.append(f"Found {large_gaps} large gaps (max: {max_gap} days)")
        
        return temporal_score, issues
    
    def _detect_anomalies(self, data: pd.DataFrame, symbol: str) -> List[str]:
        """Detect potential data anomalies"""
        warnings = []
        
        if 'Close' not in data.columns or len(data) < 10:
            return warnings
        
        # Statistical outlier detection using IQR method
        close_prices = data['Close']
        Q1 = close_prices.quantile(0.25)
        Q3 = close_prices.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((close_prices < lower_bound) | (close_prices > upper_bound)).sum()
        if outliers > 0:
            warnings.append(f"Found {outliers} statistical outliers in closing prices")
        
        # Volume anomalies
        if 'Volume' in data.columns:
            volume = data['Volume']
            median_volume = volume.median()
            
            # Very high volume (>10x median)
            high_volume = (volume > median_volume * 10).sum()
            if high_volume > 0:
                warnings.append(f"Found {high_volume} days with very high volume (>10x median)")
            
            # Zero volume days
            zero_volume = (volume == 0).sum()
            if zero_volume > 0:
                warnings.append(f"Found {zero_volume} days with zero volume")
        
        # Price spike detection
        if len(data) > 1:
            returns = close_prices.pct_change().abs()
            spike_threshold = returns.quantile(0.99)  # Top 1% of moves
            
            if spike_threshold > 0.20:  # If top 1% moves are >20%
                spikes = (returns > 0.20).sum()
                if spikes > 0:
                    warnings.append(f"Found {spikes} potential price spikes (>20% moves)")
        
        return warnings
    
    def _calculate_quality_score(self, metrics: Dict[str, float], issue_count: int) -> float:
        """Calculate overall quality score"""
        
        # Base score from metrics
        base_score = 0.0
        weight_sum = 0.0
        
        if "completeness" in metrics:
            base_score += metrics["completeness"] * 0.4
            weight_sum += 0.4
        
        if "consistency" in metrics:
            base_score += metrics["consistency"] * 0.4
            weight_sum += 0.4
        
        if "temporal" in metrics:
            base_score += metrics["temporal"] * 0.2
            weight_sum += 0.2
        
        if weight_sum > 0:
            base_score /= weight_sum
        
        # Penalty for issues
        issue_penalty = min(0.5, issue_count * 0.1)  # Max 50% penalty
        
        final_score = max(0.0, base_score - issue_penalty)
        
        return final_score
    
    def validate_symbol_data(self, symbol: str, storage_manager) -> Dict[str, ValidationResult]:
        """Validate all timeframes for a symbol"""
        
        timeframes = ["1d", "1wk", "1mo", "3mo", "1y", "1m", "5m", "15m", "30m"]
        results = {}
        
        for timeframe in timeframes:
            try:
                data = storage_manager.load_from_parquet(symbol, timeframe)
                
                if data.empty:
                    continue
                
                result = self.validate_dataframe(data, symbol, timeframe)
                results[timeframe] = result
                
                # Log validation results
                if result.is_valid:
                    logger.info(f"✅ {symbol} {timeframe}: Valid (quality: {result.quality_score:.3f})")
                else:
                    logger.warning(f"⚠️ {symbol} {timeframe}: Issues found (quality: {result.quality_score:.3f})")
                    for issue in result.issues:
                        logger.warning(f"   Issue: {issue}")
                
            except Exception as e:
                logger.error(f"Failed to validate {symbol} {timeframe}: {e}")
                results[timeframe] = ValidationResult(
                    is_valid=False,
                    quality_score=0.0,
                    issues=[f"Validation error: {str(e)}"],
                    warnings=[],
                    metrics={}
                )
        
        return results
    
    def generate_validation_report(self, validation_results: Dict[str, Dict[str, ValidationResult]]) -> Dict:
        """Generate comprehensive validation report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_symbols": len(validation_results),
                "total_timeframes": 0,
                "valid_timeframes": 0,
                "average_quality": 0.0
            },
            "symbol_results": {},
            "issues_summary": {},
            "recommendations": []
        }
        
        all_quality_scores = []
        issue_counts = {}
        
        for symbol, timeframe_results in validation_results.items():
            symbol_summary = {
                "timeframes": len(timeframe_results),
                "valid_timeframes": 0,
                "average_quality": 0.0,
                "issues": [],
                "warnings": []
            }
            
            timeframe_qualities = []
            
            for timeframe, result in timeframe_results.items():
                report["summary"]["total_timeframes"] += 1
                
                if result.is_valid:
                    report["summary"]["valid_timeframes"] += 1
                    symbol_summary["valid_timeframes"] += 1
                
                timeframe_qualities.append(result.quality_score)
                all_quality_scores.append(result.quality_score)
                
                # Collect issues
                symbol_summary["issues"].extend(result.issues)
                symbol_summary["warnings"].extend(result.warnings)
                
                # Count issue types
                for issue in result.issues:
                    issue_type = issue.split(":")[0]  # Get issue type
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            if timeframe_qualities:
                symbol_summary["average_quality"] = sum(timeframe_qualities) / len(timeframe_qualities)
            
            report["symbol_results"][symbol] = symbol_summary
        
        # Overall summary
        if all_quality_scores:
            report["summary"]["average_quality"] = sum(all_quality_scores) / len(all_quality_scores)
        
        report["issues_summary"] = issue_counts
        
        # Generate recommendations
        recommendations = []
        
        if report["summary"]["average_quality"] < 0.8:
            recommendations.append("Overall data quality is below 80%. Consider re-collecting problematic datasets.")
        
        if issue_counts:
            top_issue = max(issue_counts.items(), key=lambda x: x[1])
            recommendations.append(f"Most common issue: {top_issue[0]} ({top_issue[1]} occurrences)")
        
        valid_ratio = report["summary"]["valid_timeframes"] / max(1, report["summary"]["total_timeframes"])
        if valid_ratio < 0.9:
            recommendations.append(f"Only {valid_ratio:.1%} of timeframes are valid. Review data collection process.")
        
        report["recommendations"] = recommendations
        
        return report

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    validator = DataValidator()
    
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1D')
    sample_data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(95, 115, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Test validation
    result = validator.validate_dataframe(sample_data, "TEST.TO", "1d")
    
    print(f"🧪 Validation Test Results:")
    print(f"Valid: {result.is_valid}")
    print(f"Quality Score: {result.quality_score:.3f}")
    print(f"Issues: {len(result.issues)}")
    print(f"Warnings: {len(result.warnings)}")
    
    if result.issues:
        print("Issues found:")
        for issue in result.issues:
            print(f"  - {issue}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")