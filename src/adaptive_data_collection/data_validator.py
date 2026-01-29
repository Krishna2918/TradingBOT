"""
Data validation and cleaning pipeline for market data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from .interfaces import DataProcessor


class MarketDataValidator(DataProcessor):
    """Validator for OHLCV market data with comprehensive quality checks."""
    
    def __init__(self, drop_empty: bool = True, sort_by_date: bool = True):
        self.drop_empty = drop_empty
        self.sort_by_date = sort_by_date
        self.logger = logging.getLogger(__name__)
        
        # Required columns for OHLCV data
        self.required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        self.numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        
        self.logger.info(f"Initialized data validator: drop_empty={drop_empty}, sort_by_date={sort_by_date}")
    
    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean raw market data."""
        if data is None or len(data) == 0:
            raise ValueError("Cannot validate empty or None data")
        
        self.logger.info(f"Validating data: {len(data)} rows, {len(data.columns)} columns")
        
        # Store original row count for reporting
        original_rows = len(data)
        
        # 1. Validate required columns
        data = self._validate_columns(data)
        
        # 2. Validate and convert data types
        data = self._validate_data_types(data)
        
        # 3. Remove empty/null rows if configured
        if self.drop_empty:
            data = self._remove_empty_rows(data)
        
        # 4. Validate business rules (OHLCV constraints)
        data = self._validate_business_rules(data)
        
        # 5. Remove duplicates
        data = self._remove_duplicates(data)
        
        # Log validation results
        final_rows = len(data)
        removed_rows = original_rows - final_rows
        if removed_rows > 0:
            self.logger.warning(f"Validation removed {removed_rows} invalid rows ({removed_rows/original_rows*100:.1f}%)")
        else:
            self.logger.info("All data passed validation")
        
        return data
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators (placeholder - implemented in Task 6)."""
        # This will be implemented in Task 6
        self.logger.debug("Technical indicators calculation not yet implemented")
        return data
    
    def sort_and_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sort data by date and perform final cleaning."""
        if len(data) == 0:
            return data
        
        # Sort by date ascending if configured
        if self.sort_by_date:
            if 'date' in data.columns:
                data = data.sort_values('date', ascending=True).reset_index(drop=True)
                self.logger.debug(f"Sorted data by date: {data['date'].min()} to {data['date'].max()}")
        
        # Final cleanup
        data = self._final_cleanup(data)
        
        return data
    
    def _validate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate that all required columns are present."""
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Reorder columns to standard format
        other_columns = [col for col in data.columns if col not in self.required_columns]
        ordered_columns = self.required_columns + other_columns
        data = data[ordered_columns]
        
        self.logger.debug(f"Column validation passed: {len(data.columns)} columns")
        return data
    
    def _validate_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        # Convert date column
        if data['date'].dtype == 'object':
            try:
                data['date'] = pd.to_datetime(data['date'])
                self.logger.debug("Converted date column to datetime")
            except Exception as e:
                raise ValueError(f"Failed to convert date column: {e}")
        
        # Convert numeric columns
        for col in self.numeric_columns:
            if col in data.columns:
                try:
                    # Convert to numeric, coercing errors to NaN
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Failed to convert {col} to numeric: {e}")
        
        self.logger.debug("Data type validation completed")
        return data
    
    def _remove_empty_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with empty or null values in critical columns."""
        original_rows = len(data)
        
        # Remove rows where any numeric column is NaN
        critical_columns = ['open', 'high', 'low', 'close']
        data = data.dropna(subset=critical_columns)
        
        # Remove rows where date is NaT
        data = data.dropna(subset=['date'])
        
        removed_rows = original_rows - len(data)
        if removed_rows > 0:
            self.logger.info(f"Removed {removed_rows} rows with empty/null values")
        
        return data
    
    def _validate_business_rules(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV business rules."""
        original_rows = len(data)
        
        # Rule 1: High >= Low
        invalid_high_low = data['high'] < data['low']
        if invalid_high_low.any():
            invalid_count = invalid_high_low.sum()
            self.logger.warning(f"Found {invalid_count} rows where high < low")
            data = data[~invalid_high_low]
        
        # Rule 2: Prices must be positive
        price_columns = ['open', 'high', 'low', 'close', 'adj_close']
        for col in price_columns:
            if col in data.columns:
                negative_prices = data[col] <= 0
                if negative_prices.any():
                    invalid_count = negative_prices.sum()
                    self.logger.warning(f"Found {invalid_count} rows with non-positive {col}")
                    data = data[~negative_prices]
        
        # Rule 3: Volume must be non-negative
        if 'volume' in data.columns:
            negative_volume = data['volume'] < 0
            if negative_volume.any():
                invalid_count = negative_volume.sum()
                self.logger.warning(f"Found {invalid_count} rows with negative volume")
                data = data[~negative_volume]
        
        # Rule 4: Reasonable price ranges (detect obvious errors)
        for col in ['open', 'high', 'low', 'close', 'adj_close']:
            if col in data.columns:
                # Remove prices that are unreasonably high (> $10,000) or low (< $0.01)
                unreasonable = (data[col] > 10000) | (data[col] < 0.01)
                if unreasonable.any():
                    invalid_count = unreasonable.sum()
                    self.logger.warning(f"Found {invalid_count} rows with unreasonable {col} prices")
                    data = data[~unreasonable]
        
        # Rule 5: OHLC relationships
        # Open, High, Low, Close should be within reasonable bounds
        ohlc_invalid = (
            (data['open'] > data['high']) |
            (data['open'] < data['low']) |
            (data['close'] > data['high']) |
            (data['close'] < data['low'])
        )
        if ohlc_invalid.any():
            invalid_count = ohlc_invalid.sum()
            self.logger.warning(f"Found {invalid_count} rows with invalid OHLC relationships")
            data = data[~ohlc_invalid]
        
        removed_rows = original_rows - len(data)
        if removed_rows > 0:
            self.logger.info(f"Business rule validation removed {removed_rows} invalid rows")
        
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows based on symbol and date."""
        original_rows = len(data)
        
        # Remove duplicates based on symbol and date
        data = data.drop_duplicates(subset=['symbol', 'date'], keep='last')
        
        removed_rows = original_rows - len(data)
        if removed_rows > 0:
            self.logger.info(f"Removed {removed_rows} duplicate rows")
        
        return data
    
    def _final_cleanup(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform final cleanup operations."""
        # Reset index after all operations
        data = data.reset_index(drop=True)
        
        # Ensure consistent data types
        for col in self.numeric_columns:
            if col in data.columns:
                data[col] = data[col].astype('float64')
        
        # Ensure volume is integer where possible
        if 'volume' in data.columns:
            # Convert to int64, but handle NaN values
            data['volume'] = data['volume'].fillna(0).astype('int64')
        
        self.logger.debug("Final cleanup completed")
        return data
    
    def get_validation_report(self, original_data: pd.DataFrame, 
                            validated_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a detailed validation report."""
        original_rows = len(original_data)
        final_rows = len(validated_data)
        removed_rows = original_rows - final_rows
        
        report = {
            "original_rows": original_rows,
            "final_rows": final_rows,
            "removed_rows": removed_rows,
            "removal_percentage": round(removed_rows / original_rows * 100, 2) if original_rows > 0 else 0,
            "data_quality_score": round(final_rows / original_rows * 100, 2) if original_rows > 0 else 0,
            "validation_timestamp": datetime.now().isoformat(),
            "issues_found": []
        }
        
        # Check for specific issues in original data
        if original_rows > 0:
            # Check for missing values
            missing_values = original_data[self.numeric_columns].isnull().sum()
            for col, count in missing_values.items():
                if count > 0:
                    report["issues_found"].append(f"{col}: {count} missing values")
            
            # Check for business rule violations
            if 'high' in original_data.columns and 'low' in original_data.columns:
                high_low_violations = (original_data['high'] < original_data['low']).sum()
                if high_low_violations > 0:
                    report["issues_found"].append(f"High < Low violations: {high_low_violations}")
            
            # Check for negative prices
            for col in ['open', 'high', 'low', 'close', 'adj_close']:
                if col in original_data.columns:
                    negative_count = (original_data[col] <= 0).sum()
                    if negative_count > 0:
                        report["issues_found"].append(f"Non-positive {col}: {negative_count}")
            
            # Check for duplicates
            duplicates = original_data.duplicated(subset=['symbol', 'date']).sum()
            if duplicates > 0:
                report["issues_found"].append(f"Duplicate rows: {duplicates}")
        
        return report
    
    def validate_symbol_data(self, symbol: str, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate data for a specific symbol and return cleaned data with report."""
        self.logger.info(f"Validating data for {symbol}: {len(data)} rows")
        
        # Store original data for reporting
        original_data = data.copy()
        
        try:
            # Perform validation
            validated_data = self.validate_data(data)
            validated_data = self.sort_and_clean(validated_data)
            
            # Generate report
            report = self.get_validation_report(original_data, validated_data)
            report["symbol"] = symbol
            report["validation_success"] = True
            
            self.logger.info(f"Validation completed for {symbol}: {len(validated_data)} rows remaining "
                           f"({report['data_quality_score']:.1f}% quality score)")
            
            return validated_data, report
            
        except Exception as e:
            self.logger.error(f"Validation failed for {symbol}: {e}")
            
            # Return error report
            error_report = {
                "symbol": symbol,
                "validation_success": False,
                "error": str(e),
                "original_rows": len(original_data),
                "final_rows": 0,
                "validation_timestamp": datetime.now().isoformat()
            }
            
            raise ValueError(f"Data validation failed for {symbol}: {e}")


class DataQualityAnalyzer:
    """Analyzer for data quality metrics and reporting."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_data_quality(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Analyze data quality and generate comprehensive metrics."""
        if len(data) == 0:
            return {
                "symbol": symbol,
                "quality_score": 0.0,
                "issues": ["No data available"],
                "metrics": {}
            }
        
        quality_issues = []
        metrics = {}
        
        # 1. Completeness Analysis
        total_rows = len(data)
        numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        
        for col in numeric_cols:
            if col in data.columns:
                missing_count = data[col].isnull().sum()
                missing_pct = missing_count / total_rows * 100
                metrics[f"{col}_missing_pct"] = round(missing_pct, 2)
                
                if missing_pct > 5:  # More than 5% missing
                    quality_issues.append(f"{col}: {missing_pct:.1f}% missing values")
        
        # 2. Data Range Analysis
        for col in ['open', 'high', 'low', 'close', 'adj_close']:
            if col in data.columns and not data[col].empty:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    metrics[f"{col}_min"] = float(col_data.min())
                    metrics[f"{col}_max"] = float(col_data.max())
                    metrics[f"{col}_mean"] = float(col_data.mean())
                    metrics[f"{col}_std"] = float(col_data.std())
                    
                    # Check for extreme values
                    if col_data.min() <= 0:
                        quality_issues.append(f"{col}: Contains non-positive values")
                    
                    if col_data.max() > 10000:
                        quality_issues.append(f"{col}: Contains extremely high values (>${col_data.max():.2f})")
        
        # 3. Volume Analysis
        if 'volume' in data.columns:
            volume_data = data['volume'].dropna()
            if len(volume_data) > 0:
                metrics["volume_min"] = int(volume_data.min())
                metrics["volume_max"] = int(volume_data.max())
                metrics["volume_mean"] = float(volume_data.mean())
                
                zero_volume_pct = (volume_data == 0).sum() / len(volume_data) * 100
                metrics["zero_volume_pct"] = round(zero_volume_pct, 2)
                
                if zero_volume_pct > 10:
                    quality_issues.append(f"Volume: {zero_volume_pct:.1f}% zero volume days")
        
        # 4. Date Analysis
        if 'date' in data.columns:
            date_data = data['date'].dropna()
            if len(date_data) > 0:
                metrics["date_range_start"] = date_data.min().isoformat()
                metrics["date_range_end"] = date_data.max().isoformat()
                metrics["date_range_days"] = (date_data.max() - date_data.min()).days
                
                # Check for date gaps
                date_diff = date_data.diff().dt.days
                large_gaps = (date_diff > 7).sum()  # Gaps > 1 week
                if large_gaps > 0:
                    quality_issues.append(f"Date gaps: {large_gaps} gaps > 7 days")
        
        # 5. OHLC Relationship Analysis
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # Check OHLC relationships
            ohlc_violations = (
                (data['high'] < data['low']) |
                (data['open'] > data['high']) |
                (data['open'] < data['low']) |
                (data['close'] > data['high']) |
                (data['close'] < data['low'])
            ).sum()
            
            if ohlc_violations > 0:
                violation_pct = ohlc_violations / total_rows * 100
                quality_issues.append(f"OHLC violations: {ohlc_violations} rows ({violation_pct:.1f}%)")
                metrics["ohlc_violations_pct"] = round(violation_pct, 2)
        
        # 6. Calculate overall quality score
        quality_score = self._calculate_quality_score(metrics, quality_issues, total_rows)
        
        return {
            "symbol": symbol,
            "total_rows": total_rows,
            "quality_score": round(quality_score, 2),
            "issues": quality_issues,
            "metrics": metrics,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_quality_score(self, metrics: Dict[str, Any], 
                               issues: List[str], total_rows: int) -> float:
        """Calculate overall data quality score (0-100)."""
        if total_rows == 0:
            return 0.0
        
        score = 100.0
        
        # Deduct points for missing data
        for key, value in metrics.items():
            if key.endswith('_missing_pct') and isinstance(value, (int, float)):
                score -= value * 0.5  # Deduct 0.5 points per % missing
        
        # Deduct points for violations
        if 'ohlc_violations_pct' in metrics:
            score -= metrics['ohlc_violations_pct'] * 2  # Deduct 2 points per % violations
        
        # Deduct points for zero volume
        if 'zero_volume_pct' in metrics:
            score -= metrics['zero_volume_pct'] * 0.1  # Deduct 0.1 points per % zero volume
        
        # Deduct points for each quality issue
        score -= len(issues) * 5  # Deduct 5 points per issue
        
        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, score))
    
    def generate_quality_summary(self, validation_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of data quality across all symbols."""
        if not validation_reports:
            return {"error": "No validation reports provided"}
        
        total_symbols = len(validation_reports)
        successful_validations = sum(1 for report in validation_reports 
                                   if report.get("validation_success", False))
        
        # Calculate aggregate metrics
        total_original_rows = sum(report.get("original_rows", 0) for report in validation_reports)
        total_final_rows = sum(report.get("final_rows", 0) for report in validation_reports)
        
        # Quality scores for successful validations
        quality_scores = [report.get("data_quality_score", 0) 
                         for report in validation_reports 
                         if report.get("validation_success", False)]
        
        summary = {
            "total_symbols": total_symbols,
            "successful_validations": successful_validations,
            "failed_validations": total_symbols - successful_validations,
            "validation_success_rate": round(successful_validations / total_symbols * 100, 2),
            "total_original_rows": total_original_rows,
            "total_final_rows": total_final_rows,
            "overall_data_retention": round(total_final_rows / total_original_rows * 100, 2) if total_original_rows > 0 else 0,
            "average_quality_score": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0,
            "min_quality_score": min(quality_scores) if quality_scores else 0,
            "max_quality_score": max(quality_scores) if quality_scores else 0,
            "summary_timestamp": datetime.now().isoformat()
        }
        
        # Identify common issues
        all_issues = []
        for report in validation_reports:
            all_issues.extend(report.get("issues_found", []))
        
        if all_issues:
            issue_counts = {}
            for issue in all_issues:
                # Extract issue type (before the colon)
                issue_type = issue.split(':')[0] if ':' in issue else issue
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            # Sort by frequency
            common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            summary["common_issues"] = [{"issue": issue, "count": count} for issue, count in common_issues]
        
        return summary