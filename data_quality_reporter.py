#!/usr/bin/env python3
"""
Data quality analysis and reporting script for continuous data collection system.

This script analyzes the quality of collected stock data, generates quality reports,
and identifies data issues that may affect model training.

Usage:
    python data_quality_reporter.py [--analysis TYPE] [--symbols SYMBOLS]

Requirements: 4.1, 4.2, 10.1, 10.5
"""

import asyncio
import argparse
import json
import logging
import pandas as pd
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the continuous_data_collection package to the path
sys.path.insert(0, str(Path(__file__).parent))

from continuous_data_collection.core.config import ConfigLoader
from continuous_data_collection.core.logging_config import setup_logging
from continuous_data_collection.core.data_validator import DataValidator
from continuous_data_collection.storage.parquet_storage import ParquetStorage


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data quality analysis for continuous data collection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Types:
    overview        - Overall data quality summary
    completeness    - Data completeness analysis
    integrity       - Data integrity and consistency checks
    anomalies       - Detect data anomalies and outliers
    coverage        - Analyze data coverage and gaps
    validation      - Comprehensive validation report

Examples:
    python data_quality_reporter.py --analysis overview
    python data_quality_reporter.py --analysis completeness --symbols AAPL,MSFT,GOOGL
    python data_quality_reporter.py --analysis validation --output quality_report.json
        """
    )
    
    parser.add_argument(
        "--analysis", "-a",
        choices=["overview", "completeness", "integrity", "anomalies", "coverage", "validation"],
        default="overview",
        help="Type of analysis to perform (default: overview)"
    )
    
    parser.add_argument(
        "--symbols", "-s",
        type=str,
        help="Comma-separated list of symbols to analyze (default: all available)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/development.yaml",
        help="Configuration file path (default: config/development.yaml)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for analysis results"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json", "txt", "html"],
        default="txt",
        help="Output format (default: txt)"
    )
    
    parser.add_argument(
        "--min-years", "-m",
        type=int,
        default=10,
        help="Minimum years of data required (default: 10)"
    )
    
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Include detailed per-symbol analysis"
    )
    
    return parser.parse_args()


async def load_available_symbols(storage: ParquetStorage) -> List[str]:
    """Load list of available symbols from storage."""
    try:
        storage_stats = await storage.get_storage_stats()
        
        # Get all parquet files
        data_dir = Path(storage.config.storage.data_directory)
        if not data_dir.exists():
            return []
        
        symbols = []
        for file_path in data_dir.glob("*.parquet"):
            # Extract symbol from filename (assuming format: SYMBOL.parquet)
            symbol = file_path.stem
            symbols.append(symbol)
        
        return sorted(symbols)
        
    except Exception as e:
        logging.error(f"Failed to load available symbols: {e}")
        return []


async def analyze_symbol_data(symbol: str, storage: ParquetStorage, validator: DataValidator) -> Dict[str, Any]:
    """Analyze data quality for a single symbol."""
    try:
        # Load data
        data = await storage.load_stock_data(symbol)
        if data is None or data.empty:
            return {
                "symbol": symbol,
                "error": "No data found",
                "has_data": False
            }
        
        # Basic data info
        analysis = {
            "symbol": symbol,
            "has_data": True,
            "record_count": len(data),
            "date_range": {
                "start": data.index.min().isoformat() if not data.empty else None,
                "end": data.index.max().isoformat() if not data.empty else None,
                "years": 0
            },
            "columns": list(data.columns),
            "data_quality": {},
            "completeness": {},
            "integrity": {},
            "anomalies": []
        }
        
        # Calculate years of data
        if not data.empty:
            date_range = data.index.max() - data.index.min()
            analysis["date_range"]["years"] = date_range.days / 365.25
        
        # Validate data using the validator
        validation_result = validator.validate_stock_data(data, symbol)
        
        analysis["data_quality"] = {
            "is_valid": validation_result.is_valid,
            "quality_score": validation_result.quality_score,
            "years_of_data": validation_result.years_of_data,
            "missing_dates_count": validation_result.missing_dates_count,
            "missing_dates_percent": validation_result.missing_dates_percent,
            "issues": validation_result.issues
        }
        
        # Completeness analysis
        analysis["completeness"] = analyze_completeness(data)
        
        # Integrity analysis
        analysis["integrity"] = analyze_integrity(data)
        
        # Anomaly detection
        analysis["anomalies"] = detect_anomalies(data)
        
        return analysis
        
    except Exception as e:
        logging.error(f"Failed to analyze {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": str(e),
            "has_data": False
        }


def analyze_completeness(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data completeness."""
    if data.empty:
        return {"error": "No data to analyze"}
    
    completeness = {
        "total_records": len(data),
        "missing_values": {},
        "missing_percentage": {},
        "date_gaps": [],
        "completeness_score": 0.0
    }
    
    # Check missing values in each column
    for column in data.columns:
        missing_count = data[column].isna().sum()
        missing_pct = (missing_count / len(data)) * 100
        
        completeness["missing_values"][column] = missing_count
        completeness["missing_percentage"][column] = missing_pct
    
    # Check for date gaps (assuming daily data)
    if not data.empty:
        date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        missing_dates = date_range.difference(data.index)
        
        # Group consecutive missing dates
        if len(missing_dates) > 0:
            gaps = []
            current_gap_start = missing_dates[0]
            current_gap_end = missing_dates[0]
            
            for i in range(1, len(missing_dates)):
                if (missing_dates[i] - current_gap_end).days == 1:
                    current_gap_end = missing_dates[i]
                else:
                    gaps.append({
                        "start": current_gap_start.isoformat(),
                        "end": current_gap_end.isoformat(),
                        "days": (current_gap_end - current_gap_start).days + 1
                    })
                    current_gap_start = missing_dates[i]
                    current_gap_end = missing_dates[i]
            
            # Add the last gap
            gaps.append({
                "start": current_gap_start.isoformat(),
                "end": current_gap_end.isoformat(),
                "days": (current_gap_end - current_gap_start).days + 1
            })
            
            completeness["date_gaps"] = gaps
    
    # Calculate overall completeness score
    total_possible_values = len(data) * len(data.columns)
    total_missing = sum(completeness["missing_values"].values())
    completeness["completeness_score"] = ((total_possible_values - total_missing) / total_possible_values) * 100
    
    return completeness


def analyze_integrity(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data integrity and consistency."""
    if data.empty:
        return {"error": "No data to analyze"}
    
    integrity = {
        "ohlcv_consistency": {},
        "price_relationships": {},
        "volume_analysis": {},
        "data_types": {},
        "integrity_score": 0.0,
        "issues": []
    }
    
    # Check OHLCV consistency
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        # High should be >= Open, Low, Close
        high_violations = ((data['High'] < data['Open']) | 
                          (data['High'] < data['Low']) | 
                          (data['High'] < data['Close'])).sum()
        
        # Low should be <= Open, High, Close
        low_violations = ((data['Low'] > data['Open']) | 
                         (data['Low'] > data['High']) | 
                         (data['Low'] > data['Close'])).sum()
        
        integrity["ohlcv_consistency"] = {
            "high_violations": high_violations,
            "low_violations": low_violations,
            "total_violations": high_violations + low_violations,
            "violation_percentage": ((high_violations + low_violations) / len(data)) * 100
        }
        
        if high_violations > 0:
            integrity["issues"].append(f"Found {high_violations} records where High < other prices")
        if low_violations > 0:
            integrity["issues"].append(f"Found {low_violations} records where Low > other prices")
    
    # Check for negative values
    numeric_columns = data.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        negative_count = (data[column] < 0).sum()
        if negative_count > 0:
            integrity["issues"].append(f"Found {negative_count} negative values in {column}")
    
    # Volume analysis
    if 'Volume' in data.columns:
        zero_volume = (data['Volume'] == 0).sum()
        negative_volume = (data['Volume'] < 0).sum()
        
        integrity["volume_analysis"] = {
            "zero_volume_days": zero_volume,
            "negative_volume_days": negative_volume,
            "zero_volume_percentage": (zero_volume / len(data)) * 100
        }
        
        if negative_volume > 0:
            integrity["issues"].append(f"Found {negative_volume} days with negative volume")
    
    # Data type consistency
    for column in data.columns:
        dtype_info = {
            "dtype": str(data[column].dtype),
            "is_numeric": pd.api.types.is_numeric_dtype(data[column]),
            "has_nulls": data[column].isna().any()
        }
        integrity["data_types"][column] = dtype_info
    
    # Calculate integrity score
    total_issues = len(integrity["issues"])
    ohlcv_violations = integrity.get("ohlcv_consistency", {}).get("total_violations", 0)
    
    # Score based on violations per record
    violations_per_record = (total_issues + ohlcv_violations) / len(data)
    integrity["integrity_score"] = max(0, 100 - (violations_per_record * 100))
    
    return integrity


def detect_anomalies(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect data anomalies and outliers."""
    if data.empty:
        return []
    
    anomalies = []
    
    # Price-based anomalies
    price_columns = ['Open', 'High', 'Low', 'Close']
    available_price_cols = [col for col in price_columns if col in data.columns]
    
    for column in available_price_cols:
        if data[column].dtype in ['float64', 'int64']:
            # Calculate z-scores
            mean_val = data[column].mean()
            std_val = data[column].std()
            
            if std_val > 0:
                z_scores = abs((data[column] - mean_val) / std_val)
                outliers = data[z_scores > 3]  # 3 standard deviations
                
                if len(outliers) > 0:
                    anomalies.append({
                        "type": "price_outlier",
                        "column": column,
                        "count": len(outliers),
                        "percentage": (len(outliers) / len(data)) * 100,
                        "description": f"Found {len(outliers)} price outliers in {column}",
                        "sample_dates": outliers.index[:5].strftime('%Y-%m-%d').tolist()
                    })
    
    # Volume anomalies
    if 'Volume' in data.columns and data['Volume'].dtype in ['float64', 'int64']:
        # Extremely high volume (more than 10x median)
        median_volume = data['Volume'].median()
        if median_volume > 0:
            high_volume = data[data['Volume'] > median_volume * 10]
            if len(high_volume) > 0:
                anomalies.append({
                    "type": "volume_spike",
                    "column": "Volume",
                    "count": len(high_volume),
                    "percentage": (len(high_volume) / len(data)) * 100,
                    "description": f"Found {len(high_volume)} days with extremely high volume",
                    "sample_dates": high_volume.index[:5].strftime('%Y-%m-%d').tolist()
                })
    
    # Price movement anomalies
    if 'Close' in data.columns:
        # Calculate daily returns
        returns = data['Close'].pct_change().dropna()
        
        # Extreme daily movements (>20% in a day)
        extreme_moves = returns[abs(returns) > 0.20]
        if len(extreme_moves) > 0:
            anomalies.append({
                "type": "extreme_price_movement",
                "column": "Close",
                "count": len(extreme_moves),
                "percentage": (len(extreme_moves) / len(returns)) * 100,
                "description": f"Found {len(extreme_moves)} days with >20% price movements",
                "sample_dates": extreme_moves.index[:5].strftime('%Y-%m-%d').tolist()
            })
    
    # Consecutive identical values (potential data issues)
    for column in available_price_cols:
        consecutive_same = 0
        max_consecutive = 0
        
        for i in range(1, len(data)):
            if data[column].iloc[i] == data[column].iloc[i-1]:
                consecutive_same += 1
                max_consecutive = max(max_consecutive, consecutive_same)
            else:
                consecutive_same = 0
        
        if max_consecutive > 5:  # More than 5 consecutive identical values
            anomalies.append({
                "type": "consecutive_identical",
                "column": column,
                "max_consecutive": max_consecutive,
                "description": f"Found {max_consecutive} consecutive identical values in {column}"
            })
    
    return anomalies


async def generate_overview_report(symbols: List[str], storage: ParquetStorage, validator: DataValidator) -> Dict[str, Any]:
    """Generate overall data quality overview."""
    report = {
        "analysis_type": "overview",
        "timestamp": datetime.utcnow().isoformat(),
        "total_symbols": len(symbols),
        "summary": {
            "symbols_with_data": 0,
            "symbols_without_data": 0,
            "avg_quality_score": 0.0,
            "avg_years_of_data": 0.0,
            "total_records": 0
        },
        "quality_distribution": {
            "excellent": 0,  # >90%
            "good": 0,       # 70-90%
            "fair": 0,       # 50-70%
            "poor": 0        # <50%
        },
        "common_issues": [],
        "recommendations": []
    }
    
    quality_scores = []
    years_of_data = []
    all_issues = []
    
    print(f"Analyzing {len(symbols)} symbols...")
    
    for i, symbol in enumerate(symbols):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(symbols)} symbols analyzed")
        
        analysis = await analyze_symbol_data(symbol, storage, validator)
        
        if analysis["has_data"]:
            report["summary"]["symbols_with_data"] += 1
            
            quality_score = analysis["data_quality"]["quality_score"]
            years = analysis["data_quality"]["years_of_data"]
            
            quality_scores.append(quality_score)
            years_of_data.append(years)
            report["summary"]["total_records"] += analysis["record_count"]
            
            # Categorize quality
            if quality_score >= 0.9:
                report["quality_distribution"]["excellent"] += 1
            elif quality_score >= 0.7:
                report["quality_distribution"]["good"] += 1
            elif quality_score >= 0.5:
                report["quality_distribution"]["fair"] += 1
            else:
                report["quality_distribution"]["poor"] += 1
            
            # Collect issues
            all_issues.extend(analysis["data_quality"]["issues"])
        else:
            report["summary"]["symbols_without_data"] += 1
    
    # Calculate averages
    if quality_scores:
        report["summary"]["avg_quality_score"] = sum(quality_scores) / len(quality_scores)
        report["summary"]["avg_years_of_data"] = sum(years_of_data) / len(years_of_data)
    
    # Find common issues
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    # Sort by frequency
    common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    report["common_issues"] = [{"issue": issue, "count": count} for issue, count in common_issues]
    
    # Generate recommendations
    if report["summary"]["avg_quality_score"] < 0.7:
        report["recommendations"].append("Overall data quality is below target - investigate common issues")
    
    if report["quality_distribution"]["poor"] > len(symbols) * 0.1:
        report["recommendations"].append("More than 10% of symbols have poor quality data - consider re-collection")
    
    if report["summary"]["avg_years_of_data"] < 10:
        report["recommendations"].append("Average data history is below 10 years - collect more historical data")
    
    print(f"Analysis complete: {report['summary']['symbols_with_data']} symbols analyzed")
    
    return report


def format_report_output(report: Dict[str, Any], format_type: str, detailed: bool) -> str:
    """Format report output for display or file export."""
    if format_type == "json":
        return json.dumps(report, indent=2, default=str)
    
    elif format_type == "html":
        return generate_html_report(report)
    
    else:  # txt format
        lines = []
        analysis_type = report.get("analysis_type", "unknown").upper()
        
        lines.extend([
            f"DATA QUALITY REPORT - {analysis_type}",
            "=" * 60,
            f"Generated: {report.get('timestamp', 'Unknown')}",
            ""
        ])
        
        if analysis_type == "OVERVIEW":
            summary = report.get("summary", {})
            lines.extend([
                "SUMMARY:",
                f"  Total Symbols:        {report.get('total_symbols', 0)}",
                f"  Symbols with Data:    {summary.get('symbols_with_data', 0)}",
                f"  Symbols without Data: {summary.get('symbols_without_data', 0)}",
                f"  Average Quality:      {summary.get('avg_quality_score', 0):.1%}",
                f"  Average Years:        {summary.get('avg_years_of_data', 0):.1f}",
                f"  Total Records:        {summary.get('total_records', 0):,}",
                ""
            ])
            
            quality_dist = report.get("quality_distribution", {})
            lines.extend([
                "QUALITY DISTRIBUTION:",
                f"  Excellent (>90%):     {quality_dist.get('excellent', 0)}",
                f"  Good (70-90%):        {quality_dist.get('good', 0)}",
                f"  Fair (50-70%):        {quality_dist.get('fair', 0)}",
                f"  Poor (<50%):          {quality_dist.get('poor', 0)}",
                ""
            ])
            
            common_issues = report.get("common_issues", [])
            if common_issues:
                lines.append("COMMON ISSUES:")
                for issue_info in common_issues[:5]:
                    lines.append(f"  • {issue_info['issue']} ({issue_info['count']} symbols)")
                lines.append("")
            
            recommendations = report.get("recommendations", [])
            if recommendations:
                lines.append("RECOMMENDATIONS:")
                for rec in recommendations:
                    lines.append(f"  • {rec}")
        
        return "\n".join(lines)


def generate_html_report(report: Dict[str, Any]) -> str:
    """Generate HTML data quality report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ margin: 10px 0; }}
            .issue {{ padding: 5px; margin: 2px 0; background: #fff3cd; border-left: 4px solid #ffc107; }}
            .recommendation {{ padding: 5px; margin: 2px 0; background: #d4edda; border-left: 4px solid #28a745; }}
            .quality-excellent {{ color: #28a745; }}
            .quality-good {{ color: #17a2b8; }}
            .quality-fair {{ color: #ffc107; }}
            .quality-poor {{ color: #dc3545; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Data Quality Report</h1>
            <p>Generated: {report.get('timestamp', 'Unknown')}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
    """
    
    if report.get("analysis_type") == "overview":
        summary = report.get("summary", {})
        html += f"""
            <div class="metric">Total Symbols: {report.get('total_symbols', 0)}</div>
            <div class="metric">Symbols with Data: {summary.get('symbols_with_data', 0)}</div>
            <div class="metric">Average Quality Score: {summary.get('avg_quality_score', 0):.1%}</div>
            <div class="metric">Average Years of Data: {summary.get('avg_years_of_data', 0):.1f}</div>
        """
        
        quality_dist = report.get("quality_distribution", {})
        html += """
        </div>
        
        <div class="section">
            <h2>Quality Distribution</h2>
        """
        html += f"""
            <div class="metric quality-excellent">Excellent (>90%): {quality_dist.get('excellent', 0)}</div>
            <div class="metric quality-good">Good (70-90%): {quality_dist.get('good', 0)}</div>
            <div class="metric quality-fair">Fair (50-70%): {quality_dist.get('fair', 0)}</div>
            <div class="metric quality-poor">Poor (<50%): {quality_dist.get('poor', 0)}</div>
        """
        
        common_issues = report.get("common_issues", [])
        if common_issues:
            html += """
        </div>
        
        <div class="section">
            <h2>Common Issues</h2>
            """
            for issue_info in common_issues[:5]:
                html += f'<div class="issue">{issue_info["issue"]} ({issue_info["count"]} symbols)</div>'
        
        recommendations = report.get("recommendations", [])
        if recommendations:
            html += """
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            """
            for rec in recommendations:
                html += f'<div class="recommendation">{rec}</div>'
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html


async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(level="WARNING")
        
        print("📊 DATA QUALITY REPORTER")
        print("=" * 50)
        
        # Check if config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
        
        # Load configuration and create components
        config_loader = ConfigLoader()
        config = config_loader.load_config(str(config_path))
        
        storage = ParquetStorage(config)
        validator = DataValidator(config)
        
        # Get symbols to analyze
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(',')]
        else:
            print("Loading available symbols...")
            symbols = await load_available_symbols(storage)
            if not symbols:
                print("❌ No symbols found in storage")
                sys.exit(1)
        
        print(f"Found {len(symbols)} symbols to analyze")
        
        # Perform analysis
        if args.analysis == "overview":
            report = await generate_overview_report(symbols, storage, validator)
        else:
            # For other analysis types, we would implement similar functions
            print(f"⚠️  Analysis type '{args.analysis}' not yet implemented")
            print("Available: overview")
            sys.exit(1)
        
        # Format and display/save results
        output_text = format_report_output(report, args.format, args.detailed)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"✅ Report saved to: {args.output}")
        else:
            print("\n" + output_text)
        
    except Exception as e:
        print(f"\n❌ Data quality analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())