"""
Comprehensive Data Audit Script for NYSE/NASDAQ Stock Data
Audits 164 existing stock data files for completeness and quality
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

def audit_stock_data():
    """Audit all stock data files"""

    data_dir = Path('TrainingData/daily')
    results = []

    print("=" * 100)
    print("COMPREHENSIVE NYSE/NASDAQ DATA AUDIT")
    print("=" * 100)
    print()

    # Get all parquet files
    parquet_files = sorted(data_dir.glob('*.parquet'))
    total_files = len(parquet_files)

    print(f"Found {total_files} stock data files")
    print()
    print(f"{'Symbol':<10} {'Start Date':<12} {'End Date':<12} {'Days':<8} {'Years':<8} {'Quality':<10} {'Issues'}")
    print("-" * 100)

    # Today's date for reference
    today = datetime.now().date()
    target_start = datetime(2005, 1, 1).date()  # 20 years ago

    total_days = 0
    stocks_with_20_years = 0
    stocks_with_issues = 0

    for file in parquet_files:
        symbol = file.stem.replace('_daily', '')

        try:
            df = pd.read_parquet(file)

            # Get date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                dates = df['date']
            elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
                dates = pd.to_datetime(df.index)
            else:
                print(f"{symbol:<10} ERROR: No date column found")
                continue

            start_date = dates.min().date()
            end_date = dates.max().date()
            num_days = len(df)
            num_years = (end_date - start_date).days / 365.25

            # Check data quality
            issues = []

            # Check if we have 20+ years
            if start_date <= datetime(2005, 1, 1).date():
                stocks_with_20_years += 1
            else:
                years_missing = (datetime(2005, 1, 1).date() - start_date).days / 365.25
                if years_missing > 1:
                    issues.append(f"Missing {years_missing:.1f}y")

            # Check if data is recent
            days_old = (today - end_date).days
            if days_old > 7:
                issues.append(f"Stale({days_old}d)")

            # Check for missing values
            if 'close' in df.columns:
                missing_pct = (df['close'].isna().sum() / len(df)) * 100
                if missing_pct > 1:
                    issues.append(f"Missing:{missing_pct:.1f}%")

            # Check data gaps (should be ~252 trading days per year)
            expected_days = int(num_years * 252)
            coverage = (num_days / expected_days) * 100
            if coverage < 90:
                issues.append(f"Gaps:{100-coverage:.0f}%")

            # Quality assessment
            if not issues:
                quality = "✓ GOOD"
            elif len(issues) == 1 and "Stale" in issues[0]:
                quality = "⚠ OK"
            else:
                quality = "✗ POOR"
                stocks_with_issues += 1

            issues_str = ", ".join(issues) if issues else "None"

            print(f"{symbol:<10} {start_date} {end_date} {num_days:<8} {num_years:<8.1f} {quality:<10} {issues_str}")

            total_days += num_days

            results.append({
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'num_days': num_days,
                'num_years': num_years,
                'quality': quality,
                'issues': issues_str
            })

        except Exception as e:
            print(f"{symbol:<10} ERROR: {str(e)[:50]}")
            stocks_with_issues += 1

    # Summary statistics
    print()
    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print(f"Total stocks: {total_files}")
    print(f"Stocks with 20+ years data: {stocks_with_20_years} ({stocks_with_20_years/total_files*100:.1f}%)")
    print(f"Stocks with issues: {stocks_with_issues} ({stocks_with_issues/total_files*100:.1f}%)")
    print(f"Total data points: {total_days:,}")
    print(f"Average days per stock: {total_days/total_files:.0f}")
    print(f"Average years per stock: {total_days/total_files/252:.1f}")
    print()

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('data_audit_results.csv', index=False)
    print("Detailed results saved to: data_audit_results.csv")
    print()

    # Recommendations
    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    if stocks_with_20_years < total_files * 0.8:
        print("⚠ Less than 80% of stocks have 20+ years of data")
        print("   → Recommend: Extend historical data collection for older stocks")
    else:
        print("✓ Good historical coverage (80%+ stocks have 20+ years)")

    if stocks_with_issues > total_files * 0.2:
        print(f"⚠ {stocks_with_issues} stocks have data quality issues")
        print("   → Recommend: Update/clean problematic stocks")
    else:
        print("✓ Good data quality (fewer than 20% issues)")

    # Check if we need more stocks
    if total_files < 200:
        print(f"⚠ Current stock count: {total_files} (target: 200+ for robust training)")
        print(f"   → Recommend: Add {200 - total_files} more NYSE/NASDAQ stocks")
    else:
        print(f"✓ Sufficient stocks: {total_files} (exceeds 200 target)")

    print()
    print("=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print("1. Review data_audit_results.csv for detailed per-stock analysis")
    print("2. Update stale data (stocks with end_date > 7 days old)")
    print("3. Fill gaps for stocks missing 20+ years of history")
    print("4. Consider adding more stocks to reach 200+ total")
    print()

if __name__ == '__main__':
    audit_stock_data()
