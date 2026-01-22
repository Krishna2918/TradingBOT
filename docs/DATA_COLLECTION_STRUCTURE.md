# Training Data Collection Structure

## Overview

This document describes the comprehensive data collection structure for AI training, designed to collect 20+ years of multi-source financial data using the 4-key Alpha Vantage system.

## Directory Structure

```
TrainingData/
├── market_data/                    # Core OHLCV data (Premium key)
│   ├── AAPL_daily.parquet         # 20+ years daily OHLCV
│   ├── AAPL_1min.parquet          # 1-year 1-minute intraday
│   ├── AAPL_5min.parquet          # 1-year 5-minute intraday
│   └── ... (200 symbols)
├── fundamentals/                   # Company fundamentals (Free key #1)
│   ├── AAPL_overview.parquet      # Company overview
│   ├── AAPL_income.parquet        # Income statements (20 years)
│   ├── AAPL_balance.parquet       # Balance sheets (20 years)
│   ├── AAPL_cashflow.parquet      # Cash flow statements (20 years)
│   ├── AAPL_earnings.parquet      # Earnings data
│   ├── AAPL_insider.parquet       # Insider transactions
│   └── ... (200 symbols)
├── macro_economics/                # Global macro data (Free key #2)
│   ├── GDP.parquet                 # US Real GDP (quarterly)
│   ├── CPI.parquet                 # Consumer Price Index
│   ├── UNEMPLOYMENT.parquet        # Unemployment rate
│   ├── FEDERAL_FUNDS_RATE.parquet  # Interest rates
│   ├── WTI_OIL.parquet            # Oil prices
│   ├── GOLD.parquet               # Gold prices
│   ├── DXY.parquet                # Dollar index
│   └── ... (20+ indicators)
├── sentiment/                      # Sentiment & intelligence (Free key #3)
│   ├── news_sentiment/            # Daily news sentiment by symbol
│   ├── earnings_calendar.parquet  # Earnings announcements
│   ├── top_movers.parquet         # Daily market movers
│   └── market_sentiment.parquet   # Overall market sentiment
├── technical_indicators/           # Technical analysis data
│   ├── AAPL_RSI.parquet           # RSI indicator
│   ├── AAPL_SMA.parquet           # Simple moving averages
│   ├── AAPL_MACD.parquet          # MACD indicator
│   └── ... (9 indicators × 200 symbols)
├── processed/                      # Processed feature sets
│   ├── features/                  # Engineered features
│   ├── labels/                    # Training labels
│   └── datasets/                  # Final training datasets
├── collection_progress.json        # Collection progress tracking
├── orchestrator_status.json       # Orchestrator status
└── validation_report.json         # Data validation results
```

## Data Sources and API Key Usage

### Premium Key (75 RPM) - Core Market Data
- **Purpose**: Real-time and historical market data
- **Rate Limit**: 75 requests per minute (4,500/hour)
- **Data Types**:
  - `TIME_SERIES_DAILY_ADJUSTED`: 20+ years OHLCV with splits/dividends
  - `TIME_SERIES_INTRADAY`: 1min, 5min intraday data (1 year)
  - Technical indicators: RSI, SMA, EMA, MACD, STOCH, ADX, CCI, AROON, BBANDS
- **Compliance**: Uses `entitlement=delayed` for 15-minute delayed data

### Free Key #1 (25/day) - Fundamentals
- **Purpose**: Company financial data and fundamentals
- **Rate Limit**: 25 requests per day
- **Data Types**:
  - `OVERVIEW`: Company overview, market cap, sector, industry
  - `INCOME_STATEMENT`: Annual and quarterly income statements (20 years)
  - `BALANCE_SHEET`: Annual and quarterly balance sheets (20 years)
  - `CASH_FLOW`: Annual and quarterly cash flow statements (20 years)
  - `EARNINGS`: Historical earnings with EPS and revenue
  - `INSIDER_TRANSACTIONS`: Insider buying/selling activity

### Free Key #2 (25/day) - Macro Economics
- **Purpose**: Global economic indicators and commodities
- **Rate Limit**: 25 requests per day
- **Data Types**:
  - `REAL_GDP`: US Real GDP quarterly data
  - `INFLATION`: CPI, core CPI, PCE inflation
  - `UNEMPLOYMENT`: Unemployment rate, payroll data
  - `FEDERAL_FUNDS_RATE`: Interest rates, yield curve
  - `WTI`: Crude oil prices
  - `GOLD`: Precious metals prices
  - `DXY`: Dollar index and currency data

### Free Key #3 (25/day) - Sentiment & Intelligence
- **Purpose**: Market sentiment and intelligence data
- **Rate Limit**: 25 requests per day
- **Data Types**:
  - `NEWS_SENTIMENT`: Daily news sentiment for symbols
  - `EARNINGS_CALENDAR`: Earnings announcements and guidance
  - `TOP_GAINERS_LOSERS`: Daily market movers
  - `MARKET_NEWS_SENTIMENT`: Overall market sentiment

## Symbol Universe

### Target: Top 200 US Stocks
- **Mega Cap** (>$500B): AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, etc.
- **Large Cap** ($100B-$500B): JNJ, V, PG, JPM, HD, MA, CVX, ABBV, etc.
- **Growth & Tech**: NOW, PANW, KLAC, AMAT, WDAY, TEAM, DDOG, SNOW, etc.
- **Financial Services**: WFC, MS, C, BLK, CME, ICE, MCO, TRV, etc.
- **Healthcare & Biotech**: UNH, JNJ, PFE, ABBV, TMO, ABT, DHR, etc.
- **Consumer & Retail**: AMZN, WMT, HD, PG, KO, PEP, COST, TJX, etc.
- **Energy & Utilities**: XOM, CVX, COP, EOG, NEE, DUK, SO, etc.
- **ETFs**: SPY, QQQ, IWM, VTI, VOO (for market context)

## Data Collection Process

### Phase 1: Market Data Collection (Premium Key)
1. **Daily Data**: 20+ years of OHLCV for all 200 symbols
2. **Intraday Data**: 1-year of 1min and 5min data
3. **Technical Indicators**: 9 indicators for all symbols
4. **Rate Limiting**: 0.8 seconds between requests (75 RPM compliance)

### Phase 2: Fundamentals Collection (Free Key #1)
1. **Company Overview**: Basic company information
2. **Financial Statements**: Income, balance sheet, cash flow (20 years)
3. **Earnings Data**: Historical earnings and guidance
4. **Insider Transactions**: Insider trading activity
5. **Rate Limiting**: 25 requests per day (1 request per symbol every 8 days)

### Phase 3: Macro Economics Collection (Free Key #2)
1. **Economic Indicators**: GDP, CPI, unemployment (20+ years)
2. **Interest Rates**: Fed funds rate, treasury yields
3. **Commodities**: Oil, gold, silver, copper prices
4. **Currency Data**: DXY, major currency pairs
5. **Rate Limiting**: 25 requests per day

### Phase 4: Sentiment Collection (Free Key #3)
1. **News Sentiment**: Daily sentiment for all symbols
2. **Market Intelligence**: Top movers, earnings calendar
3. **Analyst Data**: Recommendations and estimates
4. **Rate Limiting**: 25 requests per day

### Phase 5: Validation & Quality Control
1. **Completeness Check**: Verify all expected files exist
2. **Quality Validation**: Check data integrity and consistency
3. **Training Readiness**: Ensure >95% completeness
4. **Report Generation**: Comprehensive validation report

## Data Formats

### Parquet Files
- **Compression**: Snappy compression for optimal performance
- **Index**: DateTime index for time series data
- **Columns**: Standardized column names across all datasets
- **Metadata**: Embedded metadata for data lineage

### Market Data Schema
```python
# Daily OHLCV
columns = ['Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume', 'Dividend_Amount', 'Split_Coefficient']
index = DatetimeIndex

# Intraday OHLCV
columns = ['Open', 'High', 'Low', 'Close', 'Volume']
index = DatetimeIndex
```

### Fundamentals Schema
```python
# Income Statement
columns = ['totalRevenue', 'costOfRevenue', 'grossProfit', 'operatingIncome', 'netIncome', ...]
index = DatetimeIndex (quarterly/annual)

# Balance Sheet
columns = ['totalAssets', 'totalLiabilities', 'totalShareholderEquity', 'cash', 'totalDebt', ...]
index = DatetimeIndex (quarterly/annual)
```

### Macro Economics Schema
```python
# Economic Indicators
columns = ['value', 'unit']
index = DatetimeIndex

# Commodities
columns = ['price', 'currency', 'unit']
index = DatetimeIndex
```

## Quality Control

### Data Validation Rules
1. **Completeness**: >95% of expected files must exist
2. **Consistency**: OHLC relationships must be logical (High >= Low, etc.)
3. **Continuity**: Time series should have minimal gaps
4. **Range Validation**: Prices and volumes within reasonable ranges
5. **Duplicate Detection**: No duplicate timestamps in time series

### Error Handling
1. **Retry Logic**: Up to 3 retries for failed requests
2. **Graceful Degradation**: Continue collection even if some symbols fail
3. **Progress Tracking**: Persistent progress tracking across restarts
4. **Error Logging**: Comprehensive error logging and reporting

## Usage Instructions

### Starting Data Collection
```bash
# Start full collection process
python start_data_collection.py

# Check current status
python start_data_collection.py status
```

### Monitoring Progress
```python
from src.data_collection.training_data_orchestrator import get_collection_status

status = get_collection_status()
print(f"Progress: {status['orchestrator']['total_progress']:.1f}%")
```

### Validation
```python
from src.data_collection.training_data_orchestrator import TrainingDataOrchestrator

orchestrator = TrainingDataOrchestrator()
validation = await orchestrator._final_validation()
print(f"Training ready: {validation['training_ready']}")
```

## Expected Timeline

### Collection Phases
- **Phase 1 (Market Data)**: 2-3 hours (200 symbols × 3 datasets + indicators)
- **Phase 2 (Fundamentals)**: 8-10 days (25 requests/day limit)
- **Phase 3 (Macro Economics)**: 1 day (20+ indicators)
- **Phase 4 (Sentiment)**: 8-10 days (25 requests/day limit)
- **Phase 5 (Validation)**: 30 minutes

### Total Duration: 10-14 days
- **Bottleneck**: Free key daily limits (25 requests/day)
- **Optimization**: Parallel collection where possible
- **Monitoring**: Continuous progress tracking and reporting

## Success Criteria

### Data Completeness
- ✅ 200 US stocks with 20+ years daily data
- ✅ 200 US stocks with 1-year intraday data (1min, 5min)
- ✅ Complete fundamental data (income, balance, cash flow)
- ✅ 20+ macro economic indicators
- ✅ Comprehensive sentiment and intelligence data

### Quality Metrics
- ✅ >95% data completeness across all categories
- ✅ <5% missing values in core datasets
- ✅ Logical consistency in OHLC data
- ✅ Proper time series continuity

### Training Readiness
- ✅ All data in standardized Parquet format
- ✅ Consistent datetime indexing
- ✅ Validated data quality and integrity
- ✅ Ready for feature engineering pipeline

This comprehensive data collection structure ensures that the AI training pipeline has access to the highest quality, most complete dataset possible for developing robust trading models.