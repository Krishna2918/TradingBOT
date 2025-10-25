# Data Collection Implementation Plan

## 📊 Analysis: Final Plan vs Current System

### Current System Status (from SOURCE_OF_TRUTH.md)
- ✅ Yahoo Finance integration working (real-time data)
- ✅ Data pipeline exists (`src/data_pipeline/`)
- ✅ Event calendar with TSX holidays
- ❌ **CRITICAL GAP**: No historical data collection system
- ❌ **CRITICAL GAP**: ML models not trained (requires historical data)
- ❌ **CRITICAL GAP**: No systematic data storage in Parquet format

### Implementation Priority: **CRITICAL** 
**Reason**: This addresses the #1 blocking issue for live trading - untrained ML models need historical data.

---

## 🎯 Implementation Strategy

### Phase 1: Foundation (Day 1-2)
**Goal**: Set up data collection infrastructure

#### 1.1 Create Data Collection Module
```
src/data_collection/
├── __init__.py
├── historical_appender.py      # 20-year backfill
├── intraday_appender.py        # Real-time session data
├── symbol_manager.py           # TSX/TSXV symbol lists
├── progress_tracker.py         # JSON logging system
├── data_validator.py           # Quality checks
└── storage_manager.py          # Parquet file management
```

#### 1.2 Create Storage Structure
```
PastData/
├── daily/
│   ├── RY.TO.parquet
│   ├── TD.TO.parquet
│   └── ...
├── intraday/
│   ├── RY.TO/
│   │   ├── 1min.parquet
│   │   ├── 5min.parquet
│   │   ├── 15min.parquet
│   │   └── 30min.parquet
│   └── ...
├── weekly/
├── monthly/
├── quarterly/
└── yearly/
```

#### 1.3 Progress Logging System
```
logs/
├── data_progress.json          # AI-readable progress
├── data_collection.log         # Human-readable logs
└── data_errors.log            # Error tracking
```

### Phase 2: Historical Backfill (Day 3-5)
**Goal**: Collect 20 years of daily/weekly/monthly data

#### 2.1 Symbol Selection (100 TSX/TSXV stocks)
**S&P/TSX 60 Core (Top 30)**:
- RY.TO, TD.TO, SHOP.TO, CNQ.TO, SU.TO
- ENB.TO, TRP.TO, BNS.TO, BMO.TO, CM.TO
- WCN.TO, CSU.TO, ATD.TO, MFC.TO, SLF.TO
- CCL-B.TO, DOL.TO, QSR.TO, WSP.TO, GIB-A.TO
- AEM.TO, K.TO, NTR.TO, PKI.TO, WPM.TO
- IFC.TO, EMA.TO, FNV.TO, TIH.TO, CAR-UN.TO

**High-Volume TSXV (20 stocks)**:
- HUT.TO, BITF.TO, HIVE.TO (crypto miners)
- MMED.TO, NUMI.TO (psychedelics)
- Plus 15 other high-volume TSXV stocks

**Growth/Tech Stocks (30)**:
- LSPD.TO, NVEI.TO, DCBO.TO, TOI.TO
- Plus 26 other growth stocks

**Dividend/Utilities (20)**:
- FTS.TO, H.TO, CU.TO, EIF.TO
- Plus 16 other dividend stocks

#### 2.2 Execution Timeline
- **Day 3**: Daily data (20 years, 100 symbols) - ~3 hours
- **Day 4**: Weekly/Monthly aggregation - ~1 hour  
- **Day 5**: Quarterly/Yearly aggregation - ~1 hour

### Phase 3: Intraday Collection (Day 6-7)
**Goal**: Collect recent intraday data (limited by yfinance)

#### 3.1 Intraday Limits (yfinance constraints)
- **1min data**: Last 7 days only
- **5min data**: Last 60 days only
- **15min data**: Last 60 days only
- **30min data**: Last 60 days only

#### 3.2 Real-Time Appender Setup
- Run during TSX hours (9:30 AM - 4:00 PM EDT)
- Poll every 1-5 minutes based on volatility
- Append to existing files without duplication

### Phase 4: AI Model Training (Day 8-14)
**Goal**: Train ML models with collected data

#### 4.1 LSTM Model Training
- **Data**: 1min intraday (last 7 days) + daily (20 years)
- **Features**: OHLCV + technical indicators
- **Target**: Next 1-minute price movement
- **Training Time**: ~2-4 hours

#### 4.2 GRU Model Training  
- **Data**: 5-15min intraday (last 60 days) + daily (20 years)
- **Features**: OHLCV + macro indicators + options data
- **Target**: Next 5-15 minute price movement
- **Training Time**: ~3-6 hours

#### 4.3 RL Agent Training
- **Environment**: Historical trading simulation
- **Data**: Daily data (20 years) for backtesting
- **Algorithms**: PPO, DQN
- **Training Time**: ~6-12 hours

---

## 🛠️ Technical Implementation

### Core Components to Build

#### 1. Historical Appender (`src/data_collection/historical_appender.py`)
```python
class HistoricalAppender:
    def __init__(self):
        self.symbols = load_tsx_symbols()
        self.progress_tracker = ProgressTracker()
        
    def fetch_daily_data(self, symbol: str, years: int = 20):
        """Fetch 20 years of daily data for symbol"""
        
    def fetch_all_symbols(self):
        """Fetch data for all 100 symbols with rate limiting"""
        
    def aggregate_timeframes(self):
        """Create weekly/monthly/quarterly/yearly from daily"""
```

#### 2. Progress Tracker (`src/data_collection/progress_tracker.py`)
```python
class ProgressTracker:
    def __init__(self):
        self.log_file = "logs/data_progress.json"
        
    def log_progress(self, symbol: str, timeframe: str, rows: int):
        """Log collection progress for AI analysis"""
        
    def get_completion_status(self) -> Dict:
        """Return overall completion percentage"""
        
    def generate_next_steps(self) -> List[str]:
        """AI-readable recommendations for next actions"""
```

#### 3. Storage Manager (`src/data_collection/storage_manager.py`)
```python
class StorageManager:
    def __init__(self):
        self.base_path = Path("PastData")
        
    def save_to_parquet(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """Save data to organized Parquet files"""
        
    def append_to_parquet(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """Append new data without duplication"""
        
    def validate_data_quality(self, symbol: str, timeframe: str) -> bool:
        """Check for gaps and data quality issues"""
```

---

## 📈 Expected Outcomes

### Data Volume Estimates
- **Daily Data (20 years, 100 symbols)**: ~500MB-1GB
- **Intraday Data (60 days, 100 symbols)**: ~2-5GB  
- **Total Storage**: ~3-6GB (highly compressed in Parquet)

### Training Data Availability
- **LSTM Training**: ✅ Ready after Phase 2 (daily) + Phase 3 (1min)
- **GRU Training**: ✅ Ready after Phase 3 (5-15min intraday)
- **RL Training**: ✅ Ready after Phase 2 (20 years daily for backtesting)

### Performance Impact
- **Collection Speed**: ~500-1000 symbols/hour (with rate limiting)
- **Storage Efficiency**: Parquet format ~80% smaller than CSV
- **Query Performance**: Sub-second data loading for model training

---

## 🚨 Risk Mitigation

### Rate Limiting Protection
- Random delays (1-5 seconds between requests)
- User-agent rotation (20 different browser strings)
- Request monitoring (max 2000-3000/day)
- Automatic backoff on errors

### Data Quality Assurance
- Duplicate detection by timestamp
- Gap analysis (missing trading days)
- Outlier detection (impossible price movements)
- Validation against known market events

### Recovery Mechanisms
- Progress checkpointing (resume from interruption)
- Error logging and retry logic
- Incremental updates (append-only)
- Data backup and versioning

---

## 🎯 Success Metrics

### Phase 1 Success Criteria
- [ ] Data collection infrastructure created
- [ ] Storage structure established
- [ ] Progress logging system operational
- [ ] Symbol list validated (100 TSX/TSXV stocks)

### Phase 2 Success Criteria  
- [ ] 20 years daily data collected (100 symbols)
- [ ] Weekly/monthly/quarterly/yearly aggregated
- [ ] Data quality validated (no major gaps)
- [ ] Progress logs show >95% completion

### Phase 3 Success Criteria
- [ ] 60 days intraday data collected (5min, 15min, 30min)
- [ ] 7 days 1min data collected
- [ ] Real-time appender operational during market hours
- [ ] No data duplication or loss

### Phase 4 Success Criteria
- [ ] LSTM model trained and validated (accuracy >55%)
- [ ] GRU model trained and validated (accuracy >55%)
- [ ] RL agents trained (Sharpe ratio >0.8 in backtests)
- [ ] Models integrated into trading system

---

## 🚀 Next Steps

### Immediate Actions (Today)
1. **Create data collection module structure**
2. **Set up PastData directory structure** 
3. **Implement symbol list (100 TSX/TSXV stocks)**
4. **Create progress tracking system**

### This Week
1. **Implement Historical Appender**
2. **Start 20-year daily data collection**
3. **Set up intraday collection for recent data**
4. **Validate data quality and completeness**

### Next Week  
1. **Begin ML model training with collected data**
2. **Integrate trained models into trading system**
3. **Validate model performance in paper trading**
4. **Update SOURCE_OF_TRUTH.md with new capabilities**

This implementation directly addresses the critical gap identified in SOURCE_OF_TRUTH.md and enables the transition from paper trading to live trading with properly trained AI models.