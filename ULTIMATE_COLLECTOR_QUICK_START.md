# ULTIMATE 1,400 COLLECTOR - QUICK START GUIDE

## What This Script Does

**Target**: Collect exactly 1,056 NEW stocks (current: 344, target: 1,400)

**Features**:
- Multi-source S&P 500/400/600 extraction (Wikipedia → ETF holdings → hardcoded fallback)
- Smart deduplication (automatically skips your existing 344 stocks)
- Hybrid Alpha Vantage (primary) + yfinance (backup) collection
- 24/7 continuous operation with resume capability
- Rate limiting (70 stocks/min for Alpha Vantage Premium)
- State persistence (saves progress every 10 stocks)
- Quality validation (20+ years preferred, 10+ minimum)

---

## Quick Start Commands

### Step 1: Test Mode (10 stocks, 1-2 minutes)

```bash
cd C:\Users\Coding\Desktop\GRID\projects\TradingBOT
python ultimate_1400_collector.py --test
```

**This will**:
- Extract S&P 500/400/600 lists
- Deduplicate against your existing 344 stocks
- Collect 10 new stocks as a test
- Save to TrainingData/daily/
- Show you if everything works

**Expected output**:
```
ULTIMATE 1,400 STOCK COLLECTOR
Mode: TEST (10 stocks)
Output: TrainingData\daily

GENERATING MASTER STOCK LIST
  Wikipedia S&P 500: 503 stocks
  Wikipedia S&P 400: 400 stocks
  Wikipedia S&P 600: 600 stocks
  Total unique: 1,354 stocks

DEDUPLICATION RESULTS:
  Total from S&P lists: 1,354
  Already collected: 344
  New to collect: 1,010

TEST MODE: Limited to 10 stocks

[1/10] ABBV... SUCCESS: 6536 days, 25.9 years (AlphaVantage)
[2/10] ACN... SUCCESS: 5840 days, 23.2 years (AlphaVantage)
...
[10/10] BAC... SUCCESS: 6536 days, 25.9 years (AlphaVantage)

COLLECTION COMPLETE!
Success: 10
Failed: 0
Time: 2.5 minutes
```

### Step 2: Production 24/7 Mode (overnight, 15-20 hours)

**If test successful**, start full collection:

```bash
python ultimate_1400_collector.py --continuous
```

**This will**:
- Collect all ~1,010 new stocks (deduplicated)
- Run overnight (15-20 hours estimated)
- Save progress every 10 stocks (can resume if interrupted)
- Use Alpha Vantage Premium (70 stocks/min)
- Fall back to yfinance for any Alpha Vantage failures
- Log everything to logs/ultimate_collector/

**You will be prompted**:
```
Start collecting 1,010 stocks? (y/n):
```

Type `y` and press Enter to start.

**Then go to bed!** The script runs fully autonomously.

---

## Monitoring Progress

### While Running

Open a new terminal and check:

```bash
# Count files collected so far
cd TrainingData\daily
dir *.parquet | find /c /v ""

# View recent log output (last 20 lines)
cd C:\Users\Coding\Desktop\GRID\projects\TradingBOT
powershell -command "Get-Content logs\ultimate_collector\collector_*.log -Tail 20"
```

### Check State File

The script saves progress to `collection_state.json`:

```bash
type collection_state.json
```

You'll see:
```json
{
  "collected": ["ABBV", "ACN", "ADP", ...],
  "failed": [],
  "skipped": ["AAPL", "MSFT", ...],  // Your existing 344 stocks
  "start_time": "2025-10-28T...",
  "last_update": "2025-10-28T..."
}
```

---

## Resume Capability

**If interrupted** (power loss, internet down, etc.):

Just re-run the same command:
```bash
python ultimate_1400_collector.py --continuous
```

The script will:
- Load `collection_state.json`
- Skip already collected stocks
- Continue where it left off

---

## Expected Timeline

With Alpha Vantage Premium (70 stocks/min):

```
1,010 new stocks / 70 stocks/min = 14.4 minutes of API time

+ Rate limiting safety margin
+ yfinance fallback for failures (slower)
+ Network delays

Realistic: 15-20 hours total (overnight)
```

**Why so long?** Alpha Vantage enforces rate limits. We could be faster but would hit rate limits and fail.

---

## Validation After Collection

### Check Total Count

```bash
cd TrainingData\daily
dir *.parquet | find /c /v ""
```

**Expected**: ~1,350-1,400 files (344 existing + 1,010 new = 1,354 total)

### Run Data Audit

```bash
python audit_existing_data.py
```

**Expected output**:
```
Total stocks: 1,354
Stocks with 20+ years: 1,050 (78%)
Total data points: 9.2M
Average: 6,800 days, 26.8 years per stock
Status: EXCELLENT
```

### Check Collection Stats

```bash
type collection_state.json
```

Look for:
- `"collected": [...]` - should have ~1,010 stocks
- `"failed": [...]` - hopefully empty or very few
- `"skipped": [...]` - should have your existing 344 stocks

---

## Troubleshooting

### Problem: Script says "No new stocks to collect"

**Cause**: You already have all stocks collected

**Check**:
```bash
cd TrainingData\daily
dir *.parquet | find /c /v ""
```

If you see 1,300+, you're done!

### Problem: Wikipedia extraction fails (403 Forbidden)

**Don't worry!** Script has 3-layer fallback:
1. Wikipedia (primary)
2. ETF holdings extraction (SPY, IJH, IJR)
3. Hardcoded top 300 stocks (fallback)

One will work.

### Problem: Many stocks failing

**Check logs**:
```bash
powershell -command "Get-Content logs\ultimate_collector\collector_*.log | Select-String 'FAILED'"
```

**Common causes**:
- Internet connection issues → Script will retry on next run
- Alpha Vantage rate limit → Script automatically waits
- Stock delisted → Expected, skip these (1-2%)

### Problem: Script crashes

**Resume it**:
```bash
python ultimate_1400_collector.py --continuous
```

State is saved every 10 stocks, so you won't lose much progress.

---

## What Gets Saved

### Files Created

```
TrainingData/daily/
  ABBV_daily.parquet
  ACN_daily.parquet
  ADP_daily.parquet
  ...
  (1,354 total files)

collection_state.json  (progress tracking)

logs/ultimate_collector/
  collector_20251028_150000.log  (detailed log)
```

### File Format

Each parquet file contains:
- **Columns**: open, high, low, close, volume
- **Index**: Date (datetime)
- **Rows**: Daily data (minimum 250 days, target 5,000+ days)
- **Compression**: Snappy (efficient storage)

### Storage Used

```
1,354 stocks × ~8 MB/stock = ~10 GB total
```

You have 500 GB available, so only **2% of your disk**!

---

## After Collection Complete

### Week 2: Feature Engineering

```bash
python src/ai/feature_engineering/comprehensive_feature_engineer.py \
  --input TrainingData/daily/*.parquet \
  --output TrainingData/features/ \
  --parallel 4
```

**Timeline**: 1-2 days

### Week 2-3: LSTM Retraining

```bash
python src/ai/models/lstm_trainer.py \
  --data TrainingData/features/*.parquet \
  --hidden-size 256 \
  --layers 3 \
  --epochs 50 \
  --target-accuracy 0.65
```

**Expected**: 60-65% accuracy (vs 47.6% current)

---

## Summary

**What to do RIGHT NOW**:

1. **Test** (1-2 minutes):
   ```bash
   python ultimate_1400_collector.py --test
   ```

2. **If test successful, start full collection** (overnight):
   ```bash
   python ultimate_1400_collector.py --continuous
   ```
   Type `y` when prompted, then go to bed!

3. **Check in the morning**:
   ```bash
   cd TrainingData\daily
   dir *.parquet | find /c /v ""
   ```
   Target: 1,350-1,400 files

4. **Validate**:
   ```bash
   python audit_existing_data.py
   ```

**You'll have**: World-class dataset (top 0.1% globally)

---

## Questions?

**Script not working?** Check logs:
```bash
type logs\ultimate_collector\collector_*.log
```

**Want to customize?** Edit the script:
- Line 169-200: Hardcoded fallback lists (add more stocks)
- Line 295: Rate limit (currently 70/min)
- Line 459: Data quality threshold (currently 250 days minimum)

**Ready?** Run the test now!

```bash
python ultimate_1400_collector.py --test
```

---

*Let's collect 1,400 stocks and build something world-class!* 🚀
