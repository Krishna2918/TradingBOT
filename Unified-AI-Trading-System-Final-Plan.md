# Unified AI Trading System - Final Plan

## Final Plan for Collecting 20 Years of Historical and Intraday Data Using yfinance (Updated with Logging Feature)

### Overview
This plan details a complete data collection strategy for your Canadian AI Trading Bot, using only the `yfinance` library to fetch comprehensive financial data for 100 TSX/TSXV symbols (e.g., top S&P/TSX 60 like RY.TO, TD.TO, SHOP.TO, plus high-volume stocks) over 20 years (January 1, 2005, to October 25, 2025). 

The system will collect intraday (1min, 5min, 15min, 30min), daily, weekly, monthly, quarterly, and yearly data, storing everything in "C:\Users\Coding\Desktop\TradingBOT\PastData" in an organized, appendable format (e.g., Parquet for efficiency, with subfolders like /intraday/RY.TO/5min/ for easy querying and appending without duplication or loss). 

The architecture focuses on two core components: an **Intraday Appender** for real-time session data and a **Historical Appender** for backfilling higher time frames. 

To enable AI learning from the dataset, a new **Logging Feature** is added: It tracks data collection progress (e.g., total rows/symbols fetched, completion percentage, error rates) and outlines future steps (e.g., next symbols to fetch, recommended training actions based on data quality), stored in a machine-readable format (e.g., JSON logs in `logs/data_progress.json`) for AI analysis and self-optimization (e.g., adjusting fetch priorities based on past gaps or failures).

### Preparation Phase (30-60 Minutes)

1. **Symbol Selection**: Create a configurable list of 100 TSX/TSXV symbols, starting with high-priority ones (e.g., S&P/TSX 60). Verify availability by testing a sample fetch.

2. **Environment Setup**: Install `yfinance` (if not already in your requirements.txt). Prepare the storage directory with subfolders for each time frame and symbol. Use Parquet for appendable storage to handle incremental updates efficiently.

3. **Rate Limit Safeguards**: Implement user-agent rotation (e.g., from a list of 10-20 common browsers) and randomized delays (1-5 seconds per request) to prevent IP blocks. Optionally prepare free proxies (e.g., from public lists) for high-volume days, though start without to test.

4. **Session Detection**: Define TSX market hours (09:30–16:00 EDT, adjusted for DST via a dynamic calendar check) and holidays (from your `data/event_calendar.json`) to trigger intraday appending only during active sessions.

5. **Progress Tracking**: Set up a shared log or SQLite database (e.g., in `data/change_log.db`) to record completed fetches, enabling resumption from interruptions.

6. **Logging Feature Setup**: Initialize a JSON log file (`logs/data_progress.json`) to record real-time metrics (e.g., fetched rows per symbol/time frame, total data size in GB, completion percentage, error counts) and generate future step recommendations (e.g., "Prioritize RY.TO 5min due to 20% gap; next: train LSTM on recent 12 months"). Ensure logs are structured for AI parsing (e.g., key-value pairs like {"symbol": "RY.TO", "fetched_rows": 6000, "completion_pct": 95, "future_steps": ["Fetch missing 2025-10", "Validate for gaps", "Initiate model training"]}) to support self-optimization in future runs.

### Execution Phase

1. **Historical Appender (Backfill Focus)**: This component fetches and appends 20 years of data for daily, weekly, monthly, quarterly, and yearly time frames (derived from daily data where needed). Run it off-hours or anytime, as these require fewer requests.
   - Prioritize recent data first (e.g., last 5 years) for quick training wins.
   - Fetch daily data with `period='max'` (covers 20+ years for most symbols), then aggregate to weekly/monthly/quarterly/yearly dynamically.
   - For intraday historical, fetch the maximum available (60 days for 5min/15min/30min, 7 days for 1min) and append to files.
   - Pace requests with 1-5 second randomized delays to stay under safe IP limits (~2,000-3,000 requests/day).
   - Resume incomplete fetches using progress logs to avoid re-downloading.

2. **Intraday Appender (Real-Time Focus)**: This runs only during TSX sessions, polling for 1min/5min/15min/30min data to append live updates without loss.
   - Poll every 1-5 minutes (dynamically based on interval, e.g., more frequent for 1min during volatile periods detected via recent volume).
   - Use `period='60d'` for backfill within limits to supplement historical intraday.
   - Append incrementally to existing files, checking for duplicates (e.g., by timestamp) to ensure seamless integration with historical data.
   - If a block occurs (e.g., CAPTCHA), pause and retry with increased delays or proxy rotation.

3. **Parallelization and Optimization**: Run multiple threads or processes (e.g., 5-10 concurrent) to fetch different symbols or time frames simultaneously, maximizing throughput while keeping total requests paced to avoid IP throttling. Dynamically adjust based on response times (e.g., increase delays if errors rise).

4. **Daily/Off-Hours EOD Appending**: Outside sessions, append daily data (fetched at close) to higher time frames, ensuring weekly/monthly/quarterly/yearly aggregates update without gaps.

5. **Logging Feature Integration**: During execution, log progress after each append (e.g., rows added, data size growth) and generate future steps (e.g., based on completion gaps or quality checks). Update the JSON log in real-time for AI-accessible insights, enabling future self-optimization (e.g., AI could parse logs to prioritize unfetched symbols or suggest retraining on complete datasets).

### Post-Collection Phase (30 Minutes)

1. **Aggregation and Validation**: Merge appended data into unified Parquet files per symbol/time frame, validating for completeness (e.g., check row counts against expected trading days) and quality (no unexplained gaps).

2. **Logging Finalization**: Summarize overall progress in the JSON log (e.g., total data collected in GB, average fetch speed, unresolved gaps) and append high-level future steps (e.g., "Analyze dataset for anomalies; train models on 20-year daily data; expand to 200 symbols").

3. **Cleanup**: Compress older data and archive logs for future audits.

### Time Estimate

- **Historical Higher Time Frames (Daily/Weekly/Monthly/Quarterly/Yearly, 20 Years)**: ~2-3 hours for 100 symbols (one bulk request per symbol/time frame, ~500 total requests with 2-3 second delays).
- **Historical Intraday (Limited to Recent 60 Days for 5min/15min/30min, 7 Days for 1min)**: ~1-2 hours (100 symbols x 4 intervals, ~400 requests with delays).
- **Real-Time Intraday Appending**: Ongoing during sessions (~6.5 hours/day, polling 100-400 times/day with minimal delays, adding ~1-2 hours setup/monitoring per day; full 20-year intraday not feasible due to 60-day limit, but recent appending completes in minutes per session).
- **Total for Full 20-Year Dataset**: ~3-5 hours initial backfill (limited intraday depth), plus ongoing appending for intraday. The 20-year scope is fully achievable for higher time frames but capped for intraday due to yfinance limits; expect daily EOD updates to take seconds per run.