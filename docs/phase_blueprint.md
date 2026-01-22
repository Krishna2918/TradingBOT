# 🎯 CODEX-OPTIMIZED PHASE BLUEPRINT

## **EXECUTION ORDER (Rate-Limit Optimized)**

### **Phase 1: Foundation & Caching (0-30 min)**
1. **Database Bootstrap** - Create SQLite + DuckDB schemas
2. **Environment Validation** - Verify all API keys and Ollama models
3. **Caching Layer Setup** - Implement rate-limit protection
4. **Dependency Installation** - Ensure all packages are available

### **Phase 2: Data Pipeline (30-90 min)**
5. **Universe Ingestion** - Load TSX/TSXV stock universe (497 stocks)
6. **Feature Engineering** - Technical indicators with DuckDB persistence
7. **Sentiment Pipeline** - NewsAPI integration with backoff
8. **Fundamental Data** - Finnhub integration with caching

### **Phase 3: AI Integration (90-120 min)**
9. **Ollama Health Check** - Verify all 3 models responding
10. **Ensemble Scoring** - Multi-factor AI analysis
11. **Risk Assessment** - Kelly Criterion position sizing
12. **Selection Logic** - Top 5 stock selection

### **Phase 4: Trading Engine (120-150 min)**
13. **Order Management** - Paper trading execution
14. **Position Tracking** - Real-time P&L monitoring
15. **Risk Management** - Stop-loss and circuit breakers
16. **Performance Metrics** - Daily return calculations

### **Phase 5: Dashboard Integration (150-180 min)**
17. **Data Connectors** - SQLite + DuckDB readers
18. **UI Components** - Real-time updates
19. **Visualization** - Charts and metrics display
20. **Error Handling** - Graceful degradation

### **Phase 6: Testing & Validation (180-210 min)**
21. **Unit Tests** - All modules tested
22. **Integration Tests** - End-to-end workflow
23. **Performance Tests** - Load and stress testing
24. **Final Validation** - Complete system check

## **RATE-LIMIT PROTECTION**

### **API Limits & Batching:**
- **Alpha Vantage**: 5 calls/minute, 500/day → Batch 20 stocks per call
- **NewsAPI**: 1000 calls/day → Cache results, batch by sector
- **Finnhub**: 60 calls/minute → Batch 10 stocks per call
- **Questrade**: No limits → Use for real-time data

### **Caching Strategy:**
- **DuckDB**: Store all market data with TTL
- **SQLite**: Store trading state and positions
- **Redis**: Cache API responses (if available)
- **Local Files**: Cache universe and configurations

### **Error Handling:**
- **Rate Limits**: Exponential backoff with tenacity
- **API Failures**: Fallback to cached data
- **Ollama Errors**: Retry with different model
- **Database Locks**: Close handles, retry connection

## **SELF-HEALING PROTOCOL**

### **When Stuck:**
1. **DIAGNOSE**: Capture error, timeout, or hang
2. **PATCH**: Apply minimal fix (dep, rate-limit, DB lock)
3. **RETRY**: Resume from failed step only
4. **RECORD**: Log root cause and fix

### **Common Fixes:**
- **Missing Deps**: `pip install <package>` + update requirements.txt
- **Rate Limits**: Reduce batch size, add backoff
- **DB Locks**: Remove *.wal/*.shm files, retry
- **Ollama Down**: Start server, health check
- **Path Issues**: Use correct separators for OS

## **SUCCESS METRICS**

### **Performance Targets:**
- **Universe Processing**: < 2 hours for 497 stocks
- **AI Response Time**: < 30 seconds per model
- **Memory Usage**: < 8GB during operation
- **CPU Usage**: < 80% during heavy processing

### **Quality Gates:**
- **All Tests Pass**: pytest with 100% success
- **No Placeholders**: All code implemented
- **Real Data**: No mock or hardcoded values
- **Error Recovery**: Graceful handling of all failures

## **EXECUTION COMMANDS**

### **Windows (PowerShell):**
```powershell
cd ai-trading-system
.\.venv\Scripts\Activate.ps1
python scripts/run_all.ps1
```

### **Linux/Mac (Bash):**
```bash
cd ai-trading-system
source .venv/bin/activate
bash scripts/run_all.sh
```

### **Final Validation:**
```bash
python scripts/final_validation.py
# Expected output: ✅✅✅ ALL TESTS PASSED — SYSTEM READY FOR BUILD 🚀
```

## **BUILD LOG TRACKING**

All actions, patches, and retries will be logged to:
- `docs/build_log.md` - Detailed execution log
- `logs/system.log` - System-level logging
- `logs/ai.log` - AI model interactions
- `logs/trading.log` - Trading decisions and execution

## **COMMIT STRATEGY**

- **Phase 1**: Foundation setup
- **Phase 2**: Data pipeline
- **Phase 3**: AI integration
- **Phase 4**: Trading engine
- **Phase 5**: Dashboard integration
- **Phase 6**: Testing and validation

Each phase commits with clear messages and working state.