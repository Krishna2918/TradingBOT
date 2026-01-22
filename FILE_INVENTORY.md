# GRID Root Directory - File Inventory

**Purpose:** Categorize all files at GRID root level by function

**Last Updated:** 2025-10-29

---

## 📊 Stock Data Collection Scripts (Core Functionality)

These files are **CORRECTLY** at root - they handle stock data collection:

### Main Collectors
| File | Purpose |
|------|---------|
| `full_production_collector.py` | Main production data collector (1400+ stocks) |
| `production_collector_fixed.py` | Fixed/improved production collector |
| `real_1400_stocks_collector.py` | Real-time 1400 stock collector |
| `demo_data_collection.py` | Demo/test data collection |
| `create_full_stock_list.py` | Stock list generation |
| `expand_stock_universe.py` | Expand stock coverage |

### System Management
| File | Purpose |
|------|---------|
| `automated_maintenance.py` | Automated system maintenance |
| `check_progress.py` | Collection progress monitoring |
| `start_collection.py` | Start data collection |
| `stop_collection.py` | Stop data collection |
| `start_production_monitoring.py` | Start monitoring services |

### Configuration & Setup
| File | Purpose |
|------|---------|
| `config_migration_tool.py` | Migrate between config versions |
| `config_validator.py` | Validate configuration files |
| `deployment_setup.py` | Automated deployment setup |
| `system_requirements_checker.py` | Check system requirements |

### Monitoring & Diagnostics
| File | Purpose |
|------|---------|
| `system_health_dashboard.py` | System health monitoring dashboard |
| `performance_analyzer.py` | Analyze collection performance |
| `performance_capacity_monitor.py` | Monitor system capacity |
| `diagnostic_toolkit.py` | Diagnostic tools |
| `data_quality_reporter.py` | Data quality reports |

### Operations & Recovery
| File | Purpose |
|------|---------|
| `emergency_recovery.py` | Emergency system recovery |
| `log_rotation_manager.py` | Manage log rotation |

---

## 🧪 Test Files (Data Collection System)

These test files are for the **data collection system** and should stay at root:

| File | Tests What |
|------|-----------|
| `test_monitoring_system.py` | Data collection monitoring system |
| `test_orchestrator_basic.py` | ContinuousCollector orchestrator |
| `test_orchestrator_implementation.py` | Orchestrator implementation |
| `test_worker_pool_integration.py` | Data collection worker pool |
| `run_tests.py` | Test runner |

---

## 📁 Directories

| Directory | Purpose | Belongs To |
|-----------|---------|------------|
| `continuous_data_collection/` | Main data collection package | Data Collection ✓ |
| `config/` | Configuration files | Data Collection ✓ |
| `monitoring/` | Prometheus, Grafana configs | Data Collection ✓ |
| `scripts/` | Deployment scripts | Data Collection ✓ |
| `tests/` | Integration tests | Data Collection ✓ |
| `data/` | Data storage | Data Collection ✓ |
| `logs/` | Log files | Data Collection ✓ |
| `projects/` | Sub-projects (TradingBOT) | Container ✓ |
| `temp/` | Temporary files | Shared ✓ |
| `__pycache__/` | Python cache | Auto-generated |
| `tools/` | Development tools | Development ⚠️ |

### ⚠️ tools/ Directory
Contains development tools (7-zip, cmake, cuda, git, leptonica, etc.)
- **Not specific to data collection or trading**
- General development utilities
- Could potentially be moved outside GRID or kept as-is
- Size consideration: These tools may take significant space

---

## 📄 Documentation Files

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Main repository overview | ✓ Correct |
| `PROJECT_STRUCTURE.md` | Directory structure guide | ✓ Correct |
| `REORGANIZATION_SUMMARY.md` | Reorganization changelog | ✓ Correct |
| `FILE_INVENTORY.md` | This file - file categorization | ✓ Correct |
| `PRODUCTION_DEPLOYMENT.md` | Production deployment guide | ✓ Correct |
| `PRODUCTION_READINESS_CHECKLIST.md` | Pre-launch checklist | ✓ Correct |
| `README_PRODUCTION.md` | Production features | ✓ Correct |
| `DEPLOYMENT_GUIDE.md` | General deployment guide | ✓ Correct |
| `OPERATIONAL_PROCEDURES.md` | Operations manual | ✓ Correct |
| `PERFORMANCE_OPTIMIZATION_GUIDE.md` | Performance tuning | ✓ Correct |
| `SYSTEM_REQUIREMENTS.md` | System specifications | ✓ Correct |
| `TROUBLESHOOTING_GUIDE.md` | Problem solving guide | ✓ Correct |

---

## 🐳 Containerization Files

| File | Purpose | Status |
|------|---------|--------|
| `Dockerfile` | Container image definition | ✓ Correct |
| `docker-compose.yml` | Multi-service orchestration | ✓ Correct |
| `.dockerignore` | Docker build exclusions | ✓ Correct |
| `.env.example` | Environment template | ✓ Correct |
| `requirements.txt` | Python dependencies | ✓ Correct |

---

## ✅ Files Moved to TradingBOT

These files were **successfully moved** to `projects/TradingBOT/`:

| Original Location | New Location | Type |
|------------------|--------------|------|
| `models/` | `models_archive/` | Directory |
| `alerts.db` | `alerts.db` | Database |
| `feature_manifest_20251027_115407.json` | Same name | Config |
| `feature_manifest_20251027_115437.json` | Same name | Config |
| `test_core_functionality.py` | Same name | Test |
| `AI_TRAINING_REPORT.md` | Deleted (duplicate) | Doc |

---

## 🎯 Summary

### Total Files at GRID Root

- **Python Scripts:** 27 (all data collection related ✓)
- **Test Files:** 5 (all data collection tests ✓)
- **Documentation:** 12 (all data collection docs ✓)
- **Config Files:** 4 (Docker, env, requirements ✓)
- **Directories:** 11 (1 is projects/ containing TradingBOT)

### All Files Are Correctly Placed ✅

After the reorganization and moving `test_core_functionality.py`:
- ✅ All root-level files are data collection related
- ✅ All trading bot files are in `projects/TradingBOT/`
- ✅ Clear separation of concerns
- ✅ No trading bot code at root level

---

## ⚠️ Optional: tools/ Directory

The `tools/` directory contains general development utilities:
- 7-zip compression tools
- CMake build system
- CUDA development toolkit
- Git version control
- Leptonica image processing
- Various build tools

**Recommendation Options:**
1. **Keep as-is** - Convenient for development
2. **Move to parent directory** - Outside GRID entirely
3. **Document and ignore** - Add to .gitignore if not already
4. **Remove if not needed** - If these tools aren't actively used

**Question for you:** Are these tools actively used for development, or can they be removed/moved?

---

## 📊 Verification Commands

Check what's at root level:
```bash
cd C:\Users\Coding\Desktop\GRID
ls -la | grep -E "\.py$"  # List Python files
ls -la | grep -E "\.md$"  # List documentation
```

Check TradingBOT:
```bash
cd C:\Users\Coding\Desktop\GRID\projects\TradingBOT
ls -la
```

---

**Conclusion:** After analysis, all files at GRID root are correctly placed for the stock data collection system. The only exception is the `tools/` directory which contains general development utilities not specific to either project.
