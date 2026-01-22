# Configuration Migration Guide

## 🎯 Objective: Eliminate ALL Hardcoded Values

This guide shows how to migrate from hardcoded constants to the centralized configuration system.

---

## 📁 Configuration Files

### 1. **`config/trading_config.yaml`** - Main Configuration
All tunable parameters live here. Edit this file to change behavior.

### 2. **`config/settings.py`** - Configuration Loader
Pydantic-based configuration loader with validation.

### 3. **Environment Variables** (Optional)
Override YAML values using `.env` file.

---

## 🔄 Migration Examples

### ❌ BEFORE (Hardcoded)

```python
# interactive_trading_dashboard.py
max_position_pct = 0.05  # Hardcoded!
position_size = 0.02     # Hardcoded!
trading_interval = 5000  # Hardcoded!

# src/ai/autonomous_trading_ai.py
if rsi_val < 45 and price <= sma20_val * 1.05:  # Hardcoded thresholds!
    signal_type = 'BUY'
    score = 0.65 + (45 - rsi_val) / 70

# src/dashboard/portfolio.py
target_price = price * 1.02  # Hardcoded 2% profit target
```

### ✅ AFTER (Configuration-Driven)

```python
from config.settings import settings

# Get mode-specific values
max_position_pct = settings.get_max_position_pct(mode)
position_size = settings.get_position_size(mode)
trading_interval = settings.data.trading_interval_seconds * 1000  # Convert to ms

# Use configured thresholds
if rsi_val < settings.signals.rsi_oversold and price <= sma20_val * 1.05:
    signal_type = 'BUY'
    score = 0.65 + (settings.signals.rsi_oversold - rsi_val) / 70

# Use configured profit target
target_price = price * (1 + settings.risk.profit_target_pct)
```

---

## 📋 Migration Checklist

### Phase 1: Core Trading Logic ✅ PRIORITY
- [ ] **Capital & Position Sizing**
  - [ ] `max_position_pct` → `settings.get_max_position_pct(mode)`
  - [ ] `position_size` → `settings.get_position_size(mode)`
  - [ ] Min capital → `settings.capital.min_capital`
  
- [ ] **Risk Management**
  - [ ] Kill switch thresholds → `settings.risk.max_drawdown_pct`
  - [ ] Stop loss → `settings.risk.stop_loss_pct`
  - [ ] Profit target → `settings.risk.profit_target_pct`
  
- [ ] **Signal Thresholds**
  - [ ] RSI levels → `settings.signals.rsi_oversold`, `.rsi_overbought`
  - [ ] Sentiment cutoffs → `settings.signals.sentiment_*`
  - [ ] AI confidence → `settings.signals.min_ai_model_confidence`
  - [ ] ChatGPT weights → `settings.signals.chatgpt_weight`

### Phase 2: Data & Intervals
- [ ] **Fetch Intervals**
  - [ ] Trading interval → `settings.data.trading_interval_seconds`
  - [ ] Chart interval → `settings.data.chart_update_interval_seconds`
  - [ ] Log monitor → `settings.data.log_monitor_interval_seconds`
  
- [ ] **Batch Sizes**
  - [ ] Questrade batch → `settings.data.questrade_batch_size`
  - [ ] Yahoo batch → `settings.data.yahoo_batch_size`
  - [ ] Yahoo delay → `settings.data.yahoo_delay_seconds`

### Phase 3: UI & Display
- [ ] **Branding**
  - [ ] App title → `settings.ui.app_title`
  - [ ] Mode badges → `settings.ui.demo_mode_badge`, `.live_mode_badge`
  
- [ ] **Charts**
  - [ ] Height → `settings.ui.chart_height`
  - [ ] Noise amplitude → `settings.ui.chart_noise_amplitude`
  
- [ ] **Tables**
  - [ ] Page sizes → `settings.ui.trades_table_page_size`

### Phase 4: Broker & Data Sources
- [ ] **Broker Selection**
  - [ ] Primary → `settings.broker.primary`
  - [ ] Fallback → `settings.broker.fallback`
  
- [ ] **Symbol Universe**
  - [ ] CANADIAN_STOCKS → `settings.market.primary_symbols`
  - [ ] Full universe → Import from `tsx_stock_universe.py` if `settings.market.use_full_universe`

---

## 🔧 How to Update a File

### Step 1: Add Import
```python
from config.settings import settings
```

### Step 2: Replace Hardcoded Values
Search for constants and replace with `settings.*`

### Step 3: Handle Mode-Specific Logic
```python
# Get current mode from state/context
mode = trading_state.get('mode', 'demo')

# Use mode-specific config
mode_config = settings.get_mode_config(mode)
if mode_config.aggressive_trading:
    # Apply aggressive strategy
    pass
```

### Step 4: Test
```python
# Verify configuration loads correctly
python -c "from config.settings import settings; print(settings.signals.rsi_oversold)"
```

---

## 📝 Files to Update (Priority Order)

### 🔴 Critical (Affects Trading Logic)
1. `src/ai/autonomous_trading_ai.py` - Signal thresholds, RSI, sentiment
2. `interactive_trading_dashboard.py` - Position sizing, intervals, kill switch
3. `src/dashboard/portfolio.py` - Profit targets, stop loss
4. `src/dashboard/services.py` - Fallback logic, sizing

### 🟡 Important (Affects UX)
5. `src/dashboard/sections.py` - UI copy, alerts, intervals
6. `src/dashboard/charts.py` - Chart settings, noise
7. `src/dashboard/ui_components.py` - Display settings

### 🟢 Optional (Cleanup)
8. `src/data_pipeline/comprehensive_data_pipeline.py` - Batch sizes, delays
9. Various utils and helpers

---

## 🧪 Testing Configuration Changes

### 1. Test Configuration Loading
```bash
python config/settings.py
```

### 2. Test Individual Values
```python
from config.settings import settings

# Should print 45
print(settings.signals.rsi_oversold)

# Should print 0.05 for demo, 0.03 for live
print(settings.get_max_position_pct('demo'))
print(settings.get_max_position_pct('live'))
```

### 3. Test YAML Override
Edit `config/trading_config.yaml`:
```yaml
signals:
  rsi_oversold: 30  # Change from 45 to 30
```

Reload and verify:
```python
from config.settings import Settings
new_settings = Settings.load_from_yaml()
print(new_settings.signals.rsi_oversold)  # Should print 30
```

---

## 🚨 Common Pitfalls

### ❌ Don't Do This
```python
# Hardcoding after migration!
max_pos = 0.05  # NO!
if rsi < 45:    # NO!
```

### ✅ Do This
```python
# Always use settings
max_pos = settings.get_max_position_pct(mode)
if rsi < settings.signals.rsi_oversold:
```

### ❌ Don't Cache Settings
```python
# Settings might change, don't cache!
RSI_OVERSOLD = settings.signals.rsi_oversold  # NO!
```

### ✅ Always Read Fresh
```python
# Read from settings each time
if rsi < settings.signals.rsi_oversold:  # YES!
```

---

## 📊 Benefits After Migration

### ✅ Tuning Made Easy
Change one value in `trading_config.yaml` → instant effect across entire system

### ✅ A/B Testing
```bash
cp trading_config.yaml trading_config_v1.yaml
# Edit v2 parameters
python interactive_trading_dashboard.py --config=trading_config_v2.yaml
```

### ✅ Environment-Specific Configs
```bash
# Development
export CONFIG_FILE=config/dev_config.yaml

# Production
export CONFIG_FILE=config/prod_config.yaml
```

### ✅ No More Magic Numbers
Every threshold, interval, and constant is documented and centralized

---

## 🎯 Next Steps

1. **Start Migration**: Begin with `src/ai/autonomous_trading_ai.py` (highest impact)
2. **Add Tests**: Create `tests/test_configuration.py` to validate settings
3. **Document Changes**: Update this guide as you find new hardcoded values
4. **Create Presets**: Add config files for different strategies (aggressive, conservative, etc.)

---

## 📞 Support

If you find a hardcoded value that's not documented here, add it to this guide!

**Pattern to follow:**
1. Identify the hardcoded value
2. Add it to `trading_config.yaml`
3. Add the field to appropriate config class in `settings.py`
4. Replace the hardcoded value in the source file
5. Update this migration guide

