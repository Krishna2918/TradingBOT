# 📊 Trading Bot Dashboard - Groww Style UI

A beautiful, real-time trading dashboard inspired by Groww's clean and intuitive design.

## 🎨 Features

### 1. **Portfolio Overview**
- Real-time portfolio value tracking
- Today's P&L with percentage changes
- Total trades count and win/loss ratio
- Active AI signals counter

### 2. **Performance Chart**
- 30-day portfolio value visualization
- Clean area chart with smooth animations
- Interactive hover information

### 3. **Holdings Table**
- Complete stock portfolio view
- Average price vs current price
- Real-time P&L calculation
- Percentage gains/losses

### 4. **AI Trading Signals Panel**
- Live AI-generated trading signals
- Confidence scores for each signal
- Buy/Sell recommendations
- Strategy attribution
- Execution status tracking

### 5. **Recent Trades**
- Complete trade history
- Strategy-wise categorization
- Real-time execution status
- P&L for each trade

### 6. **Strategy Performance**
- Visual performance comparison
- Returns percentage for each strategy
- Trade count per strategy

## 🚀 Quick Start

### Start the Dashboard

```bash
# Option 1: Direct Python
python start_dashboard.py

# Option 2: From source
python src/dashboard/app.py
```

### Access the Dashboard

Open your browser and navigate to:
```
http://localhost:8050
```

## 🎨 Design Philosophy

### Color Palette (Groww-inspired)
- **Primary Green**: `#00D09C` - Success, Buy signals
- **Danger Red**: `#EB5B3C` - Losses, Sell signals
- **Info Blue**: `#5367FE` - Information, Charts
- **Warning Orange**: `#FDB022` - Alerts, Pending actions
- **Background**: `#FAFAFA` - Clean light grey
- **Cards**: `#FFFFFF` - Pure white for clarity

### UI Components

#### Summary Cards
- Large, clear metrics
- Icon-based visual indicators
- Color-coded values
- Percentage changes

#### Tables
- Clean, hover-enabled rows
- Color-coded P&L values
- Badge-based status indicators
- Responsive design

#### Charts
- Interactive Plotly visualizations
- Smooth animations
- Clean grid lines
- Hover information

## 📱 Responsive Design

The dashboard is fully responsive and works on:
- 💻 Desktop (optimal experience)
- 📱 Tablets
- 📱 Mobile devices

## 🔄 Real-time Updates

The dashboard auto-refreshes every 5 seconds:
- Live portfolio values
- New trading signals
- Updated P&L
- Fresh trade entries

## 🎯 Dashboard Sections

### 1. Top Summary Bar
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Total Value │ Today's P&L │Trades Today │Active Signals│
│   $82,450   │   $737.50   │     18      │      4       │
│  +3.06% 📈  │   +0.92%    │ 14W / 4L    │ 5 strategies │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### 2. Portfolio Chart
30-day performance visualization with smooth area fill

### 3. Holdings Grid
| Stock | Qty | Avg Price | LTP | P&L |
|-------|-----|-----------|-----|-----|
| RY.TO | 50  | $125.50   |$132.80| +$365 (5.8%) |

### 4. AI Signals Feed
```
┌─────────────────────────────────────┐
│ 🟢 BUY  SHOP.TO    Confidence: 87%  │
│ Strategy: AI/ML Patterns            │
│ Reason: Bullish pattern detected    │
│ Status: ⚡ Active                    │
└─────────────────────────────────────┘
```

### 5. Recent Trades Table
| Time  | Stock | Side | Qty | Price | P&L | Status |
|-------|-------|------|-----|-------|-----|--------|
| 14:35 | RY.TO | BUY  | 50  |$125.50|+$450|✅ Done |

### 6. Strategy Performance
Bar chart showing returns for each of the 5 strategies

## 🛠️ Customization

### Change Colors
Edit `COLORS` dictionary in `src/dashboard/app.py`:

```python
COLORS = {
    'primary': '#00D09C',    # Your primary color
    'danger': '#EB5B3C',     # Your danger color
    'background': '#FAFAFA', # Background color
    # ... more colors
}
```

### Modify Refresh Rate
Change the interval in milliseconds:

```python
dcc.Interval(
    id='interval-component',
    interval=5000,  # 5 seconds (5000ms)
    n_intervals=0
)
```

### Add New Metrics
Create new summary cards following the pattern in `create_summary_cards()`.

## 📊 Data Integration

### Connect Real Trading Data

Replace mock data functions with real data:

```python
# Instead of generate_mock_trades()
def get_real_trades():
    # Connect to your trading system
    trades = trading_system.get_recent_trades()
    return pd.DataFrame(trades)
```

### Connect to Trading Bot

```python
# Import your trading bot modules
from risk_management.capital_allocation import CapitalAllocator
from strategies.strategy_manager import StrategyManager

# Initialize in callbacks
capital_allocator = CapitalAllocator("config/risk_config.yaml")
strategy_manager = StrategyManager("config/strategy_config.yaml")

# Use real data in callbacks
@app.callback(...)
def update_portfolio():
    state = capital_allocator.get_capital_state()
    return f"${state.total_capital:,.2f}"
```

## 🔒 Security Notes

### Production Deployment

1. **Disable Debug Mode**:
   ```python
   app.run_server(debug=False, host='0.0.0.0', port=8050)
   ```

2. **Add Authentication**:
   ```python
   import dash_auth
   
   VALID_USERNAME_PASSWORD_PAIRS = {
       'admin': 'password123'
   }
   
   auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
   ```

3. **Use HTTPS**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8050 start_dashboard:app --certfile=cert.pem --keyfile=key.pem
   ```

## 🚀 Deployment

### Local Development
```bash
python start_dashboard.py
```

### Production with Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:8050 'src.dashboard.app:app.server'
```

### Docker Deployment
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8050
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8050", "src.dashboard.app:app.server"]
```

## 📱 Mobile Experience

The dashboard is optimized for mobile:
- Responsive cards that stack vertically
- Touch-friendly buttons and links
- Optimized table scrolling
- Readable fonts on small screens

## 🎨 UI Screenshots

### Desktop View
- Full-width layout
- Side-by-side panels
- Large charts
- Multiple columns

### Mobile View
- Stacked cards
- Collapsible sections
- Horizontal scrolling tables
- Touch-optimized buttons

## 🔧 Troubleshooting

### Dashboard Won't Start
```bash
# Check if port is in use
netstat -ano | findstr :8050

# Use a different port
python start_dashboard.py --port 8051
```

### Import Errors
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep dash
```

### Slow Performance
- Reduce refresh interval
- Limit data points in charts
- Use data pagination for large tables

## 📚 Technologies Used

- **Dash**: Python framework for web apps
- **Plotly**: Interactive charts
- **Bootstrap**: Responsive CSS framework
- **Font Awesome**: Icons
- **Pandas**: Data manipulation

## 🎯 Future Enhancements

- [ ] WebSocket for instant updates
- [ ] Advanced filtering and sorting
- [ ] Export data to CSV/Excel
- [ ] Dark mode toggle
- [ ] Multi-user support
- [ ] Mobile app integration
- [ ] Advanced charting tools
- [ ] Trade execution from UI

## 💡 Tips

1. **Keep it running**: Use `screen` or `tmux` on Linux
2. **Monitor performance**: Check RAM and CPU usage
3. **Backup data**: Regular database backups
4. **Update regularly**: Keep dependencies updated
5. **Test mobile**: Always test on mobile devices

## 🆘 Support

For issues or questions:
1. Check logs in `logs/dashboard.log`
2. Review error messages in console
3. Check browser console for JavaScript errors
4. Verify all dependencies are installed

---

**Built with ❤️ inspired by Groww's amazing UI design**

🎨 Clean • 🚀 Fast • 📱 Responsive • 🔒 Secure

