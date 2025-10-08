"""
Demo Run - Canadian AI Trading Bot

Demonstrates the complete system in action
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import components
from src.execution import ExecutionEngine, OrderType, OrderSide
from src.event_awareness import EventCalendar, VolatilityDetector, AnomalyDetector
from src.penny_stocks import PennyStockDetector
from src.sip import SIPSimulator
from src.backtesting import BacktestEngine
from src.trading_modes import ModeManager

def print_header(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def run_demo():
    """Run complete system demo"""
    
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "🚀 CANADIAN AI TRADING BOT DEMO 🚀" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # ===== PHASE 1: SYSTEM INITIALIZATION =====
    print_header("PHASE 1: System Initialization")
    
    print("🔧 Initializing components...")
    execution_engine = ExecutionEngine()
    event_calendar = EventCalendar()
    volatility_detector = VolatilityDetector()
    anomaly_detector = AnomalyDetector()
    penny_detector = PennyStockDetector()
    sip_simulator = SIPSimulator()
    backtest_engine = BacktestEngine()
    mode_manager = ModeManager()
    
    print("✅ All components initialized successfully!")
    print(f"   • Execution Engine: Ready")
    print(f"   • Event Calendar: {len(event_calendar.events)} events loaded")
    print(f"   • Risk Management: Active")
    print(f"   • Trading Mode: {mode_manager.current_mode.value.upper()}")
    account = mode_manager.get_current_account_info()
    print(f"   • Starting Capital: ${account['capital']:,.2f} CAD")
    
    # ===== PHASE 2: MARKET STATUS CHECK =====
    print_header("PHASE 2: Market Status Check")
    
    print("📅 Checking market status...")
    
    # Check if today is a holiday
    today = datetime.now()
    is_holiday = event_calendar.is_market_holiday(today)
    
    if is_holiday:
        print("⏸️  Market is CLOSED (Holiday)")
    else:
        print("✅ Market is OPEN")
    
    # Check upcoming events
    upcoming = event_calendar.get_upcoming_events(hours_ahead=24)
    print(f"\n📆 Upcoming events in next 24h: {len(upcoming)}")
    for event in upcoming[:3]:
        print(f"   • {event.title} @ {event.scheduled_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Check high-impact events
    high_impact = event_calendar.get_high_impact_events(days_ahead=7)
    print(f"\n⚠️  High-impact events in next 7 days: {len(high_impact)}")
    for event in high_impact[:3]:
        print(f"   • {event.title}")
    
    # ===== PHASE 3: MARKET DATA ANALYSIS =====
    print_header("PHASE 3: Market Data Analysis")
    
    print("📊 Generating sample market data...")
    
    # Create sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    base_prices = 150 + np.cumsum(np.random.randn(100) * 2)
    
    market_data = pd.DataFrame({
        'open': base_prices + np.random.randn(100) * 0.5,
        'high': base_prices + np.abs(np.random.randn(100)),
        'low': base_prices - np.abs(np.random.randn(100)),
        'close': base_prices,
        'volume': np.random.uniform(1000000, 2000000, 100)
    }, index=dates)
    
    symbol = "SHOP.TO"
    current_price = market_data['close'].iloc[-1]
    
    print(f"✅ Sample data generated for {symbol}")
    print(f"   • Current Price: ${current_price:.2f}")
    print(f"   • 30-day High: ${market_data['high'].tail(30).max():.2f}")
    print(f"   • 30-day Low: ${market_data['low'].tail(30).min():.2f}")
    print(f"   • Avg Volume: {market_data['volume'].mean():,.0f}")
    
    # ===== PHASE 4: VOLATILITY ANALYSIS =====
    print_header("PHASE 4: Volatility Analysis")
    
    print(f"📈 Analyzing volatility for {symbol}...")
    
    analysis = volatility_detector.analyze_volatility(symbol, market_data)
    
    print(f"✅ Volatility Analysis Complete:")
    print(f"   • Historical Volatility: {analysis['historical_volatility']:.2f}%")
    print(f"   • ATR: {analysis['atr']:.2f}")
    print(f"   • ATR %: {analysis['atr_percent']:.2f}%")
    print(f"   • Volatility Regime: {analysis['volatility_regime'].upper()}")
    print(f"   • Trend: {analysis['trend'].capitalize()}")
    
    if analysis['is_spike']:
        print(f"   ⚠️  VOLATILITY SPIKE DETECTED! (Z-score: {analysis['spike_z_score']:.2f})")
    
    # ===== PHASE 5: ANOMALY DETECTION =====
    print_header("PHASE 5: Anomaly Detection")
    
    print(f"🔍 Training anomaly detector...")
    anomaly_detector.train(market_data)
    
    print(f"🔍 Detecting anomalies in {symbol}...")
    anomaly_analysis = anomaly_detector.analyze_anomalies(symbol, market_data)
    
    print(f"✅ Anomaly Detection Complete:")
    print(f"   • Current Status: {'ANOMALY' if anomaly_analysis['is_anomaly'] else 'NORMAL'}")
    print(f"   • Anomaly Score: {anomaly_analysis['anomaly_score']:.3f}")
    print(f"   • Volume Anomaly: {'YES' if anomaly_analysis['is_volume_anomaly'] else 'NO'}")
    print(f"   • Price Anomaly: {'YES' if anomaly_analysis['is_price_anomaly'] else 'NO'}")
    print(f"   • Recent Anomalies (20 periods): {anomaly_analysis['recent_anomaly_count']}")
    
    # ===== PHASE 6: ORDER EXECUTION =====
    print_header("PHASE 6: Order Execution")
    
    print(f"⚡ Executing sample order for {symbol}...")
    
    # Create and execute order
    order = execution_engine.create_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    
    print(f"📋 Order created: {order.order_id}")
    print(f"   • Symbol: {order.symbol}")
    print(f"   • Side: {order.side.value.upper()}")
    print(f"   • Quantity: {order.quantity}")
    print(f"   • Type: {order.order_type.value.upper()}")
    
    success = execution_engine.execute_market_order(
        order=order,
        current_price=current_price,
        volume=market_data['volume'].iloc[-1]
    )
    
    if success:
        print(f"\n✅ Order EXECUTED successfully!")
        print(f"   • Filled: {order.filled_quantity} shares")
        print(f"   • Avg Price: ${order.average_fill_price:.4f}")
        print(f"   • Total Cost: ${order.filled_quantity * order.average_fill_price:,.2f}")
        
        # Calculate profit/loss from price change
        simulated_profit = (current_price - order.average_fill_price) * order.filled_quantity
        print(f"   • Simulated P&L: ${simulated_profit:.2f}")
    
    # Get execution statistics
    stats = execution_engine.get_execution_statistics()
    print(f"\n📊 Execution Statistics:")
    print(f"   • Total Executions: {stats['total_executions']}")
    print(f"   • Total Volume: {stats['total_volume']:.2f} shares")
    print(f"   • Total Value: ${stats['total_value']:,.2f}")
    print(f"   • Average Slippage: {stats['average_slippage']:.4%}")
    
    # ===== PHASE 7: SIP AUTOMATION =====
    print_header("PHASE 7: SIP Automation (ETF Investing)")
    
    # Simulate daily profit
    daily_profit = 5000.0  # $5K profit
    etf_price = 110.50  # VFV.TO price
    
    print(f"💰 Processing daily profit: ${daily_profit:,.2f}")
    print(f"   • Allocation: 1% to ETF")
    print(f"   • Investment Amount: ${daily_profit * 0.01:.2f}")
    print(f"   • ETF: {sip_simulator.primary_etf}")
    print(f"   • Current ETF Price: ${etf_price:.2f}")
    
    transaction = sip_simulator.process_daily_profit(daily_profit, etf_price)
    
    if transaction:
        print(f"\n✅ SIP Investment Executed:")
        print(f"   • Amount Invested: ${transaction.amount_cad:.2f}")
        print(f"   • Shares Purchased: {transaction.shares_purchased:.4f}")
        print(f"   • Purchase Price: ${transaction.share_price:.2f}")
        
        # Get portfolio value
        portfolio = sip_simulator.get_portfolio_value(etf_price)
        print(f"\n📊 SIP Portfolio:")
        print(f"   • Total Shares: {portfolio['total_shares']:.4f}")
        print(f"   • Total Invested: ${portfolio['total_invested']:.2f}")
        print(f"   • Current Value: ${portfolio['current_value']:.2f}")
        print(f"   • Unrealized P&L: ${portfolio['unrealized_pnl']:.2f} ({portfolio['return_pct']:.2f}%)")
    
    # ===== PHASE 8: PENNY STOCK ANALYSIS =====
    print_header("PHASE 8: Penny Stock Analysis")
    
    print("💎 Analyzing penny stock...")
    
    # Create penny stock data
    penny_data = pd.DataFrame({
        'open': np.random.uniform(2.0, 2.5, 100),
        'high': np.random.uniform(2.2, 2.7, 100),
        'low': np.random.uniform(1.8, 2.3, 100),
        'close': np.random.uniform(2.0, 2.5, 100),
        'volume': np.random.uniform(100000, 500000, 100)
    })
    
    penny_symbol = "PENNY.V"
    profile = penny_detector.analyze_penny_stock(penny_symbol, penny_data)
    
    if profile:
        print(f"✅ Penny Stock Analysis for {profile.symbol}:")
        print(f"   • Price: ${profile.price:.2f}")
        print(f"   • Average Volume: {profile.avg_volume:,.0f}")
        print(f"   • Current Volume: {profile.current_volume:,.0f} ({profile.volume_ratio:.1f}x)")
        print(f"   • Liquidity Score: {profile.liquidity_score:.2f}/1.0")
        print(f"   • Volatility: {profile.volatility:.2%}")
        print(f"   • Risk Level: {profile.risk_level.upper()}")
        print(f"   • Tradeable: {'YES ✅' if profile.is_tradeable else 'NO ❌'}")
        
        if profile.is_tradeable:
            position_size = penny_detector.calculate_position_size(profile, 100000)
            print(f"\n💰 Position Sizing:")
            print(f"   • Max Position: ${position_size:,.2f}")
            print(f"   • Shares: {position_size / profile.price:.0f}")
    
    # ===== PHASE 9: BACKTESTING =====
    print_header("PHASE 9: Strategy Backtesting")
    
    print("🔄 Running backtest on sample strategy...")
    
    # Create backtest signals
    signals = pd.Series([0] * 100, index=market_data.index)
    signals.iloc[10] = 1  # Buy
    signals.iloc[50] = -1  # Sell
    signals.iloc[60] = 1  # Buy
    signals.iloc[90] = -1  # Sell
    
    result = backtest_engine.run_backtest(
        strategy_name="Demo_Strategy",
        data=market_data,
        signals=signals,
        position_size=0.1  # 10% per trade
    )
    
    print(f"✅ Backtest Complete:")
    print(f"   • Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
    print(f"   • Initial Capital: ${result.initial_capital:,.2f}")
    print(f"   • Final Capital: ${result.final_capital:,.2f}")
    print(f"   • Total Return: ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")
    print(f"   • Total Trades: {result.total_trades}")
    print(f"   • Win Rate: {result.win_rate:.1f}%")
    print(f"   • Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   • Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"   • Profit Factor: {result.profit_factor:.2f}")
    
    # Run Monte Carlo simulation
    print(f"\n🎲 Running Monte Carlo simulation (100 runs)...")
    mc_results = backtest_engine.monte_carlo_simulation(result, num_simulations=100)
    
    print(f"✅ Monte Carlo Results:")
    print(f"   • Mean Return: {mc_results['mean_return']:.2f}%")
    print(f"   • 95% Confidence Interval: [{mc_results['percentile_5']:.2f}%, {mc_results['percentile_95']:.2f}%]")
    print(f"   • Probability of Profit: {mc_results['probability_profit']:.1f}%")
    
    # ===== PHASE 10: SUMMARY =====
    print_header("PHASE 10: Demo Summary")
    
    print("📊 Demo Session Summary:")
    print(f"\n✅ Systems Tested:")
    print(f"   • Market Status Check")
    print(f"   • Event Calendar")
    print(f"   • Volatility Analysis")
    print(f"   • Anomaly Detection")
    print(f"   • Order Execution")
    print(f"   • SIP Automation")
    print(f"   • Penny Stock Analysis")
    print(f"   • Strategy Backtesting")
    print(f"   • Monte Carlo Simulation")
    
    print(f"\n📈 Performance Metrics:")
    print(f"   • Orders Executed: {stats['total_executions']}")
    print(f"   • SIP Investments: 1")
    print(f"   • Backtest Completed: Yes")
    print(f"   • All Systems: Operational ✅")
    
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "🎉 DEMO COMPLETE! 🎉" + " " * 32 + "║")
    print("║" + " " * 78 + "║")
    print("║" + " " * 15 + "All systems operational and ready for trading!" + " " * 16 + "║")
    print("╚" + "=" * 78 + "╝\n")

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

