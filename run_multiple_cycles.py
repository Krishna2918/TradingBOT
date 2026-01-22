#!/usr/bin/env python3
"""Run multiple trading cycles and display results"""
import sys
sys.path.insert(0, '.')
from src.demo.demo_trading_engine import DemoTradingEngine
import time

# Initialize engine
engine = DemoTradingEngine()
print('='*60)
print('RUNNING 5 TRADING CYCLES')
print('='*60)

# Get initial state
prices = engine.current_prices or {}
initial_summary = engine.account.get_summary(prices)
print(f'\nSTARTING STATE:')
print(f'  Cash: ${initial_summary["cash"]:,.2f}')
print(f'  Portfolio Value: ${initial_summary["total_value"]:,.2f}')

for cycle in range(1, 6):
    print(f'\n{"="*60}')
    print(f'CYCLE {cycle}')
    print(f'{"="*60}')

    # Run cycle
    engine.run_demo_cycle()

    # Get state
    prices = engine.current_prices or {}
    summary = engine.account.get_summary(prices)

    print(f'  Cash: ${summary["cash"]:,.2f}')
    print(f'  Portfolio: ${summary["total_value"]:,.2f}')
    print(f'  Positions: {summary["num_positions"]}')
    print(f'  Trades: {summary["num_trades"]}')
    print(f'  P&L: ${summary["total_pnl"]:+,.2f} ({summary["total_return_pct"]:+.2f}%)')

    time.sleep(1)  # Brief pause between cycles

# Final summary
print('\n' + '='*60)
print('FINAL RESULTS')
print('='*60)
prices = engine.current_prices or {}
final = engine.account.get_summary(prices)
print(f'  Starting Capital: $100,000.00')
print(f'  Final Portfolio: ${final["total_value"]:,.2f}')
print(f'  Total P&L: ${final["total_pnl"]:+,.2f}')
print(f'  Return: {final["total_return_pct"]:+.2f}%')
print(f'  Total Trades: {final["num_trades"]}')
print(f'  Open Positions: {final["num_positions"]}')

# Show all positions
if engine.account.positions:
    print('\n' + '-'*60)
    print('OPEN POSITIONS')
    print('-'*60)
    for symbol, pos in engine.account.positions.items():
        current_price = prices.get(symbol, pos['avg_price'])
        pnl = (current_price - pos['avg_price']) * pos['quantity']
        pnl_pct = ((current_price - pos['avg_price']) / pos['avg_price']) * 100
        print(f'  {symbol}: {pos["quantity"]} @ ${pos["avg_price"]:.2f} -> ${current_price:.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)')

# Show all trades
if engine.account.trade_history:
    print('\n' + '-'*60)
    print('TRADE HISTORY')
    print('-'*60)
    for trade in engine.account.trade_history:
        pnl_str = f' | P&L: ${trade["pnl"]:+,.2f}' if trade.get('pnl') else ''
        print(f'  {trade["timestamp"].strftime("%H:%M:%S")} {trade["side"]:4} {trade["quantity"]:4} {trade["symbol"]:8} @ ${trade["price"]:8.2f}{pnl_str}')
        print(f'    Strategy: {trade.get("strategy", "N/A")}')

print('\n' + '='*60)
