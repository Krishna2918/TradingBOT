#!/usr/bin/env python3
"""Run a single trading cycle and display results"""
import sys
sys.path.insert(0, '.')
from src.demo.demo_trading_engine import DemoTradingEngine

# Initialize engine
engine = DemoTradingEngine()
print('='*60)
print('RUNNING TRADING CYCLE')
print('='*60)

# Get initial state
prices = engine.current_prices or {}
initial_summary = engine.account.get_summary(prices)
print(f'\nBEFORE CYCLE:')
print(f'  Cash: ${initial_summary["cash"]:,.2f}')
print(f'  Portfolio Value: ${initial_summary["total_value"]:,.2f}')
print(f'  Positions: {initial_summary["num_positions"]}')
print(f'  Total Trades: {initial_summary["num_trades"]}')

# Fetch prices first
print('\n' + '-'*60)
print('Fetching market prices...')
print('-'*60)
engine.current_prices = engine.fetch_real_prices()
print(f'Got prices for {len(engine.current_prices)} symbols:')
for symbol, price in list(engine.current_prices.items())[:5]:
    print(f'  {symbol}: ${price:.2f}')
if len(engine.current_prices) > 5:
    print(f'  ... and {len(engine.current_prices) - 5} more')

# Generate signals
print('\n' + '-'*60)
print('Generating signals...')
print('-'*60)
signals = engine.generate_ai_signals()
print(f'Generated {len(signals)} signals:')
for sig in signals[:5]:
    print(f'  {sig["action"]} {sig["symbol"]} @ ${sig["price"]:.2f}')
    print(f'    Strategy: {sig["strategy"]} | Confidence: {sig["confidence"]:.2f}')

# Execute signals
print('\n' + '-'*60)
print('Executing signals...')
print('-'*60)
for signal in signals:
    result = engine.execute_signal(signal)
    status = "SUCCESS" if result else "FAILED"
    print(f'  {signal["action"]} {signal["symbol"]}: {status}')

# Get updated state
prices = engine.current_prices or {}
final_summary = engine.account.get_summary(prices)

print('\n' + '='*60)
print('AFTER CYCLE')
print('='*60)
print(f'  Cash: ${final_summary["cash"]:,.2f}')
print(f'  Portfolio Value: ${final_summary["total_value"]:,.2f}')
print(f'  Positions: {final_summary["num_positions"]}')
print(f'  Total Trades: {final_summary["num_trades"]}')
print(f'  Total P&L: ${final_summary["total_pnl"]:,.2f}')
print(f'  Return: {final_summary["total_return_pct"]:.2f}%')

# Show positions if any
if engine.account.positions:
    print('\n' + '='*60)
    print('CURRENT POSITIONS')
    print('='*60)
    for symbol, pos in engine.account.positions.items():
        current_price = prices.get(symbol, pos['avg_price'])
        pnl = (current_price - pos['avg_price']) * pos['quantity']
        pnl_pct = ((current_price - pos['avg_price']) / pos['avg_price']) * 100
        print(f'  {symbol}: {pos["quantity"]} shares @ ${pos["avg_price"]:.2f}')
        print(f'    Current: ${current_price:.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)')

# Show recent trades
if engine.account.trade_history:
    print('\n' + '='*60)
    print('RECENT TRADES')
    print('='*60)
    for trade in engine.account.trade_history[-5:]:
        pnl_str = f' | P&L: ${trade["pnl"]:+,.2f}' if trade.get('pnl') else ''
        print(f'  {trade["timestamp"].strftime("%H:%M:%S")} | {trade["side"]:4} {trade["quantity"]} {trade["symbol"]} @ ${trade["price"]:.2f}{pnl_str}')

print('\n' + '='*60)
print('CYCLE COMPLETE')
print('='*60)
